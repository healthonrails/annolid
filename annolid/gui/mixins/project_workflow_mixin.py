from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional

from qtpy import QtCore, QtWidgets

from annolid.domain import (
    DEFAULT_SCHEMA_FILENAME,
    default_behavior_spec,
    load_behavior_spec,
    save_behavior_spec,
)
from annolid.core.behavior.catalog import (
    behavior_catalog_entries,
    delete_behavior_definition,
    find_behavior,
    normalize_behavior_code,
    update_behavior_definition,
    upsert_behavior_definition,
)
from annolid.services.project_workspace import (
    DEFAULT_VIDEO_EXTENSIONS,
    export_project_bundle,
    prune_missing_recent_entries,
    resolve_project_dir,
    upsert_recent_project_entries,
)
from annolid.utils.logger import logger


class _ProjectExportWorker(QtCore.QObject):
    """Background worker exporting an Annolid project folder or atomic ZIP archive."""

    progressChanged = QtCore.Signal(int, int, str)  # done, total, rel_path
    finished = QtCore.Signal(bool, str)  # ok, message

    def __init__(
        self,
        *,
        project_dir: Path,
        export_path: Path,
        config: Optional[Dict[str, Any]] = None,
        include_data: bool = True,
        include_models: bool = False,
        as_zip: bool = False,
    ) -> None:
        super().__init__()
        self._project_dir = Path(project_dir)
        self._export_path = Path(export_path)
        self._config = dict(config) if config else None
        self._include_data = bool(include_data)
        self._include_models = bool(include_models)
        self._as_zip = bool(as_zip)
        self._cancel_event = threading.Event()

    @QtCore.Slot()
    def cancel(self) -> None:
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    @QtCore.Slot()
    def run(self) -> None:
        ok = False
        message = "Export failed."
        try:
            ok, message = export_project_bundle(
                self._project_dir,
                self._export_path,
                config=self._config,
                include_data=self._include_data,
                include_models=self._include_models,
                as_zip=self._as_zip,
                progress_callback=lambda done, total, rel: self.progressChanged.emit(
                    int(done), int(total), str(rel)
                ),
                cancel_check=self._cancel_event.is_set,
            )
        except Exception as exc:
            ok = False
            message = f"Failed to export project: {exc}"
        finally:
            self.finished.emit(bool(ok), str(message))
            app = QtCore.QCoreApplication.instance()
            if app is not None and self.thread() != app.thread():
                try:
                    self.moveToThread(app.thread())
                except Exception:
                    pass


class _ProjectExportBridge(QtCore.QObject):
    """GUI-thread QObject receiver ensuring worker signals never run on the worker thread."""

    def __init__(
        self,
        *,
        on_progress,
        on_finished,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._on_progress = on_progress
        self._on_finished = on_finished

    @QtCore.Slot(int, int, str)
    def handle_progress(self, done: int, total: int, label: str) -> None:
        self._on_progress(done, total, label)

    @QtCore.Slot(bool, str)
    def handle_finished(self, ok: bool, message: str) -> None:
        self._on_finished(ok, message)


class ProjectWorkflowMixin:
    """Project schema, wizard workflow, and DINO tool proxy helpers."""

    def _behavior_catalog_save_path(self) -> Path:
        target_path = getattr(self, "project_schema_path", None)
        if target_path is not None:
            return Path(target_path)
        project_root = None
        try:
            project_controller = getattr(self, "project_controller", None)
            if project_controller is not None:
                project_root = project_controller.get_current_project_path()
        except Exception:
            project_root = None
        if project_root is not None:
            return Path(project_root) / DEFAULT_SCHEMA_FILENAME
        video_file = str(getattr(self, "video_file", "") or "").strip()
        if video_file:
            return Path(video_file).with_suffix("") / DEFAULT_SCHEMA_FILENAME
        return Path.cwd() / DEFAULT_SCHEMA_FILENAME

    def _refresh_behavior_catalog_views(self) -> None:
        schema = getattr(self, "project_schema", None)
        behavior_controller = getattr(self, "behavior_controller", None)
        if behavior_controller is not None:
            try:
                behavior_controller.configure_from_schema(schema)
            except Exception:
                pass
        try:
            self._populate_behavior_controls_from_schema(schema)
        except Exception:
            pass
        try:
            self._sync_behavior_flags_from_schema(schema)
        except Exception:
            pass
        try:
            self._update_modifier_controls_for_behavior(
                getattr(self, "event_type", None)
            )
        except Exception:
            pass
        timeline_panel = getattr(self, "timeline_panel", None)
        if timeline_panel is not None:
            try:
                timeline_panel.refresh_behavior_catalog()
            except Exception:
                pass
        try:
            refresh_overlay = getattr(self, "_refresh_behavior_overlay", None)
            if callable(refresh_overlay):
                refresh_overlay()
        except Exception:
            pass

    def behavior_catalog_entries(self) -> List[Dict[str, Any]]:
        return behavior_catalog_entries(getattr(self, "project_schema", None))

    def _behavior_catalog_entry_for_code(self, code: str) -> Dict[str, Any]:
        target = find_behavior(getattr(self, "project_schema", None), code)
        if target is None:
            return {}
        return {
            "code": target.code,
            "name": target.name,
            "description": target.description or "",
            "category_id": target.category_id or "",
            "modifier_ids": list(target.modifier_ids or []),
            "key_binding": target.key_binding or "",
            "is_state": bool(target.is_state),
            "exclusive_with": list(target.exclusive_with or []),
            "color": getattr(target, "color", None) or "",
        }

    def sync_behavior_catalog_from_ui(self, *, save: bool = True) -> Dict[str, Any]:
        schema = getattr(self, "project_schema", None)
        if schema is None:
            return {"ok": False, "error": "No project schema is loaded."}

        candidate_names: List[str] = []
        flag_widget = getattr(self, "flag_widget", None)
        if flag_widget is not None:
            try:
                candidate_names.extend(list(flag_widget.behavior_names()))
            except Exception:
                pass
        timeline_panel = getattr(self, "timeline_panel", None)
        if timeline_panel is not None:
            try:
                candidate_names.extend(
                    list(timeline_panel._timeline_behavior_catalog())
                )
            except Exception:
                pass
        behavior_controller = getattr(self, "behavior_controller", None)
        if behavior_controller is not None:
            try:
                candidate_names.extend(
                    list(getattr(behavior_controller, "behavior_names", []) or [])
                )
            except Exception:
                pass
        try:
            candidate_names.extend(
                [
                    str(behavior.code)
                    for behavior in getattr(schema, "behaviors", [])
                    if getattr(behavior, "code", "")
                ]
            )
        except Exception:
            pass

        unique_codes: List[str] = []
        for candidate in candidate_names:
            normalized = normalize_behavior_code(candidate)
            if normalized and normalized not in unique_codes:
                unique_codes.append(normalized)

        updated_schema = schema
        created_codes: List[str] = []
        for code in unique_codes:
            if find_behavior(updated_schema, code) is not None:
                continue
            updated_schema, ok, _message = upsert_behavior_definition(
                updated_schema,
                code=code,
                name=code,
                replace_existing=False,
            )
            if ok:
                created_codes.append(code)

        if created_codes:
            self.project_schema = updated_schema
            self._refresh_behavior_catalog_views()

        payload: Dict[str, Any] = {
            "ok": True,
            "created_count": len(created_codes),
            "created_codes": created_codes,
            "behavior_catalog": self.behavior_catalog_entries(),
        }
        if save and created_codes:
            save_result = self.save_behavior_catalog()
            payload["saved"] = bool(save_result.get("ok", False))
            if not save_result.get("ok", False):
                payload["save_error"] = save_result.get("error")
            else:
                payload["path"] = save_result.get("path")
        else:
            payload["saved"] = False
        return payload

    def list_behavior_catalog(self) -> Dict[str, Any]:
        schema = getattr(self, "project_schema", None)
        entries = self.behavior_catalog_entries()
        return {
            "ok": True,
            "count": len(entries),
            "behavior_catalog": entries,
            "project_schema_path": str(getattr(self, "project_schema_path", "") or ""),
            "behavior_codes": [
                str(item.get("code") or "").strip()
                for item in entries
                if str(item.get("code") or "").strip()
            ],
            "behavior_names": [
                str(item.get("name") or "").strip()
                for item in entries
                if str(item.get("name") or "").strip()
            ],
            "project_loaded": bool(schema is not None),
        }

    def save_behavior_catalog(self) -> Dict[str, Any]:
        schema = getattr(self, "project_schema", None)
        if schema is None:
            return {"ok": False, "error": "No project schema is loaded."}
        target_path = self._behavior_catalog_save_path()
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            save_behavior_spec(schema, target_path)
            self.project_schema_path = target_path
            self._refresh_behavior_catalog_views()
            return {
                "ok": True,
                "path": str(target_path),
                "rows": len(schema.behaviors),
                "behavior_catalog": self.behavior_catalog_entries(),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def create_behavior_catalog_item(
        self,
        *,
        code: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        category_id: Optional[str] = None,
        modifier_ids: Optional[List[str]] = None,
        key_binding: Optional[str] = None,
        is_state: Optional[bool] = None,
        exclusive_with: Optional[List[str]] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        schema = getattr(self, "project_schema", None)
        if schema is None:
            return {"ok": False, "error": "No project schema is loaded."}
        updated_schema, ok, message = upsert_behavior_definition(
            schema,
            code=code,
            name=name,
            description=description,
            category_id=category_id,
            modifier_ids=modifier_ids,
            key_binding=key_binding,
            is_state=is_state,
            exclusive_with=exclusive_with,
        )
        if not ok:
            return {"ok": False, "error": message}
        self.project_schema = updated_schema
        self._refresh_behavior_catalog_views()
        payload = {
            "ok": True,
            "message": message,
            "behavior": self._behavior_catalog_entry_for_code(code),
        }
        if save:
            save_result = self.save_behavior_catalog()
            payload["saved"] = bool(save_result.get("ok", False))
            if not save_result.get("ok", False):
                payload["save_error"] = save_result.get("error")
            else:
                payload["path"] = save_result.get("path")
        return payload

    def update_behavior_catalog_item(
        self,
        *,
        code: str,
        updates: Dict[str, Any],
        save: bool = True,
    ) -> Dict[str, Any]:
        schema = getattr(self, "project_schema", None)
        if schema is None:
            return {"ok": False, "error": "No project schema is loaded."}
        updated_schema, ok, message = update_behavior_definition(
            schema,
            code=code,
            updates=updates,
        )
        if not ok:
            return {"ok": False, "error": message}
        self.project_schema = updated_schema
        self._refresh_behavior_catalog_views()
        payload = {
            "ok": True,
            "message": message,
            "behavior": self._behavior_catalog_entry_for_code(code),
        }
        if save:
            save_result = self.save_behavior_catalog()
            payload["saved"] = bool(save_result.get("ok", False))
            if not save_result.get("ok", False):
                payload["save_error"] = save_result.get("error")
            else:
                payload["path"] = save_result.get("path")
        return payload

    def delete_behavior_catalog_item(
        self,
        *,
        code: str,
        save: bool = True,
    ) -> Dict[str, Any]:
        schema = getattr(self, "project_schema", None)
        if schema is None:
            return {"ok": False, "error": "No project schema is loaded."}
        target_behavior = find_behavior(schema, code)
        if target_behavior is None:
            return {"ok": False, "error": f"Behavior '{code}' not found."}
        updated_schema, ok, message = delete_behavior_definition(schema, code=code)
        if not ok:
            return {"ok": False, "error": message}
        self.project_schema = updated_schema
        try:
            if hasattr(self, "pinned_flags"):
                pinned_flags = dict(getattr(self, "pinned_flags", {}) or {})
                if target_behavior.code in pinned_flags:
                    pinned_flags.pop(target_behavior.code, None)
                    self.loadFlags(pinned_flags)
        except Exception:
            pass
        self._refresh_behavior_catalog_views()
        payload: Dict[str, Any] = {
            "ok": True,
            "message": message,
            "code": target_behavior.code,
        }
        if save:
            save_result = self.save_behavior_catalog()
            payload["saved"] = bool(save_result.get("ok", False))
            if not save_result.get("ok", False):
                payload["save_error"] = save_result.get("error")
            else:
                payload["path"] = save_result.get("path")
        return payload

    def open_project_schema_dialog(self) -> None:
        """Open the schema editor dialog and persist changes."""
        from annolid.gui.widgets.project_dialog import ProjectDialog

        schema = self.project_schema or default_behavior_spec()
        dialog = ProjectDialog(schema, parent=self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            new_schema = dialog.get_schema()
            self.project_schema = new_schema
            self._refresh_behavior_catalog_views()

            try:
                save_result = self.save_behavior_catalog()
            except OSError as exc:
                QtWidgets.QMessageBox.critical(
                    self,
                    self.tr("Project Schema"),
                    self.tr("Failed to save schema:\n%s") % exc,
                )
            else:
                target_path = str(save_result.get("path") or self.project_schema_path)
                self.statusBar().showMessage(
                    self.tr("Project schema saved to %s") % Path(target_path).name,
                    4000,
                )

            imported = getattr(dialog, "imported_boris_project", None)
            if imported is not None and imported.observations:
                reply = QtWidgets.QMessageBox.question(
                    self,
                    self.tr("Load BORIS Observations"),
                    self.tr(
                        "The imported BORIS project contains %d observation(s). "
                        "Load an observation's behavior events into the timeline now?"
                    )
                    % len(imported.observations),
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.Yes,
                )
                if reply == QtWidgets.QMessageBox.Yes:
                    chosen_obs = imported.observations[0]
                    if len(imported.observations) > 1:
                        obs_ids = [o.observation_id for o in imported.observations]
                        choice, ok = QtWidgets.QInputDialog.getItem(
                            self,
                            self.tr("Select BORIS Observation"),
                            self.tr("Observation:"),
                            obs_ids,
                            0,
                            False,
                        )
                        if not ok:
                            return
                        chosen_obs = next(
                            (
                                o
                                for o in imported.observations
                                if o.observation_id == choice
                            ),
                            imported.observations[0],
                        )
                    fps = (
                        float(self.fps)
                        if getattr(self, "fps", None) and self.fps > 0
                        else float(chosen_obs.fps or 29.97)
                    )
                    self.behavior_controller.load_events_from_rows(
                        chosen_obs.events,
                        time_to_frame=lambda t: int(round(float(t) * fps)),
                    )
                    if getattr(self, "seekbar", None) is not None:
                        self.behavior_controller.attach_slider(self.seekbar)
                    refresh_log = getattr(self, "_refresh_behavior_log", None)
                    if callable(refresh_log):
                        refresh_log()
                    refresh_badges = getattr(
                        self, "_refresh_ethogram_active_badges", None
                    )
                    if callable(refresh_badges):
                        refresh_badges()

    def open_new_project_wizard(self) -> None:
        """Open the new project wizard for guided project creation."""
        from annolid.gui.widgets.project_wizard import ProjectWizard

        wizard = ProjectWizard(parent=self)
        wizard.project_created.connect(
            lambda path, schema, videos, _w=wizard: self._on_project_created(
                path,
                schema,
                videos,
                open_first_video=bool(_w.should_open_video()),
            )
        )
        wizard.exec_()

    def _register_videos_in_manager(self, videos: List[str]) -> None:
        video_mgr = getattr(self, "video_manager_widget", None)
        if video_mgr is None:
            return
        add_fn = getattr(video_mgr, "add_video", None)
        if not callable(add_fn):
            return
        for video in videos or []:
            path_str = str(video or "").strip()
            if path_str and Path(path_str).exists():
                try:
                    add_fn(path_str)
                except Exception:
                    logger.debug("Failed adding video %s to manager", path_str)

    def get_recent_projects(
        self, *, prune_missing: bool = True
    ) -> List[Dict[str, str]]:
        """Return persisted recent projects as normalized dicts."""
        settings = getattr(self, "settings", None)
        if settings is None:
            return []
        raw = settings.value("recentProjects", "[]")
        entries: list[Any] = []
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    entries = parsed
            except json.JSONDecodeError:
                entries = []
        elif isinstance(raw, list):
            entries = raw
        if prune_missing:
            cleaned = prune_missing_recent_entries(entries)
            if len(cleaned) != len(entries):
                try:
                    settings.setValue("recentProjects", json.dumps(cleaned))
                except Exception:
                    pass
            return cleaned
        return [
            {
                "name": str(e.get("name") or "") if isinstance(e, dict) else "",
                "path": str(e.get("path") or e or ""),
                "last_opened": str(e.get("last_opened") or "")
                if isinstance(e, dict)
                else "",
            }
            for e in entries
            if (isinstance(e, dict) and e.get("path"))
            or (isinstance(e, str) and e.strip())
        ]

    def _record_recent_project(self, project_path: Path | str, name: str = "") -> None:
        settings = getattr(self, "settings", None)
        if settings is None:
            return
        updated = upsert_recent_project_entries(
            self.get_recent_projects(prune_missing=False),
            project_path,
            name=name,
            max_entries=12,
        )
        try:
            settings.setValue("recentProjects", json.dumps(updated))
        except Exception:
            logger.debug("Failed persisting recentProjects", exc_info=True)

    def clear_recent_history(self) -> None:
        """Clear both recent projects and recent files from memory and QSettings."""
        if hasattr(self, "recentFiles"):
            self.recentFiles = []
        settings = getattr(self, "settings", None)
        if settings is not None:
            try:
                settings.setValue("recentFiles", [])
                settings.setValue("recentProjects", "[]")
            except Exception:
                pass
        hub = getattr(self, "_project_hub_dialog", None)
        if hub is not None:
            try:
                hub.refresh()
            except Exception:
                pass
        status_bar = self.statusBar() if hasattr(self, "statusBar") else None
        if status_bar is not None:
            status_bar.showMessage(self.tr("Cleared recent projects and files"), 2500)

    def _on_project_created(
        self,
        project_path: Path,
        schema,
        videos: list,
        *,
        open_first_video: bool = True,
    ) -> None:
        """Handle project creation from the wizard."""
        project_dir = resolve_project_dir(project_path)
        controller = getattr(self, "project_controller", None)
        if controller is not None:
            try:
                controller.load_project(project_dir)
            except Exception:
                logger.debug(
                    "ProjectController failed to load newly created project %s",
                    project_dir,
                    exc_info=True,
                )

        self.project_schema = schema
        self.project_schema_path = project_dir / DEFAULT_SCHEMA_FILENAME
        self._refresh_behavior_catalog_views()
        configure_pose = getattr(self, "_configure_pose_schema_from_project", None)
        if callable(configure_pose):
            configure_pose(schema, project_dir=project_dir)

        self._record_recent_project(project_dir, name=project_dir.name)
        video_list = [str(v) for v in (videos or []) if str(v or "").strip()]
        self._register_videos_in_manager(video_list)

        if video_list and open_first_video and hasattr(self, "openVideo"):
            self.openVideo(from_video_list=True, video_path=video_list[0])

        hub = getattr(self, "_project_hub_dialog", None)
        if hub is not None:
            try:
                hub.refresh()
            except Exception:
                pass

        self.statusBar().showMessage(
            self.tr("Project '%s' created successfully") % project_dir.name,
            5000,
        )

    def open_project(
        self,
        project_path: Optional[str | Path] = None,
        *,
        open_first_video: bool = True,
    ) -> bool:
        """Open an existing Annolid project folder (or `project.annolid.json`)."""
        target = project_path
        if not target:
            start_dir = str(getattr(self, "lastOpenDir", None) or Path.home())
            chosen = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                self.tr("Open Annolid Project Folder"),
                start_dir,
            )
            if not chosen:
                return False
            target = chosen

        project_dir = resolve_project_dir(target)
        controller = getattr(self, "project_controller", None)
        if controller is None:
            return False

        loaded = controller.load_project(project_dir)
        if not loaded:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Open Project"),
                self.tr("Could not open Annolid project at:\n%s") % str(project_dir),
            )
            return False

        info = controller.get_project_info() or {}
        schema_file = project_dir / DEFAULT_SCHEMA_FILENAME
        if schema_file.is_file():
            try:
                schema, resolved_schema_path = load_behavior_spec(path=schema_file)
                self.project_schema = schema
                self.project_schema_path = resolved_schema_path or schema_file
                self._refresh_behavior_catalog_views()
                configure_pose = getattr(
                    self, "_configure_pose_schema_from_project", None
                )
                if callable(configure_pose):
                    configure_pose(schema, project_dir=project_dir)
            except Exception as exc:
                logger.debug("Failed loading behavior schema from project: %s", exc)

        self.lastOpenDir = str(project_dir)
        project_name = str(info.get("name") or project_dir.name)
        self._record_recent_project(project_dir, name=project_name)

        project_videos = [
            v for v in controller.get_project_videos() if Path(v).is_file()
        ]
        self._register_videos_in_manager(project_videos)
        if open_first_video and project_videos and hasattr(self, "openVideo"):
            self.openVideo(from_video_list=True, video_path=project_videos[0])
        elif not project_videos and hasattr(self, "importDirImages"):
            images_dir = controller.get_project_directory("images")
            annotations_dir = controller.get_project_directory("annotations")
            browse_dir = None
            if (
                images_dir is not None
                and images_dir.is_dir()
                and any(images_dir.iterdir())
            ):
                browse_dir = images_dir
            elif (
                annotations_dir is not None
                and annotations_dir.is_dir()
                and any(annotations_dir.iterdir())
            ):
                browse_dir = annotations_dir
            if browse_dir is not None:
                self.importDirImages(str(browse_dir), load=False)

        hub = getattr(self, "_project_hub_dialog", None)
        if hub is not None:
            try:
                hub.refresh()
            except Exception:
                pass

        status_bar = self.statusBar() if hasattr(self, "statusBar") else None
        if status_bar is not None:
            status_bar.showMessage(
                self.tr("Opened project '%s'") % project_name,
                4000,
            )
        return True

    def add_videos_to_active_project(
        self,
        video_paths: Optional[List[str]] = None,
        *,
        copy_into_project: bool = False,
    ) -> List[str]:
        """Add video files to the active project and Video Manager."""
        controller = getattr(self, "project_controller", None)
        if controller is None or not controller.is_project_loaded():
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Add Videos"),
                self.tr("Open or create an Annolid project first."),
            )
            return []

        selected = list(video_paths or [])
        if not selected:
            files, _ = QtWidgets.QFileDialog.getOpenFileNames(
                self,
                self.tr("Add Videos to Project"),
                str(controller.get_current_project_path() or Path.home()),
                "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)",
            )
            selected = list(files or [])
        if not selected:
            return []

        ok, registered = controller.register_project_videos(
            selected, copy_into_project=copy_into_project
        )
        if not ok:
            return []

        self._register_videos_in_manager(registered)
        hub = getattr(self, "_project_hub_dialog", None)
        if hub is not None:
            try:
                hub.refresh()
            except Exception:
                pass
        status_bar = self.statusBar() if hasattr(self, "statusBar") else None
        if status_bar is not None:
            status_bar.showMessage(
                self.tr("Added %d video(s) to project") % len(registered),
                3500,
            )
        return registered

    def close_active_project(self) -> None:
        """Close the currently loaded project and refresh Project Hub views."""
        controller = getattr(self, "project_controller", None)
        if controller is None:
            return
        active_path = controller.get_current_project_path()
        controller.close_project()
        schema_path = getattr(self, "project_schema_path", None)
        if active_path is not None and schema_path is not None:
            try:
                Path(schema_path).resolve().relative_to(Path(active_path).resolve())
            except ValueError:
                pass
            else:
                self.project_schema_path = None
        hub = getattr(self, "_project_hub_dialog", None)
        if hub is not None:
            try:
                hub.refresh()
            except Exception:
                pass
        status_bar = self.statusBar() if hasattr(self, "statusBar") else None
        if status_bar is not None:
            status_bar.showMessage(self.tr("Closed active project"), 3000)

    def _cancel_active_project_export(self, *, wait_ms: int = 2000) -> None:
        """Cooperatively cancel and join any running project export worker."""
        worker = getattr(self, "_project_export_worker", None)
        thread = getattr(self, "_project_export_thread", None)
        progress = getattr(self, "_project_export_progress", None)
        if worker is not None:
            try:
                worker.cancel()
            except RuntimeError:
                pass
        if progress is not None:
            try:
                progress.canceled.disconnect()
            except Exception:
                pass
            try:
                progress.close()
                progress.deleteLater()
            except RuntimeError:
                pass
            self._project_export_progress = None
        if thread is not None:
            try:
                if thread.isRunning() and wait_ms > 0:
                    thread.quit()
                    thread.wait(wait_ms)
            except RuntimeError:
                pass
            if wait_ms > 0:
                try:
                    thread.deleteLater()
                except RuntimeError:
                    pass
                self._project_export_thread = None
        if worker is not None and wait_ms > 0:
            try:
                worker.deleteLater()
            except RuntimeError:
                pass
            self._project_export_worker = None
        bridge = getattr(self, "_project_export_bridge", None)
        if bridge is not None and wait_ms > 0:
            try:
                bridge.deleteLater()
            except RuntimeError:
                pass
            self._project_export_bridge = None

    def export_current_project(
        self,
        *,
        target_path: Optional[str | Path] = None,
        include_data: bool = True,
        include_models: bool = False,
        async_export: bool = True,
    ) -> bool:
        """Export the active Annolid project to a `.zip` archive or folder.

        Runs on a cancellable background `QThread` with a progress dialog by default
        so multi-GB video exports never block the UI thread.
        """
        controller = getattr(self, "project_controller", None)
        if controller is None or not controller.is_project_loaded():
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Export Project"),
                self.tr("Open or create an Annolid project first."),
            )
            return False

        existing_thread = getattr(self, "_project_export_thread", None)
        if existing_thread is not None:
            try:
                if existing_thread.isRunning():
                    QtWidgets.QMessageBox.information(
                        self,
                        self.tr("Export Project"),
                        self.tr("A project export is already in progress."),
                    )
                    return False
            except RuntimeError:
                self._project_export_thread = None

        info = controller.get_project_info() or {}
        project_dir = controller.get_current_project_path()
        if project_dir is None:
            return False

        if target_path is None:
            project_name = str(info.get("name") or "annolid_project")
            default_zip = str(Path.home() / f"{project_name}_bundle.zip")
            chosen, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                self.tr("Export Project Bundle"),
                default_zip,
                "ZIP Archive (*.zip);;Folder (*)",
            )
            if not chosen:
                return False
            target = Path(chosen)
        else:
            target = Path(target_path)

        as_zip = target.suffix.lower() == ".zip"
        if not async_export:
            ok = controller.export_project(
                target,
                include_data=include_data,
                include_models=include_models,
                as_zip=as_zip,
            )
            if ok:
                status_bar = self.statusBar() if hasattr(self, "statusBar") else None
                if status_bar is not None:
                    status_bar.showMessage(
                        self.tr("Exported project to %s") % target.name,
                        4000,
                    )
            return bool(ok)

        config_snapshot: Dict[str, Any] = {
            "directories": dict(info.get("directories") or {})
        }
        worker = _ProjectExportWorker(
            project_dir=Path(project_dir),
            export_path=target,
            config=config_snapshot,
            include_data=include_data,
            include_models=include_models,
            as_zip=as_zip,
        )
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)

        progress = QtWidgets.QProgressDialog(
            self.tr("Preparing project export…"),
            self.tr("Cancel"),
            0,
            0,
            self,
        )
        progress.setWindowTitle(self.tr("Export Project Bundle"))
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setMinimumDuration(0)
        progress.canceled.connect(worker.cancel, QtCore.Qt.DirectConnection)

        def _on_progress(done: int, total: int, label: str) -> None:
            if getattr(self, "_project_export_progress", None) is not progress:
                return
            try:
                if total > 0:
                    if progress.maximum() != total:
                        progress.setRange(0, total)
                    progress.setValue(min(done, total))
                    progress.setLabelText(
                        self.tr("Exporting (%d/%d): %s") % (done, total, label)
                    )
                else:
                    progress.setLabelText(label or self.tr("Exporting project…"))
            except RuntimeError:
                pass

        def _on_finished(ok: bool, message: str) -> None:
            cancelled = worker.is_cancelled()
            if getattr(self, "_project_export_progress", None) is progress:
                self._project_export_progress = None
            try:
                progress.canceled.disconnect()
            except Exception:
                pass
            try:
                progress.close()
                progress.deleteLater()
            except RuntimeError:
                pass

            if getattr(self, "_project_export_thread", None) is thread:
                self._project_export_thread = None
            if getattr(self, "_project_export_worker", None) is worker:
                self._project_export_worker = None
            if getattr(self, "_project_export_bridge", None) is bridge:
                self._project_export_bridge = None
            try:
                thread.quit()
                thread.wait(2000)
                thread.deleteLater()
            except RuntimeError:
                pass
            try:
                worker.deleteLater()
            except RuntimeError:
                pass
            try:
                bridge.deleteLater()
            except RuntimeError:
                pass

            status_bar = self.statusBar() if hasattr(self, "statusBar") else None
            if ok:
                if status_bar is not None:
                    status_bar.showMessage(
                        self.tr("Exported project to %s") % target.name,
                        4000,
                    )
            elif cancelled:
                if status_bar is not None:
                    status_bar.showMessage(self.tr("Project export cancelled"), 3500)
            else:
                if status_bar is not None:
                    status_bar.showMessage(
                        self.tr("Project export failed: %s") % message, 5000
                    )
                if controller is not None and hasattr(controller, "project_error"):
                    try:
                        controller.project_error.emit(message)
                    except Exception:
                        pass
                QtWidgets.QMessageBox.warning(
                    self,
                    self.tr("Export Project"),
                    message,
                )

            hub = getattr(self, "_project_hub_dialog", None)
            if hub is not None:
                try:
                    hub.refresh()
                except Exception:
                    pass

        bridge = _ProjectExportBridge(
            on_progress=_on_progress,
            on_finished=_on_finished,
            parent=self,
        )
        worker.progressChanged.connect(
            bridge.handle_progress, QtCore.Qt.QueuedConnection
        )
        worker.finished.connect(bridge.handle_finished, QtCore.Qt.QueuedConnection)
        thread.started.connect(worker.run)

        self._project_export_bridge = bridge
        self._project_export_worker = worker
        self._project_export_thread = thread
        self._project_export_progress = progress
        progress.show()
        thread.start()
        return True

    def _open_audited_file(self, path_str: str) -> bool:
        """Open a file selected from FileAuditWidget, FileAuditDialog, or recent history."""
        raw = str(path_str or "").strip()
        if not raw:
            return False
        target = Path(raw).expanduser()
        if not target.exists():
            status_bar = self.statusBar() if hasattr(self, "statusBar") else None
            if status_bar is not None:
                status_bar.showMessage(self.tr("Missing file: %s") % str(target), 4000)
            return False

        opened = False
        if target.suffix.lower() in DEFAULT_VIDEO_EXTENSIONS and hasattr(
            self, "openVideo"
        ):
            self.openVideo(from_video_list=True, video_path=str(target))
            opened = True
        elif hasattr(self, "loadFile"):
            result = self.loadFile(str(target))
            opened = True if result is None else bool(result)

        if opened:
            try:
                self.raise_()
                self.activateWindow()
            except Exception:
                pass
        return opened

    def open_annotation_audit_dialog(
        self,
        initial_dir: Optional[str | Path] = None,
        *,
        auto_scan: bool = True,
    ):
        """Open (or raise) the project-scoped Annotation Audit dialog."""
        from annolid.gui.widgets.file_audit_widget import FileAuditDialog

        controller = getattr(self, "project_controller", None)
        project_info = (
            controller.get_project_info()
            if controller is not None and controller.is_project_loaded()
            else None
        )
        project_config = (
            {"directories": dict(project_info.get("directories") or {})}
            if project_info
            else None
        )

        resolved_dir: Optional[Path] = None
        if initial_dir:
            resolved_dir = Path(initial_dir).expanduser()
        elif project_info and project_info.get("path"):
            resolved_dir = Path(str(project_info["path"]))
        elif getattr(self, "video_results_folder", None):
            resolved_dir = Path(str(self.video_results_folder))
        elif getattr(self, "annotation_dir", None):
            resolved_dir = Path(str(self.annotation_dir))
        elif getattr(self, "lastOpenDir", None):
            resolved_dir = Path(str(self.lastOpenDir))

        dialog = getattr(self, "_file_audit_dialog", None)
        if dialog is None:
            dialog = FileAuditDialog(
                initial_dir=resolved_dir,
                project_config=project_config,
                auto_scan=auto_scan,
                parent=self,
            )
            dialog.fileOpenRequested.connect(self._open_audited_file)
            dialog.finished.connect(
                lambda *_: setattr(self, "_file_audit_dialog", None)
            )
            dialog.destroyed.connect(
                lambda *_: setattr(self, "_file_audit_dialog", None)
            )
            self._file_audit_dialog = dialog
        elif resolved_dir is not None and resolved_dir.is_dir() and auto_scan:
            dialog.scan_directory(
                resolved_dir,
                project_config=project_config,
            )

        dialog.show()
        dialog.setWindowState(dialog.windowState() & ~QtCore.Qt.WindowMinimized)
        dialog.raise_()
        dialog.activateWindow()
        return dialog

    def _dispatch_workflow_step(self, step_id: str) -> None:
        step = str(step_id or "").strip().lower()
        controller = getattr(self, "project_controller", None)
        has_project = bool(controller is not None and controller.is_project_loaded())
        if step == "project":
            if has_project:
                self.open_project_schema_dialog()
            else:
                self.open_new_project_wizard()
            return
        if step == "collect":
            if has_project:
                self.add_videos_to_active_project()
            elif hasattr(self, "openDirDialog"):
                self.openDirDialog()
            return
        if step == "annotate":
            videos = controller.get_project_videos() if has_project else []
            existing = [v for v in videos if Path(v).is_file()]
            if existing and hasattr(self, "openVideo"):
                self.openVideo(from_video_list=True, video_path=existing[0])
            elif hasattr(self, "openVideo"):
                self.openVideo()
            return
        if step == "train":
            self.open_training_wizard()
            return
        if step == "inference":
            self.open_inference_wizard()
            return
        if step == "analyze":
            if hasattr(self, "open_analysis_studio"):
                self.open_analysis_studio()
            else:
                self.open_export_dataset_wizard()

    def open_project_hub(self):
        """Open (or raise) the Project & Workflow Hub dialog."""
        from annolid.gui.widgets.project_hub_dialog import ProjectHubDialog

        dialog = getattr(self, "_project_hub_dialog", None)
        if dialog is None:
            dialog = ProjectHubDialog(
                project_controller=getattr(self, "project_controller", None),
                recent_projects_getter=lambda: self.get_recent_projects(
                    prune_missing=True
                ),
                recent_files_getter=lambda: list(
                    getattr(self, "recentFiles", []) or []
                ),
                parent=self,
            )
            dialog.newProjectRequested.connect(self.open_new_project_wizard)
            dialog.openProjectRequested.connect(
                lambda path: self.open_project(path or None)
            )
            dialog.closeProjectRequested.connect(self.close_active_project)
            dialog.exportProjectRequested.connect(self.export_current_project)
            dialog.addVideosRequested.connect(
                lambda: self.add_videos_to_active_project()
            )
            dialog.editSchemaRequested.connect(self.open_project_schema_dialog)
            dialog.auditAnnotationsRequested.connect(self.open_annotation_audit_dialog)
            dialog.openVideoRequested.connect(
                lambda path: self._open_audited_file(path)
                if path
                else (self.openVideo() if hasattr(self, "openVideo") else None)
            )
            dialog.workflowStepRequested.connect(self._dispatch_workflow_step)
            dialog.finished.connect(
                lambda *_: setattr(self, "_project_hub_dialog", None)
            )
            dialog.destroyed.connect(
                lambda *_: setattr(self, "_project_hub_dialog", None)
            )
            self._project_hub_dialog = dialog

        dialog.refresh()
        dialog.show()
        dialog.setWindowState(dialog.windowState() & ~QtCore.Qt.WindowMinimized)
        dialog.raise_()
        dialog.activateWindow()
        return dialog

    def open_export_dataset_wizard(self) -> None:
        """Open the dataset export wizard."""
        from annolid.gui.widgets.dataset_wizard import DatasetExportWizard

        source_dir = None
        if self.video_file:
            potential_dir = Path(self.video_file).with_suffix("")
            if potential_dir.is_dir():
                source_dir = str(potential_dir)

        wizard = DatasetExportWizard(source_dir=source_dir, parent=self)
        wizard.export_complete.connect(self._on_dataset_exported)
        if hasattr(wizard, "file_open_requested"):
            wizard.file_open_requested.connect(self._open_audited_file)
        wizard.exec_()

    def _on_dataset_exported(self, output_path: Path, format_type: str) -> None:
        """Handle dataset export completion."""
        self.statusBar().showMessage(
            self.tr("Dataset exported to %s (%s format)")
            % (output_path.name, format_type.upper()),
            5000,
        )

    def open_training_wizard(self) -> None:
        """Open the training wizard for guided model training."""
        from annolid.gui.widgets.training_wizard import TrainingWizard

        dataset_path = None
        if self.video_file:
            potential_yaml = Path(self.video_file).with_suffix("") / "data.yaml"
            if potential_yaml.exists():
                dataset_path = str(potential_yaml)

        wizard = TrainingWizard(dataset_path=dataset_path, parent=self)
        wizard.training_requested.connect(self._on_training_requested)
        if hasattr(self, "open_colab_studio"):
            wizard.colab_requested.connect(
                lambda cfg: self.open_colab_studio(
                    initial_tab="train", training_config=cfg
                )
            )
        wizard.exec_()

    def _on_training_requested(self, config: dict) -> None:
        """Handle training request from the wizard."""
        backend = config.get("backend", "yolo")

        if backend == "yolo":
            self._start_yolo_training_from_wizard(config)
        elif backend == "dino_kpseg":
            self._start_dino_training_from_wizard(config)
        else:
            self._start_maskrcnn_training_from_wizard(config)

        open_tb = bool(config.get("open_tensorboard"))
        if config.get("open_dashboard") or open_tb:
            self._open_training_dashboard(
                initial_tab="tensorboard" if open_tb else "monitor",
                start_tensorboard=open_tb,
            )

    def _start_yolo_training_from_wizard(self, config: dict) -> None:
        """Start YOLO training with wizard configuration."""
        try:
            from annolid.gui.widgets.yolo_training_ui_shared import (
                default_yolo_plots_enabled,
            )

            manager = (
                self.ensure_yolo_training_manager()
                if hasattr(self, "ensure_yolo_training_manager")
                else getattr(self, "yolo_training_manager", None)
            )
            if manager is None:
                return

            selected_model = str(config.get("model") or "yolo11n-seg.pt").strip()
            selected_candidate = Path(selected_model).expanduser()
            if selected_candidate.is_file():
                model_path: Optional[str] = str(selected_candidate.resolve())
                yolo_model_file = selected_candidate.name
            else:
                model_path = None
                yolo_model_file = selected_model

            train_overrides: dict = {}
            for key in (
                "lr0",
                "lrf",
                "weight_decay",
                "patience",
                "close_mosaic",
                "optimizer",
                "cos_lr",
                "cache",
            ):
                if key in config and config[key] is not None:
                    train_overrides[key] = config[key]
            freeze_val = config.get("freeze")
            if freeze_val is not None and int(freeze_val) > 0:
                train_overrides["freeze"] = int(freeze_val)

            plots_val = (
                bool(config["plots"])
                if "plots" in config and config["plots"] is not None
                else default_yolo_plots_enabled()
            )
            resume_val = bool(config.get("resume", False))

            started = manager.start_training(
                yolo_model_file=yolo_model_file,
                model_path=model_path,
                data_config_path=config.get("dataset_path"),
                epochs=int(config.get("epochs", 100)),
                image_size=int(config.get("imgsz", 640)),
                batch_size=int(config.get("batch", 8)),
                device=config.get("device", ""),
                plots=plots_val,
                train_overrides=train_overrides or None,
                out_dir=config.get("output_dir"),
                resume=resume_val,
            )
            if started:
                self.statusBar().showMessage(self.tr("YOLO training started"), 3000)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                self.tr("Training Error"),
                self.tr("Failed to start training: %s") % str(e),
            )

    def _start_dino_training_from_wizard(self, config: dict) -> None:
        """Start DINO KPSEG training with wizard configuration."""
        try:
            from annolid.segmentation.dino_kpseg import defaults as dino_defaults

            manager = (
                self.ensure_dino_kpseg_training_manager()
                if hasattr(self, "ensure_dino_kpseg_training_manager")
                else getattr(self, "dino_kpseg_training_manager", None)
            )
            if manager is None:
                return
            manager.start_training(
                data_config_path=config.get("dataset_path"),
                data_format=str(config.get("data_format", "auto") or "auto"),
                out_dir=config.get("output_dir"),
                model_name=config.get("model"),
                short_side=config.get("short_side", dino_defaults.SHORT_SIDE),
                layers=config.get("layers", dino_defaults.LAYERS),
                radius_px=config.get("radius_px", dino_defaults.RADIUS_PX),
                hidden_dim=config.get("hidden_dim", dino_defaults.HIDDEN_DIM),
                lr=config.get("lr", dino_defaults.LR),
                epochs=config.get("epochs", dino_defaults.EPOCHS),
                batch_size=config.get("batch", dino_defaults.BATCH),
                threshold=config.get("threshold", dino_defaults.THRESHOLD),
                bce_type=config.get("bce_type", dino_defaults.BCE_TYPE),
                focal_alpha=config.get("focal_alpha", dino_defaults.FOCAL_ALPHA),
                focal_gamma=config.get("focal_gamma", dino_defaults.FOCAL_GAMMA),
                device=config.get("device", ""),
                cache_features=config.get("cache_features", True),
                head_type=config.get("head_type", dino_defaults.HEAD_TYPE),
                attn_heads=config.get("attn_heads", dino_defaults.ATTN_HEADS),
                attn_layers=config.get("attn_layers", dino_defaults.ATTN_LAYERS),
                obj_loss_weight=config.get(
                    "obj_loss_weight", dino_defaults.OBJ_LOSS_WEIGHT
                ),
                box_loss_weight=config.get(
                    "box_loss_weight", dino_defaults.BOX_LOSS_WEIGHT
                ),
                inst_loss_weight=config.get(
                    "inst_loss_weight", dino_defaults.INST_LOSS_WEIGHT
                ),
                multitask_aux_warmup_epochs=config.get(
                    "multitask_aux_warmup_epochs",
                    dino_defaults.MULTITASK_AUX_WARMUP_EPOCHS,
                ),
                freeze_bn=config.get("freeze_bn", None),
            )
            self.statusBar().showMessage(self.tr("DINO KPSEG training started"), 3000)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                self.tr("Training Error"),
                self.tr("Failed to start training: %s") % str(e),
            )

    def _start_maskrcnn_training_from_wizard(self, config: dict) -> None:
        """Start MaskRCNN training with wizard configuration."""
        self.statusBar().showMessage(
            self.tr("MaskRCNN training: use the legacy Train Models dialog"), 5000
        )

    def _open_training_dashboard(
        self,
        *,
        initial_tab: Optional[str] = None,
        run_dir: Optional[str | Path] = None,
        start_tensorboard: bool = False,
    ) -> None:
        """Open the unified Training Monitor & Visualization Dashboard."""
        try:
            open_fn = getattr(self, "open_training_dashboard", None)
            if callable(open_fn):
                open_fn(
                    initial_tab=initial_tab,
                    run_dir=run_dir,
                    start_tensorboard=start_tensorboard,
                )
                return

            from annolid.gui.widgets.training_dashboard import TrainingDashboardDialog

            dialog = getattr(self, "_training_dashboard_dialog", None)
            if dialog is None:
                yolo_manager = (
                    self.ensure_yolo_training_manager()
                    if hasattr(self, "ensure_yolo_training_manager")
                    else getattr(self, "yolo_training_manager", None)
                )
                dino_manager = (
                    self.ensure_dino_kpseg_training_manager()
                    if hasattr(self, "ensure_dino_kpseg_training_manager")
                    else getattr(self, "dino_kpseg_training_manager", None)
                )
                dialog = TrainingDashboardDialog(settings=self.settings, parent=self)
                if yolo_manager is not None:
                    dialog.dashboard.register_training_manager(yolo_manager)
                if dino_manager is not None:
                    dialog.dashboard.register_training_manager(dino_manager)
                dialog.finished.connect(
                    lambda *_: setattr(self, "_training_dashboard_dialog", None)
                )
                dialog.destroyed.connect(
                    lambda *_: setattr(self, "_training_dashboard_dialog", None)
                )
                self._training_dashboard_dialog = dialog

            dialog.show()
            dialog.setWindowState(dialog.windowState() & ~QtCore.Qt.WindowMinimized)
            dialog.raise_()
            dialog.activateWindow()
            QtWidgets.QApplication.processEvents()
            if run_dir is not None:
                dialog.dashboard.select_run_dir(run_dir)
            if initial_tab:
                dialog.dashboard.select_tab(initial_tab)
            if start_tensorboard:
                dialog.dashboard.start_tensorboard_for_current_scope(switch_to_tab=True)
        except Exception as e:
            logger.debug(f"Could not open training dashboard: {e}")

    def open_labeling_progress_dashboard(self) -> None:
        """Open the labeling progress dashboard (project stats + gamification)."""
        try:
            from annolid.gui.widgets.labeling_progress_dashboard import (
                LabelingProgressDashboardDialog,
            )

            dialog = getattr(self, "_labeling_progress_dashboard_dialog", None)
            if dialog is None:
                project_root = self.project_controller.get_current_project_path()
                annotation_root = None
                try:
                    annotation_root = self.project_controller.get_project_directory(
                        "annotations"
                    )
                except Exception:
                    annotation_root = None
                if project_root is None and getattr(self, "video_results_folder", None):
                    project_root = Path(self.video_results_folder).resolve()
                if annotation_root is None and getattr(self, "annotation_dir", None):
                    annotation_root = Path(self.annotation_dir).resolve()
                if project_root is None:
                    project_root = Path.cwd()
                if annotation_root is None:
                    annotation_root = project_root

                dialog = LabelingProgressDashboardDialog(
                    initial_project_root=project_root,
                    initial_annotation_root=annotation_root,
                    parent=self,
                )
                dialog.finished.connect(
                    lambda *_: setattr(
                        self, "_labeling_progress_dashboard_dialog", None
                    )
                )
                dialog.destroyed.connect(
                    lambda *_: setattr(
                        self, "_labeling_progress_dashboard_dialog", None
                    )
                )
                self._labeling_progress_dashboard_dialog = dialog

            dialog.show()
            dialog.setWindowState(dialog.windowState() & ~QtCore.Qt.WindowMinimized)
            dialog.raise_()
            dialog.activateWindow()
            QtWidgets.QApplication.processEvents()
        except Exception as exc:
            logger.debug("Could not open labeling progress dashboard: %s", exc)

    def open_inference_wizard(self, model_path: Optional[str] = None) -> None:
        """Open the inference wizard for running model predictions."""
        from annolid.gui.widgets.inference_wizard import InferenceWizard
        from annolid.yolo import discover_yolo_checkpoints

        video_path = self.video_file if self.video_file else None
        recent_models: list[str] = []
        seen: set[str] = set()

        def _append_model(candidate: Optional[str]) -> None:
            raw = str(candidate or "").strip()
            if not raw or raw in seen:
                return
            seen.add(raw)
            recent_models.append(raw)

        if model_path:
            _append_model(model_path)

        ai_mgr = getattr(self, "ai_model_manager", None)
        if ai_mgr is not None:
            try:
                current_weight = ai_mgr.get_current_weight()
                if current_weight and Path(current_weight).expanduser().is_file():
                    if not model_path:
                        model_path = str(Path(current_weight).expanduser().resolve())
                    _append_model(current_weight)
            except Exception:
                pass
            try:
                for cfg in ai_mgr.all_model_configs:
                    wf = str(getattr(cfg, "weight_file", "") or "").strip()
                    if wf and Path(wf).expanduser().is_file():
                        _append_model(wf)
            except Exception:
                pass

        try:
            for ckpt in discover_yolo_checkpoints(limit=12):
                _append_model(str(ckpt.path))
        except Exception:
            pass

        wizard = InferenceWizard(
            video_path=video_path,
            model_path=model_path,
            recent_models=recent_models,
            parent=self,
        )
        if hasattr(self, "open_colab_studio"):
            wizard.colab_requested.connect(
                lambda cfg: self.open_colab_studio(
                    initial_tab="predict", inference_config=cfg
                )
            )
        wizard.exec_()

    def open_polygon_classifier_workbench(self) -> None:
        """Open the polygon classifier dataset/training/inference workbench."""
        from annolid.gui.widgets.polygon_classifier_workbench import (
            PolygonClassifierWorkbench,
        )

        source_dir = None
        if self.video_file:
            candidate = Path(self.video_file).with_suffix("")
            if candidate.is_dir():
                source_dir = str(candidate)

        dialog = getattr(self, "_polygon_classifier_workbench", None)
        if dialog is None:
            dialog = PolygonClassifierWorkbench(
                default_source_dir=source_dir,
                parent=self,
            )
            dialog.finished.connect(
                lambda *_: setattr(self, "_polygon_classifier_workbench", None)
            )
            dialog.destroyed.connect(
                lambda *_: setattr(self, "_polygon_classifier_workbench", None)
            )
            self._polygon_classifier_workbench = dialog

        dialog.show()
        dialog.setWindowState(dialog.windowState() & ~QtCore.Qt.WindowMinimized)
        dialog.raise_()
        dialog.activateWindow()

    def _load_labels(self, labels_csv_file):
        """Load labels from the given CSV file."""
        import pandas as pd

        self._df = pd.read_csv(labels_csv_file)
        self._df.rename(columns={"Unnamed: 0": "frame_number"}, inplace=True)

    def _toggle_patch_similarity_tool(self, checked=False):
        controller = getattr(self, "dino_controller", None)
        if controller is None and hasattr(self, "ensure_dino_controller"):
            try:
                controller = self.ensure_dino_controller()
            except Exception:
                controller = None
        if controller is not None:
            controller.toggle_patch_similarity(checked)

    def _toggle_pca_map_tool(self, checked=False):
        controller = getattr(self, "dino_controller", None)
        if controller is None and hasattr(self, "ensure_dino_controller"):
            try:
                controller = self.ensure_dino_controller()
            except Exception:
                controller = None
        if controller is not None:
            controller.toggle_pca_map(checked)

    def _deactivate_pca_map(self):
        controller = getattr(self, "dino_controller", None)
        if controller is not None:
            controller.deactivate_pca_map()

    def _request_pca_map(self) -> None:
        controller = getattr(self, "dino_controller", None)
        if controller is None and hasattr(self, "ensure_dino_controller"):
            try:
                controller = self.ensure_dino_controller()
            except Exception:
                controller = None
        if controller is not None:
            controller.request_pca_map()

    def _open_patch_similarity_settings(self):
        controller = getattr(self, "dino_controller", None)
        if controller is None and hasattr(self, "ensure_dino_controller"):
            try:
                controller = self.ensure_dino_controller()
            except Exception:
                controller = None
        if controller is not None:
            controller.open_patch_similarity_settings()
