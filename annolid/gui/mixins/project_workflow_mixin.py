from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from qtpy import QtCore, QtWidgets

from annolid.domain import (
    DEFAULT_SCHEMA_FILENAME,
    default_behavior_spec,
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
from annolid.segmentation.dino_kpseg import defaults as dino_defaults
from annolid.gui.widgets import (
    LabelingProgressDashboardDialog,
    TrainingDashboardDialog,
)
from annolid.utils.logger import logger


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

    def open_new_project_wizard(self) -> None:
        """Open the new project wizard for guided project creation."""
        from annolid.gui.widgets.project_wizard import ProjectWizard

        wizard = ProjectWizard(parent=self)
        wizard.project_created.connect(self._on_project_created)
        wizard.exec_()

    def _on_project_created(self, project_path: Path, schema, videos: list) -> None:
        """Handle project creation from the wizard."""
        self.project_schema = schema
        self.project_schema_path = project_path / "project.annolid.json"
        self.behavior_controller.configure_from_schema(schema)
        self._populate_behavior_controls_from_schema(schema)
        self._sync_behavior_flags_from_schema(schema)

        if videos:
            self.openVideo(from_video_list=True, video_path=videos[0])

        self.statusBar().showMessage(
            self.tr("Project '%s' created successfully") % project_path.name,
            5000,
        )

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

        if config.get("open_dashboard"):
            self._open_training_dashboard()

    def _start_yolo_training_from_wizard(self, config: dict) -> None:
        """Start YOLO training with wizard configuration."""
        try:
            self.yolo_training_manager.start_training(
                yolo_model_file=config.get("model", "yolo11n-seg.pt"),
                model_path=None,
                data_config_path=config.get("dataset_path"),
                epochs=config.get("epochs", 100),
                image_size=config.get("imgsz", 640),
                batch_size=config.get("batch", 8),
                device=config.get("device", ""),
                plots=True,
                train_overrides=None,
                out_dir=config.get("output_dir"),
            )
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
            self.dino_kpseg_training_manager.start_training(
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

    def _open_training_dashboard(self) -> None:
        """Open the training dashboard for monitoring."""
        try:
            dialog = getattr(self, "_training_dashboard_dialog", None)
            if dialog is None:
                dialog = TrainingDashboardDialog(settings=self.settings, parent=self)
                dialog.dashboard.register_training_manager(self.yolo_training_manager)
                dialog.dashboard.register_training_manager(
                    self.dino_kpseg_training_manager
                )
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
        except Exception as e:
            logger.debug(f"Could not open training dashboard: {e}")

    def open_labeling_progress_dashboard(self) -> None:
        """Open the labeling progress dashboard (project stats + gamification)."""
        try:
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

    def open_inference_wizard(self) -> None:
        """Open the inference wizard for running model predictions."""
        from annolid.gui.widgets.inference_wizard import InferenceWizard

        video_path = self.video_file if self.video_file else None

        wizard = InferenceWizard(video_path=video_path, parent=self)
        wizard.exec_()

    def _load_labels(self, labels_csv_file):
        """Load labels from the given CSV file."""
        self._df = pd.read_csv(labels_csv_file)
        self._df.rename(columns={"Unnamed: 0": "frame_number"}, inplace=True)

    def _toggle_patch_similarity_tool(self, checked=False):
        self.dino_controller.toggle_patch_similarity(checked)

    def _toggle_pca_map_tool(self, checked=False):
        self.dino_controller.toggle_pca_map(checked)

    def _deactivate_pca_map(self):
        self.dino_controller.deactivate_pca_map()

    def _request_pca_map(self) -> None:
        self.dino_controller.request_pca_map()

    def _open_patch_similarity_settings(self):
        self.dino_controller.open_patch_similarity_settings()
