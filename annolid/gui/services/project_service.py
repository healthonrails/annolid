"""
Project Service for Annolid GUI Application.

Handles project management, configuration, workflow status, and file organization.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from annolid.infrastructure import atomic_write_json
from annolid.services.project_workspace import (
    PROJECT_CONFIG_FILENAME,
    compute_project_workflow_status,
    current_iso_timestamp,
    deep_merge_dict,
    default_project_config,
    ensure_project_directories,
    export_project_bundle,
    is_schema_only_project_config,
    normalize_project_config,
    register_project_videos,
    resolve_project_dir,
    resolve_project_video_paths,
    scan_project_statistics,
    validate_project_name,
    write_project_ignore_file,
)

logger = logging.getLogger(__name__)

_SCHEMA_ENVELOPE_KEYS = (
    "categories",
    "modifiers",
    "subjects",
    "behaviors",
    "pose_schema_path",
    "pose_schema",
    "schema_version",
)


class ProjectService:
    """
    Domain service for project management operations.

    Provides business logic for creating, loading, validating, and exporting
    Annolid annotation projects.
    """

    def __init__(self):
        """Initialize the project service."""
        self._current_project: Optional[Dict[str, Any]] = None
        self._project_config: Optional[Dict[str, Any]] = None
        self._explicitly_updated_schema_keys: Set[str] = set()

    def create_project(
        self,
        project_path: Union[str, Path],
        project_name: str,
        project_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """Create a new Annolid project."""
        try:
            valid_name, reason = validate_project_name(project_name)
            if not valid_name:
                return False, f"Invalid project name: {reason}"

            clean_name = str(project_name).strip()
            project_root_parent = Path(project_path).expanduser()
            project_dir = project_root_parent / clean_name

            config = default_project_config(clean_name)
            if project_config:
                deep_merge_dict(config, project_config)

            ensure_project_directories(project_dir, config.get("directories"))

            # Preserve any existing behavior/pose schema keys if project.annolid.json already exists.
            config_path = project_dir / PROJECT_CONFIG_FILENAME
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    existing_disk_config = json.load(f)
                if isinstance(existing_disk_config, dict):
                    for key, value in existing_disk_config.items():
                        if key not in config:
                            config[key] = value
            atomic_write_json(
                config_path,
                config,
                indent=2,
                ensure_ascii=False,
            )

            write_project_ignore_file(project_dir, overwrite=False)

            self._current_project = {
                "path": str(project_dir),
                "config": config,
            }
            self._project_config = config
            self._explicitly_updated_schema_keys.clear()

            return (
                True,
                f"Project '{clean_name}' created successfully at {project_dir}",
            )

        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            return False, f"Failed to create project: {str(e)}"

    def load_project(self, project_path: Union[str, Path]) -> Tuple[bool, str]:
        """Load an existing Annolid project (directory or `project.annolid.json`)."""
        try:
            resolved_dir = resolve_project_dir(project_path)

            if not resolved_dir.exists():
                return False, f"Project path does not exist: {resolved_dir}"

            config_path = resolved_dir / PROJECT_CONFIG_FILENAME
            if not config_path.exists():
                return False, f"Project configuration file not found: {config_path}"

            with open(config_path, "r", encoding="utf-8") as f:
                raw_config = json.load(f)
            if not isinstance(raw_config, dict):
                return (
                    False,
                    f"Invalid project configuration file: expected JSON object in {config_path}",
                )

            schema_only = is_schema_only_project_config(raw_config)
            if schema_only:
                config, _upgraded = normalize_project_config(resolved_dir, raw_config)
                ensure_project_directories(resolved_dir, config.get("directories"))
                write_project_ignore_file(resolved_dir, overwrite=False)
            else:
                config = raw_config

            is_valid, errors = self.validate_project_structure(resolved_dir, config)
            if not is_valid:
                return False, f"Invalid project structure: {'; '.join(errors)}"

            self._current_project = {
                "path": str(resolved_dir),
                "config": config,
            }
            self._project_config = config
            self._explicitly_updated_schema_keys.clear()

            return True, f"Project loaded successfully: {config.get('name', 'Unknown')}"

        except json.JSONDecodeError as e:
            return False, f"Invalid project configuration file: {e}"
        except Exception as e:
            logger.error(f"Failed to load project: {e}")
            return False, f"Failed to load project: {str(e)}"

    def close_project(self) -> None:
        """Clear the currently loaded project state."""
        self._current_project = None
        self._project_config = None
        self._explicitly_updated_schema_keys.clear()

    def validate_project_structure(
        self, project_path: Path, config: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate project directory structure and configuration."""
        errors: List[str] = []
        root = resolve_project_dir(project_path).resolve()

        required_config_keys = ["name", "directories"]
        for key in required_config_keys:
            if key not in config:
                errors.append(f"Required configuration key missing: {key}")

        dirs_config = config.get("directories")
        if isinstance(dirs_config, dict):
            for required_key in ("images", "annotations"):
                rel_dir = str(dirs_config.get(required_key) or required_key)
                if not (root / rel_dir).exists():
                    errors.append(f"Required directory missing: {rel_dir}")
            for dir_key, dir_name in dirs_config.items():
                rel_name = str(dir_name or "").strip()
                if not rel_name:
                    errors.append(f"Empty directory mapping for: {dir_key}")
                    continue
                dir_path = (root / rel_name).resolve()
                try:
                    dir_path.relative_to(root)
                except ValueError:
                    errors.append(
                        f"Configured directory escapes project root: {rel_name} ({dir_key})"
                    )
                    continue
                if not dir_path.exists():
                    errors.append(
                        f"Configured directory does not exist: {rel_name} ({dir_key})"
                    )
        else:
            for dir_name in ("images", "annotations"):
                if not (root / dir_name).exists():
                    errors.append(f"Required directory missing: {dir_name}")

        # Deduplicate error messages while preserving order.
        deduped = list(dict.fromkeys(errors))
        return len(deduped) == 0, deduped

    def save_project_config(self) -> Tuple[bool, str]:
        """Save the current project configuration atomically, preserving schema keys."""
        if not self._current_project:
            return False, "No project loaded"

        try:
            config_path = Path(self._current_project["path"]) / PROJECT_CONFIG_FILENAME
            config = self._current_project["config"]
            config["modified"] = self._get_current_timestamp()

            merged_config: Dict[str, Any] = {}
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    disk_config = json.load(f)
                if isinstance(disk_config, dict):
                    merged_config.update(disk_config)
                else:
                    disk_config = {}
            else:
                disk_config = {}

            merged_config.update(config)
            for key in _SCHEMA_ENVELOPE_KEYS:
                if key in self._explicitly_updated_schema_keys:
                    continue
                if key in disk_config:
                    merged_config[key] = disk_config[key]
                elif config_path.exists() and key not in disk_config:
                    merged_config.pop(key, None)

            atomic_write_json(
                config_path,
                merged_config,
                indent=2,
                ensure_ascii=False,
            )
            self._current_project["config"] = merged_config
            self._project_config = merged_config
            self._explicitly_updated_schema_keys.clear()

            return True, "Project configuration saved successfully"

        except Exception as e:
            logger.error(f"Failed to save project config: {e}")
            return False, f"Failed to save project configuration: {str(e)}"

    def get_project_info(self) -> Optional[Dict[str, Any]]:
        """Get rich metadata, directory mappings, settings, stats, and workflow status."""
        if not self._current_project:
            return None

        config = self._current_project["config"]
        project_path = Path(self._current_project["path"])
        statistics = self._get_project_statistics(project_path, config)
        workflow_status = compute_project_workflow_status(
            project_path, config, statistics
        )
        videos = [str(p) for p in resolve_project_video_paths(project_path, config)]

        return {
            "name": config.get("name", "Unknown"),
            "path": str(project_path),
            "version": config.get("version", "Unknown"),
            "description": config.get("description", ""),
            "created": config.get("created", "Unknown"),
            "modified": config.get("modified", "Unknown"),
            "directories": dict(config.get("directories") or {}),
            "settings": dict(config.get("settings") or {}),
            "videos": videos,
            "statistics": statistics,
            "workflow_status": workflow_status,
        }

    def get_workflow_status(self) -> Dict[str, Any]:
        """Return live 6-step workflow status for the currently loaded project."""
        if not self._current_project:
            return compute_project_workflow_status(None)
        project_path = Path(self._current_project["path"])
        config = self._current_project["config"]
        stats = self._get_project_statistics(project_path, config)
        return compute_project_workflow_status(project_path, config, stats)

    def get_project_videos(self) -> List[str]:
        """Return resolved paths of all videos registered or stored in the project."""
        if not self._current_project:
            return []
        project_path = Path(self._current_project["path"])
        config = self._current_project["config"]
        return [str(p) for p in resolve_project_video_paths(project_path, config)]

    def register_videos(
        self,
        video_paths: Sequence[Union[str, Path]],
        *,
        copy_into_project: bool = False,
    ) -> Tuple[bool, str, List[str]]:
        """Register video files in the active project and persist `project.annolid.json`."""
        if not self._current_project:
            return False, "No project loaded", []
        try:
            project_path = Path(self._current_project["path"])
            config = self._current_project["config"]
            registered = register_project_videos(
                project_path,
                config,
                video_paths,
                copy_into_project=copy_into_project,
            )
            ok, message = self.save_project_config()
            if not ok:
                return False, message, []
            return True, f"Registered {len(registered)} video(s)", registered
        except Exception as exc:
            logger.error("Failed to register project videos: %s", exc)
            return False, f"Failed to register videos: {exc}", []

    def _get_project_statistics(
        self, project_path: Path, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get project statistics including nested folders and AnnotationStore files."""
        return scan_project_statistics(project_path, config)

    def update_project_config(self, updates: Dict[str, Any]) -> Tuple[bool, str]:
        """Update project configuration with new values."""
        if not self._current_project:
            return False, "No project loaded"

        try:
            config = self._current_project["config"]
            for key in _SCHEMA_ENVELOPE_KEYS:
                if key in updates:
                    self._explicitly_updated_schema_keys.add(key)
            self._deep_merge(config, updates)

            success, message = self.save_project_config()
            if success:
                return True, "Project configuration updated successfully"
            return False, message

        except Exception as e:
            logger.error(f"Failed to update project config: {e}")
            return False, f"Failed to update project configuration: {str(e)}"

    def add_project_class(
        self, class_name: str, class_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """Add a new class to the project."""
        if not self._current_project:
            return False, "No project loaded"

        try:
            config = self._current_project["config"]
            if "classes" not in config:
                config["classes"] = []

            existing_classes = [cls.get("name", "") for cls in config["classes"]]
            if class_name in existing_classes:
                return False, f"Class '{class_name}' already exists"

            new_class = {"name": class_name}
            if class_config:
                new_class.update(class_config)

            config["classes"].append(new_class)

            success, message = self.save_project_config()
            if success:
                return True, f"Class '{class_name}' added successfully"
            return False, message

        except Exception as e:
            logger.error(f"Failed to add project class: {e}")
            return False, f"Failed to add class: {str(e)}"

    def get_project_classes(self) -> List[Dict[str, Any]]:
        """Get all classes defined in the project."""
        if not self._current_project:
            return []

        config = self._current_project["config"]
        return config.get("classes", [])

    def export_project(
        self,
        export_path: Union[str, Path],
        include_data: bool = True,
        include_models: bool = False,
        *,
        as_zip: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Tuple[bool, str]:
        """Export the current project to a directory or atomic `.zip` archive."""
        if not self._current_project:
            return False, "No project loaded"

        return export_project_bundle(
            Path(self._current_project["path"]),
            Path(export_path),
            config=self._current_project.get("config"),
            include_data=include_data,
            include_models=include_models,
            as_zip=as_zip,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return current_iso_timestamp()

    def _deep_merge(
        self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]
    ) -> None:
        """Deep merge update_dict into base_dict."""
        deep_merge_dict(base_dict, update_dict)
