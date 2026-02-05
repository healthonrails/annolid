"""
Project Service for Annolid GUI Application.

Handles project management, configuration, and file organization.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class ProjectService:
    """
    Domain service for project management operations.

    Provides business logic for creating, loading, and managing
    Annolid annotation projects.
    """

    def __init__(self):
        """Initialize the project service."""
        self._current_project: Optional[Dict[str, Any]] = None
        self._project_config: Optional[Dict[str, Any]] = None

    def create_project(
        self,
        project_path: Union[str, Path],
        project_name: str,
        project_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """
        Create a new Annolid project.

        Args:
            project_path: Path where project should be created
            project_name: Name of the project
            project_config: Optional project configuration

        Returns:
            Tuple of (success, message)
        """
        try:
            project_path = Path(project_path)
            project_dir = project_path / project_name

            # Create project directory structure
            directories = [
                project_dir,
                project_dir / "images",
                project_dir / "annotations",
                project_dir / "models",
                project_dir / "videos",
                project_dir / "exports",
            ]

            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)

            # Create default project configuration
            default_config = {
                "name": project_name,
                "version": "1.0.0",
                "description": f"Annolid project: {project_name}",
                "created": self._get_current_timestamp(),
                "modified": self._get_current_timestamp(),
                "directories": {
                    "images": "images",
                    "annotations": "annotations",
                    "models": "models",
                    "videos": "videos",
                    "exports": "exports",
                },
                "settings": {
                    "annotation_format": "labelme",
                    "image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
                    "video_extensions": [".mp4", ".avi", ".mov", ".mkv"],
                    "auto_save": True,
                    "backup_interval": 300,  # seconds
                },
                "classes": [],
                "models": [],
            }

            # Merge with provided config
            if project_config:
                self._deep_merge(default_config, project_config)

            # Save project configuration
            config_path = project_dir / "project.annolid.json"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)

            # Create .annolidignore file
            ignore_path = project_dir / ".annolidignore"
            ignore_content = """# Annolid ignore file
# Files and directories to ignore during project operations

# Temporary files
*.tmp
*.temp
*~

# System files
.DS_Store
Thumbs.db

# Cache directories
__pycache__/
*.pyc
*.pyo

# Log files
*.log

# Backup files
*.bak
*.backup

# Export directories (can be large)
exports/
"""
            with open(ignore_path, "w", encoding="utf-8") as f:
                f.write(ignore_content)

            self._current_project = {
                "path": str(project_dir),
                "config": default_config,
            }

            return (
                True,
                f"Project '{project_name}' created successfully at {project_dir}",
            )

        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            return False, f"Failed to create project: {str(e)}"

    def load_project(self, project_path: Union[str, Path]) -> Tuple[bool, str]:
        """
        Load an existing Annolid project.

        Args:
            project_path: Path to the project directory

        Returns:
            Tuple of (success, message)
        """
        try:
            project_path = Path(project_path)

            if not project_path.exists():
                return False, f"Project path does not exist: {project_path}"

            config_path = project_path / "project.annolid.json"
            if not config_path.exists():
                return False, f"Project configuration file not found: {config_path}"

            # Load project configuration
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            # Validate project structure
            is_valid, errors = self.validate_project_structure(project_path, config)
            if not is_valid:
                return False, f"Invalid project structure: {'; '.join(errors)}"

            self._current_project = {
                "path": str(project_path),
                "config": config,
            }
            self._project_config = config

            return True, f"Project loaded successfully: {config.get('name', 'Unknown')}"

        except json.JSONDecodeError as e:
            return False, f"Invalid project configuration file: {e}"
        except Exception as e:
            logger.error(f"Failed to load project: {e}")
            return False, f"Failed to load project: {str(e)}"

    def validate_project_structure(
        self, project_path: Path, config: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate project directory structure and configuration.

        Args:
            project_path: Path to the project
            config: Project configuration

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check required directories
        required_dirs = ["images", "annotations"]
        for dir_name in required_dirs:
            dir_path = project_path / dir_name
            if not dir_path.exists():
                errors.append(f"Required directory missing: {dir_name}")

        # Check configuration requirements
        required_config_keys = ["name", "directories"]
        for key in required_config_keys:
            if key not in config:
                errors.append(f"Required configuration key missing: {key}")

        # Validate directory mappings
        if "directories" in config:
            dirs_config = config["directories"]
            for dir_key, dir_name in dirs_config.items():
                dir_path = project_path / dir_name
                if not dir_path.exists():
                    errors.append(
                        f"Configured directory does not exist: {dir_name} ({dir_key})"
                    )

        return len(errors) == 0, errors

    def save_project_config(self) -> Tuple[bool, str]:
        """
        Save the current project configuration.

        Returns:
            Tuple of (success, message)
        """
        if not self._current_project:
            return False, "No project loaded"

        try:
            config_path = Path(self._current_project["path"]) / "project.annolid.json"
            config = self._current_project["config"]

            # Update modification timestamp
            config["modified"] = self._get_current_timestamp()

            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            return True, "Project configuration saved successfully"

        except Exception as e:
            logger.error(f"Failed to save project config: {e}")
            return False, f"Failed to save project configuration: {str(e)}"

    def get_project_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the current project.

        Returns:
            Project information dictionary or None if no project loaded
        """
        if not self._current_project:
            return None

        config = self._current_project["config"]
        project_path = Path(self._current_project["path"])

        # Gather project statistics
        info = {
            "name": config.get("name", "Unknown"),
            "path": str(project_path),
            "version": config.get("version", "Unknown"),
            "description": config.get("description", ""),
            "created": config.get("created", "Unknown"),
            "modified": config.get("modified", "Unknown"),
            "statistics": self._get_project_statistics(project_path, config),
        }

        return info

    def _get_project_statistics(
        self, project_path: Path, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get project statistics."""
        stats = {
            "total_images": 0,
            "total_annotations": 0,
            "total_videos": 0,
            "total_models": 0,
        }

        try:
            # Count images
            images_dir = project_path / config.get("directories", {}).get(
                "images", "images"
            )
            if images_dir.exists():
                image_extensions = config.get("settings", {}).get(
                    "image_extensions", [".jpg", ".png"]
                )
                stats["total_images"] = len(
                    [
                        f
                        for f in images_dir.iterdir()
                        if f.is_file() and f.suffix.lower() in image_extensions
                    ]
                )

            # Count annotations
            annotations_dir = project_path / config.get("directories", {}).get(
                "annotations", "annotations"
            )
            if annotations_dir.exists():
                stats["total_annotations"] = len(
                    [
                        f
                        for f in annotations_dir.iterdir()
                        if f.is_file() and f.suffix.lower() == ".json"
                    ]
                )

            # Count videos
            videos_dir = project_path / config.get("directories", {}).get(
                "videos", "videos"
            )
            if videos_dir.exists():
                video_extensions = config.get("settings", {}).get(
                    "video_extensions", [".mp4", ".avi"]
                )
                stats["total_videos"] = len(
                    [
                        f
                        for f in videos_dir.iterdir()
                        if f.is_file() and f.suffix.lower() in video_extensions
                    ]
                )

            # Count models
            models_dir = project_path / config.get("directories", {}).get(
                "models", "models"
            )
            if models_dir.exists():
                stats["total_models"] = len(
                    [f for f in models_dir.iterdir() if f.is_file()]
                )

        except Exception as e:
            logger.warning(f"Failed to gather project statistics: {e}")

        return stats

    def update_project_config(self, updates: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Update project configuration with new values.

        Args:
            updates: Dictionary of configuration updates

        Returns:
            Tuple of (success, message)
        """
        if not self._current_project:
            return False, "No project loaded"

        try:
            config = self._current_project["config"]
            self._deep_merge(config, updates)

            # Save updated configuration
            success, message = self.save_project_config()
            if success:
                return True, "Project configuration updated successfully"
            else:
                return False, message

        except Exception as e:
            logger.error(f"Failed to update project config: {e}")
            return False, f"Failed to update project configuration: {str(e)}"

    def add_project_class(
        self, class_name: str, class_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Add a new class to the project.

        Args:
            class_name: Name of the class to add
            class_config: Optional class configuration

        Returns:
            Tuple of (success, message)
        """
        if not self._current_project:
            return False, "No project loaded"

        try:
            config = self._current_project["config"]
            if "classes" not in config:
                config["classes"] = []

            # Check if class already exists
            existing_classes = [cls.get("name", "") for cls in config["classes"]]
            if class_name in existing_classes:
                return False, f"Class '{class_name}' already exists"

            # Add new class
            new_class = {"name": class_name}
            if class_config:
                new_class.update(class_config)

            config["classes"].append(new_class)

            # Save configuration
            success, message = self.save_project_config()
            if success:
                return True, f"Class '{class_name}' added successfully"
            else:
                return False, message

        except Exception as e:
            logger.error(f"Failed to add project class: {e}")
            return False, f"Failed to add class: {str(e)}"

    def get_project_classes(self) -> List[Dict[str, Any]]:
        """
        Get all classes defined in the project.

        Returns:
            List of class configurations
        """
        if not self._current_project:
            return []

        config = self._current_project["config"]
        return config.get("classes", [])

    def export_project(
        self,
        export_path: Union[str, Path],
        include_data: bool = True,
        include_models: bool = False,
    ) -> Tuple[bool, str]:
        """
        Export project to a specified location.

        Args:
            export_path: Path to export the project
            include_data: Whether to include image and annotation data
            include_models: Whether to include trained models

        Returns:
            Tuple of (success, message)
        """
        if not self._current_project:
            return False, "No project loaded"

        try:
            export_path = Path(export_path)
            project_path = Path(self._current_project["path"])

            # Create export directory
            export_path.mkdir(parents=True, exist_ok=True)

            # Copy project configuration
            import shutil

            config_src = project_path / "project.annolid.json"
            config_dst = export_path / "project.annolid.json"
            shutil.copy2(config_src, config_dst)

            # Copy directories based on options
            dirs_to_copy = ["annotations"]
            if include_data:
                dirs_to_copy.extend(["images", "videos"])
            if include_models:
                dirs_to_copy.append("models")

            for dir_name in dirs_to_copy:
                src_dir = project_path / dir_name
                dst_dir = export_path / dir_name
                if src_dir.exists():
                    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

            return True, f"Project exported successfully to {export_path}"

        except Exception as e:
            logger.error(f"Failed to export project: {e}")
            return False, f"Failed to export project: {str(e)}"

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime

        return datetime.now().isoformat()

    def _deep_merge(
        self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]
    ) -> None:
        """
        Deep merge update_dict into base_dict.

        Args:
            base_dict: Dictionary to update
            update_dict: Dictionary with updates
        """
        for key, value in update_dict.items():
            if (
                key in base_dict
                and isinstance(base_dict[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
