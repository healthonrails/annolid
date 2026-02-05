"""
Project Controller for Annolid GUI Application.

Handles UI interactions and coordinates project management operations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from qtpy import QtCore

from ..interfaces.services import IProjectService
from ..services import ProjectService

logger = logging.getLogger(__name__)


class ProjectController(QtCore.QObject):
    """
    Controller for project-related UI operations.

    Coordinates between the UI and project service, handling
    user interactions and business logic orchestration.
    """

    # Signals
    project_created = QtCore.Signal(
        dict
    )  # Emitted when project is created (project_info)
    project_loaded = QtCore.Signal(
        dict
    )  # Emitted when project is loaded (project_info)
    project_saved = QtCore.Signal()  # Emitted when project is saved
    project_error = QtCore.Signal(str)  # Emitted on project errors
    project_updated = QtCore.Signal(
        dict
    )  # Emitted when project config is updated (project_info)
    progress_updated = QtCore.Signal(int, str)  # Progress updates

    def __init__(
        self,
        project_service: Optional[IProjectService] = None,
        parent: Optional[QtCore.QObject] = None,
    ):
        """
        Initialize the project controller.

        Args:
            project_service: Project service instance
            parent: Parent QObject
        """
        super().__init__(parent)
        self._project_service = project_service or ProjectService()
        self._current_project_info: Optional[Dict[str, Any]] = None

    def create_new_project(
        self,
        project_path: Union[str, Path],
        project_name: str,
        project_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Create a new project.

        Args:
            project_path: Path where project should be created
            project_name: Name of the project
            project_config: Optional project configuration

        Returns:
            True if created successfully, False otherwise
        """
        try:
            success, message = self._project_service.create_project(
                project_path, project_name, project_config
            )

            if success:
                # Load the newly created project
                project_full_path = Path(project_path) / project_name
                return self.load_project(project_full_path)
            else:
                self.project_error.emit(message)
                return False

        except Exception as e:
            error_msg = f"Failed to create project: {str(e)}"
            self.project_error.emit(error_msg)
            logger.error(error_msg)
            return False

    def load_project(self, project_path: Union[str, Path]) -> bool:
        """
        Load an existing project.

        Args:
            project_path: Path to the project directory

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            success, message = self._project_service.load_project(project_path)

            if success:
                project_info = self._project_service.get_project_info()
                if project_info:
                    self._current_project_info = project_info
                    self.project_loaded.emit(project_info)
                    logger.info(
                        f"Project loaded: {project_info.get('name', 'Unknown')}"
                    )
                    return True
                else:
                    self.project_error.emit("Failed to get project information")
                    return False
            else:
                self.project_error.emit(message)
                return False

        except Exception as e:
            error_msg = f"Failed to load project: {str(e)}"
            self.project_error.emit(error_msg)
            logger.error(error_msg)
            return False

    def save_project(self) -> bool:
        """
        Save the current project configuration.

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            success, message = self._project_service.save_project_config()

            if success:
                self.project_saved.emit()
                logger.info("Project saved successfully")
                return True
            else:
                self.project_error.emit(message)
                return False

        except Exception as e:
            error_msg = f"Failed to save project: {str(e)}"
            self.project_error.emit(error_msg)
            logger.error(error_msg)
            return False

    def update_project_config(self, updates: Dict[str, Any]) -> bool:
        """
        Update project configuration.

        Args:
            updates: Configuration updates

        Returns:
            True if updated successfully, False otherwise
        """
        try:
            success, message = self._project_service.update_project_config(updates)

            if success:
                project_info = self._project_service.get_project_info()
                if project_info:
                    self._current_project_info = project_info
                    self.project_updated.emit(project_info)
                    logger.info("Project configuration updated")
                    return True
                else:
                    self.project_error.emit("Failed to get updated project information")
                    return False
            else:
                self.project_error.emit(message)
                return False

        except Exception as e:
            error_msg = f"Failed to update project config: {str(e)}"
            self.project_error.emit(error_msg)
            logger.error(error_msg)
            return False

    def add_project_class(
        self, class_name: str, class_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a new class to the project.

        Args:
            class_name: Name of the class to add
            class_config: Optional class configuration

        Returns:
            True if added successfully, False otherwise
        """
        try:
            success, message = self._project_service.add_project_class(
                class_name, class_config
            )

            if success:
                # Update project info
                project_info = self._project_service.get_project_info()
                if project_info:
                    self._current_project_info = project_info
                    self.project_updated.emit(project_info)
                    logger.info(f"Class added to project: {class_name}")
                    return True
                else:
                    self.project_error.emit("Failed to get updated project information")
                    return False
            else:
                self.project_error.emit(message)
                return False

        except Exception as e:
            error_msg = f"Failed to add project class: {str(e)}"
            self.project_error.emit(error_msg)
            logger.error(error_msg)
            return False

    def get_project_classes(self) -> List[Dict[str, Any]]:
        """
        Get all classes defined in the project.

        Returns:
            List of class configurations
        """
        try:
            return self._project_service.get_project_classes()
        except Exception as e:
            logger.error(f"Failed to get project classes: {e}")
            return []

    def get_project_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the current project.

        Returns:
            Project information dictionary or None
        """
        try:
            return self._project_service.get_project_info()
        except Exception as e:
            logger.error(f"Failed to get project info: {e}")
            return None

    def get_project_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for the current project.

        Returns:
            Project statistics dictionary
        """
        try:
            project_info = self.get_project_info()
            if project_info and "statistics" in project_info:
                return project_info["statistics"]
            return {}
        except Exception as e:
            logger.error(f"Failed to get project statistics: {e}")
            return {}

    def export_project(
        self,
        export_path: Union[str, Path],
        include_data: bool = True,
        include_models: bool = False,
    ) -> bool:
        """
        Export the current project.

        Args:
            export_path: Path to export the project
            include_data: Whether to include image and annotation data
            include_models: Whether to include trained models

        Returns:
            True if exported successfully, False otherwise
        """
        try:
            success, message = self._project_service.export_project(
                export_path, include_data, include_models
            )

            if success:
                logger.info(f"Project exported to: {export_path}")
                return True
            else:
                self.project_error.emit(message)
                return False

        except Exception as e:
            error_msg = f"Failed to export project: {str(e)}"
            self.project_error.emit(error_msg)
            logger.error(error_msg)
            return False

    def validate_project_structure(
        self, project_path: Union[str, Path]
    ) -> Tuple[bool, List[str]]:
        """
        Validate project directory structure.

        Args:
            project_path: Path to the project

        Returns:
            Tuple of (is_valid, error_messages)
        """
        try:
            # Get project config to validate against
            project_info = self.get_project_info()
            if not project_info:
                return False, ["No project loaded"]

            config = {"directories": project_info.get("directories", {})}
            is_valid, errors = self._project_service.validate_project_structure(
                Path(project_path), config
            )

            return is_valid, errors

        except Exception as e:
            error_msg = f"Failed to validate project structure: {str(e)}"
            return False, [error_msg]

    def get_project_directory(self, dir_type: str) -> Optional[Path]:
        """
        Get a specific project directory path.

        Args:
            dir_type: Type of directory (images, annotations, models, etc.)

        Returns:
            Directory path or None if not found
        """
        try:
            project_info = self.get_project_info()
            if not project_info or "path" not in project_info:
                return None

            project_path = Path(project_info["path"])
            directories = project_info.get("directories", {})

            if dir_type in directories:
                return project_path / directories[dir_type]
            else:
                # Try common directory names
                common_dirs = {
                    "images": "images",
                    "annotations": "annotations",
                    "models": "models",
                    "videos": "videos",
                    "exports": "exports",
                }
                if dir_type in common_dirs:
                    return project_path / common_dirs[dir_type]

            return None

        except Exception as e:
            logger.error(f"Failed to get project directory: {e}")
            return None

    def is_project_loaded(self) -> bool:
        """
        Check if a project is currently loaded.

        Returns:
            True if project is loaded, False otherwise
        """
        return self._current_project_info is not None

    def get_current_project_path(self) -> Optional[Path]:
        """
        Get the current project path.

        Returns:
            Current project path or None
        """
        if self._current_project_info and "path" in self._current_project_info:
            return Path(self._current_project_info["path"])
        return None

    def get_project_settings(self) -> Dict[str, Any]:
        """
        Get project settings.

        Returns:
            Project settings dictionary
        """
        try:
            project_info = self.get_project_info()
            if project_info and "settings" in project_info:
                return project_info["settings"]
            return {}
        except Exception as e:
            logger.error(f"Failed to get project settings: {e}")
            return {}

    def update_project_settings(self, settings: Dict[str, Any]) -> bool:
        """
        Update project settings.

        Args:
            settings: New settings

        Returns:
            True if updated successfully, False otherwise
        """
        try:
            # Wrap settings in the expected structure
            updates = {"settings": settings}
            return self.update_project_config(updates)

        except Exception as e:
            error_msg = f"Failed to update project settings: {str(e)}"
            self.project_error.emit(error_msg)
            logger.error(error_msg)
            return False
