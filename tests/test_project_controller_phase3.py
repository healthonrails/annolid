from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
from pathlib import Path

from annolid.gui.controllers.project_controller import ProjectController


class _FakeProjectService:
    def __init__(
        self,
        *,
        create_result: Tuple[bool, str] = (True, "ok"),
        load_result: Tuple[bool, str] = (True, "ok"),
        project_info: Optional[Dict[str, Any]] = None,
        raise_on_load: bool = False,
    ) -> None:
        self._create_result = create_result
        self._load_result = load_result
        self._project_info = project_info
        self._raise_on_load = raise_on_load

    def create_project(
        self,
        project_path: Union[str, Path],
        project_name: str,
        project_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        _ = project_path, project_name, project_config
        return self._create_result

    def load_project(self, project_path: Union[str, Path]) -> Tuple[bool, str]:
        _ = project_path
        if self._raise_on_load:
            raise RuntimeError("boom-load")
        return self._load_result

    def get_project_info(self) -> Optional[Dict[str, Any]]:
        return self._project_info

    def save_project_config(self) -> Tuple[bool, str]:
        return True, "ok"

    def update_project_config(self, updates: Dict[str, Any]) -> Tuple[bool, str]:
        _ = updates
        return True, "ok"

    def add_project_class(
        self, class_name: str, class_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        _ = class_name, class_config
        return True, "ok"

    def get_project_classes(self):
        return []

    def export_project(
        self,
        export_path: Union[str, Path],
        include_data: bool = True,
        include_models: bool = False,
    ) -> Tuple[bool, str]:
        _ = export_path, include_data, include_models
        return True, "ok"

    def validate_project_structure(self, project_path: Path, config: Dict[str, Any]):
        _ = project_path, config
        return True, []


def test_create_new_project_propagates_service_error_message() -> None:
    controller = ProjectController(
        project_service=_FakeProjectService(create_result=(False, "create-failed"))
    )
    errors: list[str] = []
    controller.project_error.connect(errors.append)

    ok = controller.create_new_project("/tmp", "demo")

    assert ok is False
    assert errors == ["create-failed"]


def test_load_project_propagates_exception_as_project_error() -> None:
    controller = ProjectController(
        project_service=_FakeProjectService(raise_on_load=True)
    )
    errors: list[str] = []
    controller.project_error.connect(errors.append)

    ok = controller.load_project("/tmp/demo")

    assert ok is False
    assert errors
    assert "Failed to load project: boom-load" in errors[-1]


def test_load_project_reports_missing_project_info() -> None:
    controller = ProjectController(
        project_service=_FakeProjectService(
            load_result=(True, "ok"),
            project_info=None,
        )
    )
    errors: list[str] = []
    controller.project_error.connect(errors.append)

    ok = controller.load_project("/tmp/demo")

    assert ok is False
    assert errors == ["Failed to get project information"]
