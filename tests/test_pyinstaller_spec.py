from __future__ import annotations

import ast
from pathlib import Path


def _analysis_hiddenimports(spec_path: Path) -> list[str]:
    tree = ast.parse(spec_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "Analysis":
            continue
        for keyword in node.keywords:
            if keyword.arg != "hiddenimports":
                continue
            if not isinstance(keyword.value, ast.List):
                raise AssertionError("Analysis hiddenimports must be a literal list")
            return [
                element.value
                for element in keyword.value.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            ]
    raise AssertionError("Analysis hiddenimports not found")


def test_pyinstaller_spec_includes_launcher_dynamic_gui_import() -> None:
    hiddenimports = _analysis_hiddenimports(Path("annolid.spec"))

    assert "annolid.gui.app" in hiddenimports


def test_pyinstaller_spec_includes_gui_startup_lazy_namespace_imports() -> None:
    hiddenimports = _analysis_hiddenimports(Path("annolid.spec"))

    assert "annolid.domain.project_schema" in hiddenimports
    assert "annolid.infrastructure.persistence" in hiddenimports
