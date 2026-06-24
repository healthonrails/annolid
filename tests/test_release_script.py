from pathlib import Path


def test_release_twine_check_excludes_desktop_bundles() -> None:
    script = Path("scripts/release.sh").read_text(encoding="utf-8")

    assert "python -m twine check dist/*.whl dist/*.tar.gz" in script
    assert "python -m twine check dist/*\n" not in script
