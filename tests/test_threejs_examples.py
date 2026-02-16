from pathlib import Path

from annolid.gui.threejs_examples import (
    THREEJS_EXAMPLE_IDS,
    generate_threejs_example,
)


def test_generate_threejs_examples(tmp_path: Path):
    for example_id in THREEJS_EXAMPLE_IDS:
        path = generate_threejs_example(example_id, tmp_path)
        assert path.exists()
        assert path.is_file()
        assert path.stat().st_size > 0


def test_generate_cool_point_cloud_example_is_html_asset(tmp_path: Path):
    path = generate_threejs_example("cool_point_cloud_importmap_html", tmp_path)
    assert path.name == "cool_point_cloud_importmap.html"
    assert path.suffix == ".html"


def test_generate_threejs_example_invalid_id(tmp_path: Path):
    try:
        generate_threejs_example("invalid", tmp_path)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for invalid example id")
