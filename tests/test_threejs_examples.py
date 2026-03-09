import json
from pathlib import Path

from annolid.gui.threejs_examples import (
    THREEJS_EXAMPLE_IDS,
    _flybody_part_style,
    attach_flybody_floor,
    attach_flybody_mesh_parts,
    generate_threejs_example,
)


def test_generate_threejs_examples(tmp_path: Path):
    for example_id in THREEJS_EXAMPLE_IDS:
        path = generate_threejs_example(example_id, tmp_path)
        assert path.exists()
        assert path.is_file()
        assert path.stat().st_size > 0


def test_generate_threejs_example_invalid_id(tmp_path: Path):
    try:
        generate_threejs_example("invalid", tmp_path)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for invalid example id")


def test_generate_flybody_example_uses_repo_mesh_when_available(
    tmp_path: Path, monkeypatch
):
    """With a repo available the static example should use the assembled fly mesh and floor."""
    repo = tmp_path / "flybody_repo"
    assets = repo / "flybody" / "fruitfly" / "assets"
    assets.mkdir(parents=True)
    (assets / "body.obj").write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "f 1 2 3",
            ]
        ),
        encoding="utf-8",
    )
    # body must have a name= attribute so _maybe_build_flybody_mesh_parts includes it
    (assets / "fruitfly.xml").write_text(
        "\n".join(
            [
                "<mujoco>",
                "  <asset><mesh name='thorax' file='body.obj'/></asset>",
                "  <worldbody><body name='thorax'><geom mesh='thorax' pos='0 0 0' quat='1 0 0 0'/></body></worldbody>",
                "</mujoco>",
            ]
        ),
        encoding="utf-8",
    )
    (assets / "floor.xml").write_text(
        "\n".join(
            [
                "<mujoco>",
                "  <worldbody><geom name='floor' type='plane' size='5 5 .1' pos='0 0 -.132'/></worldbody>",
                "</mujoco>",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANNOLID_FLYBODY_PATH", str(repo))

    path = generate_threejs_example("flybody_simulation_json", tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["mesh"]["type"] == "obj"
    assert (tmp_path / payload["mesh"]["path"]).exists()
    assert payload["environment"]["floor"]["type"] == "plane"
    # Floor position is converted to Three.js Y-up: [0, mujoco_z * scale, 0]
    assert payload["environment"]["floor"]["position"] == [0.0, -0.132 * 7.5, 0.0]
    assert payload["display"] == {
        "show_points": False,
        "show_labels": False,
        "show_edges": False,
        "show_trails": False,
    }


def test_attach_flybody_mesh_parts_builds_body_part_payload(
    tmp_path: Path, monkeypatch
):
    repo = tmp_path / "flybody_repo"
    assets = repo / "flybody" / "fruitfly" / "assets"
    assets.mkdir(parents=True)
    (assets / "body.obj").write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "f 1 2 3",
            ]
        ),
        encoding="utf-8",
    )
    (assets / "fruitfly.xml").write_text(
        "\n".join(
            [
                "<mujoco>",
                "  <asset><mesh name='thorax' file='body.obj'/></asset>",
                "  <worldbody><body name='thorax'><geom mesh='thorax' pos='0 0 0' quat='1 0 0 0'/></body></worldbody>",
                "</mujoco>",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANNOLID_FLYBODY_PATH", str(repo))

    payload = attach_flybody_mesh_parts({"kind": "annolid-simulation-v1"}, tmp_path)

    assert payload["mesh"]["type"] == "flybody_parts"
    assert payload["mesh"]["parts"][0]["body"] == "thorax"
    assert payload["mesh"]["parts"][0]["category"] == "thorax"
    assert payload["mesh"]["parts"][0]["color"] == "#9c6b3f"
    assert (tmp_path / payload["mesh"]["parts"][0]["path"]).exists()


def test_attach_flybody_mesh_parts_creates_output_directory(
    tmp_path: Path, monkeypatch
):
    repo = tmp_path / "flybody_repo"
    assets = repo / "flybody" / "fruitfly" / "assets"
    assets.mkdir(parents=True)
    (assets / "body.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8"
    )
    (assets / "fruitfly.xml").write_text(
        "<mujoco><asset><mesh name='thorax' file='body.obj'/></asset><worldbody><body name='thorax'><geom mesh='thorax' pos='0 0 0' quat='1 0 0 0'/></body></worldbody></mujoco>",
        encoding="utf-8",
    )
    monkeypatch.setenv("ANNOLID_FLYBODY_PATH", str(repo))

    out_dir = tmp_path / "nested" / "parts"
    payload = attach_flybody_mesh_parts({"kind": "annolid-simulation-v1"}, out_dir)

    assert out_dir.exists()
    assert (out_dir / payload["mesh"]["parts"][0]["path"]).exists()


def test_attach_flybody_mesh_parts_reuses_existing_files(tmp_path: Path, monkeypatch):
    repo = tmp_path / "flybody_repo"
    assets = repo / "flybody" / "fruitfly" / "assets"
    assets.mkdir(parents=True)
    (assets / "body.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8"
    )
    (assets / "fruitfly.xml").write_text(
        "<mujoco><asset><mesh name='thorax' file='body.obj'/></asset><worldbody><body name='thorax'><geom mesh='thorax' pos='0 0 0' quat='1 0 0 0'/></body></worldbody></mujoco>",
        encoding="utf-8",
    )
    monkeypatch.setenv("ANNOLID_FLYBODY_PATH", str(repo))

    out_dir = tmp_path / "parts"
    payload = attach_flybody_mesh_parts({"kind": "annolid-simulation-v1"}, out_dir)
    part_path = out_dir / payload["mesh"]["parts"][0]["path"]
    original_mtime = part_path.stat().st_mtime_ns

    payload_2 = attach_flybody_mesh_parts({"kind": "annolid-simulation-v1"}, out_dir)
    assert payload_2["mesh"]["parts"][0]["path"] == payload["mesh"]["parts"][0]["path"]
    assert part_path.stat().st_mtime_ns == original_mtime


def test_flybody_part_style_assigns_stable_groups() -> None:
    assert _flybody_part_style("wing_left")["category"] == "wing"
    assert _flybody_part_style("antenna_right")["category"] == "antenna"
    assert _flybody_part_style("femur_T1_left")["category"] == "leg"
    assert _flybody_part_style("abdomen_4")["category"] == "abdomen"


def test_attach_flybody_floor_reads_floor_xml(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "flybody_repo"
    assets = repo / "flybody" / "fruitfly" / "assets"
    assets.mkdir(parents=True)
    (assets / "fruitfly.xml").write_text("<mujoco/>", encoding="utf-8")
    (assets / "floor.xml").write_text(
        "<mujoco><worldbody><geom type='plane' size='5 5 .1' pos='0 0 -.132'/></worldbody></mujoco>",
        encoding="utf-8",
    )
    monkeypatch.setenv("ANNOLID_FLYBODY_PATH", str(repo))

    payload = attach_flybody_floor({"kind": "annolid-simulation-v1"})

    assert payload["environment"]["floor"]["type"] == "plane"
    assert payload["environment"]["floor"]["size"] == [5.0, 5.0]
    assert payload["environment"]["floor"]["position"] == [0.0, 0.0, -0.132]
