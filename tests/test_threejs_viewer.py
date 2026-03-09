from annolid.gui.threejs_support import supports_threejs_canvas
from pathlib import Path


def test_supports_threejs_canvas_known_extensions():
    assert supports_threejs_canvas("model.stl")
    assert supports_threejs_canvas("mesh.obj")
    assert supports_threejs_canvas("cloud.ply")
    assert supports_threejs_canvas("avatar.glb")
    assert supports_threejs_canvas("scene.gltf")
    assert supports_threejs_canvas("points.csv")
    assert supports_threejs_canvas("points.xyz")


def test_supports_threejs_canvas_rejects_unknown_extensions():
    assert not supports_threejs_canvas("volume.tif")
    assert not supports_threejs_canvas("volume.nii.gz")
    assert not supports_threejs_canvas("anything")


def test_flybody_part_loader_accepts_parts_without_explicit_type():
    repo_root = Path(__file__).resolve().parents[1]
    js_path = (
        repo_root
        / "annolid"
        / "gui"
        / "assets"
        / "threejs"
        / "annolid_threejs_viewer.js"
    )
    source = js_path.read_text(encoding="utf-8")

    assert 'if (part && (!part.type || part.type === "obj"))' in source


def test_simulation_loader_preserves_world_coordinates_and_exposes_flybody_controls():
    repo_root = Path(__file__).resolve().parents[1]
    js_path = (
        repo_root
        / "annolid"
        / "gui"
        / "assets"
        / "threejs"
        / "annolid_threejs_viewer.js"
    )
    source = js_path.read_text(encoding="utf-8")

    assert "addLoadedObject(simulationRoot)" not in source
    assert "annolid://flybody-live/" in source
    assert "annolidThreeFlybodyControls" in source


def test_simulation_loader_reuses_flybody_mesh_between_motion_updates():
    repo_root = Path(__file__).resolve().parents[1]
    js_path = (
        repo_root
        / "annolid"
        / "gui"
        / "assets"
        / "threejs"
        / "annolid_threejs_viewer.js"
    )
    source = js_path.read_text(encoding="utf-8")

    assert 'let simulationMeshKey = "";' in source
    assert 'let simulationEnvironmentKey = "";' in source
    assert "const shouldReloadMesh = nextMeshKey !== simulationMeshKey;" in source
    assert (
        "const shouldReloadEnvironment = nextEnvironmentKey !== simulationEnvironmentKey;"
        in source
    )
    assert "simulationMeshKey = nextMeshKey;" in source
    assert "simulationEnvironmentKey = nextEnvironmentKey;" in source


def test_flybody_controls_snap_under_toolbar_and_are_draggable():
    repo_root = Path(__file__).resolve().parents[1]
    js_path = (
        repo_root
        / "annolid"
        / "gui"
        / "assets"
        / "threejs"
        / "annolid_threejs_viewer.js"
    )
    css_path = (
        repo_root
        / "annolid"
        / "gui"
        / "assets"
        / "threejs"
        / "annolid_threejs_viewer.css"
    )
    js_source = js_path.read_text(encoding="utf-8")
    css_source = css_path.read_text(encoding="utf-8")

    assert "positionFlybodyControls" in js_source
    assert "Drag to move FlyBody controls" in js_source
    assert "cursor: move;" in css_source
    assert "touch-action: none;" in css_source
