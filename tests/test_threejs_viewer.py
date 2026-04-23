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
    assert supports_threejs_canvas("panorama.jpg")
    assert supports_threejs_canvas("panorama.jpeg")
    assert supports_threejs_canvas("panorama.png")
    assert supports_threejs_canvas("panorama.webp")
    assert supports_threejs_canvas("atlas_interleaved_30um_image.zarr")
    assert supports_threejs_canvas("volume.tif")
    assert supports_threejs_canvas("volume.tiff")


def test_supports_threejs_canvas_rejects_unknown_extensions():
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


def test_threejs_viewer_supports_360_equirectangular_images():
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

    assert "isPanoramaImageExt" in source
    assert "SphereGeometry(500, 64, 40)" in source
    assert "Loaded 360 panorama" in source


def test_threejs_viewer_supports_zarr_gaussian_splat_render_mode():
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

    assert "gaussian_splatting" in source
    assert "renderZarrGaussianSplatPoints" in source
    assert "THREE.AdditiveBlending" in source
    assert "annolidThreeVolumePanel" in source
    assert "getVolumeRenderDefaults" in source
    assert "decodeVolumeGrid" in source
    assert "Data3DTexture" in source
    assert "renderVolumePanel" in source
    assert "volumeHistogramCanvas" in source
    assert "loadPersistedVolumeState" in source
    assert "window.localStorage" in source
    assert "isVolumePointVisible" in source
    assert "volumeClipAxis" in source
    assert "nissl_sections" in source
    assert "myelin_sections" in source
    assert "section_ink" in source
    assert "getZarrSectionTexture" in source
    assert "applyVolumeSceneStyle" in source
    assert "renderVolumeSlabPlane" in source
    assert "renderVolumeRaymarch" in source
    assert 'adapter === "tiff-volume"' in source
    assert 'adapter === "zarr-volume" || adapter === "tiff-volume"' in source
    assert "fitCameraToVolumeSlab" in source
    assert "volume_grid_base64" in source
    assert "volumeHistologyDefaults" in source
    assert "histology_defaults" in source
    assert 'value="raymarch"' in source
    assert "volumeSectionEmphasis" in source
    assert "resolveAutoSectionEmphasis" in source
    assert "volumeRaymarchSteps" in source
    assert "volumeRaymarchGradientOpacity" in source
    assert "volumeRaymarchShading" in source
    assert "volumeQuickPresetCinematic" in source
    assert "requestVolumeRender" in source
    assert "WebGPURenderer" in source
    assert "data-threejs-renderer" in source
    assert "updateAdaptiveRaymarchQuality" in source
    assert "adaptiveRaymarchFactor" in source
    assert "u_useGradientOpacity" in source
    assert "u_useShading" in source
    assert "interleaved_detected" in source
    assert "volumeFocusSlab" in source
    css_source = (
        repo_root
        / "annolid"
        / "gui"
        / "assets"
        / "threejs"
        / "annolid_threejs_viewer.css"
    ).read_text(encoding="utf-8")
    assert "cursor: move;" in css_source
    assert ".three-volume-card" in css_source
    assert ".three-volume-quick-presets" in css_source
