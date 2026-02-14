from annolid.gui.threejs_support import supports_threejs_canvas


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
