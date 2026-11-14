"""Opt-in rendering and behavior checks against the local desktop asset server.

ANNOLID_RUN_THREEJS_BROWSER_TESTS=1 .venv/bin/python -m pytest tests/test_two_mice_browser.py
Requires Playwright Chromium; runs 100% offline against bundled Three.js and lil-gui assets.
"""

import os
import math
from pathlib import Path
from urllib.parse import urlparse

import pytest


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("ANNOLID_RUN_THREEJS_BROWSER_TESTS") != "1",
        reason="Opt in to browser WebGL validation with ANNOLID_RUN_THREEJS_BROWSER_TESTS=1",
    ),
]


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        try:
            launched = playwright.chromium.launch(headless=True, channel="chromium")
        except Exception:
            launched = playwright.chromium.launch(headless=True)
        yield launched
        launched.close()


@pytest.fixture
def scene_page(browser):
    from annolid.gui.widgets.threejs_viewer_server import _ensure_threejs_http_server

    page = browser.new_page(viewport={"width": 1100, "height": 800})
    errors = []
    blocked_external_urls = []

    def _guard_offline_requests(route):
        parsed = urlparse(route.request.url)
        if parsed.scheme in {"http", "https"} and parsed.hostname not in {
            "127.0.0.1",
            "localhost",
        }:
            blocked_external_urls.append(route.request.url)
            route.abort()
            return
        route.continue_()

    page.route("**/*", _guard_offline_requests)
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on(
        "console",
        lambda message: errors.append(message.text)
        if message.type == "error"
        else None,
    )
    page.add_init_script("window.originalRandom = Math.random;")
    base = _ensure_threejs_http_server()

    def load(query):
        page.goto(f"{base}/threejs/two_mice.html?{query}", timeout=120_000)
        page.wait_for_selector("#c[data-scene-ready='true']", timeout=120_000)
        return page

    yield load
    page.close()
    assert not blocked_external_urls, (
        f"Unexpected external network requests: {blocked_external_urls}"
    )
    assert not errors, "\n".join(errors)


def test_seeded_capture_is_repeatable_and_keeps_pose_contract(scene_page):
    page = scene_page("dataset=1&seed=20260710")
    assert page.evaluate("window.originalRandom === Math.random")
    subjects = page.evaluate("window.annolidTwoMice.getDiagnostics().subjects")
    assert subjects[0]["heading"] == pytest.approx(math.atan2(1, 0.12))
    assert subjects[1]["heading"] == pytest.approx(math.atan2(-1, -0.12))
    frames = []
    for frame in range(12):
        payload = page.evaluate("i => window.annolidPoseDataset.setFrame(i)", frame)
        assert payload["image"]["frame_id"] == frame
        assert not payload["image"]["subject_overlap"]
        assert not payload["image"]["tail_overlap"]
        assert payload["image"]["maximum_foot_slip"] <= 0.075
        assert payload["image"]["maximum_paw_ground_error"] <= 0.1
        assert {item["track_id"] for item in payload["annotations"]} == {1, 2}
        assert all(len(item["keypoints"]) == 39 for item in payload["annotations"])
        frames.append(payload)
    assert frames[0]["annotations"] != frames[-1]["annotations"]
    page = scene_page("dataset=1&seed=20260710")
    repeated = page.evaluate("window.annolidPoseDataset.setFrame(11)")
    assert repeated == frames[-1]


def test_deformed_coat_normals_follow_ellipsoid_and_poles(scene_page):
    page = scene_page("dataset=1")
    result = page.evaluate(
        """async () => {
          const {THREE} = await import('./annolid_threejs_runtime.js');
          const {createDeformedSurfaceSampler} = await import('./two_mice_model.js');
          const sample = createDeformedSurfaceSampler(THREE, v => {
            v.x *= 1.27; v.y *= 1.18; v.z *= 2.68;
          });
          const p = new THREE.Vector3(), n = new THREE.Vector3();
          let minDot = 1;
          for (let i = 0; i <= 200; i++) {
            const y = 1 - i / 100;
            const radius = Math.sqrt(Math.max(0, 1-y*y));
            const source = new THREE.Vector3(radius*Math.cos(i*2.4), y, radius*Math.sin(i*2.4));
            sample(source, p, n);
            const expected = new THREE.Vector3(source.x/1.27, source.y/1.18, source.z/2.68).normalize();
            minDot = Math.min(minDot, expected.dot(n));
          }
          return minDot;
        }"""
    )
    assert result > 0.999


def test_live_pause_cameras_and_subject_resource_lifecycle(scene_page):
    page = scene_page("seed=123")
    page.wait_for_function("window.annolidTwoMice.getDiagnostics().time > 0.2")
    page.locator(".lil-gui.root > .title").click()
    pause = (
        page.locator(".controller").filter(has_text="Pause Simulation").locator("input")
    )
    pause.check()
    assert page.evaluate("window.annolidTwoMice.getDiagnostics().paused")
    before = page.evaluate("window.annolidTwoMice.getDiagnostics()")
    page.wait_for_timeout(400)
    after = page.evaluate("window.annolidTwoMice.getDiagnostics()")
    assert before["time"] == after["time"]
    assert before["subjects"] == after["subjects"]
    skeleton_toggle = (
        page.locator(".controller")
        .filter(has_text="Show Pose Skeleton")
        .locator("input")
    )
    skeleton_toggle.check()
    assert skeleton_toggle.is_checked()
    skeleton_toggle.uncheck()
    assert not skeleton_toggle.is_checked()
    camera = page.locator(".controller").filter(has_text="Camera").locator("select")
    for view in ("Follow black", "Follow white", "Top down", "Both mice"):
        camera.select_option(label=view)
    # Post-processing must preserve tone-mapped sRGB exposure rather than crushing linear RGB.
    from io import BytesIO
    from PIL import Image, ImageStat

    canvas_crop = (300, 200, 750, 650)
    off_luma = ImageStat.Stat(
        Image.open(BytesIO(page.screenshot())).crop(canvas_crop).convert("L")
    ).mean[0]
    shaders_title = page.locator(".lil-gui .title", has_text="Shaders")
    shaders_title.click()
    postfx_toggle = (
        page.locator(".controller").filter(has_text="Post-Processing").locator("input")
    )
    postfx_toggle.check()
    page.wait_for_timeout(120)
    on_luma = ImageStat.Stat(
        Image.open(BytesIO(page.screenshot())).crop(canvas_crop).convert("L")
    ).mean[0]
    assert abs(on_luma - off_luma) < 12.0
    postfx_toggle.uncheck()
    shaders_title.click()

    pause.uncheck()
    page.wait_for_function(
        "t => window.annolidTwoMice.getDiagnostics().time > t", arg=before["time"]
    )
    page.get_by_role("button", name="(!) Clear All", exact=True).click()
    page.wait_for_function(
        "window.annolidTwoMice.getDiagnostics().subjects.length === 0"
    )
    page.wait_for_timeout(100)
    empty = page.evaluate("window.annolidTwoMice.getDiagnostics().renderer")
    assert empty["geometries"] < before["renderer"]["geometries"]
    strain_select = (
        page.locator(".controller").filter(has_text="Strains").first.locator("select")
    )
    for strain_label in ("BALB/c (White)", "Nude/Hairless"):
        strain_select.select_option(label=strain_label)
        page.get_by_role("button", name="(+) Spawn Subject", exact=True).click()
        page.wait_for_function(
            "window.annolidTwoMice.getDiagnostics().subjects.length === 1"
        )
        page.get_by_role("button", name="(!) Clear All", exact=True).click()
        page.wait_for_timeout(100)
        cleared = page.evaluate("window.annolidTwoMice.getDiagnostics().renderer")
        assert cleared["geometries"] == empty["geometries"]
        assert cleared["textures"] == empty["textures"]

    # Model factory fur options and in-place shared anatomy rebuild work without leaking VBOs.
    model_checks = page.evaluate(
        """async () => {
          const {THREE} = await import('./annolid_threejs_runtime.js');
          const {AnnolidShaders} = await import('./annolid_shaders.js');
          const {createMouseModel} = await import('./two_mice_model.js');
          const {createSeededRandom} = await import('./two_mice_simulation.js');
          const model = createMouseModel(THREE, new AnnolidShaders(THREE), createSeededRandom(42));
          model.bodyGeo.computeBoundingBox();
          const initialSpanZ = model.bodyGeo.boundingBox.max.z - model.bodyGeo.boundingBox.min.z;
          model.PROP.bodyLength = 3.4;
          model.rebuildSharedAnatomyGeometries();
          const rebuiltSpanZ = model.bodyGeo.boundingBox.max.z - model.bodyGeo.boundingBox.min.z;
          const allUnder = model.generateMultiLayerFur(model.deformBody, 0.2, '#ffffff', false, {
            furGuardRatio: 0.0,
            furLength: 1.4,
            furClumpiness: 0.6,
          });
          const allGuard = model.generateMultiLayerFur(model.deformBody, 0.2, '#ffffff', false, {
            furGuardRatio: 1.0,
            furLength: 0.6,
            furClumpiness: 0.0,
          });
          const counts = [allUnder.children.length, allGuard.children.length];
          model.disposeSubject(allUnder);
          model.disposeSubject(allGuard);
          model.disposeSharedResources();
          return {changed: rebuiltSpanZ > initialSpanZ + 1.0, counts};
        }"""
    )
    assert model_checks == {"changed": True, "counts": [1, 1]}

    # Navigating away triggers pagehide cleanup cleanly with zero uncaught errors.
    page.goto("about:blank")


@pytest.mark.parametrize(
    "query",
    [
        "anatomy_qa=torso-side",
        "anatomy_qa=ears-front&anatomy_subject=white",
        "rearing_qa=unsupported",
        "rearing_qa=wall-supported",
        "grooming_qa=face-wash",
        "contact_qa=tail",
    ],
)
def test_anatomy_and_behavior_poses_render(scene_page, query):
    page = scene_page("dataset=1&" + query)
    telemetry = page.locator("#c").evaluate("el => ({...el.dataset})")
    assert float(telemetry["pixelLuminanceRange"]) > 20
    assert telemetry["subjectOverlap"] == "false"
    assert telemetry["tailOverlap"] == "false"
    if "grooming_qa" in query:
        assert telemetry["groomingActive"] == "true"
    if "rearing_qa" in query:
        assert telemetry["rearingActive"] == "true"
        frame = page.evaluate("window.annolidPoseDataset.getFrame()")
        assert frame["image"]["maximum_tail_segment_length_error"] <= 0.5
        subject_ann = frame["annotations"][0]
        kp_names = frame["category"]["keypoints"]
        kp_y = {
            name: subject_ann["keypoints"][idx * 3 + 1]
            for idx, name in enumerate(kp_names)
            if subject_ann["keypoints"][idx * 3 + 2] > 0
        }
        assert kp_y["tail_mid"] > kp_y["neck"]
        if "tail_tip" in kp_y:
            assert kp_y["tail_tip"] > kp_y["neck"]
    output = os.environ.get("ANNOLID_THREEJS_SCREENSHOT_DIR")
    if output:
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        page.screenshot(
            path=str(directory / (query.replace("&", "_").replace("=", "-") + ".png"))
        )
