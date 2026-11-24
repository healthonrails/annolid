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

    def load(query, filename="two_mice.html"):
        page.goto(f"{base}/threejs/{filename}?{query}", timeout=120_000)
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
        assert payload["image"]["maximum_tail_segment_length_error"] <= 0.05
        assert payload["image"]["maximum_tail_bend_angle_degrees"] <= 16.0
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
        "anatomy_qa=head-three-quarter",
        "anatomy_qa=mouth-front",
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
    if "contact_qa=tail" in query:
        frame = page.evaluate("window.annolidPoseDataset.getFrame()")
        assert frame["image"]["maximum_tail_segment_length_error"] <= 0.05
        assert frame["image"]["maximum_tail_bend_angle_degrees"] <= 16.0
    if "rearing_qa" in query:
        assert telemetry["rearingActive"] == "true"
        frame = page.evaluate("window.annolidPoseDataset.getFrame()")
        assert frame["image"]["maximum_tail_segment_length_error"] <= 0.05
        assert frame["image"]["maximum_tail_bend_angle_degrees"] <= 16.0
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


def test_motor_control_api_turns_in_place_and_walks_on_command(scene_page):
    page = scene_page("dataset=1&seed=321")
    result = page.evaluate(
        """() => {
          const api = window.annolidTwoMice;
          const dataset = window.annolidPoseDataset;
          const spec = api.getActionSpec();
          dataset.setFrame(0);
          const initial = api.getObservation(1);
          const errors = [];
          for (const [trackId, command] of [
            [1, { forwardSpeed: NaN }],
            [1, { turnRate: 1, gait: 'hop' }],
            [99, { forwardSpeed: 1 }],
          ]) {
            try { api.setCommand(trackId, command); errors.push(null); }
            catch (error) { errors.push(error.name); }
          }

          api.setCommand(1, { forwardSpeed: 0, turnRate: 1.5 });
          let frame = 0;
          let unwrappedTurn = 0;
          let previousHeading = initial.heading;
          let turnSteppingHz = 0;
          const turnStart = api.getObservation(1).position;
          for (let i = 0; i < 8; i++) {
            dataset.setFrame(++frame);
            const obs = api.getObservation(1);
            unwrappedTurn += Math.atan2(
              Math.sin(obs.heading - previousHeading),
              Math.cos(obs.heading - previousHeading)
            );
            previousHeading = obs.heading;
            turnSteppingHz = Math.max(turnSteppingHz, obs.motor.frequencyHz);
          }
          const turnEnd = api.getObservation(1).position;

          api.setCommand(1, { forwardSpeed: 4.0, turnRate: 0, gait: 'walk' });
          let maxForward = 0;
          let minStanceLimbs = 4;
          let maxFootSlip = 0;
          const gaits = new Set();
          for (let i = 0; i < 6; i++) {
            const payload = dataset.setFrame(++frame);
            const obs = api.getObservation(1);
            maxForward = Math.max(maxForward, obs.egocentricVelocity.forward);
            minStanceLimbs = Math.min(
              minStanceLimbs,
              Object.values(obs.motor.stance).filter(Boolean).length
            );
            maxFootSlip = Math.max(maxFootSlip, payload.image.maximum_foot_slip);
            gaits.add(obs.motor.gait);
          }
          const commanded = api.getObservation(1);

          // Extreme reverse+turn stress toward enclosure walls/corners must preserve
          // exact segment length, smooth curvature, and zero tail overlap.
          api.setCommand(1, { forwardSpeed: -4.8, turnRate: 2.0 });
          let maxReverseTailBend = 0;
          let maxReverseTailLenErr = 0;
          let reverseTailOverlap = false;
          for (let i = 0; i < 12; i++) {
            const payload = dataset.setFrame(++frame);
            maxReverseTailBend = Math.max(
              maxReverseTailBend,
              payload.image.maximum_tail_bend_angle_degrees
            );
            maxReverseTailLenErr = Math.max(
              maxReverseTailLenErr,
              payload.image.maximum_tail_segment_length_error
            );
            reverseTailOverlap = reverseTailOverlap || Boolean(payload.image.tail_overlap);
          }
          api.clearCommand(1);
          return {
            joints: spec.names.length,
            observedJoints: Object.keys(initial.jointPositions).sort(),
            specJoints: [...spec.names].sort(),
            footLimbs: Object.keys(initial.footContacts).sort(),
            errors,
            unwrappedTurn,
            turnDisplacement: Math.hypot(turnEnd[0] - turnStart[0], turnEnd[1] - turnStart[1]),
            turnSteppingHz,
            maxForward,
            minStanceLimbs,
            maxFootSlip,
            gaits: [...gaits],
            commandDuring: commanded.command,
            commandAfter: api.getObservation(1).command,
            commandedFlag: api.getDiagnostics().subjects[0].commanded,
            maxReverseTailBend,
            maxReverseTailLenErr,
            reverseTailOverlap,
            json: JSON.stringify(commanded).length > 0,
          };
        }"""
    )
    assert result["joints"] == 22
    assert result["observedJoints"] == result["specJoints"]
    assert result["footLimbs"] == ["LF", "LH", "RF", "RH"]
    assert result["errors"] == ["TypeError", "RangeError", "RangeError"]
    # 1.5 rad/s for 1.6 s, allowing for yaw inertia: a clear in-place turn.
    assert result["unwrappedTurn"] > 1.2
    assert result["turnDisplacement"] < 1.5
    assert result["turnSteppingHz"] > 0.5
    assert result["maxForward"] > 2.5
    assert result["gaits"] == ["walk"]
    # Lateral-sequence walk keeps at least two paws on the ground.
    assert result["minStanceLimbs"] >= 2
    assert result["maxFootSlip"] <= 0.075
    assert result["commandDuring"]["gait"] == "walk"
    assert result["commandAfter"] is None
    assert result["commandedFlag"] is False
    assert result["maxReverseTailBend"] <= 15.0
    assert result["maxReverseTailLenErr"] <= 0.02
    assert result["reverseTailOverlap"] is False
    assert result["json"] is True


def test_dataset_segmentation_masks_match_viewport_and_subjects(scene_page):
    import numpy as np

    page = scene_page(
        "dataset=1&seed=20260713&camera_view=oblique-close&scenario=rearing_social"
    )
    page.evaluate("window.annolidPoseDataset.setFrame(4)")
    before_gpu = page.evaluate("window.annolidTwoMice.getDiagnostics().renderer")
    seg = page.evaluate("window.annolidPoseDataset.captureSegmentationMasks()")
    after_gpu = page.evaluate("window.annolidTwoMice.getDiagnostics().renderer")
    # Repeated captures must reuse the offscreen target without leaking WebGL geometries/textures.
    seg_repeat = page.evaluate("window.annolidPoseDataset.captureSegmentationMasks()")
    repeat_gpu = page.evaluate("window.annolidTwoMice.getDiagnostics().renderer")
    assert repeat_gpu["geometries"] == after_gpu["geometries"]
    assert repeat_gpu["textures"] == after_gpu["textures"]
    assert seg == seg_repeat
    _ = before_gpu

    width = int(seg["width"])
    height = int(seg["height"])
    assert (width, height) == (1100, 800)

    rle = np.asarray(seg["rle"], dtype=np.int32).reshape(-1, 4)
    flat_rgb = np.repeat(rle[:, 1:4].astype(np.uint8), rle[:, 0], axis=0)
    assert flat_rgb.shape == (width * height, 3)
    id_img = flat_rgb.reshape((height, width, 3))
    track_map = id_img[:, :, 0]
    fine_map = id_img[:, :, 1]
    coarse_map = id_img[:, :, 2]

    assert set(np.unique(track_map).tolist()) == {0, 1, 2}
    # Background pixels must never carry body-part IDs; foreground pixels must always carry valid part IDs.
    assert not np.any((track_map == 0) & (coarse_map > 0))
    assert not np.any((track_map == 0) & (fine_map > 0))
    assert np.all(coarse_map[track_map > 0] > 0)
    assert np.all(fine_map[track_map > 0] > 0)
    # Oblique rearing scene exposes all 4 coarse parts (head, torso, limbs, tail) and fine limb IDs.
    assert {1, 2, 3, 4}.issubset(set(np.unique(coarse_map).tolist()))
    assert bool(set(np.unique(fine_map).tolist()) & {7, 8, 9, 10})


def test_fly_brain_controls_mouse_behaviors_and_plasticity(scene_page):
    # Launch via the standalone fly_brain_mouse.html wrapper to verify its redirect path too.
    page = scene_page("seed=777", filename="fly_brain_mouse.html")
    assert "two_mice.html" in page.url and "fly_brain=1" in page.url
    assert page.locator("#fly-brain-panel").is_visible()
    page.wait_for_function(
        "() => (window.annolidTwoMice.getFlyBrainState(1)?.totalSpikes || 0) > 60"
    )
    initial_state = page.evaluate("window.annolidTwoMice.getFlyBrainState(1)")
    assert initial_state["enabled"] is True
    assert initial_state["dynamics"] == "spiking"
    assert initial_state["numNeurons"] >= 200

    # Switching subject selector to 'both' attaches a fly brain to subject 2 as well.
    page.locator("#fb-target-select").select_option("both")
    assert page.evaluate("window.annolidTwoMice.getFlyBrainState(2)?.enabled") is True
    page.locator("#fb-target-select").select_option("black")
    assert page.evaluate("window.annolidTwoMice.getFlyBrainState(2)?.enabled") is False

    # Optogenetic Moonwalker (MDN) drives backward locomotion on the 3D mouse rig.
    page.locator('#fb-opto-grid button[data-preset-id="moonwalk"]').click()
    page.wait_for_function(
        """() => {
          const obs = window.annolidTwoMice.getObservation(1);
          const fb = window.annolidTwoMice.getFlyBrainState(1);
          return fb && fb.dominantCircuit === 'moonwalk_MDN' && obs.egocentricVelocity.forward < -0.4;
        }"""
    )

    output = os.environ.get("ANNOLID_THREEJS_SCREENSHOT_DIR")
    if output:
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(directory / "fly_brain_mouse.png"))

    # Optogenetic aDN1 triggers real 3D face-wash grooming posture.
    page.locator("#fb-clear-stim").click()
    page.locator('#fb-opto-grid button[data-preset-id="groom"]').click()
    page.wait_for_function(
        "() => window.annolidTwoMice.getObservation(1).behavior.grooming === 'face-wash'"
    )

    # Stonkfly-style PAM reward dopamine paired with Reward KC modulates KC -> MBON weights.
    page.locator("#fb-clear-stim").click()
    page.evaluate(
        """() => {
          const api = window.annolidTwoMice;
          api.resetFlyBrainPlasticity(1);
          api.stimulateFlyNeuron(1, 'KC_Reward_Zone', 1.4, 2.5);
          api.stimulateFlyNeuron(1, 'PAM', 1.6, 2.5);
        }"""
    )
    page.wait_for_function(
        """() => {
          const p = window.annolidTwoMice.getFlyBrainState(1)?.plasticity;
          return p && p.reward_approach.scale > 1.08 && p.reward_avoid.scale < 0.92;
        }""",
        timeout=60_000,
    )

    # Disabling Fly Brain hides the panel and returns the subject to autonomous control.
    page.evaluate("window.annolidTwoMice.disableFlyBrain(1)")
    assert not page.locator("#fly-brain-panel").is_visible()
    assert page.evaluate("window.annolidTwoMice.getObservation(1).command") is None


def test_neuroscience_arenas_switch_simulate_and_render(scene_page):
    new_arenas = (
        "circular_open_field",
        "elevated_plus_maze",
        "y_maze",
        "t_maze",
        "three_chamber",
        "light_dark_box",
        "barnes_maze",
        "novel_object",
    )
    page = scene_page("dataset=1&seed=20260719&arena=elevated_plus_maze")
    catalog = page.evaluate("window.annolidTwoMice.getArenas()")
    assert len(catalog) == 10
    assert page.evaluate("window.annolidTwoMice.getArena().id") == "elevated_plus_maze"

    invalid_error = page.evaluate(
        """() => {
          try { window.annolidTwoMice.setArena('does_not_exist'); return null; }
          catch (err) { return err.name; }
        }"""
    )
    assert invalid_error == "RangeError"

    output = os.environ.get("ANNOLID_THREEJS_SCREENSHOT_DIR")
    if output:
        Path(output).mkdir(parents=True, exist_ok=True)

    frame_cursor = 0
    for arena_id in new_arenas:
        switched = page.evaluate("id => window.annolidTwoMice.setArena(id)", arena_id)
        assert switched["id"] == arena_id
        for _ in range(4):
            frame_cursor += 1
            payload = page.evaluate(
                "f => window.annolidPoseDataset.setFrame(f)", frame_cursor
            )
        diag = page.evaluate("window.annolidTwoMice.getDiagnostics()")
        obs = page.evaluate("window.annolidTwoMice.getObservations()")
        assert diag["arena"] == arena_id
        assert len(diag["subjects"]) == 2
        assert all(s["zone"] in switched["zones"] for s in diag["subjects"])
        assert all(
            o["arena"] == arena_id and o["zone"] in switched["zones"] for o in obs
        )
        assert not payload["image"]["subject_overlap"], f"Subject overlap in {arena_id}"
        assert not payload["image"]["tail_overlap"], f"Tail overlap in {arena_id}"
        assert payload["image"]["maximum_tail_segment_length_error"] <= 0.05, (
            f"Tail stretch error in {arena_id}: {payload['image']}"
        )
        assert payload["image"]["maximum_tail_bend_angle_degrees"] <= 16.5, (
            f"Tail bend error in {arena_id}: {payload['image']}"
        )
        assert payload["image"]["maximum_paw_ground_error"] <= 0.1, (
            f"Paw ground error in {arena_id}: {payload['image']}"
        )
        if output:
            # Frame oblique view cleanly for visual inspection
            page.evaluate(
                """() => {
                  const a = window.annolidTwoMice.getArena();
                  return a.id;
                }"""
            )
            page.screenshot(path=str(Path(output) / f"arena_{arena_id}.png"))

    # Also verify live GUI dropdown arena switching works without errors
    live = scene_page("seed=42")
    live.locator(".lil-gui.root > .title").click()
    arena_select = (
        live.locator(".controller").filter(has_text="Arena").first.locator("select")
    )
    arena_select.select_option(label="Y-Maze")
    assert live.evaluate("window.annolidTwoMice.getArena().id") == "y_maze"
    assert "y-maze" in live.locator("#hud-title").inner_text().lower()
    arena_select.select_option(label="Three-Chamber Social")
    assert live.evaluate("window.annolidTwoMice.getArena().id") == "three_chamber"
    assert "three-chamber" in live.locator("#hud-title").inner_text().lower()

    # Verify Fly-Brain connectome runs cleanly inside a non-convex multi-arm maze (T-Maze)
    live.evaluate("window.annolidTwoMice.setArena('t_maze')")
    live.evaluate("window.annolidTwoMice.enableFlyBrain(1)")
    fb_meta = live.evaluate("window.annolidTwoMice.getFlyBrainMetadata()")
    assert fb_meta["zones"]["reward"]["x"] == pytest.approx(12.6, abs=0.1)
    assert fb_meta["zones"]["reward"]["z"] == pytest.approx(-8.6, abs=0.1)
    live.wait_for_function(
        "() => (window.annolidTwoMice.getFlyBrainState(1)?.totalSpikes || 0) > 20"
    )
    t_diag = live.evaluate("window.annolidTwoMice.getDiagnostics()")
    assert t_diag["arena"] == "t_maze"
    assert t_diag["subjects"][0]["zone"] in {
        "start_box",
        "start_stem",
        "choice_point",
        "left_arm",
        "right_arm",
    }
    live.evaluate("window.annolidTwoMice.disableFlyBrain(1)")
    live.goto("about:blank")
