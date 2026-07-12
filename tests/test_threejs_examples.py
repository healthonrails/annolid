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


def test_two_mice_example_preserves_subject_appearance_and_render_readiness():
    repo_root = Path(__file__).resolve().parents[1]
    source = (
        repo_root / "annolid" / "gui" / "assets" / "threejs" / "two_mice.html"
    ).read_text(encoding="utf-8")

    assert "this.appearance = Object.freeze({ furColor, skinColor, density" in source
    assert "this.fleshMat.color.set(this.guiParams.skinColor)" not in source
    assert "geo.rotateX(Math.PI / 2);" in source
    assert "canvas.dataset.sceneReady = 'true';" in source
    assert "Fur pigment applies to newly spawned subjects." in source
    assert "const portraitFit = THREE.MathUtils.clamp" in source
    assert "const cameraDistanceScale = 1.0 + portraitFit * 0.42" in source
    assert "const MOUSE_COLLISION_DISC_LAYOUT" in source
    assert "resolveAllMouseContacts(mice, dt);" in source
    assert "canvas.dataset.minimumObservedSubjectClearance" in source
    assert "canvas.dataset.subjectOverlap" in source
    assert "canvas.dataset.minimumObservedTailClearance" in source
    assert "canvas.dataset.tailOverlap" in source
    assert "canvas.dataset.observedTailOverlap" in source
    assert "function createDynamicTailTubeGeometry(" in source
    assert "function updateDynamicTailTubeGeometry(" in source
    assert "this.tailMesh.name = 'continuous-tail-tube'" in source
    assert "this.tailInstanced" not in source
    assert "const clampPointToArena" in source
    assert "Math.sin(i * 2.8)" not in source
    assert "segLen * 1.3" not in source
    assert "canvas.dataset.tailContinuityBySubject" in source
    assert "maximum_tail_segment_length_error" in source
    assert "canvas.dataset.maximumObservedFootSlip" in source
    assert "canvas.dataset.maximumObservedPawGroundError" in source
    assert "solveFootPlantConstraints(dt = 0.016)" in source
    assert "updateTailGeometry(otherMice = [])" in source
    assert "const getSegmentContact = (start, end, index)" in source
    assert "for (let pass = 0; pass < this.tailCount * 2; pass++)" in source
    assert "const CONTACT_QA = datasetQuery.get('contact_qa')" in source
    assert "const REARING_QA = datasetQuery.get('rearing_qa')" in source
    assert "const GROOMING_QA = datasetQuery.get('grooming_qa')" in source
    assert "const ANATOMY_QA = datasetQuery.get('anatomy_qa')" in source
    assert "REARING_QA === 'wall-transition'" in source
    assert "startRearing(mode, supportDirection = null, holdDuration = null)" in source
    assert "updateRearingBehavior(time, dt, context)" in source
    assert "function integrateRearAngularDynamics(" in source
    assert "captureRearFootAnchors()" in source
    assert "solveRearBalanceConstraints(dt = 0.016)" in source
    assert "solveRearSupportConstraints()" in source
    assert "const momentOfInertia" in source
    assert "const gravityTorque" in source
    assert "canvas.dataset.rearingPhysicsBySubject" in source
    assert "canvas.dataset.maximumObservedRearSupportDetails" in source
    assert "function updateVisualQaPixelTelemetry()" in source
    assert "canvas.dataset.pixelLuminanceRange" in source
    assert "rearSupportReleased" in source
    assert "mayRecoverAtRest" in source
    assert "this.rig.position.y += poseAmount * (supported ? 1.30 : 1.25)" not in source
    assert "stiffness * (target - this.rearAmount)" not in source
    assert "this.startRearing('wall-supported', supportDirection)" in source
    assert "this.startRearing('unsupported')" in source
    assert "canvas.dataset.rearingBySubject" in source
    assert "canvas.dataset.observedRearingModes" in source
    assert "rearing_mode: mouse.rearing.mode" in source
    assert "rearing_phase: mouse.rearing.phase" in source
    assert "behaviorFolder.add(guiParams, 'triggerRearing').name('Rear Now')" in source
    assert "startGrooming(mode, activeDuration = null)" in source
    assert "updateGroomingBehavior(time, dt, context)" in source
    assert "GROOMING_QA === 'face-wash'" in source
    assert "GROOMING_QA === 'flank-groom'" in source
    assert "canvas.dataset.groomingBySubject" in source
    assert "canvas.dataset.observedGroomingModes" in source
    assert "grooming_mode: mouse.grooming.mode" in source
    assert "grooming_phase: mouse.grooming.phase" in source
    assert "const leftWashStroke" in source
    assert "const rightWashStroke" in source
    assert "const pawLick" in source
    assert "const flankLick" in source
    assert "const FORELIMB_NEUTRAL = Object.freeze" in source
    assert "const uaLen = 0.42 * fl" in source
    assert "const faLen = 0.45 * fl" in source
    assert "scapular-soft-tissue-bridge" in source
    assert "chain._segmentLengths = Object.freeze" in source
    assert "chain._maximumReach = shoulderToElbow + elbowToWrist" in source
    assert "groomingContact.name = 'forepaw-palmar-contact'" in source
    assert "this.upperArmL = this.armL._upperArm" in source
    assert "[this.armL._wrist, this.armL._forearm, this.upperArmL]" in source
    assert "solveForelimbBehaviorConstraints()" in source
    assert "foot.maximumReach * 0.985" in source
    assert "const availableVerticalReach = Math.sqrt" in source
    assert "const loadedSupportSide = -this.grooming.side" in source
    assert "const flankRepositioningPaw = flankGroomSupport" in source
    assert "const maximumElbowExtension = flankGroomSupport ? -0.46 : -0.50" in source
    assert "const elbowPole = THREE.MathUtils.lerp" in source
    assert "[elbowPole, elbowPole]" in source
    assert "supported ? -0.58 : -1.20" in source
    assert "elbowX: -0.66 - leftWashStroke" in source
    assert "elbowX: 0.66 + leftWashStroke" not in source
    assert "const wristPosition = new THREE.Vector3()" in source
    assert "plant.pawReachRatio" in source
    assert "const faceWashRecovery = foot.isFore" in source
    assert "canvas.dataset.forelimbKinematicsBySubject" in source
    assert "grooming_paw_contact_error" in source
    assert "forelimb_extension_ratio" in source
    assert "this.armL.rotation.x" not in source
    assert "this.armR.rotation.x" not in source
    assert (
        "behaviorFolder.add(guiParams, 'triggerGrooming').name('Groom Now')" in source
    )
    assert "const thighMuscle = new THREE.Mesh" in source
    assert "const legLengthScale = params.legLength" in source
    assert "paw.name = isFore ? 'connected-forepaw' : 'connected-hindpaw'" in source
    assert "rootBridge.name = isFore ? 'forepaw-root-bridge'" in source
    assert "toe.add(toeTip);" in source
    assert "toe.add(toePad);" in source
    assert "toe.add(claw);" in source
    assert "paw._groundContact = groundContact;" in source
    assert "paw.add(toeTip)" not in source
    assert "paw.position.set(sign * 0.035, -0.13, 0.07)" in source
    assert "paw.position.set(sign * 0.045, -0.20, 0.31)" in source
    assert "this.hindL._paw.rotation.x = -0.10" in source
    assert "this.maxSpeed = params.maxSpeed" in source
    assert "bodyLength: 2.68, bodyGirthX: 1.27, bodyGirthY: 1.18" in source
    assert "function createPinnaShellGeometry()" in source
    assert "function createPinnaSurfaceGeometry(" in source
    assert "const lowerOpen = THREE.MathUtils.smoothstep" in source
    assert (
        "const attachmentBlend = THREE.MathUtils.smoothstep(-y, 0.58, 0.96)" in source
    )
    assert "const attachmentY = -0.70 + Math.abs(x) * 0.035" in source
    assert "const openingYaw = 0.20 +" in source
    assert "sideSign * openingYaw + asymmetryYaw" in source
    assert "ear.name = sideSign > 0 ? 'left-ear-root-pivot'" in source
    assert "pinna.position.set(sideSign * 0.012, 0.76, 0.0)" in source
    assert "0.02 + heightOffset" in source
    assert "rim.name = 'open-auricular-rim'" in source
    assert "pedicle.name = 'buried-pinna-pedicle'" in source
    assert "concha.name = 'basal-conchal-fold'" in source
    assert "tragus.name = 'auricular-tragus-fold'" in source
    assert "canal.name = 'ear-canal-shadow'" in source
    assert "baseFur.name = 'pinna-root-fur-collar'" in source
    assert "One continuous dorsal skull slope" in source
    assert "canvas.dataset.earAttachmentBySubject" in source
    assert "canvas.dataset.anatomySubject" in source
    assert "datasetQuery.get('anatomy_subject') === 'white'" in source
    assert "subject.tailMesh.visible = false" in source
    assert "ANATOMY_QA === 'ears-front'" in source
    assert "const farEarScale" not in source
    assert "earOuterColor: '#d6c9c7'" in source
    assert "function createTaperedWhiskerGeometry(" in source
    assert "function createAnatomicalWhiskers(" in source
    assert "earBackingGeo" not in source
    assert "new THREE.TubeGeometry" not in source
    assert "createTrackingNode('lforepaw', this.armL._paw" in source
    assert "createTrackingNode('lhindpaw', this.hindL._paw" in source
    assert "const legCycle = this.locomotionTime * 4.1" in source
    assert "const DATASET_CAPTURE" in source
    assert "const POSE_KEYPOINTS = Object.freeze" in source
    assert "window.annolidPoseDataset = Object.freeze" in source
    assert "setFrame: targetFrame =>" in source
    assert "track_id: mouse.trackId" in source
    assert "canvas.dataset.datasetReady = 'true';" in source


def test_two_mice_head_neck_body_transition_is_articulated_and_testable():
    repo_root = Path(__file__).resolve().parents[1]
    source = (
        repo_root / "annolid" / "gui" / "assets" / "threejs" / "two_mice.html"
    ).read_text(encoding="utf-8")

    assert "function getCervicalMantleProfile(t)" in source
    assert "function createCervicalMantleGeometry()" in source
    assert "indices.push(a, a + 1, b, b, a + 1, b + 1);" in source
    assert "neckMesh.name = 'tapered-cervical-mantle'" in source
    assert "new THREE.CapsuleGeometry(PROP.neckGirth" not in source
    assert "neckCoreMat.colorWrite = false" in source
    assert "this.cervicalUndercoat = cervicalUndercoat" in source
    assert "this.cervicalGuardHairs = cervicalGuardHairs" in source

    assert "this.headGroup.name = 'atlanto-occipital-head-pivot'" in source
    assert "this.head.position.set(0, 0.14 * chonk, PROP.bodyLength * 0.20)" in source
    assert "this.head.scale.set(" in source
    assert "this.guiParams.headSize * 0.92" in source
    assert "this.guiParams.headSize * 0.94" in source
    assert "this.guiParams.headSize * 1.03" in source
    assert "const occipitalBlend = THREE.MathUtils.smoothstep" in source
    assert "THREE.MathUtils.smoothstep(zNorm, 0.62, 0.98)" in source
    assert "neckT * (v.y >= 0 ? 0.25 : 0.10)" in source

    assert "this.head.localToWorld" in source
    assert "this.head.worldToLocal" in source
    assert "this.neckMesh.position.x += headOffsetX * 0.48" in source
    assert "this.neckMesh.rotation.y += this.headGroup.rotation.y * 0.30" in source
    assert "new THREE.Vector3(0, 0.46, -PROP.neckLength * 0.26)" in source

    assert "ANATOMY_QA === 'torso-side'" in source
    assert "ANATOMY_QA === 'torso-front'" in source
    assert "canvas.dataset.torsoProportions" in source
    assert "canvas.dataset.headNeckBodyBySubject" in source
    assert "canvas.dataset.torsoQa = IS_TORSO_ANATOMY_QA" in source
    assert "datasetQuery.get('anatomy_layer')" not in source


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
