from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from tests.test_two_mice_simulation import _find_node_binary


REPO_ROOT = Path(__file__).resolve().parents[1]
THREEJS_DIR = REPO_ROOT / "annolid" / "gui" / "assets" / "threejs"
RAT_CIRCUIT_JS = THREEJS_DIR / "rat_ncap_circuit.js"
RAT_MODEL_JS = THREEJS_DIR / "rat_ncap_model.js"
RAT_SCENE_JS = THREEJS_DIR / "rat_ncap_scene.js"
RAT_HTML = THREEJS_DIR / "rat_ncap.html"
RAT_V2_GLB = THREEJS_DIR / "rat_v2_skin.glb"


def _run_node(script: str) -> dict:
    node = _find_node_binary()
    if node is None:
        pytest.skip("Node.js binary unavailable for JS module tests.")
    completed = subprocess.run(
        [str(node), "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, (
        f"Node failed:\nSTDOUT: {completed.stdout}\nSTDERR: {completed.stderr}"
    )
    return json.loads(completed.stdout.strip())


def test_rat_ncap_js_assets_exist_and_expose_controls_and_dataset_api() -> None:
    assert RAT_CIRCUIT_JS.is_file()
    assert RAT_SCENE_JS.is_file()
    assert RAT_HTML.is_file()
    html = RAT_HTML.read_text(encoding="utf-8")
    scene = RAT_SCENE_JS.read_text(encoding="utf-8")
    for token in (
        "rat-ncap-panel",
        "rat-speed-slider",
        "rat-interface-select",
        "rat-coat-select",
        "rat-v0-slider",
        "rat-v3-slider",
        "rat-noise-slider",
        "rat-spinal-scope",
    ):
        assert token in html
    for token in (
        "window.annolidRatNCAP = Object.freeze",
        "window.annolidPoseDataset = Object.freeze",
        "captureSegmentationMasks",
        "canvas.dataset.sceneReady = 'true'",
        "canvas.dataset.datasetReady = 'true'",
    ):
        assert token in scene

    viewer_js = (THREEJS_DIR / "annolid_threejs_viewer.js").read_text(encoding="utf-8")
    for token in (
        "rat_v2_skinned",
        'import("./rat_ncap_model.js")',
        "applyRolloutFrame",
        "frameRatSimulationCamera",
        "data-threejs-rat-ready",
    ):
        assert token in viewer_js


def test_rat_ncap_js_spring_muscle_and_gait_transitions_and_ablations() -> None:
    module_url = RAT_CIRCUIT_JS.as_uri()
    script = f"""
import {{
  RAT_NCAP_SPEC,
  NeuromechanicalInterface,
  RatNCAPController,
  computeSpringMuscleParams,
  computeBumpyTerrainHeight,
}} from {json.dumps(module_url)};

// 1. Spring-muscle parameter matching identities
const low = [-1.2, -0.8];
const high = [0.9, 1.4];
const poseq = [-0.1, 0.25];
const keq = [0.6, 0.4];
const tmax = [1.5, 1.1];
const sp = computeSpringMuscleParams(low, high, poseq, keq, tmax, tmax);
const stiffErr = Math.max(
  Math.abs((sp.kf[0] + sp.ke[0]) - keq[0]),
  Math.abs((sp.kf[1] + sp.ke[1]) - keq[1]),
);
const eqTrq0 = sp.ke[0] * (high[0] - poseq[0] + sp.ce[0]) - sp.kf[0] * (poseq[0] - low[0] + sp.cf[0]);

// 2. Speed-dependent gait emergence
function runFinal(opts, steps = 130) {{
  const c = new RatNCAPController(opts);
  let last = null;
  for (let i = 0; i < steps; i++) last = c.step(0.02);
  return last;
}}
const sStand = runFinal({{ behavior: 'stand', targetSpeed: 0.0 }});
const sWalk = runFinal({{ behavior: 'walk', targetSpeed: 0.20 }});
const sTrot = runFinal({{ behavior: 'trot', targetSpeed: 0.50 }});
const sBound = runFinal({{ behavior: 'bound', targetSpeed: 0.90 }});

// 3. In-silico ablations at 0.45 m/s
const sV0Abl = runFinal({{ behavior: 'trot', targetSpeed: 0.45, ablation: 'v0v_v0d', ablationScale: 0.0 }});
const sV3Abl = runFinal({{ behavior: 'trot', targetSpeed: 0.45, ablation: 'v3f_v3a', ablationScale: 0.0 }});

// 4. Determinism under RG weight noise
const n1 = runFinal({{ behavior: 'trot', rgNoiseStd: 0.2, seed: 42 }}, 40);
const n2 = runFinal({{ behavior: 'trot', rgNoiseStd: 0.2, seed: 42 }}, 40);

// 5. All 4 interfaces produce finite torques
const ifaceOk = RAT_NCAP_SPEC.interfaces.every(name => {{
  const iface = new NeuromechanicalInterface(name);
  const tau = iface.computeTorque({{
    q: new Float64Array(iface.dof),
    dq: new Float64Array(iface.dof),
    flx: new Float64Array(iface.dof).fill(0.5),
    ext: new Float64Array(iface.dof).fill(0.5),
    qTarget: iface.poseq,
    kpScale: new Float64Array(iface.dof).fill(0.8),
    trqCmd: new Float64Array(iface.dof).fill(0.1),
  }});
  return tau.length === 31 && Array.from(tau).every(Number.isFinite);
}});

console.log(JSON.stringify({{
  totalDofs: RAT_NCAP_SPEC.totalDofs,
  locomotionDofs: RAT_NCAP_SPEC.locomotionDofs,
  stiffErr,
  eqTrq0,
  gaits: {{
    stand: sStand.gait,
    walk: sWalk.gait,
    trot: sTrot.gait,
    bound: sBound.gait,
    v0Abl: sV0Abl.gait,
    v3Abl: sV3Abl.gait,
  }},
  freqs: [sStand.stepFrequencyHz, sWalk.stepFrequencyHz, sTrot.stepFrequencyHz, sBound.stepFrequencyHz],
  detDiff: Math.abs(n1.forwardSpeedMps - n2.forwardSpeedMps) + Math.abs(n1.lumbarExtend - n2.lumbarExtend),
  bumpyH: computeBumpyTerrainHeight(0.0125, 0.0, 0.03, 0.05),
  ifaceOk,
}}));
"""
    res = _run_node(script)
    assert res["totalDofs"] == 38
    assert res["locomotionDofs"] == 31
    assert res["stiffErr"] < 1e-9
    assert abs(res["eqTrq0"]) < 1e-9
    assert res["gaits"] == {
        "stand": "stand",
        "walk": "walk",
        "trot": "trot",
        "bound": "bound",
        "v0Abl": "bound",
        "v3Abl": "trot",
    }
    freqs = res["freqs"]
    assert freqs[0] < freqs[1] < freqs[2] < freqs[3]
    assert res["detDiff"] < 1e-12
    assert res["bumpyH"] > 0.0
    assert res["ifaceOk"] is True


def test_rat_v2_glb_parses_with_41_bones_55_joints_and_16_keypoints() -> None:
    assert RAT_MODEL_JS.is_file()
    assert RAT_V2_GLB.is_file()
    model_url = RAT_MODEL_JS.as_uri()
    circuit_url = RAT_CIRCUIT_JS.as_uri()
    glb_path = str(RAT_V2_GLB)
    script = f"""
import fs from 'node:fs';
import {{ RAT_KEYPOINTS }} from {json.dumps(circuit_url)};
import {{
  RAT_V2_GLB_FILE,
  RAT_V2_COAT_TEXTURES,
  parseRatGlbBuffer,
}} from {json.dumps(model_url)};

const bytes = fs.readFileSync({json.dumps(glb_path)});
const parsed = parseRatGlbBuffer(bytes);
const vCount = parsed.positions.length / 3;
const fCount = parsed.indices.length / 3;
let maxWeightErr = 0;
for (let i = 0; i < vCount; i++) {{
  const sum = parsed.skinWeights[i * 4]
    + parsed.skinWeights[i * 4 + 1]
    + parsed.skinWeights[i * 4 + 2]
    + parsed.skinWeights[i * 4 + 3];
  maxWeightErr = Math.max(maxWeightErr, Math.abs(sum - 1.0));
}}
let fineMin = 999, fineMax = -1, coarseMin = 999, coarseMax = -1;
for (let i = 0; i < vCount; i++) {{
  const fine = parsed.partIds[i * 2];
  const coarse = parsed.partIds[i * 2 + 1];
  if (fine < fineMin) fineMin = fine;
  if (fine > fineMax) fineMax = fine;
  if (coarse < coarseMin) coarseMin = coarse;
  if (coarse > coarseMax) coarseMax = coarse;
}}
const allIbmFinite = Array.from(parsed.inverseBindMatrices).every(Number.isFinite);
const kpKeys = Object.keys(parsed.rig.keypoints);
const allKeypointsPresent = RAT_KEYPOINTS.every(k => kpKeys.includes(k));
console.log(JSON.stringify({{
  glbFile: RAT_V2_GLB_FILE,
  coatTextures: RAT_V2_COAT_TEXTURES,
  vCount,
  fCount,
  boneCount: parsed.rig.bones.length,
  jointCount: parsed.rig.joints.length,
  tendonCount: Object.keys(parsed.rig.tendons || {{}}).length,
  ibmLen: parsed.inverseBindMatrices.length,
  allIbmFinite,
  maxWeightErr,
  fineRange: [fineMin, fineMax],
  coarseRange: [coarseMin, coarseMax],
  allKeypointsPresent,
  copyright: parsed.gltf.asset?.copyright || '',
}}));
"""
    res = _run_node(script)
    assert res["glbFile"] == "rat_v2_skin.glb"
    assert set(res["coatTextures"].keys()) == {"black_hooded", "cinnamon", "gray"}
    for tex_name in res["coatTextures"].values():
        assert (THREEJS_DIR / tex_name).is_file()
    assert res["vCount"] == 14754
    assert res["fCount"] == 19608
    assert res["boneCount"] == 41
    assert res["jointCount"] == 55
    assert res["tendonCount"] == 8
    assert res["ibmLen"] == 41 * 16
    assert res["allIbmFinite"] is True
    assert res["maxWeightErr"] < 1e-4
    assert res["fineRange"] == [1, 11]
    assert res["coarseRange"] == [1, 4]
    assert res["allKeypointsPresent"] is True
    assert "MIT" in res["copyright"]
