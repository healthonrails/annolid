from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

from annolid.gui.rat_ncap_support import (
    build_live_rat_ncap_command,
    build_probe_rat_ncap_command,
    resolve_local_rat_ncap_repo,
    resolve_rat_xml_path,
)
from annolid.gui.threejs_examples import THREEJS_EXAMPLE_IDS, generate_threejs_example
from annolid.simulation import (
    RAT_V2_ACTUATOR_NAMES,
    RAT_V2_KEYPOINT_NAMES,
    RAT_V2_LOCOMOTION_ACTUATOR_NAMES,
    NeuromechanicalInterface,
    RatNCAPConfig,
    RatNCAPController,
    SimulationRunRequest,
    build_simulation_adapter,
    build_simulation_view_payload,
    classify_quadruped_gait,
    compute_mechanical_power,
    compute_spring_muscle_params,
    generate_rat_ncap_mapping_template,
    load_simulation_mapping,
    run_rat_ncap_rollout,
    run_simulation_workflow,
)
from annolid.simulation.rat_ncap_live import (
    build_live_rat_ncap_payload,
    probe_rat_ncap_runtime,
)


def test_rat_v2_body_specification_counts() -> None:
    assert len(RAT_V2_ACTUATOR_NAMES) == 38
    assert len(RAT_V2_LOCOMOTION_ACTUATOR_NAMES) == 31
    assert len(RAT_V2_KEYPOINT_NAMES) == 16


def test_spring_muscle_exact_parameter_matching_identities() -> None:
    """Verify exact SpringMuscleController parameter matching from Rat NCAP Appendix C.2."""
    low = np.array([-1.2, -0.8, -1.0])
    high = np.array([0.9, 1.4, 1.0])
    poseq = np.array([-0.1, 0.25, 0.1])
    keq = np.array([0.6, 0.4, 0.1])
    tmax_f = np.array([1.5, 1.1, 2.0])
    tmax_e = np.array([1.5, 1.1, 2.0])

    params = compute_spring_muscle_params(
        low=low,
        high=high,
        poseq=poseq,
        keq=keq,
        trq_max_flx=tmax_f,
        trq_max_ext=tmax_e,
    )
    # 1. Total stiffness at flx=ext=1 equals keq
    np.testing.assert_allclose(params.kf + params.ke, keq, atol=1e-9)

    # 2. At flx=ext=1 and q=poseq, net spring torque is identically zero
    xf_eq = poseq - low + params.cf
    xe_eq = high - poseq + params.ce
    trq_eq = params.ke * xe_eq - params.kf * xf_eq
    np.testing.assert_allclose(trq_eq, 0.0, atol=1e-9)

    # 3. At flx=1, ext=0, q=high -> torque == -trq_max_flx
    trq_flx_max = -params.kf * (high - low + params.cf)
    np.testing.assert_allclose(trq_flx_max, -tmax_f, atol=1e-9)

    # 4. At flx=0, ext=1, q=low -> torque == +trq_max_ext
    trq_ext_max = params.ke * (high - low + params.ce)
    np.testing.assert_allclose(trq_ext_max, tmax_e, atol=1e-9)


def test_all_four_neuromechanical_interfaces_compute_bounded_torques() -> None:
    q = np.zeros(31, dtype=np.float64)
    dq = np.full(31, 0.2, dtype=np.float64)
    for iface_name in ("flexor_extensor", "position_stiffness", "position", "torque"):
        iface = NeuromechanicalInterface(iface_name)
        assert iface.dof == 31
        tau = iface.compute_torque(
            q=q,
            dq=dq,
            flx=np.full(31, 0.6),
            ext=np.full(31, 0.4),
            q_target=iface.poseq + 0.1,
            kp_scale=np.full(31, 0.8),
            trq_cmd=np.full(31, 0.5),
        )
        assert tau.shape == (31,)
        assert np.all(np.abs(tau) <= iface.trq_max + 1e-9)
        power = compute_mechanical_power(tau, dq)
        assert power["positive_power_w"] >= 0.0
        assert power["negative_power_w"] <= 0.0
        assert math.isclose(
            power["net_power_w"],
            power["positive_power_w"] + power["negative_power_w"],
            abs_tol=1e-9,
        )


def test_speed_dependent_gait_emergence_stand_walk_trot_bound() -> None:
    stand = run_rat_ncap_rollout(steps=80, behavior="stand", target_speed=0.0)
    walk = run_rat_ncap_rollout(steps=120, behavior="walk", target_speed=0.20)
    trot = run_rat_ncap_rollout(steps=120, behavior="trot", target_speed=0.50)
    bound = run_rat_ncap_rollout(steps=120, behavior="bound", target_speed=0.90)

    assert stand["frames"][-1]["gait"] == "stand"
    assert walk["frames"][-1]["gait"] == "walk"
    assert trot["frames"][-1]["gait"] == "trot"
    assert bound["frames"][-1]["gait"] == "bound"
    # Step frequency increases monotonically across gaits
    assert (
        stand["frames"][-1]["step_frequency_hz"]
        < walk["frames"][-1]["step_frequency_hz"]
        < trot["frames"][-1]["step_frequency_hz"]
        < bound["frames"][-1]["step_frequency_hz"]
    )


def test_in_silico_v0_ablation_switches_trot_to_bound_while_v3_keeps_trot() -> None:
    """Reproduces Bellardita & Kiehn (2015) / Rat NCAP Fig. 5 in-silico ablation phenotypes."""
    intact = run_rat_ncap_rollout(
        steps=140, behavior="trot", target_speed=0.45, ablation="none"
    )
    v0_ablated = run_rat_ncap_rollout(
        steps=140,
        behavior="trot",
        target_speed=0.45,
        ablation="v0v_v0d",
        ablation_scale=0.0,
    )
    v3_ablated = run_rat_ncap_rollout(
        steps=140,
        behavior="trot",
        target_speed=0.45,
        ablation="v3f_v3a",
        ablation_scale=0.0,
    )
    assert intact["frames"][-1]["gait"] == "trot"
    assert v0_ablated["frames"][-1]["gait"] == "bound"
    assert v3_ablated["frames"][-1]["gait"] == "trot"


def test_rg_synaptic_weight_noise_is_sign_preserving_and_deterministic() -> None:
    c1 = RatNCAPController(RatNCAPConfig(rg_noise_std=0.25, seed=123))
    c2 = RatNCAPController(RatNCAPConfig(rg_noise_std=0.25, seed=123))
    for key, base in c1.rg._base_weights.items():
        w1 = c1.rg.weights[key]
        w2 = c2.rg.weights[key]
        assert w1 == pytest.approx(w2)
        if base > 0:
            assert w1 > 0
        elif base < 0:
            assert w1 < 0


def test_rat_ncap_workflow_fit_and_rollout_and_viewer_payload(tmp_path: Path) -> None:
    # 1. Checked-in mapping template loads cleanly
    template = load_simulation_mapping(Path("annolid/configs/rat_ncap_template.yaml"))
    assert template.backend == "rat_ncap"
    assert template.keypoint_to_site["left_forepaw"] == "finger_sensor_L"

    # 2. Programmatic template generator works
    gen_map = generate_rat_ncap_mapping_template(
        keypoints=["nose", "thorax", "left_forepaw", "custom_marker"]
    )
    assert gen_map.backend == "rat_ncap"
    assert gen_map.keypoint_to_site["left_forepaw"] == "finger_sensor_L"
    assert gen_map.keypoint_to_site["custom_marker"] == "custom_marker_site"

    # 3. Closed-loop rollout via run_simulation_workflow writes valid NDJSON + viewer payload
    out_ndjson = tmp_path / "rat_rollout.ndjson"
    written = run_simulation_workflow(
        SimulationRunRequest(
            backend="rat_ncap",
            out_ndjson=str(out_ndjson),
            rollout_steps=20,
            behavior="trot",
            interface="flexor_extensor",
            coat="cinnamon",
            seed=11,
        )
    )
    assert written.is_file()
    lines = written.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 20
    first_rec = json.loads(lines[0])
    assert first_rec["otherData"]["simulation"]["adapter"] == "rat_ncap"
    assert len(first_rec["otherData"]["simulation"]["state"]["qpos"]) == 31

    view_payload = build_simulation_view_payload(written)
    assert view_payload["kind"] == "annolid-simulation-v1"
    assert view_payload["adapter"] == "rat_ncap"
    assert len(view_payload["frames"]) == 20
    assert ["nose", "head"] in view_payload["edges"]
    assert ["thorax", "left_forepaw"] in view_payload["edges"]

    from annolid.simulation.viewer import export_simulation_view_payload

    exported_view_path = export_simulation_view_payload(written, out_dir=tmp_path)
    exported_payload = json.loads(exported_view_path.read_text(encoding="utf-8"))
    assert exported_payload["mesh"]["type"] == "rat_v2_skinned"
    assert exported_payload["mesh"]["coat"] == "cinnamon"
    assert exported_payload["metadata"]["coordinate_system"]["up_axis"] == "Y"
    assert exported_payload["playback"]["autoplay"] is True

    # 4. Pose 2D fitting mode works
    pose_json = tmp_path / "rat_pose.json"
    pose_json.write_text(
        json.dumps(
            {
                "version": "5.0",
                "imagePath": "frame_0.png",
                "imageHeight": 480,
                "imageWidth": 640,
                "shapes": [
                    {"label": "nose", "shape_type": "point", "points": [[200, 120]]},
                    {"label": "thorax", "shape_type": "point", "points": [[160, 140]]},
                    {"label": "pelvis", "shape_type": "point", "points": [[110, 145]]},
                ],
            }
        ),
        encoding="utf-8",
    )
    fit_ndjson = tmp_path / "rat_fit.ndjson"
    run_simulation_workflow(
        SimulationRunRequest(
            backend="rat_ncap",
            input_path=str(pose_json),
            mapping_path=str(Path("annolid/configs/rat_ncap_template.yaml")),
            out_ndjson=str(fit_ndjson),
            dry_run=True,
        )
    )
    fit_rec = json.loads(fit_ndjson.read_text(encoding="utf-8").splitlines()[0])
    assert "nose_site" in fit_rec["otherData"]["simulation"]["state"]["site_targets"]


def test_rat_ncap_live_probe_payload_support_and_threejs_examples(
    tmp_path: Path, monkeypatch
) -> None:
    probe = probe_rat_ncap_runtime()
    assert probe["ready"] is True
    assert probe["dof_total"] == 38
    assert probe["dof_locomotion"] == 31

    live_payload = build_live_rat_ncap_payload(
        steps=15, behavior="bound", interface="position_stiffness", coat="gray"
    )
    assert live_payload["kind"] == "annolid-simulation-v1"
    assert live_payload["adapter"] == "rat_ncap"
    assert live_payload["mesh"]["type"] == "rat_v2_skinned"
    assert live_payload["mesh"]["coat"] == "gray"
    assert live_payload["playback"]["autoplay"] is True
    assert "model_pose" in live_payload["frames"][0]
    assert len(live_payload["frames"]) == 15

    # Repo discovery helper respects ANNOLID_RAT_NCAP_PATH
    fake_repo = tmp_path / "rat-ncap"
    xml_path = fake_repo / "projects" / "rodent" / "envs" / "bodies" / "rat" / "rat.xml"
    xml_path.parent.mkdir(parents=True)
    xml_path.write_text("<mujoco model='rat'/>", encoding="utf-8")
    monkeypatch.setenv("ANNOLID_RAT_NCAP_PATH", str(fake_repo))
    assert resolve_local_rat_ncap_repo() == fake_repo
    assert resolve_rat_xml_path() == xml_path
    assert "--probe" in build_probe_rat_ncap_command(sys.executable)
    cmd = build_live_rat_ncap_command(
        sys.executable, out_path=tmp_path / "out.json", behavior="trot"
    )
    assert "annolid.simulation.rat_ncap_live" in cmd

    # Three.js examples include both rat_ncap_html and rat_ncap_simulation_json
    assert "rat_ncap_html" in THREEJS_EXAMPLE_IDS
    assert "rat_ncap_simulation_json" in THREEJS_EXAMPLE_IDS
    html_path = generate_threejs_example("rat_ncap_html", tmp_path)
    assert html_path.name == "rat_ncap.html"
    assert html_path.is_file()
    json_path = generate_threejs_example("rat_ncap_simulation_json", tmp_path)
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["adapter"] == "rat_ncap"
    assert parsed["mesh"]["type"] == "rat_v2_skinned"
    assert len(parsed["frames"]) == 360
    gaits_seen = {f["diagnostics"]["gait"] for f in parsed["frames"]}
    assert {"stand", "walk", "trot", "bound"}.issubset(gaits_seen)


def test_build_simulation_adapter_accepts_rat_alias() -> None:
    adapter = build_simulation_adapter("rat")
    assert adapter.name == "rat_ncap"


def test_threejs_server_whitelists_rat_ncap_assets() -> None:
    from annolid.gui.widgets.threejs_viewer_server import (
        _THREEJS_ALLOWED_ASSETS,
        _threejs_asset_path,
    )
    from scripts.check_distribution_artifacts import is_forbidden_member

    expected_assets = (
        "rat_ncap.html",
        "rat_ncap_scene.js",
        "rat_ncap_circuit.js",
        "rat_ncap_model.js",
        "rat_v2_skin.glb",
        "rat_blackhooded.jpg",
        "rat_cinnamon.jpg",
        "rat_gray.jpg",
    )
    for asset in expected_assets:
        assert asset in _THREEJS_ALLOWED_ASSETS
        assert _threejs_asset_path(asset).is_file()
        rel_member = f"annolid/gui/assets/threejs/{asset}"
        assert not is_forbidden_member(rel_member, artifact_kind="distribution")
        assert not is_forbidden_member(rel_member, artifact_kind="bundle")


def test_simulation_runner_cli_accepts_rat_ncap_rollout_and_template(
    tmp_path: Path,
) -> None:
    import argparse

    from annolid.engine.models.simulation_runner import SimulationRunnerPlugin

    plugin = SimulationRunnerPlugin()
    parser = argparse.ArgumentParser()
    plugin.add_predict_args(parser)

    tpl_path = tmp_path / "rat_tpl.json"
    args_tpl = parser.parse_args(
        [
            "--backend",
            "rat_ncap",
            "--template-keypoints",
            "nose,head,thorax,left_forepaw",
            "--write-mapping-template",
            str(tpl_path),
        ]
    )
    assert plugin.predict(args_tpl) == 0
    tpl_payload = json.loads(tpl_path.read_text(encoding="utf-8"))
    assert tpl_payload["backend"] == "rat_ncap"
    assert tpl_payload["keypoint_to_site"]["left_forepaw"] == "finger_sensor_L"

    out_ndjson = tmp_path / "cli_rat_rollout.ndjson"
    args_roll = parser.parse_args(
        [
            "--backend",
            "rat_ncap",
            "--rollout-steps",
            "12",
            "--behavior",
            "v0_ablation_bound",
            "--interface",
            "flexor_extensor",
            "--coat",
            "black_hooded",
            "--out-ndjson",
            str(out_ndjson),
        ]
    )
    assert plugin.predict(args_roll) == 0
    lines = out_ndjson.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 12


def test_gait_classifier_direct_cases() -> None:
    assert (
        classify_quadruped_gait(
            speed_mps=0.0, phase_offsets_rel_fr={"FL": 0.5, "HL": 0.0, "HR": 0.5}
        )
        == "stand"
    )
    assert (
        classify_quadruped_gait(
            speed_mps=0.22,
            phase_offsets_rel_fr={"FL": 0.5, "HL": 0.75, "HR": 0.25},
            stance_fraction=0.76,
        )
        == "walk"
    )
    assert (
        classify_quadruped_gait(
            speed_mps=0.50,
            phase_offsets_rel_fr={"FL": 0.5, "HL": 0.0, "HR": 0.5},
            stance_fraction=0.60,
        )
        == "trot"
    )
    assert (
        classify_quadruped_gait(
            speed_mps=0.85,
            phase_offsets_rel_fr={"FL": 0.02, "HL": 0.5, "HR": 0.52},
            stance_fraction=0.48,
        )
        == "bound"
    )
