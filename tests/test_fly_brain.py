from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from annolid.simulation.fly_brain import (
    CONNECTOME_RELEASE_METADATA,
    SUPPORTED_BRAIN_DYNAMICS,
    SUPPORTED_FLYBODY_BEHAVIORS,
    FlyBrainController,
    SparseSynapticWeights,
    build_calibrated_wingbeat_pattern,
    build_canonical_fly_connectome,
    build_spiking_fly_connectome,
    ensure_calibrated_wingbeat_path,
    load_connectome_graph,
    neurotransmitter_polarity,
)


def _walking_spec() -> SimpleNamespace:
    legs = (
        "T1_left",
        "T1_right",
        "T2_left",
        "T2_right",
        "T3_left",
        "T3_right",
    )
    names = [f"adhere_claw_{leg}" for leg in legs]
    names.extend(["head_abduct", "head_twist", "head", "abdomen_abduct", "abdomen"])
    for leg in legs:
        names.extend(
            [
                f"coxa_abduct_{leg}",
                f"coxa_twist_{leg}",
                f"coxa_{leg}",
                f"femur_twist_{leg}",
                f"femur_{leg}",
                f"tibia_{leg}",
                f"tarsus_{leg}",
                f"tarsus2_{leg}",
            ]
        )
    minimum = np.concatenate(
        [np.zeros(6, dtype=np.float64), np.full(len(names) - 6, -1.5, dtype=np.float64)]
    )
    maximum = np.concatenate(
        [np.ones(6, dtype=np.float64), np.full(len(names) - 6, 2.0, dtype=np.float64)]
    )
    return SimpleNamespace(
        name="\t".join(names),
        shape=(len(names),),
        dtype=np.float64,
        minimum=minimum,
        maximum=maximum,
    )


def _flight_spec() -> SimpleNamespace:
    names = (
        "head_abduct",
        "head_twist",
        "head",
        "wing_yaw_left",
        "wing_roll_left",
        "wing_pitch_left",
        "wing_yaw_right",
        "wing_roll_right",
        "wing_pitch_right",
        "abdomen_abduct",
        "abdomen",
        "user_0",
    )
    return SimpleNamespace(
        name="\t".join(names),
        shape=(len(names),),
        dtype=np.float64,
        minimum=np.full(len(names), -1.0, dtype=np.float64),
        maximum=np.full(len(names), 1.0, dtype=np.float64),
    )


def test_canonical_connectome_exposes_google_malecns_flywire_wiring() -> None:
    graph = build_canonical_fly_connectome()
    summary = graph.summary()

    assert summary["id"] == CONNECTOME_RELEASE_METADATA["id"]
    assert "male-cns:v1.0" in summary["reference_datasets"]
    assert summary["population_count"] >= 40
    assert summary["edge_count"] >= 50

    weights, tau, bias, idx = graph.compile_matrices()
    assert weights.shape == (len(graph.populations), len(graph.populations))
    assert tau.shape == (len(graph.populations),)
    assert bias.shape == (len(graph.populations),)
    # Excitatory CX -> DNp09 vs GABAergic Pair1 -> DNp09
    assert weights[idx["DNp09_L"], idx["CX_Forward_Goal"]] > 0.0
    assert weights[idx["DNp09_L"], idx["Pair1"]] < 0.0
    # Fraenkel tarsal contact inhibition onto flight DNg02
    assert weights[idx["DNg02_L"], idx["AN_Contact"]] < 0.0


def test_neurotransmitter_polarity_respects_drosophila_glucl_vs_nmj() -> None:
    assert neurotransmitter_polarity("acetylcholine") == 1.0
    assert neurotransmitter_polarity("gaba") == -1.0
    assert neurotransmitter_polarity("glutamate", post_region="vnc_premotor") < 0.0
    assert neurotransmitter_polarity("glutamate", post_region="motor") == 1.0


def test_load_connectome_graph_supports_csv_json_and_npz(tmp_path: Path) -> None:
    csv_path = tmp_path / "malecns_edges.csv"
    csv_path.write_text(
        "pre_type,post_type,weight,nt_type\n"
        "ORN_L,PFL3_L,600,acetylcholine\n"
        "Pair1,DNp09_L,500,gaba\n",
        encoding="utf-8",
    )
    csv_graph = load_connectome_graph(csv_path, merge_canonical=True)
    w_csv, _, _, idx_csv = csv_graph.compile_matrices()
    w_base, _, _, idx_base = build_canonical_fly_connectome().compile_matrices()
    assert (
        w_csv[idx_csv["PFL3_L"], idx_csv["ORN_L"]]
        > w_base[idx_base["PFL3_L"], idx_base["ORN_L"]]
    )

    json_path = tmp_path / "subcircuit.json"
    json_path.write_text(
        json.dumps(
            {
                "populations": [
                    {"cell_type": "Custom_Sens", "region": "sensory", "nt": "ach"},
                    {"cell_type": "Custom_DN", "region": "descending", "nt": "gaba"},
                ],
                "connections": [
                    {"pre": "Custom_Sens", "post": "Custom_DN", "synapse_count": 85}
                ],
            }
        ),
        encoding="utf-8",
    )
    json_graph = load_connectome_graph(json_path, merge_canonical=False)
    assert len(json_graph.populations) == 2
    assert len(json_graph.connections) == 1

    npz_path = tmp_path / "connectome.npz"
    np.savez(
        npz_path,
        cell_types=np.asarray(["A", "B"]),
        regions=np.asarray(["sensory", "descending"]),
        neurotransmitters=np.asarray(["ach", "ach"]),
        synapse_counts=np.asarray([[0.0, 0.0], [120.0, 0.0]]),
    )
    npz_graph = load_connectome_graph(npz_path, merge_canonical=False)
    w_npz, _, _, idx_npz = npz_graph.compile_matrices()
    assert w_npz[idx_npz["B"], idx_npz["A"]] > 0.0


def test_fly_brain_controller_behavioral_selectivity_and_bounds() -> None:
    spec = _walking_spec()
    names = spec.name.split("\t")
    act_idx = {name: i for i, name in enumerate(names)}

    # 1. Forward walk activates DNp09 while tarsal contact suppresses DNg02
    walk_ctrl = FlyBrainController(spec, behavior="walk_imitation", seed=11)
    obs_grounded = {"walker/touch": np.ones(6, dtype=np.float64)}
    for _ in range(40):
        act = walk_ctrl.step(obs_grounded, dt=0.002)
    assert np.all(act >= spec.minimum) and np.all(act <= spec.maximum)
    walk_diag = walk_ctrl.diagnostics()
    assert walk_diag["dominant_circuit"] == "dnp09_forward_walk"
    assert walk_diag["descending_neurons"]["DNp09"] > 0.5
    assert walk_diag["descending_neurons"]["DNg02"] < 0.1

    # 2. Backward walk activates Moonwalker (MDN) and suppresses DNp09
    back_ctrl = FlyBrainController(spec, behavior="backward_walk", seed=11)
    for _ in range(40):
        back_ctrl.step(obs_grounded, dt=0.002)
    back_diag = back_ctrl.diagnostics()
    assert back_diag["dominant_circuit"] == "moonwalker_backward"
    assert back_diag["descending_neurons"]["MDN"] > 0.5
    assert back_diag["descending_neurons"]["DNp09"] < 0.05

    # 3. Grooming activates aDN1 + Pair1, arrests T2/T3, and lifts T1 forelegs
    groom_ctrl = FlyBrainController(spec, behavior="grooming", seed=11)
    groom_act = np.zeros(len(names))
    for _ in range(45):
        groom_act = groom_ctrl.step(obs_grounded, dt=0.002)
    groom_diag = groom_ctrl.diagnostics()
    assert groom_diag["dominant_circuit"] == "antennal_grooming"
    assert groom_diag["descending_neurons"]["aDN1"] > 0.45
    assert groom_diag["descending_neurons"]["Pair1"] > 0.6
    assert groom_act[act_idx["adhere_claw_T1_left"]] == 0.0
    assert groom_act[act_idx["adhere_claw_T2_left"]] > 0.8
    assert groom_act[act_idx["coxa_T1_left"]] > 0.4

    # 4. Escape takeoff triggers Giant Fiber (DNp01) jump window
    esc_ctrl = FlyBrainController(spec, behavior="escape_takeoff", seed=11)
    saw_escape = False
    for _ in range(110):
        esc_act = esc_ctrl.step(obs_grounded, dt=0.002)
        if esc_ctrl.diagnostics()["dominant_circuit"] == "giant_fiber_escape":
            saw_escape = True
            assert esc_act[act_idx["femur_T2_left"]] > 1.0
            break
    assert saw_escape is True


def test_fly_brain_controller_vision_guided_flight_steers_toward_salient_eye() -> None:
    fspec = _flight_spec()
    names = fspec.name.split("\t")
    act_idx = {name: i for i, name in enumerate(names)}
    ctrl = FlyBrainController(
        fspec, behavior="vision_guided_flight", seed=5, control_timestep=0.0002
    )
    # Dark landmark on left eye (low luminance) vs bright sky on right eye
    obs = {
        "walker/left_eye": np.full((32, 32, 3), 30, dtype=np.uint8),
        "walker/right_eye": np.full((32, 32, 3), 235, dtype=np.uint8),
        "walker/velocimeter": np.asarray([1.0, 0.0, -0.2]),
    }
    for _ in range(80):
        act = ctrl.step(obs, dt=0.0002)
    assert np.all(act >= fspec.minimum) and np.all(act <= fspec.maximum)
    diag = ctrl.diagnostics()
    assert diag["dominant_circuit"] == "flight_tectulum"
    assert diag["sensory"]["visual_left"] > diag["sensory"]["visual_right"]
    assert act[act_idx["user_0"]] > 0.0


def test_fly_brain_controller_is_deterministic_across_reset() -> None:
    spec = _walking_spec()
    ctrl = FlyBrainController(spec, behavior="forage_walk", seed=23)
    seq_a = [ctrl.step({}, dt=0.002).copy() for _ in range(10)]
    ctrl.reset()
    seq_b = [ctrl.step({}, dt=0.002).copy() for _ in range(10)]
    for a, b in zip(seq_a, seq_b):
        assert np.allclose(a, b)
    assert len(SUPPORTED_FLYBODY_BEHAVIORS) == 9
    assert SUPPORTED_BRAIN_DYNAMICS == ("rate", "spiking")


def test_sparse_whole_cns_scale_and_spiking_lif_connectome(tmp_path: Path) -> None:
    # 1. Verify 166,700-neuron MaleCNS scale never allocates dense O(N^2) RAM
    n_cns = 166_700
    rng = np.random.default_rng(3)
    posts = rng.integers(0, n_cns, size=50_000, dtype=np.int32)
    pre_ids_arr = rng.integers(0, n_cns, size=50_000, dtype=np.int32)
    vals = rng.uniform(-0.5, 0.5, size=50_000)
    sparse_w = SparseSynapticWeights((n_cns, n_cns), posts, pre_ids_arr, vals)
    out = sparse_w.propagate_spikes(np.asarray([0, 10, 42], dtype=np.int32))
    assert out.shape == (n_cns,)

    # 2. Verify expanded ~1,300+ individual spiking LIF neuron connectome
    spiking_graph = build_spiking_fly_connectome(seed=7)
    assert len(spiking_graph.populations) >= 1200
    assert len(spiking_graph.connections) >= 5000

    spec = _walking_spec()
    lif_walk = FlyBrainController(
        spec,
        behavior="walk_imitation",
        connectome=spiking_graph,
        dynamics="spiking",
        seed=7,
    )
    lif_walk.warmup({"walker/touch": np.ones(6)}, duration_sec=0.04)
    act = lif_walk.step({"walker/touch": np.ones(6)}, dt=0.002)
    assert np.all(act >= spec.minimum) and np.all(act <= spec.maximum)
    diag = lif_walk.diagnostics()
    assert diag["dynamics"] == "spiking"
    assert diag["spiking"]["total_spikes"] > 0
    assert diag["dominant_circuit"] == "dnp09_forward_walk"

    # 3. Verify neuron-level bodyId pooling in sparse coordinate-list NPZ loader
    sparse_npz = tmp_path / "sparse_neurons.npz"
    np.savez(
        sparse_npz,
        neuron_ids=np.asarray(["body_1", "body_2", "body_3"]),
        cell_types=np.asarray(["ORN_L", "ORN_L", "PFL3_L"]),
        regions=np.asarray(["antennal_lobe", "antennal_lobe", "central_complex"]),
        neurotransmitters=np.asarray(["ach", "ach", "ach"]),
        pre_indices=np.asarray([0, 1], dtype=np.int32),
        post_indices=np.asarray([2, 2], dtype=np.int32),
        synapse_counts=np.asarray([90.0, 110.0], dtype=np.float64),
    )
    loaded = load_connectome_graph(sparse_npz, merge_canonical=False)
    groups = loaded.cell_type_groups()
    assert groups["ORN_L"].size == 2
    assert groups["PFL3_L"].size == 1


def test_calibrated_wingbeat_pattern_respects_mujoco_joint_limits(
    tmp_path: Path,
) -> None:
    pattern = build_calibrated_wingbeat_pattern(500)
    assert pattern.shape == (500, 3)
    # Must stay strictly inside fruitfly.xml joint ranges: yaw [-1.5, 1.5], roll [-1.0, 1.5], pitch [-1.27, 2.92]
    assert float(np.min(pattern[:, 0])) > -1.5 and float(np.max(pattern[:, 0])) < 1.5
    assert float(np.min(pattern[:, 1])) > -1.0 and float(np.max(pattern[:, 1])) < 1.5
    assert float(np.min(pattern[:, 2])) > -1.27 and float(np.max(pattern[:, 2])) < 2.92

    cached = ensure_calibrated_wingbeat_path(tmp_path)
    assert cached.exists()
    reloaded = np.load(cached)
    assert np.allclose(pattern, reloaded)
