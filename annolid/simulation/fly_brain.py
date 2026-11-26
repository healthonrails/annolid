from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

try:
    from scipy import sparse as _sp
except Exception:  # pragma: no cover - optional fallback when scipy is absent
    _sp = None


CONNECTOME_RELEASE_METADATA: Dict[str, Any] = {
    "id": "google-malecns-flywire-cns-v1",
    "title": "Google Research / HHMI Janelia / FlyWire Drosophila Brain + VNC Connectome",
    "reference_datasets": [
        "male-cns:v1.0",
        "flywire-fafb:v783",
        "manc:v1.2.1",
    ],
    "reference_scale": {
        "cns_neurons": 166700,
        "cns_synapses": 125000000,
    },
}

SUPPORTED_FLYBODY_BEHAVIORS: tuple[str, ...] = (
    "walk_imitation",
    "forage_walk",
    "backward_walk",
    "grooming",
    "escape_takeoff",
    "walk_on_ball",
    "flight_imitation",
    "vision_guided_flight",
    "template_task",
)

SUPPORTED_BRAIN_DYNAMICS: tuple[str, ...] = ("rate", "spiking")

BEHAVIOR_ENV_FACTORY_NAMES: Dict[str, str] = {
    "walk_imitation": "walk_imitation",
    "forage_walk": "template_task",
    "backward_walk": "template_task",
    "grooming": "template_task",
    "escape_takeoff": "template_task",
    "walk_on_ball": "walk_on_ball",
    "flight_imitation": "flight_imitation",
    "vision_guided_flight": "vision_guided_flight",
    "template_task": "template_task",
}

_NT_ALIASES: Dict[str, str] = {
    "ach": "ach",
    "acetylcholine": "ach",
    "cholinergic": "ach",
    "gaba": "gaba",
    "gabaergic": "gaba",
    "glu": "glu",
    "glut": "glu",
    "glutamate": "glu",
    "glutamatergic": "glu",
    "oa": "oa",
    "octopamine": "oa",
    "da": "da",
    "dopamine": "da",
    "5ht": "5ht",
    "serotonin": "5ht",
}

_LEG_IDS: tuple[str, ...] = (
    "T1_left",
    "T1_right",
    "T2_left",
    "T2_right",
    "T3_left",
    "T3_right",
)

_LEG_INHIBITORY_NEIGHBORS: Dict[str, tuple[str, ...]] = {
    "T1_left": ("T1_right", "T2_left"),
    "T1_right": ("T1_left", "T2_right"),
    "T2_left": ("T2_right", "T1_left", "T3_left"),
    "T2_right": ("T2_left", "T1_right", "T3_right"),
    "T3_left": ("T3_right", "T2_left"),
    "T3_right": ("T3_left", "T2_right"),
}

_CALIBRATED_WINGBEAT_FILENAME = "annolid_drosophila_calibrated_wingbeat_v1.npy"


def build_calibrated_wingbeat_pattern(timesteps: int = 500) -> np.ndarray:
    """Build a single aerodynamically trimmed Drosophila 3D wingbeat cycle `(timesteps, 3)`.

    Columns correspond to `(wing_yaw, wing_roll, wing_pitch)` in radians.
    Calibrated against `fruitfly.xml`'s ellipsoid fluid model (`g = 981 cm/s^2`,
    `m = 0.985 mg`) so cycle-averaged vertical lift exceeds body weight
    (`Fz ~ 1.04 dynes > 0.966 dynes`) while staying strictly inside joint limits
    (`yaw in [-1.48, 0.92]`, `roll in [-0.93, 1.03]`, `pitch in [-0.68, 0.92]`).
    """
    n = max(32, int(timesteps))
    phase = np.linspace(0.0, math.tau, n, endpoint=False, dtype=np.float64)
    yaw = 1.20 * np.cos(phase) - 0.28
    roll = -0.98 * np.cos(phase) + 0.05
    pitch = -0.80 * np.sin(phase) + 0.12
    return np.column_stack((yaw, roll, pitch))


def ensure_calibrated_wingbeat_path(
    cache_dir: str | Path | None = None,
) -> Path:
    """Materialize and cache the calibrated wingbeat `.npy` file for FlyBody WBPG."""
    target_dir = (
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else Path(tempfile.gettempdir()) / "annolid_flybody_cache"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / _CALIBRATED_WINGBEAT_FILENAME
    if not out_path.exists():
        pattern = build_calibrated_wingbeat_pattern(500)
        np.save(out_path, pattern)
    return out_path


def normalize_neurotransmitter(name: str | None) -> str:
    token = str(name or "ach").strip().lower()
    return _NT_ALIASES.get(token, "ach")


def neurotransmitter_polarity(
    neurotransmitter: str | None, *, post_region: str = ""
) -> float:
    """Return signed synaptic polarity under Drosophila CNS receptor biology."""
    nt = normalize_neurotransmitter(neurotransmitter)
    if nt == "ach":
        return 1.0
    if nt == "gaba":
        return -1.0
    if nt == "glu":
        return 1.0 if str(post_region).strip().lower() == "motor" else -0.6
    if nt == "oa":
        return 0.85
    if nt == "da":
        return 0.5
    if nt == "5ht":
        return -0.3
    return 1.0


class SparseSynapticWeights:
    """Memory-efficient `O(|E|)` sparse synaptic weight operator supporting 166k+ neurons.

    Uses `scipy.sparse.csr_matrix` when available and falls back to vectorized
    NumPy coordinate-list scatter-add so large whole-CNS graphs never allocate
    `O(N^2)` RAM.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        post_indices: np.ndarray,
        pre_indices: np.ndarray,
        values: np.ndarray,
    ) -> None:
        self.shape = (int(shape[0]), int(shape[1]))
        self._post = np.asarray(post_indices, dtype=np.int32)
        self._pre = np.asarray(pre_indices, dtype=np.int32)
        self._values = np.asarray(values, dtype=np.float64)
        self._csr = (
            _sp.csr_matrix(
                (self._values, (self._post, self._pre)),
                shape=self.shape,
                dtype=np.float64,
            )
            if _sp is not None
            else None
        )

    @property
    def nnz(self) -> int:
        return int(self._csr.nnz) if self._csr is not None else int(self._values.size)

    def matvec(self, x: np.ndarray) -> np.ndarray:
        vec = np.asarray(x, dtype=np.float64).reshape(-1)
        if self._csr is not None:
            return np.asarray(self._csr @ vec, dtype=np.float64)
        out = np.zeros(self.shape[0], dtype=np.float64)
        if self._values.size > 0:
            np.add.at(out, self._post, self._values * vec[self._pre])
        return out

    def propagate_spikes(self, spike_indices: np.ndarray) -> np.ndarray:
        """Event-driven `O(|active spikes|)` postsynaptic propagation for LIF networks."""
        idx = np.asarray(spike_indices, dtype=np.int32).reshape(-1)
        if idx.size == 0:
            return np.zeros(self.shape[0], dtype=np.float64)
        vec = np.zeros(self.shape[1], dtype=np.float64)
        vec[idx] = 1.0
        return self.matvec(vec)

    def __matmul__(self, other: np.ndarray) -> np.ndarray:
        return self.matvec(other)

    def __getitem__(self, key: tuple[int, int]) -> float:
        row, col = int(key[0]), int(key[1])
        if self._csr is not None:
            return float(self._csr[row, col])
        mask = (self._post == row) & (self._pre == col)
        return float(np.sum(self._values[mask])) if np.any(mask) else 0.0

    def to_dense(self) -> np.ndarray:
        if self._csr is not None:
            return self._csr.toarray()
        dense = np.zeros(self.shape, dtype=np.float64)
        if self._values.size > 0:
            np.add.at(dense, (self._post, self._pre), self._values)
        return dense

    def with_multiplicative_noise(
        self, sigma: float, rng: np.random.Generator
    ) -> SparseSynapticWeights:
        """Return a copy with Dale's-law sign-preserving multiplicative Gaussian noise."""
        noise_std = max(0.0, float(sigma))
        if noise_std <= 0.0 or self._values.size == 0:
            return SparseSynapticWeights(
                shape=self.shape,
                post_indices=self._post.copy(),
                pre_indices=self._pre.copy(),
                values=self._values.copy(),
            )
        eps = rng.normal(0.0, 1.0, size=self._values.size)
        scale = np.maximum(0.05, 1.0 + noise_std * eps)
        return SparseSynapticWeights(
            shape=self.shape,
            post_indices=self._post.copy(),
            pre_indices=self._pre.copy(),
            values=self._values * scale,
        )


@dataclass(frozen=True)
class NeuronPopulation:
    cell_type: str
    region: str
    neurotransmitter: str = "ach"
    hemisphere: str = "bilateral"
    tau_sec: float = 0.015
    bias: float = 0.0
    neuron_count: int = 2
    neuron_id: str | None = None

    @property
    def key(self) -> str:
        return str(self.neuron_id or self.cell_type)


@dataclass(frozen=True)
class SynapticConnection:
    pre: str
    post: str
    synapse_count: float
    neurotransmitter: str | None = None
    gain: float = 1.0


@dataclass(frozen=True)
class ConnectomeGraph:
    populations: tuple[NeuronPopulation, ...]
    connections: tuple[SynapticConnection, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def population_index(self) -> Dict[str, int]:
        return {pop.key: idx for idx, pop in enumerate(self.populations)}

    def cell_type_groups(self) -> Dict[str, np.ndarray]:
        groups: Dict[str, list[int]] = {}
        for idx, pop in enumerate(self.populations):
            groups.setdefault(pop.cell_type, []).append(idx)
        return {
            name: np.asarray(indices, dtype=np.int32)
            for name, indices in groups.items()
        }

    def compile_sparse(
        self, *, synapse_norm: float = 120.0
    ) -> tuple[SparseSynapticWeights, np.ndarray, np.ndarray, Dict[str, int]]:
        """Compile sparse `(W_sparse, tau, bias, index_map)` in `O(|V| + |E|)` memory."""
        n = len(self.populations)
        idx_map = self.population_index()
        # Fallback map by cell_type when an edge references a cell_type instead of neuron_id
        groups = self.cell_type_groups()
        denom = math.log1p(max(1.0, float(synapse_norm)))

        post_list: list[int] = []
        pre_list: list[int] = []
        val_list: list[float] = []

        for edge in self.connections:
            pre_idx = idx_map.get(edge.pre)
            post_idx = idx_map.get(edge.post)
            if pre_idx is not None and post_idx is not None:
                pre_pop = self.populations[pre_idx]
                post_pop = self.populations[post_idx]
                nt = edge.neurotransmitter or pre_pop.neurotransmitter
                polarity = neurotransmitter_polarity(nt, post_region=post_pop.region)
                strength = math.log1p(max(0.0, float(edge.synapse_count))) / denom
                post_list.append(post_idx)
                pre_list.append(pre_idx)
                val_list.append(polarity * float(edge.gain) * strength)
                continue

            # Support cell-type-level edges overlaid onto neuron-level graphs
            pre_Targets = (
                [pre_idx]
                if pre_idx is not None
                else groups.get(edge.pre, np.empty(0, dtype=np.int32)).tolist()
            )
            post_Targets = (
                [post_idx]
                if post_idx is not None
                else groups.get(edge.post, np.empty(0, dtype=np.int32)).tolist()
            )
            if not pre_Targets or not post_Targets:
                continue
            scale_fan_in = 1.0 / max(1, len(pre_Targets))
            strength = math.log1p(max(0.0, float(edge.synapse_count))) / denom
            for p_i in pre_Targets:
                pre_pop = self.populations[p_i]
                nt = edge.neurotransmitter or pre_pop.neurotransmitter
                for q_i in post_Targets:
                    post_pop = self.populations[q_i]
                    polarity = neurotransmitter_polarity(
                        nt, post_region=post_pop.region
                    )
                    post_list.append(q_i)
                    pre_list.append(p_i)
                    val_list.append(
                        polarity * float(edge.gain) * strength * scale_fan_in
                    )

        sparse_w = SparseSynapticWeights(
            shape=(n, n),
            post_indices=np.asarray(post_list, dtype=np.int32),
            pre_indices=np.asarray(pre_list, dtype=np.int32),
            values=np.asarray(val_list, dtype=np.float64),
        )
        tau = np.asarray(
            [max(0.002, float(pop.tau_sec)) for pop in self.populations],
            dtype=np.float64,
        )
        bias = np.asarray(
            [float(pop.bias) for pop in self.populations],
            dtype=np.float64,
        )
        return sparse_w, tau, bias, idx_map

    def compile_matrices(
        self, *, synapse_norm: float = 120.0, max_dense_nodes: int = 2048
    ) -> tuple[
        np.ndarray | SparseSynapticWeights, np.ndarray, np.ndarray, Dict[str, int]
    ]:
        """Compile synaptic matrices; returns dense ndarray for compact circuits and `SparseSynapticWeights` above `max_dense_nodes`."""
        sparse_w, tau, bias, idx_map = self.compile_sparse(synapse_norm=synapse_norm)
        if len(self.populations) <= int(max_dense_nodes):
            return sparse_w.to_dense(), tau, bias, idx_map
        return sparse_w, tau, bias, idx_map

    def summary(self) -> Dict[str, Any]:
        total_neurons = int(sum(max(1, pop.neuron_count) for pop in self.populations))
        total_synapses = int(
            round(sum(max(0.0, float(edge.synapse_count)) for edge in self.connections))
        )
        return {
            **dict(self.metadata),
            "population_count": len(self.populations),
            "cell_type_count": len(self.cell_type_groups()),
            "edge_count": len(self.connections),
            "represented_neurons": total_neurons,
            "represented_synapses": total_synapses,
        }


def build_canonical_fly_connectome() -> ConnectomeGraph:
    """Build the canonical Drosophila Brain + VNC sensorimotor connectome graph."""
    populations = (
        # Sensory & Primary Neuropils (including 5ms low-pass Haltere gyro sensors)
        NeuronPopulation("ORN_L", "antennal_lobe", "ach", "L", 0.018, 0.0, 60),
        NeuronPopulation("ORN_R", "antennal_lobe", "ach", "R", 0.018, 0.0, 60),
        NeuronPopulation("JO_Mech", "sensory", "ach", "bilateral", 0.012, 0.0, 120),
        NeuronPopulation("Tactile_Ant", "sensory", "ach", "bilateral", 0.010, 0.0, 40),
        NeuronPopulation("Visual_Target_L", "optic_lobe", "ach", "L", 0.014, 0.0, 90),
        NeuronPopulation("Visual_Target_R", "optic_lobe", "ach", "R", 0.014, 0.0, 90),
        NeuronPopulation(
            "LC4_Looming", "optic_lobe", "ach", "bilateral", 0.006, 0.0, 110
        ),
        NeuronPopulation(
            "Ocelli_Pitch", "optic_lobe", "ach", "bilateral", 0.006, 0.0, 24
        ),
        NeuronPopulation("Ocelli_Roll_L", "optic_lobe", "ach", "L", 0.006, 0.0, 16),
        NeuronPopulation("Ocelli_Roll_R", "optic_lobe", "ach", "R", 0.006, 0.0, 16),
        NeuronPopulation(
            "Haltere_Pitch_Down", "sensory", "ach", "bilateral", 0.005, 0.0, 40
        ),
        NeuronPopulation(
            "Haltere_Pitch_Up", "sensory", "ach", "bilateral", 0.005, 0.0, 40
        ),
        NeuronPopulation("Haltere_Roll_L", "sensory", "ach", "L", 0.005, 0.0, 30),
        NeuronPopulation("Haltere_Roll_R", "sensory", "ach", "R", 0.005, 0.0, 30),
        NeuronPopulation("FeCO_Load_L", "sensory", "ach", "L", 0.010, 0.0, 70),
        NeuronPopulation("FeCO_Load_R", "sensory", "ach", "R", 0.010, 0.0, 70),
        # Central Complex (CX), Lateral Accessory Lobe (LAL), and SEZ
        NeuronPopulation(
            "CX_Forward_Goal", "central_complex", "ach", "bilateral", 0.025, 0.0, 32
        ),
        NeuronPopulation("PFL3_L", "central_complex", "ach", "L", 0.016, 0.0, 12),
        NeuronPopulation("PFL3_R", "central_complex", "ach", "R", 0.016, 0.0, 12),
        NeuronPopulation(
            "PFL2_Brake", "central_complex", "ach", "bilateral", 0.018, 0.0, 12
        ),
        NeuronPopulation("LAL_Inh_L", "lal", "gaba", "L", 0.014, 0.0, 18),
        NeuronPopulation("LAL_Inh_R", "lal", "gaba", "R", 0.014, 0.0, 18),
        NeuronPopulation("SEZ_Groom", "sez", "ach", "bilateral", 0.020, 0.0, 16),
        NeuronPopulation(
            "Flight_Command", "central_complex", "oa", "bilateral", 0.015, 0.0, 14
        ),
        # Ascending Neurons (VNC -> Brain)
        NeuronPopulation("AN_Contact", "ascending", "ach", "bilateral", 0.012, 0.0, 20),
        NeuronPopulation("AN_Stride_L", "ascending", "ach", "L", 0.015, 0.0, 12),
        NeuronPopulation("AN_Stride_R", "ascending", "ach", "R", 0.015, 0.0, 12),
        # Neck-Connective Descending Neurons (Brain -> VNC)
        NeuronPopulation("DNp09_L", "descending", "ach", "L", 0.015, 0.0, 2),
        NeuronPopulation("DNp09_R", "descending", "ach", "R", 0.015, 0.0, 2),
        NeuronPopulation("DNa01_L", "descending", "ach", "L", 0.014, 0.0, 2),
        NeuronPopulation("DNa01_R", "descending", "ach", "R", 0.014, 0.0, 2),
        NeuronPopulation("DNa02_L", "descending", "ach", "L", 0.010, 0.0, 2),
        NeuronPopulation("DNa02_R", "descending", "ach", "R", 0.010, 0.0, 2),
        NeuronPopulation("MDN", "descending", "ach", "bilateral", 0.015, 0.0, 4),
        NeuronPopulation("Pair1", "descending", "gaba", "bilateral", 0.014, 0.0, 2),
        NeuronPopulation("aDN1", "descending", "ach", "bilateral", 0.016, 0.0, 4),
        NeuronPopulation("DNp01", "descending", "ach", "bilateral", 0.005, -0.18, 2),
        NeuronPopulation("DNg02_L", "descending", "ach", "L", 0.010, 0.0, 15),
        NeuronPopulation("DNg02_R", "descending", "ach", "R", 0.010, 0.0, 15),
        NeuronPopulation(
            "DNb01_Pitch", "descending", "ach", "bilateral", 0.008, 0.0, 4
        ),
        # Ventral Nerve Cord (VNC) Premotor & Motor Pools
        NeuronPopulation(
            "VNC_Walk_Drive", "vnc_premotor", "ach", "bilateral", 0.014, 0.0, 48
        ),
        NeuronPopulation("VNC_Turn_L", "vnc_premotor", "ach", "L", 0.012, 0.0, 24),
        NeuronPopulation("VNC_Turn_R", "vnc_premotor", "ach", "R", 0.012, 0.0, 24),
        NeuronPopulation(
            "VNC_Reverse", "vnc_premotor", "ach", "bilateral", 0.012, 0.0, 16
        ),
        NeuronPopulation(
            "VNC_MDN_Inh", "vnc_premotor", "gaba", "bilateral", 0.012, 0.0, 12
        ),
        NeuronPopulation(
            "VNC_Groom_T1", "vnc_premotor", "ach", "bilateral", 0.014, 0.0, 20
        ),
        NeuronPopulation(
            "VNC_Jump", "vnc_premotor", "ach", "bilateral", 0.006, 0.0, 16
        ),
        NeuronPopulation("WTct_Wing_L", "motor", "glu", "L", 0.008, 0.0, 24),
        NeuronPopulation("WTct_Wing_R", "motor", "glu", "R", 0.008, 0.0, 24),
        NeuronPopulation("MN_Pitch_Up", "motor", "glu", "bilateral", 0.006, 0.0, 12),
        NeuronPopulation("MN_Pitch_Down", "motor", "glu", "bilateral", 0.006, 0.0, 12),
        NeuronPopulation("MN_Roll_L", "motor", "glu", "L", 0.006, 0.0, 12),
        NeuronPopulation("MN_Roll_R", "motor", "glu", "R", 0.006, 0.0, 12),
        NeuronPopulation("MN_Stride_L", "motor", "glu", "L", 0.010, 0.0, 36),
        NeuronPopulation("MN_Stride_R", "motor", "glu", "R", 0.010, 0.0, 36),
    )

    connections = (
        # Chemotaxis & Visual Target -> Central Complex & LAL Steering
        SynapticConnection("ORN_L", "PFL3_L", 130),
        SynapticConnection("ORN_R", "PFL3_R", 130),
        SynapticConnection("ORN_L", "CX_Forward_Goal", 75),
        SynapticConnection("ORN_R", "CX_Forward_Goal", 75),
        SynapticConnection("Visual_Target_L", "PFL3_L", 165),
        SynapticConnection("Visual_Target_R", "PFL3_R", 165),
        SynapticConnection("Visual_Target_L", "DNa02_L", 110),
        SynapticConnection("Visual_Target_R", "DNa02_R", 110),
        SynapticConnection("Visual_Target_L", "CX_Forward_Goal", 60),
        SynapticConnection("Visual_Target_R", "CX_Forward_Goal", 60),
        # Central Complex Forward Goal -> DNp09 Bilateral Forward Locomotion
        SynapticConnection("CX_Forward_Goal", "DNp09_L", 210),
        SynapticConnection("CX_Forward_Goal", "DNp09_R", 210),
        SynapticConnection("DNp09_L", "DNp09_R", 45),
        SynapticConnection("DNp09_R", "DNp09_L", 45),
        # LAL Flip-Flop Mutual Inhibition & Asymmetric Steering DNs
        SynapticConnection("PFL3_L", "DNa01_L", 180),
        SynapticConnection("PFL3_R", "DNa01_R", 180),
        SynapticConnection("PFL3_L", "DNa02_L", 145),
        SynapticConnection("PFL3_R", "DNa02_R", 145),
        SynapticConnection("PFL3_L", "LAL_Inh_L", 120),
        SynapticConnection("PFL3_R", "LAL_Inh_R", 120),
        SynapticConnection("LAL_Inh_L", "DNa01_R", 135, "gaba"),
        SynapticConnection("LAL_Inh_R", "DNa01_L", 135, "gaba"),
        SynapticConnection("LAL_Inh_L", "DNa02_R", 120, "gaba"),
        SynapticConnection("LAL_Inh_R", "DNa02_L", 120, "gaba"),
        SynapticConnection("PFL2_Brake", "Pair1", 110),
        # Antennal Mechanosensory -> SEZ -> aDN1 Grooming + Pair1 Locomotor Halt
        SynapticConnection("JO_Mech", "SEZ_Groom", 190),
        SynapticConnection("SEZ_Groom", "aDN1", 220),
        SynapticConnection("SEZ_Groom", "Pair1", 200),
        SynapticConnection("aDN1", "Pair1", 140),
        SynapticConnection("aDN1", "VNC_Groom_T1", 260),
        # Pair1 GABAergic Arrest of Walking DNs and VNC Drive
        SynapticConnection("Pair1", "DNp09_L", 240, "gaba"),
        SynapticConnection("Pair1", "DNp09_R", 240, "gaba"),
        SynapticConnection("Pair1", "VNC_Walk_Drive", 260, "gaba"),
        # Anterior Aversive Tactile -> Moonwalker (MDN) Backward Walking
        SynapticConnection("Tactile_Ant", "MDN", 250),
        SynapticConnection("MDN", "VNC_MDN_Inh", 200),
        SynapticConnection("VNC_MDN_Inh", "DNp09_L", 220, "gaba"),
        SynapticConnection("VNC_MDN_Inh", "DNp09_R", 220, "gaba"),
        SynapticConnection("MDN", "VNC_Reverse", 280),
        SynapticConnection("MDN", "VNC_Walk_Drive", 230),
        # Looming Visual Threat -> LC4 -> Giant Fiber (DNp01) Escape Jump & Takeoff
        SynapticConnection("LC4_Looming", "DNp01", 340),
        SynapticConnection("LC4_Looming", "Flight_Command", 160),
        SynapticConnection("DNp01", "VNC_Jump", 420),
        SynapticConnection("DNp01", "DNg02_L", 180),
        SynapticConnection("DNp01", "DNg02_R", 180),
        # Flight Command, Ocelli & Haltere Attitude Stabilization -> DNg02 & Steering MNs
        SynapticConnection("Flight_Command", "DNg02_L", 230, "oa"),
        SynapticConnection("Flight_Command", "DNg02_R", 230, "oa"),
        SynapticConnection("Visual_Target_L", "DNg02_R", 110),
        SynapticConnection("Visual_Target_R", "DNg02_L", 110),
        SynapticConnection("Ocelli_Pitch", "DNb01_Pitch", 180),
        SynapticConnection("Haltere_Pitch_Down", "DNb01_Pitch", 220),
        SynapticConnection("Haltere_Pitch_Down", "MN_Pitch_Up", 260),
        SynapticConnection("DNb01_Pitch", "MN_Pitch_Up", 210),
        SynapticConnection("Haltere_Pitch_Up", "MN_Pitch_Down", 240),
        SynapticConnection("Ocelli_Roll_L", "MN_Roll_L", 200),
        SynapticConnection("Haltere_Roll_L", "MN_Roll_L", 240),
        SynapticConnection("Ocelli_Roll_R", "MN_Roll_R", 200),
        SynapticConnection("Haltere_Roll_R", "MN_Roll_R", 240),
        SynapticConnection("DNg02_L", "WTct_Wing_L", 260),
        SynapticConnection("DNg02_R", "WTct_Wing_R", 260),
        # Proprioceptive Tarsal Load -> Ascending & VNC Stance Feedback (Fraenkel tarsal reflex)
        SynapticConnection("FeCO_Load_L", "AN_Contact", 140),
        SynapticConnection("FeCO_Load_R", "AN_Contact", 140),
        SynapticConnection("AN_Contact", "DNg02_L", 190, "gaba"),
        SynapticConnection("AN_Contact", "DNg02_R", 190, "gaba"),
        SynapticConnection("FeCO_Load_L", "MN_Stride_L", 55),
        SynapticConnection("FeCO_Load_R", "MN_Stride_R", 55),
        SynapticConnection("AN_Stride_L", "PFL3_R", 30),
        SynapticConnection("AN_Stride_R", "PFL3_L", 30),
        # Descending Locomotor & Steering Drive -> VNC Thoracic Motor Pools
        SynapticConnection("DNp09_L", "VNC_Walk_Drive", 200),
        SynapticConnection("DNp09_R", "VNC_Walk_Drive", 200),
        SynapticConnection("DNa01_L", "VNC_Turn_L", 190),
        SynapticConnection("DNa02_L", "VNC_Turn_L", 175),
        SynapticConnection("DNa01_R", "VNC_Turn_R", 190),
        SynapticConnection("DNa02_R", "VNC_Turn_R", 175),
        SynapticConnection("VNC_Walk_Drive", "MN_Stride_L", 220),
        SynapticConnection("VNC_Walk_Drive", "MN_Stride_R", 220),
        SynapticConnection("VNC_Turn_L", "MN_Stride_R", 150),
        SynapticConnection("VNC_Turn_L", "MN_Stride_L", 130, "gaba"),
        SynapticConnection("VNC_Turn_R", "MN_Stride_L", 150),
        SynapticConnection("VNC_Turn_R", "MN_Stride_R", 130, "gaba"),
    )

    return ConnectomeGraph(
        populations=populations,
        connections=connections,
        metadata=dict(CONNECTOME_RELEASE_METADATA),
    )


def build_spiking_fly_connectome(
    *,
    seed: int = 7,
    connection_probability: float = 0.38,
) -> ConnectomeGraph:
    """Expand the Brain + VNC connectome into ~1,200+ individual spiking LIF neurons.

    Each biological population is instantiated at its `neuron_count` multiplicity
    with sparse, seeded, heterogeneous synaptic connectivity preserving exact
    cell-type neurotransmitter identities and pathway strengths.
    """
    canonical = build_canonical_fly_connectome()
    rng = np.random.default_rng(int(seed))
    prob = float(np.clip(connection_probability, 0.05, 1.0))

    indiv_pops: list[NeuronPopulation] = []
    ids_by_type: Dict[str, list[str]] = {}
    for pop in canonical.populations:
        count = max(1, int(pop.neuron_count))
        member_ids: list[str] = []
        for k in range(count):
            nid = f"{pop.cell_type}#{k}"
            tau_jitter = float(pop.tau_sec) * float(rng.uniform(0.90, 1.10))
            indiv_pops.append(
                NeuronPopulation(
                    cell_type=pop.cell_type,
                    region=pop.region,
                    neurotransmitter=pop.neurotransmitter,
                    hemisphere=pop.hemisphere,
                    tau_sec=max(0.003, tau_jitter),
                    bias=pop.bias,
                    neuron_count=1,
                    neuron_id=nid,
                )
            )
            member_ids.append(nid)
        ids_by_type[pop.cell_type] = member_ids

    indiv_edges: list[SynapticConnection] = []
    for edge in canonical.connections:
        pre_ids = ids_by_type.get(edge.pre, [])
        post_ids = ids_by_type.get(edge.post, [])
        if not pre_ids or not post_ids:
            continue
        n_pre = len(pre_ids)
        n_post = len(post_ids)
        if n_pre * n_post <= 8:
            # Small command bottlenecks (e.g., Giant Fiber DNp01, DNa01, Pair1): all-to-all
            fan_in = float(n_pre)
            for q_id in post_ids:
                for p_id in pre_ids:
                    indiv_edges.append(
                        SynapticConnection(
                            pre=p_id,
                            post=q_id,
                            synapse_count=float(edge.synapse_count),
                            neurotransmitter=edge.neurotransmitter,
                            gain=float(edge.gain) / fan_in,
                        )
                    )
        else:
            mask = rng.random((n_post, n_pre)) < prob
            # Guarantee every postsynaptic neuron receives at least one input
            empty_rows = np.where(~np.any(mask, axis=1))[0]
            for r in empty_rows.tolist():
                mask[r, int(rng.integers(0, n_pre))] = True
            for q_idx, q_id in enumerate(post_ids):
                active_pre = np.where(mask[q_idx])[0]
                fan_in = float(active_pre.size)
                for p_idx in active_pre.tolist():
                    syn_jitter = float(edge.synapse_count) * float(
                        rng.uniform(0.85, 1.15)
                    )
                    indiv_edges.append(
                        SynapticConnection(
                            pre=pre_ids[p_idx],
                            post=q_id,
                            synapse_count=syn_jitter,
                            neurotransmitter=edge.neurotransmitter,
                            gain=float(edge.gain) / fan_in,
                        )
                    )

    meta = {
        **dict(CONNECTOME_RELEASE_METADATA),
        "mode": "expanded_individual_neurons",
        "seed": int(seed),
    }
    return ConnectomeGraph(
        populations=tuple(indiv_pops),
        connections=tuple(indiv_edges),
        metadata=meta,
    )


def build_plastic_fly_connectome(
    *,
    spiking: bool = False,
    seed: int = 7,
    connection_probability: float = 0.38,
) -> ConnectomeGraph:
    """Build a Drosophila connectome augmented with a Mushroom Body associative learning circuit.

    Adds Kenyon cells (`KC_Odor_A`, `KC_Odor_B`), Mushroom Body Output Neurons
    (`MBON_Approach`, `MBON_Avoid`), and dopaminergic teaching neurons (`PAM` reward,
    `PPL1` aversive) wired into Central Complex forward pursuit (`CX_Forward_Goal`),
    Moonwalker retreat (`MDN`), and braking (`PFL2_Brake`).
    """
    base = (
        build_spiking_fly_connectome(
            seed=seed, connection_probability=connection_probability
        )
        if spiking
        else build_canonical_fly_connectome()
    )
    mb_pops = (
        NeuronPopulation(
            "KC_Odor_A", "mushroom_body", "ach", "bilateral", 0.018, 0.0, 16
        ),
        NeuronPopulation(
            "KC_Odor_B", "mushroom_body", "ach", "bilateral", 0.018, 0.0, 16
        ),
        NeuronPopulation(
            "MBON_Approach", "mushroom_body", "ach", "bilateral", 0.016, 0.0, 8
        ),
        NeuronPopulation(
            "MBON_Avoid", "mushroom_body", "ach", "bilateral", 0.016, 0.0, 8
        ),
        NeuronPopulation("PAM", "dopamine", "da", "bilateral", 0.020, 0.0, 8),
        NeuronPopulation("PPL1", "dopamine", "da", "bilateral", 0.020, 0.0, 8),
    )
    # Note: plastic KC -> MBON drive is modulated online by FlyBrainController's 3-factor rule,
    # while MBON -> downstream valence pathways are fixed connectome projections.
    mb_edges = (
        SynapticConnection("MBON_Approach", "CX_Forward_Goal", 210),
        SynapticConnection("MBON_Approach", "PFL3_L", 90),
        SynapticConnection("MBON_Avoid", "MDN", 240),
        SynapticConnection("MBON_Avoid", "PFL2_Brake", 140),
    )

    if not spiking:
        meta = dict(base.metadata)
        meta["mushroom_body_plasticity"] = True
        return ConnectomeGraph(
            populations=base.populations + mb_pops,
            connections=base.connections + mb_edges,
            metadata=meta,
        )

    rng = np.random.default_rng(int(seed) + 101)
    expanded_mb_pops: list[NeuronPopulation] = []
    for pop in mb_pops:
        for k in range(max(1, int(pop.neuron_count))):
            tau_jitter = float(pop.tau_sec) * float(rng.uniform(0.90, 1.10))
            expanded_mb_pops.append(
                NeuronPopulation(
                    cell_type=pop.cell_type,
                    region=pop.region,
                    neurotransmitter=pop.neurotransmitter,
                    hemisphere=pop.hemisphere,
                    tau_sec=max(0.003, tau_jitter),
                    bias=pop.bias,
                    neuron_count=1,
                    neuron_id=f"{pop.cell_type}#{k}",
                )
            )
    meta = dict(base.metadata)
    meta["mushroom_body_plasticity"] = True
    return ConnectomeGraph(
        populations=base.populations + tuple(expanded_mb_pops),
        connections=base.connections + mb_edges,
        metadata=meta,
    )


def _first_present(
    row: Mapping[str, Any], keys: Sequence[str], default: Any = None
) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


def load_connectome_graph(
    path: str | Path,
    *,
    merge_canonical: bool = True,
) -> ConnectomeGraph:
    """Load a connectome graph from JSON, CSV/TSV, NPZ, Parquet, or Feather."""
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Connectome file not found: {source}")

    canonical = build_canonical_fly_connectome()
    pop_map: Dict[str, NeuronPopulation] = (
        {pop.key: pop for pop in canonical.populations} if merge_canonical else {}
    )
    edges: list[SynapticConnection] = (
        list(canonical.connections) if merge_canonical else []
    )
    extra_meta: Dict[str, Any] = {"custom_connectome_path": str(source)}

    suffix = source.suffix.lower()
    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            edge_rows: Iterable[Mapping[str, Any]] = payload
            pop_rows: Iterable[Mapping[str, Any]] = ()
        elif isinstance(payload, dict):
            pop_rows = payload.get("populations") or payload.get("neurons") or ()
            edge_rows = (
                payload.get("connections")
                or payload.get("edges")
                or payload.get("synapses")
                or ()
            )
            if isinstance(payload.get("metadata"), dict):
                extra_meta.update(payload["metadata"])
        else:
            raise ValueError(f"Unsupported JSON connectome structure in {source}")
        for raw_pop in pop_rows:
            if not isinstance(raw_pop, Mapping):
                continue
            neuron_id_raw = _first_present(
                raw_pop, ("neuron_id", "bodyId", "root_id"), None
            )
            cell_type = str(
                _first_present(raw_pop, ("cell_type", "type", "name", "id"), "")
            ).strip()
            if not cell_type and neuron_id_raw is not None:
                cell_type = str(neuron_id_raw).strip()
            if not cell_type:
                continue
            nid = str(neuron_id_raw).strip() if neuron_id_raw is not None else None
            pop = NeuronPopulation(
                cell_type=cell_type,
                region=str(
                    _first_present(raw_pop, ("region", "roi"), "central_complex")
                ),
                neurotransmitter=normalize_neurotransmitter(
                    _first_present(
                        raw_pop, ("neurotransmitter", "nt_type", "nt"), "ach"
                    )
                ),
                hemisphere=str(
                    _first_present(raw_pop, ("hemisphere", "side"), "bilateral")
                ),
                tau_sec=float(_first_present(raw_pop, ("tau_sec", "tau"), 0.015)),
                bias=float(_first_present(raw_pop, ("bias",), 0.0)),
                neuron_count=int(_first_present(raw_pop, ("neuron_count", "count"), 1)),
                neuron_id=nid,
            )
            pop_map[pop.key] = pop
        parsed_edges = _parse_edge_mappings(edge_rows, pop_map)
        edges = _merge_or_replace_edges(
            edges, parsed_edges, merge_canonical=merge_canonical
        )
    elif suffix in (".csv", ".tsv"):
        delimiter = "\t" if suffix == ".tsv" else ","
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            parsed_edges = _parse_edge_mappings(reader, pop_map)
        edges = _merge_or_replace_edges(
            edges, parsed_edges, merge_canonical=merge_canonical
        )
    elif suffix in (".parquet", ".feather"):
        import pandas as pd

        frame = (
            pd.read_parquet(source) if suffix == ".parquet" else pd.read_feather(source)
        )
        parsed_edges = _parse_edge_mappings(frame.to_dict(orient="records"), pop_map)
        edges = _merge_or_replace_edges(
            edges, parsed_edges, merge_canonical=merge_canonical
        )
    elif suffix == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            names_arr = (
                archive["cell_types"] if "cell_types" in archive else archive["names"]
            )
            cell_types = [str(item) for item in names_arr.tolist()]
            neuron_ids = (
                [str(item) for item in archive["neuron_ids"].tolist()]
                if "neuron_ids" in archive
                else [None] * len(cell_types)
            )
            nt_arr = (
                [str(item) for item in archive["neurotransmitters"].tolist()]
                if "neurotransmitters" in archive
                else ["ach"] * len(cell_types)
            )
            regions_arr = (
                [str(item) for item in archive["regions"].tolist()]
                if "regions" in archive
                else ["central_complex"] * len(cell_types)
            )
            keys_in_order: list[str] = []
            for cell_type, nid, region, nt in zip(
                cell_types, neuron_ids, regions_arr, nt_arr
            ):
                pop = NeuronPopulation(
                    cell_type=cell_type,
                    region=region,
                    neurotransmitter=normalize_neurotransmitter(nt),
                    neuron_id=nid,
                )
                if pop.key not in pop_map:
                    pop_map[pop.key] = pop
                keys_in_order.append(pop.key)

            if "post_indices" in archive and "pre_indices" in archive:
                # Sparse coordinate-list NPZ format for 100k+ whole-CNS connectomes
                posts = np.asarray(archive["post_indices"], dtype=np.int64)
                pre_ids_arr = np.asarray(archive["pre_indices"], dtype=np.int64)
                counts = np.asarray(archive["synapse_counts"], dtype=np.float64)
                parsed_edges = [
                    SynapticConnection(
                        pre=keys_in_order[int(pre_i)],
                        post=keys_in_order[int(post_i)],
                        synapse_count=abs(float(val)),
                        neurotransmitter=("gaba" if float(val) < 0 else None),
                    )
                    for post_i, pre_i, val in zip(posts, pre_ids_arr, counts)
                    if float(val) != 0.0
                ]
            else:
                matrix = np.asarray(archive["synapse_counts"], dtype=np.float64)
                rows, cols = np.nonzero(matrix)
                parsed_edges = [
                    SynapticConnection(
                        pre=keys_in_order[pre_i],
                        post=keys_in_order[post_i],
                        synapse_count=abs(float(matrix[post_i, pre_i])),
                        neurotransmitter=(
                            "gaba" if float(matrix[post_i, pre_i]) < 0 else None
                        ),
                    )
                    for post_i, pre_i in zip(rows.tolist(), cols.tolist())
                ]
            edges = _merge_or_replace_edges(
                edges, parsed_edges, merge_canonical=merge_canonical
            )
    else:
        raise ValueError(
            f"Unsupported connectome file extension {suffix!r}; expected .json, .csv, .tsv, .parquet, .feather, or .npz"
        )

    merged_meta = dict(CONNECTOME_RELEASE_METADATA)
    merged_meta.update(extra_meta)
    return ConnectomeGraph(
        populations=tuple(pop_map.values()),
        connections=tuple(edges),
        metadata=merged_meta,
    )


def _parse_edge_mappings(
    rows: Iterable[Mapping[str, Any]],
    pop_map: Dict[str, NeuronPopulation],
) -> list[SynapticConnection]:
    parsed: list[SynapticConnection] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        pre_id = _first_present(row, ("pre_id", "bodyId_pre", "pre_root_id"), None)
        post_id = _first_present(row, ("post_id", "bodyId_post", "post_root_id"), None)
        pre_type = str(
            _first_present(
                row,
                ("pre", "pre_type", "type_pre", "pre_cell_type", "source"),
                pre_id or "",
            )
        ).strip()
        post_type = str(
            _first_present(
                row,
                ("post", "post_type", "type_post", "post_cell_type", "target"),
                post_id or "",
            )
        ).strip()
        if not pre_type or not post_type:
            continue
        pre_key = str(pre_id).strip() if pre_id is not None else pre_type
        post_key = str(post_id).strip() if post_id is not None else post_type

        syn_raw = _first_present(
            row,
            ("synapse_count", "weight", "syn_count", "n_synapses", "count"),
            0.0,
        )
        synapse_count = float(syn_raw)
        nt_raw = _first_present(
            row,
            ("neurotransmitter", "nt_type", "nt", "top_nt"),
            None,
        )
        nt = normalize_neurotransmitter(nt_raw) if nt_raw is not None else None
        gain = float(_first_present(row, ("gain",), 1.0))

        if pre_key not in pop_map:
            pop_map[pre_key] = NeuronPopulation(
                cell_type=pre_type,
                region=str(_first_present(row, ("pre_region",), "central_complex")),
                neurotransmitter=nt or "ach",
                neuron_count=1,
                neuron_id=(str(pre_id).strip() if pre_id is not None else None),
            )
        if post_key not in pop_map:
            pop_map[post_key] = NeuronPopulation(
                cell_type=post_type,
                region=str(_first_present(row, ("post_region",), "central_complex")),
                neurotransmitter="ach",
                neuron_count=1,
                neuron_id=(str(post_id).strip() if post_id is not None else None),
            )
        parsed.append(
            SynapticConnection(
                pre=pre_key,
                post=post_key,
                synapse_count=abs(synapse_count),
                neurotransmitter=("gaba" if synapse_count < 0 and nt is None else nt),
                gain=gain,
            )
        )
    return parsed


def _merge_or_replace_edges(
    base_edges: list[SynapticConnection],
    custom_edges: list[SynapticConnection],
    *,
    merge_canonical: bool,
) -> list[SynapticConnection]:
    if not merge_canonical:
        return custom_edges
    by_pair: Dict[tuple[str, str], SynapticConnection] = {
        (edge.pre, edge.post): edge for edge in base_edges
    }
    for edge in custom_edges:
        by_pair[(edge.pre, edge.post)] = edge
    return list(by_pair.values())


class SparseMushroomBodyMemory:
    """High-dimensional Drosophila Mushroom Body associative memory (e.g. 2,000 KCs).

    Models antennal-lobe Projection Neuron (PN) -> Kenyon Cell (KC) random sparse
    expansion (~6 PN claws per KC), global APL GABAergic winner-take-all sparsification
    (~5% active KCs per odor), presynaptic eligibility traces, and three-factor
    PAM/PPL1 dopaminergic plasticity across all `2 * n_kcs` KC -> MBON synapses.
    Pure NumPy, zero optional ML dependencies.
    """

    def __init__(
        self,
        *,
        n_pns: int = 50,
        n_kcs: int = 2000,
        claws_per_kc: int = 6,
        active_fraction: float = 0.05,
        learning_rate: float = 2.6,
        eligibility_tau_sec: float = 0.09,
        seed: int = 7,
    ) -> None:
        self.n_pns = max(4, int(n_pns))
        self.n_kcs = max(32, int(n_kcs))
        self.claws_per_kc = min(max(2, int(claws_per_kc)), self.n_pns)
        self.active_fraction = float(np.clip(active_fraction, 0.01, 0.5))
        self.k_active = max(2, int(round(self.n_kcs * self.active_fraction)))
        self.learning_rate = float(learning_rate)
        self.eligibility_tau_sec = max(1e-3, float(eligibility_tau_sec))
        self.seed = int(seed)

        rng = np.random.default_rng(self.seed + 31337)
        # Random PN -> KC claw connectivity (each KC samples `claws_per_kc` distinct PNs)
        claw_idx = np.empty((self.n_kcs, self.claws_per_kc), dtype=np.int32)
        for kc in range(self.n_kcs):
            claw_idx[kc] = rng.choice(self.n_pns, size=self.claws_per_kc, replace=False)
        self._claw_idx = claw_idx
        self._claw_weights = rng.uniform(
            0.85, 1.15, size=(self.n_kcs, self.claws_per_kc)
        ).astype(np.float64)

        self.w_approach = np.ones(self.n_kcs, dtype=np.float64)
        self.w_avoid = np.ones(self.n_kcs, dtype=np.float64)
        self.eligibility = np.zeros(self.n_kcs, dtype=np.float64)
        self.last_kc_activity = np.zeros(self.n_kcs, dtype=np.float64)

    def reset(self) -> None:
        self.w_approach.fill(1.0)
        self.w_avoid.fill(1.0)
        self.eligibility.fill(0.0)
        self.last_kc_activity.fill(0.0)

    def encode_odor(self, pn_pattern: Sequence[float] | np.ndarray) -> np.ndarray:
        """Project a PN odor vector into APL-sparsified KC activity (top-k active)."""
        pn = np.asarray(pn_pattern, dtype=np.float64).reshape(-1)
        if pn.size < self.n_pns:
            pn = np.pad(pn, (0, self.n_pns - pn.size))
        elif pn.size > self.n_pns:
            pn = pn[: self.n_pns]
        pn = np.maximum(0.0, pn)
        if float(np.max(pn)) <= 1e-8:
            self.last_kc_activity.fill(0.0)
            return self.last_kc_activity

        raw = np.sum(pn[self._claw_idx] * self._claw_weights, axis=1)
        # Global APL GABAergic inhibition: keep top `k_active` Kenyon cells
        thresh = float(
            np.partition(raw, self.n_kcs - self.k_active)[self.n_kcs - self.k_active]
        )
        sparse_act = np.where(raw >= thresh, np.maximum(0.0, raw - 0.82 * thresh), 0.0)
        # Enforce exact <= k_active sparsity when ties occur
        nz = np.flatnonzero(sparse_act > 0.0)
        if nz.size > self.k_active:
            order = np.argsort(sparse_act[nz])[::-1][: self.k_active]
            keep = nz[order]
            masked = np.zeros(self.n_kcs, dtype=np.float64)
            masked[keep] = sparse_act[keep]
            sparse_act = masked
        norm = float(np.sum(sparse_act))
        if norm > 1e-9:
            # Normalize L1 sum to ~1.0 so downstream MBON drive is scale-invariant in n_kcs
            sparse_act = sparse_act / norm
        self.last_kc_activity = sparse_act
        return sparse_act

    def compute_mbon_drives(self, kc_activity: np.ndarray) -> tuple[float, float]:
        app = float(np.dot(kc_activity, self.w_approach))
        avd = float(np.dot(kc_activity, self.w_avoid))
        return app, avd

    def update(
        self, kc_activity: np.ndarray, pam: float, ppl1: float, step_dt: float
    ) -> None:
        """Three-factor PAM/PPL1 dopaminergic plasticity across all `2 * n_kcs` synapses."""
        decay = math.exp(-max(1e-5, float(step_dt)) / self.eligibility_tau_sec)
        # Peak-normalized per-KC eligibility so sparse active KCs receive strong synaptic updates
        peak = float(np.max(kc_activity))
        kc_norm = (kc_activity / peak) if peak > 1e-9 else kc_activity
        self.eligibility = self.eligibility * decay + (1.0 - decay) * kc_norm
        if pam <= 0.05 and ppl1 <= 0.05:
            return
        da_diff = float(pam) - float(ppl1)
        delta = self.learning_rate * self.eligibility * da_diff * float(step_dt)
        self.w_approach = np.clip(self.w_approach + delta, 0.05, 2.50)
        self.w_avoid = np.clip(self.w_avoid - delta, 0.05, 2.50)

    def diagnostics(self) -> Dict[str, Any]:
        active_n = int(np.sum(self.last_kc_activity > 0.0))
        return {
            "n_pns": self.n_pns,
            "n_kcs": self.n_kcs,
            "plastic_synapses": 2 * self.n_kcs,
            "active_kcs": active_n,
            "active_fraction": round(active_n / float(self.n_kcs), 4),
            "mean_w_approach": round(float(np.mean(self.w_approach)), 4),
            "mean_w_avoid": round(float(np.mean(self.w_avoid)), 4),
        }


class FlyBrainController:
    """Embodied Drosophila connectome controller supporting both Rate and LIF Spiking dynamics.

    - `dynamics='rate'`: Continuous leaky synaptic firing-rate integration over sparse CSR weights.
    - `dynamics='spiking'`: Biophysical Leaky Integrate-and-Fire (LIF) spiking model
      (`V_rest = -52 mV`, `V_thresh = -45 mV`, `tau_ref = 2.2 ms`, `tau_syn = 5 ms`)
      using event-driven sparse spike propagation capable of scaling to 166,700+
      individual CNS neurons.
    """

    _LIF_V_REST_MV = -52.0
    _LIF_V_RESET_MV = -52.0
    _LIF_V_THRESH_MV = -45.0
    _LIF_REFRACTORY_SEC = 0.0022
    _LIF_TAU_SYN_SEC = 0.005
    _LIF_TAU_RATE_SEC = 0.020
    _LIF_SYN_SCALE_MV = 28.0
    _LIF_DRIVE_SCALE_MV = 16.0

    def __init__(
        self,
        action_spec: Any,
        *,
        behavior: str = "walk_imitation",
        connectome: ConnectomeGraph | None = None,
        dynamics: str = "rate",
        seed: int = 7,
        control_timestep: float = 0.002,
        enable_plasticity: bool = False,
        synaptic_noise_std: float = 0.0,
        suppress_behavior_drive: bool = False,
        sparse_mb_kcs: int = 0,
        sparse_mb_pns: int = 50,
    ) -> None:
        if behavior not in SUPPORTED_FLYBODY_BEHAVIORS:
            raise ValueError(
                f"Unsupported FlyBody behavior {behavior!r}; expected one of {SUPPORTED_FLYBODY_BEHAVIORS}"
            )
        dyn_mode = str(dynamics or "rate").strip().lower()
        if dyn_mode not in SUPPORTED_BRAIN_DYNAMICS:
            raise ValueError(
                f"Unsupported brain dynamics {dynamics!r}; expected one of {SUPPORTED_BRAIN_DYNAMICS}"
            )
        self.behavior = str(behavior)
        self.dynamics = dyn_mode
        self.enable_plasticity = bool(enable_plasticity or int(sparse_mb_kcs) > 0)
        self.suppress_behavior_drive = bool(suppress_behavior_drive)
        self.synaptic_noise_std = max(0.0, float(synaptic_noise_std))
        self.sparse_mb: SparseMushroomBodyMemory | None = (
            SparseMushroomBodyMemory(
                n_pns=int(sparse_mb_pns),
                n_kcs=int(sparse_mb_kcs),
                seed=int(seed),
            )
            if int(sparse_mb_kcs) > 0
            else None
        )
        if connectome is not None:
            self.connectome = connectome
        elif self.enable_plasticity:
            self.connectome = build_plastic_fly_connectome(
                spiking=(dyn_mode == "spiking"), seed=seed
            )
        else:
            self.connectome = build_canonical_fly_connectome()
        self.seed = int(seed)
        self.default_dt = max(1e-5, float(control_timestep))

        self._base_weights, self._tau, self._bias, self._idx = (
            self.connectome.compile_sparse()
        )
        self._weights = self._base_weights
        self._cell_groups = self.connectome.cell_type_groups()
        self._num_nodes = len(self.connectome.populations)

        self._node_activity_scale = np.ones(self._num_nodes, dtype=np.float64)
        self._ablations: Dict[str, float] = {}
        self._stimulations: list[Dict[str, Any]] = []

        # Three-factor Mushroom Body associative plasticity (KC_Odor_A/B -> MBON_Approach/Avoid)
        self._plastic_weights: Dict[str, float] = {
            "odor_a_approach": 1.0,
            "odor_a_avoid": 1.0,
            "odor_b_approach": 1.0,
            "odor_b_avoid": 1.0,
        }
        self._plastic_eligibility: Dict[str, float] = {
            "KC_Odor_A": 0.0,
            "KC_Odor_B": 0.0,
        }
        self._plasticity_lr = 1.85
        self._eligibility_tau_sec = 0.09

        self._voltage = np.zeros(self._num_nodes, dtype=np.float64)
        self._rates = np.zeros(self._num_nodes, dtype=np.float64)
        self._syn_conductance = np.zeros(self._num_nodes, dtype=np.float64)
        self._refractory_rem = np.zeros(self._num_nodes, dtype=np.float64)
        self._last_step_spikes = 0
        self._total_spikes = 0

        self._action_dtype = getattr(action_spec, "dtype", np.float64)
        self._action_shape = tuple(getattr(action_spec, "shape", ()))
        self._action_min = np.asarray(action_spec.minimum, dtype=np.float64).reshape(-1)
        self._action_max = np.asarray(action_spec.maximum, dtype=np.float64).reshape(-1)
        raw_name = str(getattr(action_spec, "name", "") or "")
        parsed_names = [part.strip() for part in raw_name.split("\t") if part.strip()]
        if len(parsed_names) == self._action_min.size:
            self._actuator_names = tuple(parsed_names)
        else:
            self._actuator_names = tuple(
                f"actuator_{idx}" for idx in range(self._action_min.size)
            )
        self._act_index: Dict[str, int] = {
            name: idx for idx, name in enumerate(self._actuator_names)
        }
        self._has_legs = any(name.startswith("coxa_") for name in self._actuator_names)
        self._has_wings = any(name.startswith("wing_") for name in self._actuator_names)
        self._uses_wbpg = "user_0" in self._act_index
        self._is_flight_spec = self._has_wings and not self._has_legs

        self._rng = np.random.default_rng(self.seed)
        self._sim_time = 0.0
        self._leg_phases = np.zeros(len(_LEG_IDS), dtype=np.float64)
        self._wing_phase = 0.0
        self._groom_phase = 0.0
        self._jump_timer = 0.0
        self._jump_triggered = False
        self._gyro_lp = np.zeros(3, dtype=np.float64)
        self._world_z_lp = np.asarray([0.737, 0.0, 0.676], dtype=np.float64)
        self._altitude_integral = 0.0
        self._last_sensory_summary: Dict[str, float] = {}
        if self.synaptic_noise_std > 0.0:
            self.apply_synaptic_noise(self.synaptic_noise_std, seed=self.seed)
        self.reset()

    def apply_synaptic_noise(self, sigma: float, *, seed: int | None = None) -> None:
        """Apply Dale's-law sign-preserving multiplicative synaptic noise to connectome weights."""
        self.synaptic_noise_std = max(0.0, float(sigma))
        if self.synaptic_noise_std <= 0.0:
            self._weights = self._base_weights
            return
        noise_rng = np.random.default_rng(
            int(self.seed if seed is None else seed) + 7919
        )
        self._weights = self._base_weights.with_multiplicative_noise(
            self.synaptic_noise_std, noise_rng
        )

    def stimulate_cell_type(
        self,
        cell_type: str,
        current: float,
        *,
        duration_sec: float = 0.2,
    ) -> None:
        """Inject a timed depolarizing/hyperpolarizing optogenetic current (e.g. Chrimson)."""
        name = str(cell_type or "").strip()
        if not name:
            return
        self._stimulations.append(
            {
                "cell_type": name,
                "current": float(current),
                "end_time": float(self._sim_time + max(0.0, float(duration_sec))),
            }
        )

    def clear_stimulations(self) -> None:
        self._stimulations.clear()

    def set_ablation(self, cell_type: str, scale: float = 0.0) -> None:
        """Scale activity of a cell type in `[0, 1]` (`0.0` = full Kir2.1/TNT silencing)."""
        name = str(cell_type or "").strip()
        if not name:
            return
        k = float(np.clip(scale, 0.0, 1.0))
        self._ablations[name] = k
        self._rebuild_ablation_mask()

    def set_ablations(self, scales: Mapping[str, float]) -> None:
        self._ablations = {
            str(k).strip(): float(np.clip(v, 0.0, 1.0))
            for k, v in scales.items()
            if str(k).strip()
        }
        self._rebuild_ablation_mask()

    def clear_ablations(self) -> None:
        self._ablations.clear()
        self._node_activity_scale.fill(1.0)

    def _rebuild_ablation_mask(self) -> None:
        self._node_activity_scale.fill(1.0)
        for name, scale in self._ablations.items():
            # Allow bilateral prefix shortcuts like "DNp09", "LAL_Inh", "DNa02"
            targets = []
            if name in self._cell_groups:
                targets.append(name)
            else:
                for suffix in ("_L", "_R"):
                    if f"{name}{suffix}" in self._cell_groups:
                        targets.append(f"{name}{suffix}")
            for target in targets:
                self._node_activity_scale[self._cell_groups[target]] = scale
            if not targets and name in self._idx:
                self._node_activity_scale[self._idx[name]] = scale

    def plasticity_weights(self) -> Dict[str, float]:
        return {k: round(float(v), 4) for k, v in self._plastic_weights.items()}

    def reset_plasticity(self) -> None:
        for k in self._plastic_weights:
            self._plastic_weights[k] = 1.0
        for k in self._plastic_eligibility:
            self._plastic_eligibility[k] = 0.0
        if self.sparse_mb is not None:
            self.sparse_mb.reset()

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        if self.dynamics == "spiking":
            # Heterogeneous resting initialization around V_rest (-52 mV) to prevent artificial synchrony
            self._voltage = self._LIF_V_REST_MV + self._rng.uniform(
                -1.5, 1.5, size=self._num_nodes
            )
        else:
            self._voltage.fill(0.0)
        self._rates.fill(0.0)
        self._syn_conductance.fill(0.0)
        self._refractory_rem.fill(0.0)
        self._last_step_spikes = 0
        self._total_spikes = 0
        self._sim_time = 0.0
        tripod_offsets = np.asarray(
            [0.0, math.pi, math.pi, 0.0, 0.0, math.pi],
            dtype=np.float64,
        )
        jitter = self._rng.uniform(-0.08, 0.08, size=len(_LEG_IDS))
        self._leg_phases = np.mod(tripod_offsets + jitter, math.tau)
        self._wing_phase = 0.0
        self._groom_phase = 0.0
        self._jump_timer = 0.0
        self._jump_triggered = False
        self._gyro_lp = np.zeros(3, dtype=np.float64)
        self._world_z_lp = np.asarray([0.737, 0.0, 0.676], dtype=np.float64)
        self._altitude_integral = 0.0
        self._last_sensory_summary = {}
        self._stimulations.clear()
        self.reset_plasticity()

    def _integrate_neural_state(self, ext_drive: np.ndarray, total_dt: float) -> int:
        has_ablation = bool(self._ablations)
        if self.dynamics == "spiking":
            sub_dt = 0.0002  # 0.2 ms biophysical LIF integration step
            substeps = max(1, int(round(total_dt / sub_dt)))
            dt = total_dt / float(substeps)
            syn_decay = math.exp(-dt / self._LIF_TAU_SYN_SEC)
            rate_decay = math.exp(-dt / self._LIF_TAU_RATE_SEC)
            # Normalize spike impulse so ~90 Hz sustained firing maps to rate ~ 1.0
            spike_rate_jump = 1.0 / (90.0 * self._LIF_TAU_RATE_SEC)
            drive_mv = (self._bias + ext_drive) * self._LIF_DRIVE_SCALE_MV
            step_spikes = 0

            for _ in range(substeps):
                self._syn_conductance *= syn_decay
                self._refractory_rem = np.maximum(0.0, self._refractory_rem - dt)
                active = self._refractory_rem <= 0.0
                if has_ablation:
                    active = active & (self._node_activity_scale > 0.0)
                dv = (
                    (self._LIF_V_REST_MV - self._voltage)
                    + self._syn_conductance
                    + drive_mv
                ) / self._tau
                self._voltage = np.where(
                    active,
                    np.clip(self._voltage + dt * dv, -75.0, -35.0),
                    self._LIF_V_RESET_MV,
                )
                spiked_idx = np.flatnonzero(self._voltage >= self._LIF_V_THRESH_MV)
                self._rates *= rate_decay
                if spiked_idx.size > 0:
                    if has_ablation:
                        # Partial knockdown attenuates effective spike transmission
                        scales = self._node_activity_scale[spiked_idx]
                        keep = (scales >= 0.999) | (
                            self._rng.random(spiked_idx.size) < scales
                        )
                        spiked_idx = spiked_idx[keep]
                if spiked_idx.size > 0:
                    step_spikes += int(spiked_idx.size)
                    self._voltage[spiked_idx] = self._LIF_V_RESET_MV
                    self._refractory_rem[spiked_idx] = self._LIF_REFRACTORY_SEC
                    self._rates[spiked_idx] = np.minimum(
                        1.0, self._rates[spiked_idx] + spike_rate_jump
                    )
                    postsyn_mv = (
                        self._weights.propagate_spikes(spiked_idx)
                        * self._LIF_SYN_SCALE_MV
                    )
                    self._syn_conductance = np.clip(
                        self._syn_conductance + postsyn_mv, -35.0, 45.0
                    )
            if has_ablation:
                self._rates *= self._node_activity_scale
            self._last_step_spikes = step_spikes
            self._total_spikes += step_spikes
            return step_spikes

        neural_substeps = max(1, int(math.ceil(total_dt / 0.001)))
        sub_dt = total_dt / float(neural_substeps)
        for _ in range(neural_substeps):
            synaptic_input = (self._weights @ self._rates) + self._bias + ext_drive
            dv = (-self._voltage + synaptic_input) / self._tau
            self._voltage = np.clip(self._voltage + sub_dt * dv, -4.0, 6.0)
            self._rates = np.tanh(np.maximum(0.0, self._voltage))
            if has_ablation:
                self._rates *= self._node_activity_scale
        self._last_step_spikes = 0
        return 0

    def warmup(
        self,
        observation: Mapping[str, Any] | None = None,
        *,
        duration_sec: float = 0.04,
    ) -> None:
        """Equilibrate neural membrane potentials/spiking rates on initial observation without advancing behavioral timers."""
        obs = observation if isinstance(observation, Mapping) else {}
        if "walker/world_zaxis" in obs:
            wz0 = np.asarray(obs["walker/world_zaxis"], dtype=np.float64).reshape(-1)
            if wz0.size >= 3:
                self._world_z_lp = wz0[:3].copy()
        ext_drive = self._encode_sensory_drive(obs, 0.001)
        self._integrate_neural_state(ext_drive, max(0.002, float(duration_sec)))

    def rate(self, cell_type: str) -> float:
        group = self._cell_groups.get(cell_type)
        if group is not None and group.size > 0:
            return float(np.mean(self._rates[group]))
        idx = self._idx.get(cell_type)
        if idx is None:
            return 0.0
        return float(self._rates[idx])

    def step(
        self,
        observation: Mapping[str, Any] | None = None,
        *,
        dt: float | None = None,
    ) -> np.ndarray:
        step_dt = max(1e-5, float(dt if dt is not None else self.default_dt))
        obs = observation if isinstance(observation, Mapping) else {}
        ext_drive = self._encode_sensory_drive(obs, step_dt)
        self._integrate_neural_state(ext_drive, step_dt)
        if self.enable_plasticity:
            self._update_mushroom_body_plasticity(step_dt)
        self._advance_vnc_oscillators(obs, step_dt)
        self._sim_time += step_dt

        if self._is_flight_spec:
            command = self._decode_flight_actuators(obs, step_dt)
        else:
            command = self._decode_walking_and_posture_actuators(obs)
            if self._has_wings:
                self._overlay_wing_actuators(
                    command, obs, step_dt, include_posture=False
                )

        clipped = np.clip(command, self._action_min, self._action_max)
        return clipped.reshape(self._action_shape).astype(self._action_dtype)

    def _update_mushroom_body_plasticity(self, step_dt: float) -> None:
        """Three-factor PAM/PPL1 dopaminergic eligibility-trace update on KC -> MBON synapses."""
        decay = math.exp(-step_dt / self._eligibility_tau_sec)
        for kc_name in ("KC_Odor_A", "KC_Odor_B"):
            r_kc = self.rate(kc_name)
            self._plastic_eligibility[kc_name] = (
                self._plastic_eligibility[kc_name] * decay + (1.0 - decay) * r_kc
            )
        pam = self.rate("PAM")
        ppl1 = self.rate("PPL1")
        if self.sparse_mb is not None:
            self.sparse_mb.update(
                self.sparse_mb.last_kc_activity, pam=pam, ppl1=ppl1, step_dt=step_dt
            )
        if pam <= 0.05 and ppl1 <= 0.05:
            return
        for prefix, kc_name in (("odor_a", "KC_Odor_A"), ("odor_b", "KC_Odor_B")):
            elig = self._plastic_eligibility[kc_name]
            if elig <= 0.02:
                continue
            # PAM reward dopamine strengthens approach & depresses avoidance;
            # PPL1 aversive dopamine strengthens avoidance & depresses approach.
            delta_app = self._plasticity_lr * elig * (pam - ppl1) * step_dt
            delta_avd = self._plasticity_lr * elig * (ppl1 - pam) * step_dt
            self._plastic_weights[f"{prefix}_approach"] = float(
                np.clip(
                    self._plastic_weights[f"{prefix}_approach"] + delta_app, 0.15, 2.50
                )
            )
            self._plastic_weights[f"{prefix}_avoid"] = float(
                np.clip(
                    self._plastic_weights[f"{prefix}_avoid"] + delta_avd, 0.15, 2.50
                )
            )

    def _set_drive(self, vec: np.ndarray, cell_type: str, value: float) -> None:
        group = self._cell_groups.get(cell_type)
        if group is not None and group.size > 0:
            vec[group] += float(value)
            return
        idx = self._idx.get(cell_type)
        if idx is not None:
            vec[idx] += float(value)
            return
        # Allow bilateral prefix shortcuts like "DNp09" -> DNp09_L + DNp09_R
        for suffix in ("_L", "_R"):
            g = self._cell_groups.get(f"{cell_type}{suffix}")
            if g is not None and g.size > 0:
                vec[g] += float(value)

    def _encode_sensory_drive(
        self, obs: Mapping[str, Any], step_dt: float
    ) -> np.ndarray:
        drive = np.zeros(self._num_nodes, dtype=np.float64)

        # 1. Proprioceptive tarsal contact -> FeCO_Load_L / FeCO_Load_R
        touch = np.asarray(obs.get("walker/touch", ()), dtype=np.float64).reshape(-1)
        if touch.size >= 6:
            left_load = float(np.mean(np.clip(touch[0::2], 0.0, 5.0))) / 2.5
            right_load = float(np.mean(np.clip(touch[1::2], 0.0, 5.0))) / 2.5
        else:
            left_load = 0.35 if not self._is_flight_spec else 0.0
            right_load = 0.35 if not self._is_flight_spec else 0.0
        self._set_drive(drive, "FeCO_Load_L", min(1.5, left_load))
        self._set_drive(drive, "FeCO_Load_R", min(1.5, right_load))

        # Ascending stride efference copy
        left_swing = float(
            np.mean([max(0.0, math.sin(self._leg_phases[i])) for i in (0, 2, 4)])
        )
        right_swing = float(
            np.mean([max(0.0, math.sin(self._leg_phases[i])) for i in (1, 3, 5)])
        )
        self._set_drive(drive, "AN_Stride_L", 0.4 * left_swing)
        self._set_drive(drive, "AN_Stride_R", 0.4 * right_swing)

        # 2. Ocellar horizon & 5ms low-pass Haltere Coriolis gyro sensors
        alpha_lp = min(1.0, max(0.02, step_dt / (0.005 + step_dt)))
        world_z = np.asarray(
            obs.get("walker/world_zaxis", ()), dtype=np.float64
        ).reshape(-1)
        gyro = np.asarray(obs.get("walker/gyro", ()), dtype=np.float64).reshape(-1)
        if world_z.size >= 3:
            self._world_z_lp = (1.0 - alpha_lp) * self._world_z_lp + alpha_lp * world_z[
                :3
            ]
        if gyro.size >= 3:
            self._gyro_lp = (1.0 - alpha_lp) * self._gyro_lp + alpha_lp * gyro[:3]

        pitch_ang = math.atan2(
            float(self._world_z_lp[0]), max(0.05, float(self._world_z_lp[2]))
        )
        roll_ang = math.atan2(
            float(self._world_z_lp[1]), max(0.05, float(self._world_z_lp[2]))
        )
        # Hover/cruise pitch setpoint ~0.73 rad (~42 deg nose-up)
        pitch_down_err = max(0.0, 0.73 - pitch_ang)
        pitch_up_err = max(0.0, pitch_ang - 0.73)
        self._set_drive(drive, "Ocelli_Pitch", min(2.0, 1.4 * pitch_down_err))
        self._set_drive(
            drive,
            "Haltere_Pitch_Down",
            min(2.0, max(0.0, 0.030 * float(self._gyro_lp[1]))),
        )
        self._set_drive(
            drive,
            "Haltere_Pitch_Up",
            min(
                2.0,
                1.4 * pitch_up_err + max(0.0, -0.030 * float(self._gyro_lp[1])),
            ),
        )
        self._set_drive(drive, "Ocelli_Roll_L", min(2.0, max(0.0, roll_ang * 1.6)))
        self._set_drive(drive, "Ocelli_Roll_R", min(2.0, max(0.0, -roll_ang * 1.6)))
        self._set_drive(
            drive,
            "Haltere_Roll_L",
            min(2.0, max(0.0, 0.024 * float(self._gyro_lp[0]))),
        )
        self._set_drive(
            drive,
            "Haltere_Roll_R",
            min(2.0, max(0.0, -0.024 * float(self._gyro_lp[0]))),
        )

        # 3. Compound-eye visual target & reference lookahead bearing
        vis_left = 0.0
        vis_right = 0.0
        ref_disp = obs.get("walker/ref_displacement")
        if ref_disp is not None:
            disp_arr = np.asarray(ref_disp, dtype=np.float64)
            if disp_arr.ndim == 2 and disp_arr.shape[0] > 0 and disp_arr.shape[1] >= 2:
                look_idx = min(4, disp_arr.shape[0] - 1)
                dx, dy = float(disp_arr[look_idx, 0]), float(disp_arr[look_idx, 1])
                bearing = math.atan2(dy, max(1e-3, abs(dx)))
                vis_left += max(0.0, bearing * 1.4)
                vis_right += max(0.0, -bearing * 1.4)

        left_eye = obs.get("walker/left_eye")
        right_eye = obs.get("walker/right_eye")
        if left_eye is not None and right_eye is not None:
            l_arr = np.asarray(left_eye, dtype=np.float64)
            r_arr = np.asarray(right_eye, dtype=np.float64)
            if l_arr.size > 0 and r_arr.size > 0:
                l_salience = 1.0 - float(np.mean(l_arr)) / 255.0
                r_salience = 1.0 - float(np.mean(r_arr)) / 255.0
                diff = l_salience - r_salience
                vis_left += max(0.0, diff * 2.2)
                vis_right += max(0.0, -diff * 2.2)

        # 4. Behavioral context & stimulus gating
        looming = 0.0
        odor_l = 0.0
        odor_r = 0.0
        antennal_dust = 0.0
        anterior_touch = 0.0
        suppress_ctx = self.suppress_behavior_drive or bool(
            obs.get("suppress_behavior_drive", False)
        )

        if not suppress_ctx:
            if self.behavior in ("walk_imitation", "walk_on_ball", "template_task"):
                self._set_drive(drive, "CX_Forward_Goal", 1.35)
                if self.behavior == "template_task":
                    odor_l = 0.35 + 0.18 * max(0.0, math.sin(self._sim_time * 3.5))
                    odor_r = 0.35 + 0.18 * max(0.0, -math.sin(self._sim_time * 3.5))
            elif self.behavior == "forage_walk":
                self._set_drive(drive, "CX_Forward_Goal", 1.20)
                wave = math.sin(self._sim_time * math.tau * 0.9)
                odor_l = 0.25 + 0.95 * max(0.0, wave)
                odor_r = 0.25 + 0.95 * max(0.0, -wave)
            elif self.behavior == "backward_walk":
                anterior_touch = 1.55
            elif self.behavior == "grooming":
                antennal_dust = 1.65
            elif self.behavior == "escape_takeoff":
                if self._sim_time < 0.14:
                    self._set_drive(drive, "CX_Forward_Goal", 0.85)
                elif self._sim_time < 0.34:
                    looming = 1.85
                    self._set_drive(drive, "CX_Forward_Goal", 1.45)
                else:
                    self._set_drive(drive, "CX_Forward_Goal", 1.65)
            elif self.behavior in ("flight_imitation", "vision_guided_flight"):
                self._set_drive(drive, "Flight_Command", 1.65)

        self._set_drive(drive, "Visual_Target_L", vis_left)
        self._set_drive(drive, "Visual_Target_R", vis_right)
        self._set_drive(drive, "ORN_L", odor_l)
        self._set_drive(drive, "ORN_R", odor_r)
        self._set_drive(drive, "JO_Mech", antennal_dust)
        self._set_drive(drive, "Tactile_Ant", anterior_touch)
        self._set_drive(drive, "LC4_Looming", looming)

        # 5. Optional Mushroom Body olfactory conditioning & dopamine inputs
        if self.enable_plasticity:
            odor_a = float(obs.get("odor_a", 0.0) or 0.0)
            odor_b = float(obs.get("odor_b", 0.0) or 0.0)
            reward_da = float(obs.get("reward_da", 0.0) or 0.0)
            punish_da = float(obs.get("punish_da", 0.0) or 0.0)
            if odor_a > 0.0:
                self._set_drive(drive, "KC_Odor_A", 1.55 * odor_a)
            if odor_b > 0.0:
                self._set_drive(drive, "KC_Odor_B", 1.55 * odor_b)
            if reward_da > 0.0:
                self._set_drive(drive, "PAM", 1.65 * reward_da)
            if punish_da > 0.0:
                self._set_drive(drive, "PPL1", 1.65 * punish_da)
            kc_a = self.rate("KC_Odor_A")
            kc_b = self.rate("KC_Odor_B")
            mbon_app_drive = (
                kc_a * self._plastic_weights["odor_a_approach"]
                + kc_b * self._plastic_weights["odor_b_approach"]
            )
            mbon_avd_drive = (
                kc_a * self._plastic_weights["odor_a_avoid"]
                + kc_b * self._plastic_weights["odor_b_avoid"]
            )
            if self.sparse_mb is not None:
                odor_pat = obs.get("odor_pattern")
                if odor_pat is not None:
                    kc_act = self.sparse_mb.encode_odor(odor_pat)
                    s_app, s_avd = self.sparse_mb.compute_mbon_drives(kc_act)
                    mbon_app_drive += s_app
                    mbon_avd_drive += s_avd
                else:
                    self.sparse_mb.last_kc_activity.fill(0.0)
            # Mutual valence competition between MBON_Approach and MBON_Avoid
            self._set_drive(
                drive,
                "MBON_Approach",
                max(0.0, 1.35 * mbon_app_drive - 0.75 * mbon_avd_drive),
            )
            self._set_drive(
                drive,
                "MBON_Avoid",
                max(0.0, 1.35 * mbon_avd_drive - 0.75 * mbon_app_drive),
            )

        # 6. Timed optogenetic current injections (Chrimson / Halorhodopsin)
        if self._stimulations:
            still_active: list[Dict[str, Any]] = []
            for stim in self._stimulations:
                if self._sim_time <= float(stim["end_time"]):
                    self._set_drive(drive, stim["cell_type"], float(stim["current"]))
                    still_active.append(stim)
            self._stimulations = still_active

        self._last_sensory_summary = {
            "tarsal_load_left": round(float(left_load), 4),
            "tarsal_load_right": round(float(right_load), 4),
            "visual_left": round(float(vis_left), 4),
            "visual_right": round(float(vis_right), 4),
            "odor_left": round(float(odor_l), 4),
            "odor_right": round(float(odor_r), 4),
            "looming": round(float(looming), 4),
        }
        return drive

    def _advance_vnc_oscillators(self, obs: Mapping[str, Any], step_dt: float) -> None:
        del obs
        walk_drive = self.rate("VNC_Walk_Drive")
        arrest_drive = self.rate("Pair1")
        groom_drive = self.rate("VNC_Groom_T1")
        jump_drive = max(self.rate("DNp01"), self.rate("VNC_Jump"))

        if jump_drive > 0.45 and not self._jump_triggered:
            self._jump_triggered = True
            self._jump_timer = 0.11
        elif self._jump_timer > 0.0:
            self._jump_timer = max(0.0, self._jump_timer - step_dt)

        effective_walk = max(0.0, walk_drive * (1.0 - 0.92 * arrest_drive))
        step_freq_hz = 11.0 * effective_walk if effective_walk > 0.05 else 0.0

        dphi = np.full(len(_LEG_IDS), math.tau * step_freq_hz, dtype=np.float64)
        if step_freq_hz > 0.1:
            leg_to_i = {leg: idx for idx, leg in enumerate(_LEG_IDS)}
            for i, leg in enumerate(_LEG_IDS):
                coupling = 0.0
                for neighbor in _LEG_INHIBITORY_NEIGHBORS[leg]:
                    j = leg_to_i[neighbor]
                    coupling -= 2.4 * math.sin(
                        self._leg_phases[i] - self._leg_phases[j]
                    )
                dphi[i] += coupling
            self._leg_phases = np.mod(self._leg_phases + step_dt * dphi, math.tau)

        if groom_drive > 0.1:
            self._groom_phase = math.fmod(
                self._groom_phase + step_dt * math.tau * (7.5 * groom_drive),
                math.tau,
            )

        wing_drive = 0.5 * (self.rate("WTct_Wing_L") + self.rate("WTct_Wing_R"))
        wing_freq_hz = 195.0 + 35.0 * wing_drive
        self._wing_phase = math.fmod(
            self._wing_phase + step_dt * math.tau * wing_freq_hz,
            math.tau,
        )

    def _set_act(self, action: np.ndarray, name: str, value: float) -> None:
        idx = self._act_index.get(name)
        if idx is not None:
            action[idx] = float(value)

    def _decode_walking_and_posture_actuators(
        self, obs: Mapping[str, Any]
    ) -> np.ndarray:
        del obs
        action = np.zeros(self._action_min.size, dtype=np.float64)

        stride_l = self.rate("MN_Stride_L")
        stride_r = self.rate("MN_Stride_R")
        arrest = self.rate("Pair1")
        groom = self.rate("VNC_Groom_T1")
        reverse_rate = self.rate("VNC_Reverse")
        turn_asym = self.rate("VNC_Turn_L") - self.rate("VNC_Turn_R")

        direction_sign = -1.0 if reverse_rate > 0.45 else 1.0

        if groom > 0.35:
            head_pitch = 0.16 * groom
            head_yaw = 0.25 * math.sin(self._groom_phase)
            head_roll = 0.08 * math.cos(self._groom_phase)
            abd_pitch = -0.08 * groom
            abd_yaw = 0.0
        else:
            head_pitch = -0.05
            head_yaw = 0.55 * turn_asym
            head_roll = 0.06 * turn_asym
            abd_pitch = -0.10
            abd_yaw = -0.25 * turn_asym

        self._set_act(action, "head", head_pitch)
        self._set_act(action, "head_twist", head_yaw)
        self._set_act(action, "head_abduct", head_roll)
        self._set_act(action, "abdomen", abd_pitch)
        self._set_act(action, "abdomen_abduct", abd_yaw)

        if self._jump_timer > 0.0:
            for side in ("left", "right"):
                self._set_act(action, f"adhere_claw_T1_{side}", 0.0)
                self._set_act(action, f"adhere_claw_T2_{side}", 0.0)
                self._set_act(action, f"adhere_claw_T3_{side}", 0.0)
                self._set_act(action, f"coxa_T1_{side}", 0.55)
                self._set_act(action, f"femur_T1_{side}", 0.10)
                self._set_act(action, f"coxa_T2_{side}", -0.18)
                self._set_act(action, f"coxa_twist_T2_{side}", -0.55)
                self._set_act(action, f"femur_twist_T2_{side}", 0.65)
                self._set_act(action, f"femur_T2_{side}", 1.35)
                self._set_act(action, f"tibia_T2_{side}", -0.65)
                self._set_act(action, f"tarsus_T2_{side}", 0.85)
                self._set_act(action, f"coxa_T3_{side}", -0.22)
                self._set_act(action, f"coxa_twist_T3_{side}", -0.12)
                self._set_act(action, f"femur_T3_{side}", 1.15)
                self._set_act(action, f"tibia_T3_{side}", 0.95)
                self._set_act(action, f"tarsus_T3_{side}", 0.75)
            return action

        for leg_idx, leg in enumerate(_LEG_IDS):
            is_left = leg.endswith("_left")
            segment = leg.split("_", 1)[0]

            if groom > 0.35:
                if segment == "T1":
                    sweep = math.sin(self._groom_phase + (0.0 if is_left else 0.65))
                    rub = math.cos(self._groom_phase * 2.0 + (0.0 if is_left else 0.65))
                    self._set_act(action, f"adhere_claw_{leg}", 0.0)
                    self._set_act(action, f"coxa_{leg}", 1.05 * groom + 0.14 * rub)
                    self._set_act(
                        action, f"coxa_twist_{leg}", 0.42 * groom + 0.22 * sweep
                    )
                    self._set_act(action, f"coxa_abduct_{leg}", -0.45 * groom)
                    self._set_act(action, f"femur_{leg}", 0.32 * groom + 0.18 * sweep)
                    self._set_act(action, f"femur_twist_{leg}", -0.35 * groom)
                    self._set_act(action, f"tibia_{leg}", 0.58 * groom + 0.32 * sweep)
                    self._set_act(action, f"tarsus_{leg}", 0.45 * groom + 0.28 * rub)
                    self._set_act(action, f"tarsus2_{leg}", 0.22 * sweep)
                else:
                    self._set_act(action, f"adhere_claw_{leg}", 0.95)
                    self._set_act(action, f"coxa_{leg}", -0.04)
                    self._set_act(action, f"femur_{leg}", 0.16)
                    self._set_act(action, f"tibia_{leg}", 0.0)
                    self._set_act(action, f"tarsus_{leg}", -0.10)
                continue

            side_gain = (stride_l if is_left else stride_r) * (1.0 - 0.9 * arrest)
            if side_gain < 0.04:
                self._set_act(action, f"adhere_claw_{leg}", 0.85)
                self._set_act(action, f"femur_{leg}", 0.12)
                continue

            phi = float(self._leg_phases[leg_idx])
            swing_sin = math.sin(phi)
            lift = max(0.0, swing_sin)
            stance = max(0.0, -swing_sin)
            stroke = -math.cos(phi) * direction_sign * min(1.15, 1.15 * side_gain)

            adhere = 1.0 if swing_sin < -0.08 else 0.0
            self._set_act(action, f"adhere_claw_{leg}", adhere)

            if segment == "T1":
                self._set_act(action, f"coxa_{leg}", 0.42 * lift - 0.06 * stance)
                self._set_act(action, f"coxa_twist_{leg}", 0.34 * stroke)
                self._set_act(action, f"coxa_abduct_{leg}", -0.20 * stroke)
                self._set_act(action, f"femur_twist_{leg}", -0.28 * stroke)
                self._set_act(action, f"femur_{leg}", 0.20 * stance - 0.06 * lift)
                self._set_act(action, f"tibia_{leg}", 0.42 * stroke + 0.10 * lift)
                self._set_act(action, f"tarsus_{leg}", 0.24 * stroke - 0.14 * stance)
                self._set_act(action, f"tarsus2_{leg}", 0.12 * stroke)
            elif segment == "T2":
                self._set_act(action, f"coxa_{leg}", 0.38 * lift - 0.06 * stance)
                self._set_act(action, f"coxa_twist_{leg}", 0.44 * stroke)
                self._set_act(action, f"coxa_abduct_{leg}", 0.16 * stroke)
                self._set_act(action, f"femur_twist_{leg}", -0.36 * stroke)
                self._set_act(action, f"femur_{leg}", 0.22 * stance - 0.06 * lift)
                self._set_act(action, f"tibia_{leg}", 0.22 * stroke + 0.08 * lift)
                self._set_act(action, f"tarsus_{leg}", -0.16 * stance + 0.10 * lift)
                self._set_act(action, f"tarsus2_{leg}", 0.08 * stroke)
            else:
                self._set_act(action, f"coxa_{leg}", 0.38 * lift - 0.06 * stance)
                self._set_act(action, f"coxa_twist_{leg}", 0.34 * stroke)
                self._set_act(action, f"coxa_abduct_{leg}", -0.26 * stroke)
                self._set_act(action, f"femur_twist_{leg}", -0.34 * stroke)
                self._set_act(action, f"femur_{leg}", 0.24 * stance - 0.08 * lift)
                self._set_act(action, f"tibia_{leg}", -0.44 * stroke + 0.08 * lift)
                self._set_act(action, f"tarsus_{leg}", -0.22 * stroke - 0.12 * stance)
                self._set_act(action, f"tarsus2_{leg}", -0.10 * stroke)

        return action

    def _decode_flight_actuators(
        self, obs: Mapping[str, Any], step_dt: float
    ) -> np.ndarray:
        action = np.zeros(self._action_min.size, dtype=np.float64)
        self._overlay_wing_actuators(action, obs, step_dt, include_posture=True)
        return action

    def _overlay_wing_actuators(
        self,
        action: np.ndarray,
        obs: Mapping[str, Any],
        step_dt: float,
        *,
        include_posture: bool,
    ) -> None:
        wing_l = self.rate("WTct_Wing_L")
        wing_r = self.rate("WTct_Wing_R")
        mean_wing = 0.5 * (wing_l + wing_r)
        if not self._is_flight_spec and mean_wing < 0.12 and self._jump_timer <= 0.0:
            self._set_act(action, "wing_yaw_left", 0.0)
            self._set_act(action, "wing_yaw_right", 0.0)
            self._set_act(action, "wing_roll_left", 0.0)
            self._set_act(action, "wing_roll_right", 0.0)
            self._set_act(action, "wing_pitch_left", 0.0)
            self._set_act(action, "wing_pitch_right", 0.0)
            return

        # Connectome Haltere + Ocellar chemical MNs + monosynaptic haltere->b1 electrical gap-junction reflex
        pitch_ang = math.atan2(
            float(self._world_z_lp[0]), max(0.05, float(self._world_z_lp[2]))
        )
        roll_ang = math.atan2(
            float(self._world_z_lp[1]), max(0.05, float(self._world_z_lp[2]))
        )
        gap_junc_gate = float(np.clip(1.5 * mean_wing, 0.25, 1.0))
        elec_pitch = gap_junc_gate * (
            0.80 * (0.73 - pitch_ang) + 0.015 * float(self._gyro_lp[1])
        )
        elec_roll = gap_junc_gate * (-0.60 * roll_ang - 0.010 * float(self._gyro_lp[0]))

        chem_pitch = 0.22 * (self.rate("MN_Pitch_Up") - self.rate("MN_Pitch_Down"))
        chem_roll = 0.18 * (self.rate("MN_Roll_R") - self.rate("MN_Roll_L"))
        visual_asym = wing_r - wing_l

        nose_up = float(np.clip(elec_pitch + chem_pitch, -0.45, 0.45))
        roll_c = float(np.clip(elec_roll + chem_roll, -0.32, 0.32))
        yaw_c = float(
            np.clip(
                0.22 * visual_asym - 0.008 * float(self._gyro_lp[2]),
                -0.25,
                0.25,
            )
        )

        # Vertical speed & target altitude hold for WBPG frequency (user_0)
        vel = np.asarray(obs.get("walker/velocimeter", ()), dtype=np.float64).reshape(
            -1
        )
        vz_world = float(np.dot(vel[:3], self._world_z_lp)) if vel.size >= 3 else 0.0
        self._altitude_integral = float(
            np.clip(self._altitude_integral + vz_world * step_dt, -0.5, 0.5)
        )

        ref_dz = 0.0
        ref_disp = obs.get("walker/ref_displacement")
        if ref_disp is not None:
            disp_arr = np.asarray(ref_disp, dtype=np.float64)
            if disp_arr.ndim == 2 and disp_arr.shape[0] > 0 and disp_arr.shape[1] >= 3:
                ref_dz = float(disp_arr[0, 2])

        if self._uses_wbpg:
            # WBPG handles the 218 Hz wingbeat carrier; connectome outputs steering & beat-frequency trim
            alt_cmd = (
                0.55
                + 0.40 * mean_wing
                + 2.0 * ref_dz
                - 1.8 * self._altitude_integral
                - 0.035 * vz_world
            )
            self._set_act(action, "wing_yaw_left", -0.40 * nose_up + yaw_c)
            self._set_act(action, "wing_yaw_right", -0.40 * nose_up - yaw_c)
            self._set_act(action, "wing_roll_left", 0.0)
            self._set_act(action, "wing_roll_right", 0.0)
            self._set_act(action, "wing_pitch_left", 0.40 * nose_up + roll_c)
            self._set_act(action, "wing_pitch_right", 0.40 * nose_up - roll_c)
            self._set_act(action, "user_0", float(np.clip(alt_cmd, -1.0, 1.0)))
        else:
            # Direct wing joint control (e.g. escape_takeoff without WBPG): drive inclined stroke plane directly
            amp_l = float(np.clip(0.35 + 0.65 * wing_l, 0.0, 1.0))
            amp_r = float(np.clip(0.35 + 0.65 * wing_r, 0.0, 1.0))
            c_wave = math.cos(self._wing_phase)
            s_wave = math.sin(self._wing_phase)
            self._set_act(action, "wing_yaw_left", 1.15 * amp_l * c_wave - 0.25 + yaw_c)
            self._set_act(
                action, "wing_yaw_right", 1.15 * amp_r * c_wave - 0.25 - yaw_c
            )
            self._set_act(action, "wing_roll_left", -0.90 * amp_l * c_wave)
            self._set_act(action, "wing_roll_right", -0.90 * amp_r * c_wave)
            self._set_act(
                action, "wing_pitch_left", -0.78 * amp_l * s_wave + 0.12 + roll_c
            )
            self._set_act(
                action, "wing_pitch_right", -0.78 * amp_r * s_wave + 0.12 - roll_c
            )

        if include_posture:
            self._set_act(action, "head", -0.15 * nose_up)
            self._set_act(action, "head_twist", 0.45 * visual_asym)
            self._set_act(action, "head_abduct", -0.15 * visual_asym)
            self._set_act(action, "abdomen", 0.50 * nose_up)
            self._set_act(action, "abdomen_abduct", -0.30 * visual_asym)

    def dominant_circuit(self) -> str:
        if self._jump_timer > 0.0 or self.rate("DNp01") > 0.45:
            return "giant_fiber_escape"
        if self._is_flight_spec or (
            0.5 * (self.rate("DNg02_L") + self.rate("DNg02_R")) > 0.45
        ):
            return "flight_tectulum"
        if self.rate("aDN1") > 0.40:
            return "antennal_grooming"
        if self.rate("MDN") > 0.45:
            return "moonwalker_backward"
        turn_diff = abs(self.rate("DNa01_L") - self.rate("DNa01_R"))
        if turn_diff > 0.18:
            return "lal_asymmetric_turn"
        if max(self.rate("DNp09_L"), self.rate("DNp09_R")) > 0.20:
            return "dnp09_forward_walk"
        return "pair1_quiet_stance"

    def diagnostics(self) -> Dict[str, Any]:
        dn_rates = {
            "DNp09": round(0.5 * (self.rate("DNp09_L") + self.rate("DNp09_R")), 4),
            "DNa01_L": round(self.rate("DNa01_L"), 4),
            "DNa01_R": round(self.rate("DNa01_R"), 4),
            "DNa02_L": round(self.rate("DNa02_L"), 4),
            "DNa02_R": round(self.rate("DNa02_R"), 4),
            "MDN": round(self.rate("MDN"), 4),
            "Pair1": round(self.rate("Pair1"), 4),
            "aDN1": round(self.rate("aDN1"), 4),
            "DNp01": round(self.rate("DNp01"), 4),
            "DNg02": round(0.5 * (self.rate("DNg02_L") + self.rate("DNg02_R")), 4),
        }
        diag: Dict[str, Any] = {
            "controller": "connectome",
            "dynamics": self.dynamics,
            "connectome_id": str(
                self.connectome.metadata.get("id") or CONNECTOME_RELEASE_METADATA["id"]
            ),
            "active_nodes": self._num_nodes,
            "synaptic_edges": self._weights.nnz,
            "dominant_circuit": self.dominant_circuit(),
            "descending_neurons": dn_rates,
            "vnc_motor": {
                "walk_drive": round(self.rate("VNC_Walk_Drive"), 4),
                "stride_left": round(self.rate("MN_Stride_L"), 4),
                "stride_right": round(self.rate("MN_Stride_R"), 4),
                "wing_left": round(self.rate("WTct_Wing_L"), 4),
                "wing_right": round(self.rate("WTct_Wing_R"), 4),
                "pitch_up": round(self.rate("MN_Pitch_Up"), 4),
                "groom_t1": round(self.rate("VNC_Groom_T1"), 4),
                "jump": round(self.rate("VNC_Jump"), 4),
            },
            "sensory": dict(self._last_sensory_summary),
        }
        if self.dynamics == "spiking":
            mean_hz = (
                self._total_spikes / max(1e-6, self._sim_time * self._num_nodes)
                if self._sim_time > 0.0
                else 0.0
            )
            diag["spiking"] = {
                "step_spikes": int(self._last_step_spikes),
                "total_spikes": int(self._total_spikes),
                "mean_firing_rate_hz": round(float(mean_hz), 2),
            }
        if self._ablations:
            diag["ablations"] = dict(self._ablations)
        if self._stimulations:
            diag["active_stimulations"] = [
                {
                    "cell_type": str(s["cell_type"]),
                    "current": round(float(s["current"]), 3),
                }
                for s in self._stimulations
            ]
        if self.enable_plasticity:
            diag["plasticity"] = {
                "weights": self.plasticity_weights(),
                "mbon_approach": round(self.rate("MBON_Approach"), 4),
                "mbon_avoid": round(self.rate("MBON_Avoid"), 4),
                "pam": round(self.rate("PAM"), 4),
                "ppl1": round(self.rate("PPL1"), 4),
            }
        return diag
