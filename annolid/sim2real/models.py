from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

SimSpecies = Literal["mouse", "rat", "fly"]
SimTaskKind = Literal["yolo_seg", "yolo_pose", "yolo_detect"]
SimExportLevel = Literal["instances", "parts"]
SimTrainingBackend = Literal["colab", "local"]

SCHEMA_VERSION = 1

MOUSE_ARENAS: tuple[str, ...] = (
    "open_field",
    "home_cage",
    "circular_open_field",
    "elevated_plus_maze",
    "y_maze",
    "t_maze",
    "three_chamber",
    "light_dark_box",
    "barnes_maze",
    "novel_object",
)

MOUSE_CAMERA_VIEWS: tuple[str, ...] = (
    "top",
    "top-close",
    "oblique",
    "oblique-close",
    "bottom-up",
    "home-cage-top",
)

MOUSE_LIGHTING_PRESETS: tuple[str, ...] = (
    "default",
    "ir_night",
    "dim_warm",
    "calms_gray",
)

MOUSE_COAT_PAIRS: tuple[str, ...] = (
    "default",
    "c57_pair",
    "balbc_pair",
    "cba_pair",
    "nude_pair",
)

MOUSE_OCCLUDERS: tuple[str, ...] = (
    "none",
    "wire_bars",
)

MOUSE_HEADSTAGES: tuple[str, ...] = (
    "none",
    "resident",
    "intruder",
    "both",
)

MOUSE_SCENARIOS: tuple[str, ...] = (
    "arena_explore",
    "social_huddle",
    "curled_groom",
    "rearing_social",
    "ventral_gait",
    "calms_resident_intruder",
    "calms_corner_huddle",
    "calms_corner_right",
)

FLY_BEHAVIORS: tuple[str, ...] = (
    "walk_imitation",
    "forage_walk",
    "grooming",
    "backward_walk",
    "escape_takeoff",
    "flight_imitation",
    "template_task",
)

FLY_CAMERA_VIEWS: tuple[str, ...] = (
    "walker/track1",
    "walker/track2",
    "walker/track3",
    "walker/hero",
    "walker/side",
    "walker/back",
    "top_camera",
)

MOUSE_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_ear",
    "right_ear",
    "neck",
    "left_shoulder",
    "right_shoulder",
    "mid_back",
    "left_hip",
    "right_hip",
    "tail_base",
    "tail_mid",
    "tail_tip",
    "center",
)

FLY_KEYPOINT_NAMES: tuple[str, ...] = (
    "head",
    "thorax",
    "abdomen",
    "wing_left",
    "wing_right",
    "leg_front_left",
    "leg_front_right",
    "leg_mid_left",
    "leg_mid_right",
    "leg_hind_left",
    "leg_hind_right",
)

RAT_ARENAS: tuple[str, ...] = (
    "treadmill",
    "open_field",
    "circular_open_field",
    "bumpy_terrain",
    "elevated_plus_maze",
)

RAT_CAMERA_VIEWS: tuple[str, ...] = (
    "oblique",
    "track_right",
    "track_left",
    "track_top",
    "track_front",
    "ventral",
)

RAT_COAT_PATTERNS: tuple[str, ...] = (
    "black_hooded",
    "cinnamon",
    "gray",
)

RAT_INTERFACES: tuple[str, ...] = (
    "flexor_extensor",
    "position_stiffness",
    "position",
    "torque",
)

RAT_SCENARIOS: tuple[str, ...] = (
    "trot",
    "walk",
    "bound",
    "gait_transition",
    "bumpy_terrain",
    "open_field_explore",
    "v0_ablation_bound",
    "v3_ablation_trot",
    "stand",
)

RAT_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "head",
    "left_ear",
    "right_ear",
    "neck",
    "thorax",
    "torso_center",
    "lumbar",
    "pelvis",
    "tail_base",
    "tail_mid",
    "tail_tip",
    "left_forepaw",
    "right_forepaw",
    "left_hindpaw",
    "right_hindpaw",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_project_id(prefix: str = "sim2real") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    clean = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in prefix)
    return f"{clean}_{stamp}_{suffix}"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


@dataclass
class VideoEnvironmentProfile:
    """Environment signature inferred from one or more real example videos."""

    example_videos: list[str] = field(default_factory=list)
    species: str = "mouse"
    species_confidence: float = 0.5
    width: int = 640
    height: int = 480
    fps: float = 30.0
    total_frames: int = 0
    sampled_frames: int = 0
    is_grayscale: bool = False
    mean_luminance: float = 128.0
    std_luminance: float = 40.0
    mean_saturation: float = 20.0
    lab_mean: list[float] = field(default_factory=lambda: [128.0, 128.0, 128.0])
    lab_std: list[float] = field(default_factory=lambda: [35.0, 8.0, 8.0])
    rgb_mean: list[float] = field(default_factory=lambda: [128.0, 128.0, 128.0])
    rgb_std: list[float] = field(default_factory=lambda: [40.0, 40.0, 40.0])
    sharpness_laplacian_var: float = 120.0
    estimated_blur_sigma: float = 0.0
    estimated_noise_sigma: float = 2.0
    vignette_strength: float = 0.0
    estimated_subject_count: int = 2
    median_subject_area_ratio: float = 0.03
    subject_luminance_mean: float = 70.0
    subject_luminance_spread: float = 30.0
    recommended_arena: str = "open_field"
    recommended_camera_view: str = "top-close"
    recommended_lighting: str = "default"
    recommended_coat_pair: str = "default"
    recommended_occluder: str = "none"
    recommended_headstage: str = "none"
    recommended_scenarios: list[str] = field(
        default_factory=lambda: ["arena_explore", "social_huddle"]
    )
    reference_background_path: str = ""
    sample_montage_path: str = ""
    match_rationale: dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoEnvironmentProfile":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in dict(data or {}).items() if k in known}
        return cls(**filtered)


@dataclass
class SimDatasetConfig:
    """Configuration controlling 3D synthetic dataset generation & domain matching."""

    species: str = "mouse"
    task: str = "yolo_seg"
    export_level: str = "instances"
    arena: str = "open_field"
    camera_view: str = "top-close"
    lighting: str = "default"
    coat_pair: str = "default"
    occluder: str = "none"
    headstage: str = "none"
    headstage_style: str = "winged_wide"
    scenarios: list[str] = field(
        default_factory=lambda: ["arena_explore", "social_huddle"]
    )
    frames_per_scenario: int = 24
    width: int = 640
    height: int = 480
    fps: int = 8
    seed: int = 42
    val_fraction: float = 0.2
    match_example_photometry: bool = True
    photometry_strength: float = 0.75
    composite_on_real_background: bool = False
    domain_randomization: float = 0.25
    real_labeled_dir: Optional[str] = None
    real_oversample_factor: int = 2
    fly_subject_count: int = 1
    fly_brain_dynamics: str = "spiking"
    rat_coat: str = "black_hooded"
    rat_interface: str = "flexor_extensor"
    rat_ablation: str = "none"
    rat_ablation_scale: float = 0.0
    rat_terrain_bump_cm: float = 0.0
    allow_fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimDatasetConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in dict(data or {}).items() if k in known}
        return cls(**filtered)


@dataclass
class SimDatasetArtifact:
    """Output artifact produced by a Mouse 3D or FlyBody dataset generation run."""

    dataset_dir: str
    data_yaml_path: str
    labelme_dir: str
    manifest_path: str
    renderer: str
    species: str
    task: str
    export_level: str
    total_frames: int
    train_frames: int
    val_frames: int
    class_names: list[str] = field(default_factory=list)
    preview_images: list[str] = field(default_factory=list)
    preview_video: str = ""
    notes: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimDatasetArtifact":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in dict(data or {}).items() if k in known}
        return cls(**filtered)


@dataclass
class SimIterationEvaluation:
    """Combined training validation metrics and real-video evaluation metrics."""

    val_box_map50: Optional[float] = None
    val_mask_map50: Optional[float] = None
    val_pose_map50: Optional[float] = None
    best_train_epoch: Optional[int] = None
    final_train_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    evaluated_frames: int = 0
    frames_with_detections: int = 0
    detection_coverage: float = 0.0
    mean_confidence: float = 0.0
    p10_confidence: float = 0.0
    mean_detections_per_frame: float = 0.0
    expected_subject_count: int = 1
    subject_count_match_rate: float = 0.0
    temporal_iou_stability: float = 0.0
    area_stability: float = 0.0
    class_detection_counts: dict[str, int] = field(default_factory=dict)
    low_confidence_frames: list[int] = field(default_factory=list)
    empty_frames: list[int] = field(default_factory=list)
    supervised_miou: Optional[float] = None
    supervised_precision: Optional[float] = None
    supervised_recall: Optional[float] = None
    supervised_f1: Optional[float] = None
    sim2real_score: float = 0.0
    overlay_contact_sheet_path: str = ""
    per_video_summaries: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimIterationEvaluation":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in dict(data or {}).items() if k in known}
        return cls(**filtered)


@dataclass
class SimIterationAdjustment:
    """Actionable diagnosis and machine-applicable adjustments for the next iteration."""

    diagnosis: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    suggested_sim_config: dict[str, Any] = field(default_factory=dict)
    suggested_train_overrides: dict[str, Any] = field(default_factory=dict)
    should_continue: bool = True
    stop_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimIterationAdjustment":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in dict(data or {}).items() if k in known}
        return cls(**filtered)


@dataclass
class Sim2RealIterationRecord:
    """Complete provenance record of one Sim-to-Real generate -> train -> evaluate cycle."""

    iteration: int
    status: str = "pending"
    started_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    sim_config: dict[str, Any] = field(default_factory=dict)
    dataset: Optional[dict[str, Any]] = None
    train_backend: str = "colab"
    train_model: str = "yolo"
    train_weights_in: str = ""
    train_epochs: int = 20
    train_imgsz: int = 640
    train_batch: Optional[int] = None
    accelerator: str = "T4"
    colab_train_job_id: str = ""
    colab_predict_job_ids: list[str] = field(default_factory=list)
    train_output_dir: str = ""
    best_weights_path: str = ""
    prediction_dirs: list[str] = field(default_factory=list)
    evaluation: Optional[dict[str, Any]] = None
    adjustment: Optional[dict[str, Any]] = None
    score_delta_vs_prev: Optional[float] = None
    error: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Sim2RealIterationRecord":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in dict(data or {}).items() if k in known}
        return cls(**filtered)


@dataclass
class Sim2RealProjectState:
    """Persisted state for a Sim-to-Real iterative training project."""

    schema_version: int = SCHEMA_VERSION
    project_id: str = field(default_factory=new_project_id)
    project_dir: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    example_videos: list[str] = field(default_factory=list)
    profile: Optional[dict[str, Any]] = None
    current_sim_config: Optional[dict[str, Any]] = None
    iterations: list[dict[str, Any]] = field(default_factory=list)
    best_iteration: Optional[int] = None
    best_score: Optional[float] = None
    best_weights_path: str = ""
    report_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Sim2RealProjectState":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in dict(data or {}).items() if k in known}
        return cls(**filtered)


class Sim2RealProjectStore:
    """Atomic JSON store for a Sim-to-Real project directory."""

    STATE_FILENAME = "sim2real_state.json"

    def __init__(self, project_dir: str | Path) -> None:
        self.project_dir = Path(project_dir).expanduser().resolve()
        self.state_path = self.project_dir / self.STATE_FILENAME

    def exists(self) -> bool:
        return self.state_path.is_file()

    def load(self) -> Sim2RealProjectState:
        if not self.state_path.is_file():
            state = Sim2RealProjectState(project_dir=str(self.project_dir))
            return state
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        state = Sim2RealProjectState.from_dict(raw)
        state.project_dir = str(self.project_dir)
        return state

    def save(self, state: Sim2RealProjectState) -> Sim2RealProjectState:
        state.project_dir = str(self.project_dir)
        state.updated_at = utc_now_iso()
        _atomic_write_json(self.state_path, state.to_dict())
        return state
