"""Simulation integration contracts for physics and kinematics backends."""

from annolid.simulation.io import (
    load_pose_schema,
    read_pose_frames,
    write_simulation_ndjson,
)
from annolid.simulation.lifting import lift_pose_frames_with_depth, load_depth_records
from annolid.simulation.mapping import (
    load_simulation_mapping,
    simulation_mapping_from_dict,
)
from annolid.simulation.smoothing import smooth_pose_frames
from annolid.simulation.templates import (
    generate_flybody_mapping_template,
    save_simulation_mapping_template,
)
from annolid.simulation.types import (
    Pose2DFrame,
    Pose3DFrame,
    SimulationAdapter,
    SimulationFrameResult,
    SimulationMapping,
    SimulationRunResult,
)
from annolid.simulation.viewer import (
    build_simulation_view_payload,
    export_simulation_view_payload,
)
from annolid.simulation.workflow import (
    SimulationRunRequest,
    build_default_output_path,
    build_simulation_adapter,
    run_simulation_workflow,
)

__all__ = [
    "Pose2DFrame",
    "Pose3DFrame",
    "SimulationAdapter",
    "SimulationFrameResult",
    "SimulationMapping",
    "SimulationRunResult",
    "load_pose_schema",
    "load_depth_records",
    "load_simulation_mapping",
    "read_pose_frames",
    "smooth_pose_frames",
    "simulation_mapping_from_dict",
    "generate_flybody_mapping_template",
    "save_simulation_mapping_template",
    "lift_pose_frames_with_depth",
    "write_simulation_ndjson",
    "build_simulation_view_payload",
    "export_simulation_view_payload",
    "SimulationRunRequest",
    "build_default_output_path",
    "build_simulation_adapter",
    "run_simulation_workflow",
]
