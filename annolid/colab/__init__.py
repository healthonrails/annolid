"""Google Colab authentication, runtime management, notebook generation, and train/predict workflows."""

from __future__ import annotations

from .auth import ColabAuthManager, ColabAuthRequiredError
from .backends import (
    ColabApiBackend,
    ColabCliBackend,
    ColabDriveNotebookBackend,
)
from .models import (
    ColabJobRecord,
    ColabJobStore,
    ColabPredictRequest,
    ColabRuntimeInfo,
    ColabRuntimeSpec,
    ColabTrainRequest,
    PREDICT_MODEL_CHOICES,
    SUPPORTED_ACCELERATORS,
    TRAIN_MODEL_CHOICES,
    colab_home_dir,
    normalize_accelerator,
)
from .monitor import (
    ColabLiveMonitorState,
    ColabLogMonitorParser,
    ColabMetricPoint,
    load_metrics_from_results_csv,
)
from .notebooks import (
    COLAB_NOTEBOOK_TEMPLATES,
    ColabNotebookTemplate,
    build_colab_drive_url,
    build_colab_github_url,
    generate_job_notebook,
    generate_predict_notebook,
    generate_train_notebook,
    get_notebook_template,
    list_notebook_templates,
    write_notebook,
)
from .packaging import (
    StagedInputBundle,
    build_pip_install_spec,
    build_remote_predict_command,
    build_remote_runner_script,
    build_remote_train_command,
    package_inference_inputs,
    package_training_dataset,
    required_extras_for_model,
    safe_extract_archive,
)
from .service import ColabWorkflowService

__all__ = [
    "COLAB_NOTEBOOK_TEMPLATES",
    "ColabApiBackend",
    "ColabAuthManager",
    "ColabAuthRequiredError",
    "ColabCliBackend",
    "ColabDriveNotebookBackend",
    "ColabJobRecord",
    "ColabJobStore",
    "ColabLiveMonitorState",
    "ColabLogMonitorParser",
    "ColabMetricPoint",
    "ColabNotebookTemplate",
    "ColabPredictRequest",
    "ColabRuntimeInfo",
    "ColabRuntimeSpec",
    "ColabTrainRequest",
    "ColabWorkflowService",
    "PREDICT_MODEL_CHOICES",
    "SUPPORTED_ACCELERATORS",
    "StagedInputBundle",
    "TRAIN_MODEL_CHOICES",
    "build_colab_drive_url",
    "build_colab_github_url",
    "build_pip_install_spec",
    "build_remote_predict_command",
    "build_remote_runner_script",
    "build_remote_train_command",
    "colab_home_dir",
    "generate_job_notebook",
    "generate_predict_notebook",
    "generate_train_notebook",
    "get_notebook_template",
    "list_notebook_templates",
    "load_metrics_from_results_csv",
    "normalize_accelerator",
    "package_inference_inputs",
    "package_training_dataset",
    "required_extras_for_model",
    "safe_extract_archive",
    "write_notebook",
]
