# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from hydra import initialize_config_module
from hydra.core.global_hydra import GlobalHydra

if not GlobalHydra.instance().is_initialized():  # pragma: no cover - init guard
    initialize_config_module(
        "annolid.segmentation.SAM.efficienttam", version_base="1.2"
    )
