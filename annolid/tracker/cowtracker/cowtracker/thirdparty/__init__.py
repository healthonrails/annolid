# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Third-party modules compatibility shim.

This keeps backward compatibility for existing ``import cowtracker.thirdparty``
callers while delegating path management to the centralized dependency gateway.
"""

from cowtracker.dependencies import ensure_vggt_importable

ensure_vggt_importable()
