# CowTracker Notes

## VGGT runtime strategy

CowTracker uses a **minimal vendored VGGT runtime subset** under:

`annolid/tracker/cowtracker/cowtracker/thirdparty/vggt`

This keeps CowTracker runnable without requiring users to install full VGGT.

At runtime, CowTracker resolves VGGT in this order:

1. Use vendored subset (preferred) if present and complete.
2. Fallback to an externally installed `vggt` package.

If neither is available, CowTracker raises a dependency error with install hints.

Vendored runtime requirements are tracked in:

`annolid/tracker/cowtracker/cowtracker/thirdparty/vggt/VENDORED_MANIFEST.json`

and enforced by:

`annolid/tracker/cowtracker/cowtracker/vendor/vggt_runtime.py`

## Required vendored files

Keep these files if you vendor VGGT:

- `vggt/__init__.py`
- `vggt/models/__init__.py`
- `vggt/models/aggregator.py`
- `vggt/heads/__init__.py`
- `vggt/heads/dpt_head.py`
- `vggt/heads/head_act.py`
- `vggt/heads/utils.py`
- `vggt/layers/__init__.py`
- `vggt/layers/attention.py`
- `vggt/layers/block.py`
- `vggt/layers/drop_path.py`
- `vggt/layers/layer_scale.py`
- `vggt/layers/mlp.py`
- `vggt/layers/patch_embed.py`
- `vggt/layers/rope.py`
- `vggt/layers/swiglu_ffn.py`
- `vggt/layers/vision_transformer.py`
- `LICENSE.txt` (license preservation)

## Update workflow

When syncing VGGT-related changes:

1. Update vendored files under `thirdparty/vggt/vggt/...`.
2. Keep `VENDORED_MANIFEST.json` in sync with actual files.
3. Keep `vendor/vggt_runtime.py` in sync with the same list.
4. Run `pytest -q tests/test_cowtracker_dependencies.py tests/test_cowtracker_vggt_manifest.py`.

## Do not vendor as nested git repo

Do not keep `.git/` inside vendored `vggt`.

Nested repos create embedded-repository issues and break normal clone behavior.
