# Installing Annolid with `uv`

This is the recommended development-style setup for this repository and matches the local validation guidance used in the repo instructions.

## Prerequisites

- Python 3.10 to 3.13 on your `PATH`
- `git`
- `ffmpeg` recommended for video-heavy workflows
- `uv` installed from <https://docs.astral.sh/uv/>

## 1. Clone the Repository

```bash
git clone --recurse-submodules https://github.com/healthonrails/annolid.git
cd annolid
```

## 2. Create `.venv`

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
uv venv .venv --python 3.11
.venv\Scripts\Activate.ps1
```

## 3. Install Annolid

GUI-capable editable install:

```bash
uv pip install -e ".[gui]"
```

If you also need Annolid Bot integrations:

```bash
uv pip install -e ".[gui,annolid_bot]"
```

## 4. Verify

```bash
annolid --help
annolid-run --help
annolid-run list-models
```

## Optional FlyBody Checkout

If you want the simulation/FlyBody plugin path in the same `.venv`, install the
optional runtime explicitly with `uv pip`:

```bash
source .venv/bin/activate
uv pip install --python .venv/bin/python dm-control mujoco dm-tree mediapy h5py
uv pip install --python .venv/bin/python --no-deps -e /path/to/flybody
python scripts/check_flybody_runtime.py
```

Equivalent helper:

```bash
scripts/setup_flybody_uv.sh --flybody-path /path/to/flybody
```

Alternative isolated environment:

```bash
scripts/setup_flybody_uv.sh --venv-dir .venv311 --python 3.11 --flybody-path /path/to/flybody
```

Notes:

- Prefer Python 3.11 for this setup.
- Python 3.13 can still require a `labmaze` compatibility workaround upstream.
- The Annolid plugin docs are in [Simulation and FlyBody](simulation_flybody.md).

## 5. Launch

```bash
annolid
```

## Notes for Current Codebase

- The current terminal command is `annolid-run`, not older `annolid-train` or `annolid-track` commands.
- The repo’s local validation instructions assume a repository-local `.venv`.
- Optional extras are defined in `pyproject.toml`; install only what you need.

## Troubleshooting

- If Qt bindings are missing, reinstall with the `gui` extra.
- If FFmpeg tools are missing, install `ffmpeg` with your package manager or conda-forge.
- If Annolid Bot features are missing, ensure you installed the `annolid_bot` extra and configured the relevant runtime files.
- If you want a fully reproducible dependency state, keep `uv.lock` in sync with your environment and use the repo’s validation checks.
