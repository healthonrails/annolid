#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_VERSION="3.11"
FLYBODY_PATH=""
ANNOLID_EXTRAS="[gui]"
RUN_PROBE=1
FORCE_ANNOLID_INSTALL=0

usage() {
  cat <<'EOF'
Usage:
  scripts/setup_flybody_uv.sh --flybody-path /path/to/flybody [options]

Options:
  --flybody-path PATH     Local FlyBody checkout to install in editable mode.
  --python VERSION        Python version for `uv venv` when creating `.venv`.
                          Default: 3.11
  --venv-dir PATH         Virtual environment directory.
                          Default: <repo>/.venv
  --annolid-extras EXTRAS Extras to install for Annolid editable mode.
                          Default: [gui]
  --force-annolid-install Reinstall Annolid even if it is already importable.
  --skip-probe            Skip `python scripts/check_flybody_runtime.py`.
  -h, --help              Show this help text.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --flybody-path)
      FLYBODY_PATH="${2:-}"
      shift 2
      ;;
    --python)
      PYTHON_VERSION="${2:-}"
      shift 2
      ;;
    --venv-dir)
      VENV_DIR="${2:-}"
      shift 2
      ;;
    --annolid-extras)
      ANNOLID_EXTRAS="${2:-}"
      shift 2
      ;;
    --force-annolid-install)
      FORCE_ANNOLID_INSTALL=1
      shift
      ;;
    --skip-probe)
      RUN_PROBE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${FLYBODY_PATH}" ]]; then
  echo "--flybody-path is required." >&2
  usage >&2
  exit 2
fi

if [[ ! -d "${FLYBODY_PATH}" ]]; then
  echo "FlyBody path does not exist: ${FLYBODY_PATH}" >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but not found on PATH." >&2
  exit 2
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Creating ${VENV_DIR} with Python ${PYTHON_VERSION}"
  uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}"
fi

PYTHON_BIN="${VENV_DIR}/bin/python"

if [[ "${FORCE_ANNOLID_INSTALL}" -eq 1 ]] || ! uv pip show --python "${PYTHON_BIN}" annolid >/dev/null 2>&1; then
  echo "Installing Annolid editable package with extras ${ANNOLID_EXTRAS}"
  uv pip install --python "${PYTHON_BIN}" -e "${ROOT_DIR}${ANNOLID_EXTRAS}"
else
  echo "Annolid package already installed in ${VENV_DIR}; skipping editable reinstall"
fi

echo "Installing FlyBody runtime prerequisites into ${VENV_DIR}"
uv pip install --python "${PYTHON_BIN}" dm-control mujoco dm-tree mediapy h5py

echo "Installing FlyBody checkout in editable mode from ${FLYBODY_PATH}"
uv pip install --python "${PYTHON_BIN}" --no-deps -e "${FLYBODY_PATH}"

if [[ "${RUN_PROBE}" -eq 1 ]]; then
  echo "Running FlyBody runtime probe"
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/check_flybody_runtime.py"
else
  echo "Skipping runtime probe"
fi
