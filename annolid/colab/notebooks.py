from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from .models import ColabPredictRequest, ColabTrainRequest, normalize_accelerator
from .packaging import build_pip_install_spec, format_command_shell

ANNOLID_GITHUB_REPO = "healthonrails/annolid"
ANNOLID_DEFAULT_BRANCH = "main"


@dataclass(frozen=True)
class ColabNotebookTemplate:
    key: str
    title: str
    description: str
    repo_path: str
    category: str  # "unified" | "train" | "inference" | "legacy"
    recommended_gpu: str = "T4"

    def colab_url(self, *, branch: str = ANNOLID_DEFAULT_BRANCH) -> str:
        return build_colab_github_url(self.repo_path, branch=branch)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["colab_url"] = self.colab_url()
        return payload


COLAB_NOTEBOOK_TEMPLATES: tuple[ColabNotebookTemplate, ...] = (
    ColabNotebookTemplate(
        key="unified_train_inference",
        title="Annolid Train & Inference (Unified CLI)",
        description=(
            "Modern Annolid notebook supporting YOLO, DINO KPSEG, Mask R-CNN, "
            "and Keypoint R-CNN training & inference via `annolid-run` + Google Drive."
        ),
        repo_path="docs/tutorials/Annolid_Colab_Train_and_Inference.ipynb",
        category="unified",
        recommended_gpu="T4",
    ),
    ColabNotebookTemplate(
        key="yolo_seg",
        title="YOLO Instance Segmentation",
        description="Train and evaluate Ultralytics YOLO segmentation models on Annolid datasets.",
        repo_path="docs/tutorials/Annolid_Instance_Segmentation_on_YOLO_Tutorial.ipynb",
        category="train",
        recommended_gpu="T4",
    ),
    ColabNotebookTemplate(
        key="yolo_pose",
        title="YOLO Pose Estimation",
        description="Train Ultralytics YOLO pose/keypoint models on Annolid pose exports.",
        repo_path="docs/tutorials/Annolid_Pose_Estimation_on_YOLO_Tutorial.ipynb",
        category="train",
        recommended_gpu="T4",
    ),
    ColabNotebookTemplate(
        key="video_batch_inference",
        title="Video Batch Inference",
        description="Run batched video inference and export tracking results on Colab GPUs.",
        repo_path="docs/tutorials/Annolid_video_batch_inference.ipynb",
        category="inference",
        recommended_gpu="T4",
    ),
    ColabNotebookTemplate(
        key="detectron2_legacy",  # gitleaks:allow
        title="Detectron2 Mask R-CNN (Legacy)",
        description="Legacy Detectron2 tutorial notebook preserved for historical workflows.",
        repo_path="docs/tutorials/Annolid_on_Detectron2_Tutorial.ipynb",
        category="legacy",
        recommended_gpu="T4",
    ),
)


def build_colab_github_url(
    repo_rel_path: str,
    *,
    repo: str = ANNOLID_GITHUB_REPO,
    branch: str = ANNOLID_DEFAULT_BRANCH,
) -> str:
    cleaned = str(repo_rel_path or "").strip().lstrip("/")
    return f"https://colab.research.google.com/github/{repo}/blob/{branch}/{cleaned}"


def build_colab_drive_url(file_id: str) -> str:
    clean_id = str(file_id or "").strip()
    if not clean_id:
        raise ValueError("Google Drive file_id is required to build a Colab Drive URL.")
    return f"https://colab.research.google.com/drive/{clean_id}"


def list_notebook_templates() -> list[dict[str, Any]]:
    return [tpl.to_dict() for tpl in COLAB_NOTEBOOK_TEMPLATES]


def get_notebook_template(key: str) -> Optional[ColabNotebookTemplate]:
    target = str(key or "").strip().lower()
    for tpl in COLAB_NOTEBOOK_TEMPLATES:
        if tpl.key.lower() == target:
            return tpl
    return None


def _md_cell(text: str) -> dict[str, Any]:
    lines = text.strip("\n").splitlines(keepends=True)
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": lines,
    }


def _code_cell(code: str) -> dict[str, Any]:
    lines = code.strip("\n").splitlines(keepends=True)
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def _colab_accelerator_metadata(accelerator: str) -> dict[str, Any]:
    acc = normalize_accelerator(accelerator)
    meta: dict[str, Any] = {
        "name": "Annolid Colab Job",
        "provenance": [],
    }
    if acc == "CPU":
        return meta
    if acc.startswith("TPU_"):
        return meta
    meta["gpuType"] = acc
    return meta


def _wrap_notebook(
    cells: list[dict[str, Any]],
    *,
    title: str,
    accelerator: str = "T4",
) -> dict[str, Any]:
    acc = normalize_accelerator(accelerator)
    colab_meta = _colab_accelerator_metadata(acc)
    colab_meta["name"] = title
    metadata: dict[str, Any] = {
        "colab": colab_meta,
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
        },
    }
    if acc.startswith("TPU_"):
        metadata["accelerator"] = "TPU"
    elif acc != "CPU":
        metadata["accelerator"] = "GPU"
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": metadata,
        "cells": cells,
    }


def generate_job_notebook(
    *,
    job_id: str,
    kind: str,
    model: str,
    install_spec: str,
    accelerator: str,
    remote_job_root: str,
    input_archive_name: str,
    remote_command: Sequence[str],
    drive_job_rel_path: Optional[str] = None,
    use_drive: bool = True,
) -> dict[str, Any]:
    """Build an executable nbformat v4 Colab notebook for an Annolid train/predict job."""
    pip_spec = build_pip_install_spec(install_spec, model)
    mode_label = "Training" if kind == "train" else "Inference"
    title = f"Annolid {mode_label} on Google Colab ({model}) — {job_id}"
    drive_path_expr = (
        f"/content/drive/MyDrive/{drive_job_rel_path.strip('/')}"
        if drive_job_rel_path
        else f"/content/drive/MyDrive/Annolid/colab/{job_id}"
    )
    cmd_json = json.dumps(list(remote_command))
    cmd_preview = format_command_shell(remote_command)

    cells: list[dict[str, Any]] = [
        _md_cell(
            f"# {title}\n\n"
            f"- **Job ID:** `{job_id}`\n"
            f"- **Mode:** `{kind}`\n"
            f"- **Model:** `{model}`\n"
            f"- **Target Accelerator:** `{normalize_accelerator(accelerator)}`\n\n"
            "Run each cell in order (or choose **Runtime -> Run all**). "
            "After completion, outputs are saved to `output_bundle.zip` and `job_result.json` "
            "so you can pull them straight back into Annolid with:\n\n"
            f"```bash\nannolid-run colab pull {job_id}\n```"
        ),
        _md_cell("## 1. Check Colab Hardware & GPU"),
        _code_cell(
            "import platform, shutil, subprocess\n"
            "print('Python:', platform.python_version())\n"
            "if shutil.which('nvidia-smi'):\n"
            "    subprocess.run(['nvidia-smi'], check=False)\n"
            "else:\n"
            "    print('No NVIDIA GPU detected (running on CPU/TPU).')\n"
        ),
        _md_cell("## 2. Install Annolid Runtime & Model Dependencies"),
        _code_cell(
            f"PIP_SPEC = {pip_spec!r}\n"
            "import subprocess, sys\n"
            "print('Installing:', PIP_SPEC)\n"
            "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', PIP_SPEC])\n"
            "import torch\n"
            "print('PyTorch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())\n"
        ),
    ]

    if use_drive:
        cells.append(_md_cell("## 3. Mount Google Drive & Stage Inputs"))
        cells.append(
            _code_cell(
                "import shutil, zipfile\n"
                "from pathlib import Path\n"
                "from google.colab import drive\n\n"
                "drive.mount('/content/drive')\n"
                f"JOB_ROOT = Path({remote_job_root!r})\n"
                f"DRIVE_JOB_DIR = Path({drive_path_expr!r})\n"
                f"INPUT_ARCHIVE_NAME = {input_archive_name!r}\n"
                "INPUT_DIR = JOB_ROOT / 'input'\n"
                "OUTPUT_DIR = JOB_ROOT / 'output'\n"
                "INPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
                "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
                "DRIVE_JOB_DIR.mkdir(parents=True, exist_ok=True)\n\n"
                "src_archive = DRIVE_JOB_DIR / INPUT_ARCHIVE_NAME\n"
                "if not src_archive.is_file():\n"
                "    from google.colab import files\n"
                "    print(f'Archive not found at {src_archive}. Upload {INPUT_ARCHIVE_NAME} below:')\n"
                "    uploaded = files.upload()\n"
                "    first_name = next(iter(uploaded.keys()))\n"
                "    src_archive = Path('/content') / first_name\n\n"
                "with zipfile.ZipFile(src_archive, 'r') as zf:\n"
                "    for info in zf.infolist():\n"
                "        dest = (INPUT_DIR / info.filename).resolve()\n"
                "        if INPUT_DIR.resolve() != dest and INPUT_DIR.resolve() not in dest.parents:\n"
                "            raise ValueError(f'Unsafe archive path: {info.filename}')\n"
                "        if info.is_dir():\n"
                "            dest.mkdir(parents=True, exist_ok=True)\n"
                "            continue\n"
                "        dest.parent.mkdir(parents=True, exist_ok=True)\n"
                "        with zf.open(info, 'r') as s, open(dest, 'wb') as d:\n"
                "            shutil.copyfileobj(s, d)\n"
                "import yaml as _yaml\n"
                "for _yml in list(INPUT_DIR.glob('data.yaml')) + list(INPUT_DIR.glob('*/data.yaml')):\n"
                "    _cfg = _yaml.safe_load(_yml.read_text(encoding='utf-8'))\n"
                "    if isinstance(_cfg, dict) and str(_cfg.get('path', '')).strip() in {'', '.'}:\n"
                "        _cfg['path'] = str(_yml.parent.resolve())\n"
                "        _yml.write_text(_yaml.safe_dump(_cfg, sort_keys=False), encoding='utf-8')\n"
                "print('Extracted input files to:', INPUT_DIR)\n"
            )
        )
    else:
        cells.append(_md_cell("## 3. Upload & Extract Staged Input Archive"))
        cells.append(
            _code_cell(
                "import shutil, zipfile\n"
                "from pathlib import Path\n"
                "from google.colab import files\n\n"
                f"JOB_ROOT = Path({remote_job_root!r})\n"
                "DRIVE_JOB_DIR = None\n"
                "INPUT_DIR = JOB_ROOT / 'input'\n"
                "OUTPUT_DIR = JOB_ROOT / 'output'\n"
                "INPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
                "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n\n"
                f"print('Please upload {input_archive_name} created by Annolid:')\n"
                "uploaded = files.upload()\n"
                "first_name = next(iter(uploaded.keys()))\n"
                "src_archive = Path('/content') / first_name\n"
                "with zipfile.ZipFile(src_archive, 'r') as zf:\n"
                "    for info in zf.infolist():\n"
                "        dest = (INPUT_DIR / info.filename).resolve()\n"
                "        if INPUT_DIR.resolve() != dest and INPUT_DIR.resolve() not in dest.parents:\n"
                "            raise ValueError(f'Unsafe archive path: {info.filename}')\n"
                "        if info.is_dir():\n"
                "            dest.mkdir(parents=True, exist_ok=True)\n"
                "            continue\n"
                "        dest.parent.mkdir(parents=True, exist_ok=True)\n"
                "        with zf.open(info, 'r') as s, open(dest, 'wb') as d:\n"
                "            shutil.copyfileobj(s, d)\n"
                "import yaml as _yaml\n"
                "for _yml in list(INPUT_DIR.glob('data.yaml')) + list(INPUT_DIR.glob('*/data.yaml')):\n"
                "    _cfg = _yaml.safe_load(_yml.read_text(encoding='utf-8'))\n"
                "    if isinstance(_cfg, dict) and str(_cfg.get('path', '')).strip() in {'', '.'}:\n"
                "        _cfg['path'] = str(_yml.parent.resolve())\n"
                "        _yml.write_text(_yaml.safe_dump(_cfg, sort_keys=False), encoding='utf-8')\n"
                "print('Extracted input files to:', INPUT_DIR)\n"
            )
        )

    cells.append(
        _md_cell(
            f"## 4. Run Annolid {mode_label}\n\nCommand:\n```bash\n{cmd_preview}\n```"
        )
    )
    cells.append(
        _code_cell(
            "import os, re, shutil, subprocess, sys, time\n\n"
            "os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')\n"
            f"MODEL_NAME = {model!r}\n"
            f"REMOTE_CMD = {cmd_json}\n"
            "if MODEL_NAME == 'yolo_labelme':\n"
            "    _qt_probe = subprocess.run([sys.executable, '-c', 'import annolid.segmentation.yolos'], capture_output=True, text=True)\n"
            "    if _qt_probe.returncode != 0 and 'QtBindingsNotFoundError' in ((_qt_probe.stderr or '') + (_qt_probe.stdout or '')):\n"
            "        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'PySide6-Essentials>=6.7.0'], check=True)\n\n"
            "cmd = list(REMOTE_CMD)\n"
            "if cmd and cmd[0] == 'annolid-run' and not shutil.which('annolid-run'):\n"
            "    cmd = [sys.executable, '-m', 'annolid.engine.cli', *cmd[1:]]\n\n"
            "sub_idx = next((i for i, t in enumerate(cmd[:-1]) if t in {'train', 'predict'}), -1)\n"
            "if sub_idx != -1 and sub_idx + 1 < len(cmd):\n"
            "    _hp = subprocess.run([*cmd[:sub_idx + 2], '--help-model'], capture_output=True, text=True)\n"
            "    _ht = (_hp.stdout or '') + '\\n' + (_hp.stderr or '')\n"
            "    if _hp.returncode == 0 and '--help' in _ht:\n"
            "        _supported = set(re.findall(r'--[a-z0-9][a-z0-9-]*', _ht))\n"
            "        _opt_val = {'--name', '--preset', '--conf', '--iou', '--imgsz', '--batch', '--save-video'}\n"
            "        _opt_bool = {'--resume', '--save-csv', '--plots'}\n"
            "        _filtered = list(cmd[:sub_idx + 2])\n"
            "        _i = sub_idx + 2\n"
            "        while _i < len(cmd):\n"
            "            _tok = cmd[_i]\n"
            "            if _tok in _opt_val and _tok not in _supported:\n"
            "                print(f'[WARN] Remote annolid does not support {_tok}; skipping.')\n"
            "                _i += 2 if _i + 1 < len(cmd) and not cmd[_i + 1].startswith('-') else 1\n"
            "                continue\n"
            "            if _tok in _opt_bool and _tok not in _supported:\n"
            "                print(f'[WARN] Remote annolid does not support {_tok}; skipping.')\n"
            "                _i += 1\n"
            "                continue\n"
            "            _filtered.append(_tok)\n"
            "            _i += 1\n"
            "        cmd = _filtered\n\n"
            "started = time.time()\n"
            "print('Executing:', ' '.join(cmd))\n"
            "proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n"
            "assert proc.stdout is not None\n"
            "for line in proc.stdout:\n"
            "    print(line, end='', flush=True)\n"
            "rc = proc.wait()\n"
            "elapsed_s = round(time.time() - started, 2)\n"
            "if rc != 0:\n"
            "    raise RuntimeError(f'Annolid command failed with exit code {rc}')\n"
            "print(f'Completed in {elapsed_s}s.')\n"
        )
    )

    cells.append(_md_cell("## 5. Package Outputs & Sync to Google Drive / Download"))
    cells.append(
        _code_cell(
            "import json, shutil, zipfile\n"
            "from pathlib import Path\n\n"
            "OUTPUT_ARCHIVE = JOB_ROOT / 'output_bundle.zip'\n"
            "RESULT_MANIFEST = JOB_ROOT / 'job_result.json'\n"
            "artifacts = []\n"
            "with zipfile.ZipFile(OUTPUT_ARCHIVE, 'w', compression=zipfile.ZIP_DEFLATED) as zf:\n"
            "    for p in sorted(OUTPUT_DIR.rglob('*')):\n"
            "        if p.is_file():\n"
            "            rel = p.relative_to(OUTPUT_DIR).as_posix()\n"
            "            zf.write(p, rel)\n"
            "            artifacts.append(rel)\n\n"
            "result = {\n"
            f"    'job_id': {job_id!r},\n"
            f"    'kind': {kind!r},\n"
            f"    'model': {model!r},\n"
            "    'status': 'completed',\n"
            "    'elapsed_seconds': elapsed_s,\n"
            "    'artifact_count': len(artifacts),\n"
            "    'artifacts': artifacts,\n"
            "}\n"
            "RESULT_MANIFEST.write_text(json.dumps(result, indent=2), encoding='utf-8')\n"
            "if DRIVE_JOB_DIR is not None:\n"
            "    DRIVE_JOB_DIR.mkdir(parents=True, exist_ok=True)\n"
            "    shutil.copy2(OUTPUT_ARCHIVE, DRIVE_JOB_DIR / OUTPUT_ARCHIVE.name)\n"
            "    shutil.copy2(RESULT_MANIFEST, DRIVE_JOB_DIR / RESULT_MANIFEST.name)\n"
            "    print('Saved outputs to Google Drive:', DRIVE_JOB_DIR)\n"
            "print(f'Packaged {len(artifacts)} artifact(s) into {OUTPUT_ARCHIVE}')\n"
        )
    )
    cells.append(
        _code_cell(
            "# Optional: download output_bundle.zip directly to your browser\n"
            "DOWNLOAD_IN_BROWSER = False\n"
            "if DOWNLOAD_IN_BROWSER:\n"
            "    from google.colab import files\n"
            "    files.download(str(OUTPUT_ARCHIVE))\n"
        )
    )

    return _wrap_notebook(cells, title=title, accelerator=accelerator)


def generate_train_notebook(
    request: ColabTrainRequest,
    *,
    job_id: str,
    remote_job_root: str,
    input_archive_name: str,
    remote_command: Sequence[str],
    drive_job_rel_path: Optional[str] = None,
    use_drive: bool = True,
) -> dict[str, Any]:
    return generate_job_notebook(
        job_id=job_id,
        kind="train",
        model=request.model,
        install_spec=request.annolid_install_spec,
        accelerator=request.accelerator,
        remote_job_root=remote_job_root,
        input_archive_name=input_archive_name,
        remote_command=remote_command,
        drive_job_rel_path=drive_job_rel_path,
        use_drive=use_drive,
    )


def generate_predict_notebook(
    request: ColabPredictRequest,
    *,
    job_id: str,
    remote_job_root: str,
    input_archive_name: str,
    remote_command: Sequence[str],
    drive_job_rel_path: Optional[str] = None,
    use_drive: bool = True,
) -> dict[str, Any]:
    return generate_job_notebook(
        job_id=job_id,
        kind="predict",
        model=request.model,
        install_spec=request.annolid_install_spec,
        accelerator=request.accelerator,
        remote_job_root=remote_job_root,
        input_archive_name=input_archive_name,
        remote_command=remote_command,
        drive_job_rel_path=drive_job_rel_path,
        use_drive=use_drive,
    )


def write_notebook(notebook_payload: dict[str, Any], output_path: str | Path) -> Path:
    dest = Path(output_path).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(
        json.dumps(notebook_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(dest)
    return dest
