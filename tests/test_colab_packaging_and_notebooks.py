"""Tests for portable Colab packaging, zip-slip protection, command builders, and notebooks."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import yaml

from annolid.colab.models import ColabPredictRequest, ColabTrainRequest
from annolid.colab.notebooks import (
    build_colab_drive_url,
    build_colab_github_url,
    generate_predict_notebook,
    generate_train_notebook,
    list_notebook_templates,
    write_notebook,
)
from annolid.colab.packaging import (
    build_pip_install_spec,
    build_remote_predict_command,
    build_remote_runner_script,
    build_remote_train_command,
    package_inference_inputs,
    package_training_dataset,
    safe_extract_archive,
)


def _create_sample_yolo_dataset(root: Path) -> Path:
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)

    (root / "images" / "train" / "f001.jpg").write_bytes(b"img-train")
    (root / "labels" / "train" / "f001.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (root / "images" / "val" / "f002.jpg").write_bytes(b"img-val")
    (root / "labels" / "val" / "f002.txt").write_text(
        "0 0.4 0.4 0.2 0.2\n", encoding="utf-8"
    )
    (root / "pose_schema.json").write_text('{"keypoints": ["nose"]}', encoding="utf-8")

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root.resolve()),  # machine-local absolute path!
                "train": "images/train",
                "val": "images/val",
                "names": {0: "mouse"},
            }
        ),
        encoding="utf-8",
    )
    return data_yaml


def test_package_yolo_dataset_rewrites_portable_paths_without_mutating_source(
    tmp_path: Path,
):
    ds_root = tmp_path / "my_yolo_ds"
    data_yaml = _create_sample_yolo_dataset(ds_root)
    original_text = data_yaml.read_text(encoding="utf-8")
    local_weights = tmp_path / "custom_init.pt"
    local_weights.write_bytes(b"pt-bytes")

    staging = tmp_path / "staging"
    bundle = package_training_dataset(
        data_yaml,
        model="yolo",
        staging_dir=staging,
        weights=str(local_weights),
    )

    # Original file is untouched
    assert data_yaml.read_text(encoding="utf-8") == original_text
    assert bundle.dataset_entry == "data.yaml"
    assert bundle.weights_entry == "weights/custom_init.pt"
    assert bundle.archive_path.is_file()

    with zipfile.ZipFile(bundle.archive_path, "r") as zf:
        names = set(zf.namelist())
        assert "data.yaml" in names
        assert "images/train/f001.jpg" in names
        assert "labels/train/f001.txt" in names
        assert "images/val/f002.jpg" in names
        assert "labels/val/f002.txt" in names
        assert "pose_schema.json" in names
        assert "weights/custom_init.pt" in names

        portable = yaml.safe_load(zf.read("data.yaml").decode("utf-8"))
        assert portable["path"] == "."
        assert portable["train"] == "images/train"
        assert portable["val"] == "images/val"


def test_package_inference_inputs_bundles_local_weights_but_not_presets(tmp_path: Path):
    video = tmp_path / "trial01.mp4"
    video.write_bytes(b"mp4-data")
    local_pt = tmp_path / "best.pt"
    local_pt.write_bytes(b"weights-data")

    bundle_local = package_inference_inputs(
        video,
        staging_dir=tmp_path / "s1",
        weights=str(local_pt),
    )
    assert bundle_local.dataset_entry == "source/trial01.mp4"
    assert bundle_local.weights_entry == "weights/best.pt"

    bundle_preset = package_inference_inputs(
        video,
        staging_dir=tmp_path / "s2",
        weights="yolo11n-seg.pt",
    )
    assert bundle_preset.dataset_entry == "source/trial01.mp4"
    assert bundle_preset.weights_entry is None


def test_safe_extract_archive_rejects_zip_slip(tmp_path: Path):
    evil_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil_zip, "w") as zf:
        zf.writestr("../escape.txt", "pwned")

    with pytest.raises(ValueError, match="outside destination"):
        safe_extract_archive(evil_zip, tmp_path / "dest")

    with pytest.raises(ValueError, match="Unsafe path traversal"):
        package_training_dataset(evil_zip, model="yolo", staging_dir=tmp_path / "stage")


def test_remote_command_builders_and_runner_script_compiles():
    train_req = ColabTrainRequest(
        model="yolo",
        dataset_path="/local/data.yaml",
        weights="yolo11s-seg.pt",
        epochs=50,
        batch=16,
        imgsz=640,
        preset="small_dataset",
        plots=True,
    )
    t_cmd = build_remote_train_command(
        train_req,
        remote_input_root="/content/annolid_jobs/j1/input",
        remote_output_root="/content/annolid_jobs/j1/output",
        dataset_entry="data.yaml",
    )
    assert t_cmd[:4] == ["annolid-run", "train", "yolo", "--data"]
    assert "/content/annolid_jobs/j1/input/data.yaml" in t_cmd
    assert "--weights" in t_cmd and "yolo11s-seg.pt" in t_cmd
    assert "--preset" in t_cmd and "small_dataset" in t_cmd

    pred_req = ColabPredictRequest(
        model="yolo_labelme",
        source_path="/local/video.mp4",
        weights="/local/best.pt",
        conf=0.35,
        save_csv=True,
        save_video=True,
    )
    p_cmd = build_remote_predict_command(
        pred_req,
        remote_input_root="/content/annolid_jobs/j2/input",
        remote_output_root="/content/annolid_jobs/j2/output",
        source_entry="source/video.mp4",
        weights_entry="weights/best.pt",
    )
    assert p_cmd[:3] == ["annolid-run", "predict", "yolo_labelme"]
    assert "/content/annolid_jobs/j2/input/weights/best.pt" in p_cmd
    assert "--save-csv" in p_cmd
    assert "--save-video" in p_cmd

    script = build_remote_runner_script(
        job_id="j1",
        kind="train",
        model="yolo",
        install_spec="annolid",
        remote_job_root="/content/annolid_jobs/j1",
        input_archive_remote_path="/content/annolid_jobs/j1/input_dataset.zip",
        remote_command=t_cmd,
    )
    compile(script, "colab_runner_test.py", "exec")
    assert "annolid[yolo]" in script
    assert "QT_QPA_PLATFORM" in script
    assert "ANNOLID_COLAB_ERROR_JSON=" in script
    assert "filter_supported_cli_flags" in script
    assert "ensure_annolid_installed" in script
    assert ".annolid_colab_env.json" in script
    assert "QtBindingsNotFoundError" in script
    assert build_pip_install_spec("annolid", "yolo_labelme") == "annolid[yolo]"
    assert build_pip_install_spec(
        "git+https://github.com/healthonrails/annolid.git", "dino_kpseg"
    ) == ("annolid[ml,training] @ git+https://github.com/healthonrails/annolid.git")


def test_shape_and_keypoints_work_headlessly_when_qtpy_has_no_bindings():
    import subprocess
    import sys

    code = (
        "import sys, types\n"
        "fake_qtpy = types.ModuleType('qtpy')\n"
        "fake_qtpy.__file__ = 'fake_qtpy.py'\n"
        "class QtBindingsNotFoundError(RuntimeError): pass\n"
        "def _fail(name):\n"
        "    if name.startswith('__'): raise AttributeError(name)\n"
        "    raise QtBindingsNotFoundError('No Qt bindings could be found')\n"
        "fake_qtpy.__getattr__ = _fail\n"
        "sys.modules['qtpy'] = fake_qtpy\n"
        "from annolid.gui.shape import Shape, _QT_AVAILABLE\n"
        "from annolid.annotation.keypoints import format_shape\n"
        "assert _QT_AVAILABLE is False\n"
        "s = Shape(label='mouse', shape_type='polygon')\n"
        "s.addPoint((1.0, 2.0)); s.addPoint((3.0, 4.0)); s.addPoint((1.0, 4.0))\n"
        "d = format_shape(s)\n"
        "assert d['label'] == 'mouse' and len(d['points']) == 3\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_colab_live_monitor_parser_and_results_csv_loader(tmp_path: Path):
    from annolid.colab.monitor import (
        ColabLogMonitorParser,
        load_metrics_from_results_csv,
    )

    parser = ColabLogMonitorParser()
    parser.reset(kind="train", model="yolo", session_name="annolid-gpu")

    sample_lines = [
        "Packaging inputs for yolo (backend=cli, accelerator=T4)...",
        "Provisioning Colab CLI session 'annolid-gpu' (T4)...",
        "Colab URL: https://colab.research.google.com/notebooks/empty.ipynb#datalabBackendUrl=https://colab.research.google.com/tun/m/gpu-t4-1",
        "|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |",
        "| N/A   46C    P8             11W /   70W |       0MiB /  15360MiB |      0%      Default |",
        "[annolid-colab][j1] Ensuring Annolid runtime (annolid[yolo])...",
        "[annolid-colab][j1] Running: annolid-run train yolo --data data.yaml",
        "        1/3     0.412G      2.100      4.200      4.100      1.600          0         41        320: 100% 9/9",
        "                   all         18        198      0.110      0.090      0.080     0.0300      0.105      0.085      0.075     0.0280",
        "        3/3     0.467G      1.754      3.534      3.752      1.485          0         41        320: 100% 9/9",
        "                   all         18        198      0.191      0.114      0.124     0.0527      0.191      0.114      0.121     0.0657",
        'ANNOLID_COLAB_RESULT_JSON={"job_id": "train_yolo_1", "artifact_count": 23, "elapsed_seconds": 39.67}',
    ]
    for line in sample_lines:
        parser.feed_line(line)

    st = parser.state
    assert st.gpu_name == "Tesla T4 (15360 MiB)"
    assert st.colab_url.startswith("https://colab.research.google.com/")
    assert st.current_step == 3
    assert st.total_steps == 3
    assert st.artifact_count == 23
    assert len(st.history) == 2
    assert st.latest_metrics is not None
    assert st.latest_metrics.map50 == pytest.approx(0.124)
    assert st.latest_metrics.mask_map50 == pytest.approx(0.121)
    assert st.latest_metrics.train_loss == pytest.approx(1.754 + 3.534 + 3.752 + 1.485)

    # Also verify results.csv replay
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "epoch,train/box_loss,train/seg_loss,train/cls_loss,train/dfl_loss,metrics/mAP50(B),metrics/mAP50(M),val/box_loss\n"
        "1,2.0,3.0,3.5,1.5,0.20,0.18,1.8\n"
        "2,1.5,2.5,2.8,1.2,0.45,0.42,1.3\n",
        encoding="utf-8",
    )
    pts = load_metrics_from_results_csv(csv_path)
    assert len(pts) == 2
    assert pts[-1].step == 2
    assert pts[-1].map50 == pytest.approx(0.45)
    assert pts[-1].mask_map50 == pytest.approx(0.42)
    assert pts[-1].val_loss == pytest.approx(1.3)


def test_generated_notebooks_and_canonical_notebook_are_valid_nbformat_v4(
    tmp_path: Path,
):
    train_req = ColabTrainRequest(
        model="yolo",
        dataset_path="/tmp/data.yaml",
        weights="yolo11n-seg.pt",
        epochs=20,
        accelerator="L4",
    )
    nb = generate_train_notebook(
        train_req,
        job_id="job_test_1",
        remote_job_root="/content/annolid_jobs/job_test_1",
        input_archive_name="input_dataset.zip",
        remote_command=[
            "annolid-run",
            "train",
            "yolo",
            "--data",
            "/content/annolid_jobs/job_test_1/input/data.yaml",
        ],
    )
    assert nb["nbformat"] == 4
    assert nb["metadata"]["accelerator"] == "GPU"
    assert nb["metadata"]["colab"]["gpuType"] == "L4"
    assert len(nb["cells"]) >= 8
    out_nb = write_notebook(nb, tmp_path / "train.ipynb")
    reloaded = json.loads(out_nb.read_text(encoding="utf-8"))
    for cell in reloaded["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "cell.py", "exec")

    pred_req = ColabPredictRequest(
        model="yolo_labelme",
        source_path="/tmp/video.mp4",
        accelerator="CPU",
    )
    nb_pred = generate_predict_notebook(
        pred_req,
        job_id="job_test_2",
        remote_job_root="/content/annolid_jobs/job_test_2",
        input_archive_name="input_inference.zip",
        remote_command=[
            "annolid-run",
            "predict",
            "yolo_labelme",
            "--source",
            "/content/annolid_jobs/job_test_2/input/source/video.mp4",
        ],
        use_drive=False,
    )
    assert "accelerator" not in nb_pred["metadata"]

    assert (
        build_colab_drive_url("abc123")
        == "https://colab.research.google.com/drive/abc123"
    )
    assert build_colab_github_url(
        "docs/tutorials/Annolid_Colab_Train_and_Inference.ipynb"
    ).startswith(
        "https://colab.research.google.com/github/healthonrails/annolid/blob/main/"
    )
    templates = list_notebook_templates()
    assert any(t["key"] == "unified_train_inference" for t in templates)

    # Validate the shipped canonical notebook file itself
    canonical_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "tutorials"
        / "Annolid_Colab_Train_and_Inference.ipynb"
    )
    assert canonical_path.is_file()
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    assert canonical["nbformat"] == 4
    for cell in canonical["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "canonical_cell.py", "exec")
