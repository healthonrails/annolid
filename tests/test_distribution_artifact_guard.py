from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

from scripts.check_distribution_artifacts import (
    find_forbidden_members,
    is_forbidden_member,
)


def test_distribution_guard_rejects_model_artifact_suffixes() -> None:
    assert is_forbidden_member("annolid-1.0.0/annolid/models/model.onnx")
    assert is_forbidden_member("annolid/segmentation/SAM/vit_b.pth")
    assert is_forbidden_member("annolid/realtime/yolo11n.mlpackage/Manifest.json")


def test_distribution_guard_allows_normal_package_files() -> None:
    assert not is_forbidden_member("annolid-1.0.0/annolid/gui/app.py")
    assert not is_forbidden_member("annolid/core/output/schema/agent_record.json")
    assert not is_forbidden_member("annolid/configs/runs/yolo_train.yaml")
    assert not is_forbidden_member("annolid/icons/icon.png")


def test_bundle_guard_rejects_heavy_runtime_modules() -> None:
    assert is_forbidden_member(
        "annolid/_internal/torch/lib/libtorch_cpu.dylib",
        artifact_kind="bundle",
    )
    assert is_forbidden_member(
        "annolid/_internal/transformers/models/auto.py",
        artifact_kind="bundle",
    )
    assert is_forbidden_member(
        "annolid/_internal/onnxruntime/capi/libonnxruntime.so",
        artifact_kind="bundle",
    )


def test_distribution_guard_does_not_use_bundle_only_runtime_names() -> None:
    assert not is_forbidden_member(
        "annolid-1.0.0/annolid/configs/runs/yolo_train.yaml",
        artifact_kind="distribution",
    )
    assert is_forbidden_member(
        "annolid/_internal/runs/yolo_train.yaml",
        artifact_kind="bundle",
    )


def test_distribution_guard_scans_tar_and_wheel(tmp_path: Path) -> None:
    tar_path = tmp_path / "annolid-1.0.0.tar.gz"
    with tarfile.open(tar_path, "w:gz") as archive:
        payload = b"{}"
        info = tarfile.TarInfo("annolid-1.0.0/annolid/segmentation/SAM/model.onnx")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    wheel_path = tmp_path / "annolid-1.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel_path, "w") as archive:
        archive.writestr("annolid/gui/app.py", "")

    matches = find_forbidden_members([tar_path, wheel_path])

    assert len(matches) == 1
    assert "model.onnx" in matches[0]


def test_bundle_guard_scans_directory(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "dist"
    runtime_dir = bundle_dir / "annolid" / "_internal" / "torch"
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "__init__.py").write_text("", encoding="utf-8")

    matches = find_forbidden_members([bundle_dir], artifact_kind="bundle")

    assert len(matches) == 2
    assert all("torch" in match for match in matches)
