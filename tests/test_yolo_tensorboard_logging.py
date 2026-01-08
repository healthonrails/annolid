from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Tuple

from annolid.yolo.tensorboard_logging import (
    YOLORunTensorBoardLogger,
    sanitize_tensorboard_tag,
)


@dataclass
class _StubWriter:
    scalars: List[Tuple[str, float, int]] = field(default_factory=list)
    texts: List[Tuple[str, str, int]] = field(default_factory=list)
    images: List[Tuple[str, Any, int, str]] = field(default_factory=list)
    flushed: int = 0
    closed: int = 0

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        self.scalars.append((tag, float(scalar_value), int(global_step)))

    def add_text(self, tag: str, text_string: str, global_step: int) -> None:
        self.texts.append((tag, str(text_string), int(global_step)))

    def add_image(self, tag: str, img_tensor: Any, global_step: int, dataformats: str = "HWC") -> None:
        self.images.append(
            (tag, img_tensor, int(global_step), str(dataformats)))

    def flush(self) -> None:
        self.flushed += 1

    def close(self) -> None:
        self.closed += 1


def _write_results_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_sanitize_tensorboard_tag_preserves_groups_and_removes_bad_chars():
    assert sanitize_tensorboard_tag("train/box_loss") == "train/box_loss"
    assert sanitize_tensorboard_tag("metrics/mAP50(B)") == "metrics/mAP50_B"


def test_tb_logger_logs_new_epoch_metrics_from_results_csv(tmp_path: Path):
    run_dir = tmp_path / "run"
    results_csv = run_dir / "results.csv"
    _write_results_csv(
        results_csv,
        rows=[
            {"epoch": "0", "train/box_loss": "1.0", "metrics/mAP50(B)": "0.1"},
            {"epoch": "1", "train/box_loss": "0.9", "metrics/mAP50(B)": "0.2"},
        ],
    )

    writer = _StubWriter()
    tb = YOLORunTensorBoardLogger(run_dir=run_dir, writer=writer)

    logged, epoch = tb.poll_and_log_metrics()
    assert logged is True
    assert epoch == 1

    tags = [t for (t, _v, _s) in writer.scalars]
    assert "ultralytics/train/box_loss" in tags
    assert "ultralytics/metrics/mAP50_B" in tags
    assert "annolid/progress/epoch" in tags

    # Calling again without changing results should not duplicate logging.
    logged2, epoch2 = tb.poll_and_log_metrics()
    assert logged2 is False
    assert epoch2 == 1


def test_tb_logger_finalize_writes_text_summaries(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "weights").mkdir()
    (run_dir / "weights" / "best.pt").write_bytes(b"")

    writer = _StubWriter()
    tb = YOLORunTensorBoardLogger(run_dir=run_dir, writer=writer)
    tb.record_output_line("hello")
    tb.finalize(ok=True)

    text_tags = [t for (t, _txt, _s) in writer.texts]
    assert "annolid/train_log_tail" in text_tags
    assert "annolid/run_tree" in text_tags
    assert "annolid/status" in text_tags


def test_tb_logger_polls_images(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        # If these deps are unavailable in some minimal environments, skip.
        return

    img = np.zeros((64, 96, 3), dtype=np.uint8)
    img[:, :, 2] = 255
    out_path = run_dir / "results.png"
    assert cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    writer = _StubWriter()
    tb = YOLORunTensorBoardLogger(run_dir=run_dir, writer=writer)
    written = tb.poll_and_log_images(step=1)
    assert written >= 1
    assert any(tag.startswith("annolid/images")
               for (tag, _arr, _step, _fmt) in writer.images)
