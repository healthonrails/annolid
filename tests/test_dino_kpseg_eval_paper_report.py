from __future__ import annotations

from pathlib import Path

from annolid.segmentation.dino_kpseg.eval import (
    build_paper_report,
    write_paper_report_files,
)


def test_build_paper_report_includes_markdown_and_ci() -> None:
    summary = {
        "images_total": 10,
        "images_used": 10,
        "instances_total": 10,
        "keypoints_visible_total": 100,
        "mean_error_px": 2.25,
        "swap_rate": 0.1,
        "pck": {"4.0": 0.7, "8.0": 0.9},
        "pck_counts": {"4.0": 70, "8.0": 90},
    }
    report = build_paper_report(
        summary=summary,
        dataset_name="test_set",
        model_name="best",
        split="test",
    )
    assert report["metadata"]["split"] == "test"
    table = report["paper_table"]
    assert "PCK@4px" in table["markdown"]
    assert "95% CI" in table["markdown"]
    assert "pck@4px_ci95_low" in table["csv"]


def test_write_paper_report_files(tmp_path: Path) -> None:
    summary = {
        "images_total": 1,
        "images_used": 1,
        "instances_total": 1,
        "keypoints_visible_total": 2,
        "mean_error_px": 1.0,
        "pck": {"4.0": 1.0},
        "pck_counts": {"4.0": 2},
    }
    report = build_paper_report(
        summary=summary,
        dataset_name="set",
        model_name="model",
        split="val",
    )
    paths = write_paper_report_files(
        report=report,
        report_dir=tmp_path,
        base_name="metrics",
    )
    assert Path(paths["json"]).exists()
    assert Path(paths["markdown"]).exists()
    assert Path(paths["csv"]).exists()
    assert Path(paths["latex"]).exists()
