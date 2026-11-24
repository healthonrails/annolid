from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from annolid.colab.models import ColabJobRecord
from annolid.core.agent.tools.sim2real import Sim2RealTool
from annolid.sim2real import (
    Sim2RealWorkflowService,
    SimDatasetConfig,
    SimIterationEvaluation,
    VideoEnvironmentProfile,
    evaluate_iteration_outputs,
    suggest_next_iteration,
)
from annolid.sim2real.cli import main as sim2real_cli_main


def _make_example_video(path: Path, n_frames: int = 10) -> Path:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240)
    )
    try:
        for i in range(n_frames):
            frame = np.full((240, 320, 3), 170, dtype=np.uint8)
            cv2.circle(frame, (80 + i * 8, 110), 18, (30, 30, 30), -1)
            cv2.circle(frame, (220 - i * 6, 130), 18, (235, 235, 235), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


def _write_labelme_preds(
    pred_dir: Path,
    *,
    n_frames: int,
    detect_every: int = 1,
    n_boxes: int = 2,
    score: float = 0.78,
) -> Path:
    pred_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        shapes: list[dict[str, Any]] = []
        if i % detect_every == 0:
            for b in range(n_boxes):
                x0 = 60 + b * 110 + i * 3
                y0 = 90 + b * 10
                shapes.append(
                    {
                        "label": "mouse",
                        "score": score,
                        "points": [
                            [x0, y0],
                            [x0 + 40, y0],
                            [x0 + 40, y0 + 35],
                            [x0, y0 + 35],
                        ],
                        "group_id": b + 1,
                        "shape_type": "polygon",
                        "flags": {},
                    }
                )
        (pred_dir / f"frame_{i:04d}.json").write_text(
            json.dumps(
                {
                    "version": "5.0.1",
                    "flags": {},
                    "shapes": shapes,
                    "imagePath": f"frame_{i:04d}.png",
                    "imageData": None,
                    "imageHeight": 240,
                    "imageWidth": 320,
                }
            ),
            encoding="utf-8",
        )
    return pred_dir


class _FakeColabService:
    """Deterministic fake ColabWorkflowService simulating improving metrics across iterations."""

    def __init__(self) -> None:
        self.train_requests: list[Any] = []
        self.predict_requests: list[Any] = []
        self.stopped_sessions: list[str] = []

    def run_train(self, req: Any, **kwargs: Any) -> ColabJobRecord:
        self.train_requests.append(req)
        idx = len(self.train_requests)
        out_dir = Path(req.output_dir)
        weights_dir = out_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        best_pt = weights_dir / "best.pt"
        best_pt.write_bytes(f"fake-weights-iter-{idx}".encode("utf-8"))

        # Write realistic Ultralytics results.csv improving across iterations
        map_val = 0.62 if idx == 1 else 0.86
        (out_dir / "results.csv").write_text(
            "epoch,train/box_loss,train/seg_loss,metrics/mAP50(B),metrics/mAP50(M),val/box_loss\n"
            f"1,1.8,1.5,{map_val - 0.15:.3f},{map_val - 0.18:.3f},1.4\n"
            f"2,1.1,0.9,{map_val:.3f},{map_val:.3f},0.8\n",
            encoding="utf-8",
        )
        return ColabJobRecord(
            job_id=f"train_fake_{idx}",
            kind="train",
            model=req.model,
            backend="cli",
            accelerator=req.accelerator,
            status="completed",
            session_name=req.session_name or f"sess_{idx}",
            local_output_dir=str(out_dir),
            downloaded_files=[str(best_pt), str(out_dir / "results.csv")],
        )

    def run_predict(self, req: Any, **kwargs: Any) -> ColabJobRecord:
        self.predict_requests.append(req)
        idx = len(self.predict_requests)
        out_dir = Path(req.output_dir)
        # Round 1: noisy / lower coverage; Round 2: clean high coverage
        if idx == 1:
            _write_labelme_preds(
                out_dir, n_frames=8, detect_every=2, n_boxes=1, score=0.48
            )
        else:
            _write_labelme_preds(
                out_dir, n_frames=8, detect_every=1, n_boxes=2, score=0.89
            )
        return ColabJobRecord(
            job_id=f"pred_fake_{idx}",
            kind="predict",
            model=req.model,
            backend="cli",
            accelerator=req.accelerator,
            status="completed",
            session_name=req.session_name or f"sess_{idx}",
            local_output_dir=str(out_dir),
        )

    def stop_session(self, session: str, *, backend: str = "auto") -> None:
        self.stopped_sessions.append(session)


def test_evaluator_supervised_and_advisor_recommendations(tmp_path: Path) -> None:
    vid = _make_example_video(tmp_path / "mouse_eval.mp4", n_frames=8)
    pred_dir = _write_labelme_preds(
        tmp_path / "preds", n_frames=8, detect_every=2, n_boxes=1, score=0.45
    )
    gt_dir = _write_labelme_preds(
        tmp_path / "gt", n_frames=8, detect_every=1, n_boxes=2, score=1.0
    )

    profile = VideoEnvironmentProfile(
        example_videos=[str(vid)],
        species="mouse",
        estimated_subject_count=2,
        reference_background_path=str(tmp_path / "bg.png"),
    )
    cv2.imwrite(
        profile.reference_background_path, np.full((240, 320, 3), 170, dtype=np.uint8)
    )

    ev = evaluate_iteration_outputs(
        example_videos=[vid],
        prediction_dirs=[pred_dir],
        profile=profile,
        ground_truth_labelme_dir=gt_dir,
        overlay_output_path=tmp_path / "overlay.png",
    )
    assert ev.evaluated_frames == 8
    assert ev.detection_coverage == pytest.approx(0.5, abs=0.01)
    assert ev.supervised_f1 is not None
    assert 0.0 < ev.supervised_f1 < 1.0
    assert Path(ev.overlay_contact_sheet_path).is_file()

    cfg = SimDatasetConfig(
        species="mouse",
        scenarios=["arena_explore"],
        frames_per_scenario=16,
        photometry_strength=0.6,
        composite_on_real_background=False,
    )
    adj = suggest_next_iteration(
        current_config=cfg,
        evaluation=ev,
        profile=profile,
        current_best_weights="/tmp/best.pt",
        current_epochs=15,
        current_imgsz=640,
    )
    assert adj.should_continue is True
    assert adj.suggested_sim_config["composite_on_real_background"] is True
    assert adj.suggested_sim_config["photometry_strength"] > cfg.photometry_strength
    assert adj.suggested_sim_config["frames_per_scenario"] > cfg.frames_per_scenario
    assert "social_huddle" in adj.suggested_sim_config["scenarios"]
    assert adj.suggested_train_overrides["weights"] == "/tmp/best.pt"
    assert adj.suggested_train_overrides["imgsz"] == 768

    # High-score convergence check
    high_ev = SimIterationEvaluation(
        evaluated_frames=10,
        frames_with_detections=10,
        detection_coverage=0.95,
        mean_confidence=0.91,
        sim2real_score=0.92,
    )
    stop_adj = suggest_next_iteration(
        current_config=cfg, evaluation=high_ev, target_score=0.88
    )
    assert stop_adj.should_continue is False
    assert stop_adj.stop_reason == "target_score_reached"


def test_closed_loop_two_iterations_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import annolid.sim2real.generators as gen_mod

    monkeypatch.setattr(gen_mod.importlib.util, "find_spec", lambda _name: None)

    vid = _make_example_video(tmp_path / "mouse_loop.mp4", n_frames=10)
    fake_colab = _FakeColabService()
    proj_dir = tmp_path / "sim2real_proj"
    svc = Sim2RealWorkflowService(proj_dir, colab_service=fake_colab)

    profile, sim_cfg = svc.analyze_videos(
        [vid], species_hint="mouse", frames_per_scenario=4
    )
    assert profile.species == "mouse"
    sim_cfg.allow_fallback = True
    svc.update_sim_config(sim_cfg)

    records = svc.run_loop(
        iterations=2,
        auto_apply_recommendations=True,
        train_backend="colab",
        colab_backend="cli",
        accelerator="T4",
        epochs=10,
        imgsz=640,
        keep_runtime=False,
    )
    assert len(records) == 2
    assert records[0].status == "completed"
    assert records[1].status == "completed"
    # Iteration 2 must warm-start from Iteration 1's best.pt and improve score!
    assert fake_colab.train_requests[1].weights == records[0].best_weights_path
    assert records[1].score_delta_vs_prev is not None
    assert records[1].score_delta_vs_prev > 0.1
    # Shared warm Colab session used across both rounds and cleaned up at the end
    assert len(set(r.session_name for r in fake_colab.train_requests)) == 1
    assert len(fake_colab.stopped_sessions) == 1

    state = svc.load_state()
    assert state.best_iteration == 2
    assert Path(state.best_weights_path).is_file()
    assert Path(state.report_paths["markdown"]).is_file()
    assert Path(state.report_paths["json"]).is_file()
    assert Path(state.report_paths["html"]).is_file()
    md_text = Path(state.report_paths["markdown"]).read_text(encoding="utf-8")
    assert "Iteration Progression Summary" in md_text
    assert "Sim2Real Score" in md_text
    html_text = Path(state.report_paths["html"]).read_text(encoding="utf-8")
    assert "<svg" in html_text
    assert "data:image/png;base64," in html_text


def test_cli_and_agent_tool_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import annolid.sim2real.generators as gen_mod

    monkeypatch.setattr(gen_mod.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(gen_mod, "_find_flybody_python", lambda: None)

    vid = _make_example_video(tmp_path / "fly_example.mp4", n_frames=8)
    proj = tmp_path / "cli_proj"

    assert (
        sim2real_cli_main(
            [
                "--json",
                "analyze",
                "--video",
                str(vid),
                "--project",
                str(proj),
                "--species",
                "fly",
                "--frames-per-scenario",
                "4",
            ]
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    assert out["profile"]["species"] == "fly"

    assert (
        sim2real_cli_main(
            [
                "--json",
                "generate",
                "--project",
                str(proj),
                "--frames-per-scenario",
                "4",
                "--allow-fallback",
            ]
        )
        == 0
    )
    gen_out = json.loads(capsys.readouterr().out)
    assert Path(gen_out["data_yaml_path"]).is_file()

    assert sim2real_cli_main(["--json", "report", "--project", str(proj)]) == 0
    rep_out = json.loads(capsys.readouterr().out)
    assert Path(rep_out["markdown"]).is_file()

    # Agent tool check
    tool = Sim2RealTool(allowed_dir=tmp_path)
    res_str = asyncio.run(tool.execute(action="status", project_dir=str(proj)))
    parsed = json.loads(res_str)
    assert parsed["profile"]["species"] == "fly"
