from __future__ import annotations

import json
from pathlib import Path

import torch

from annolid.behavior.eval import evaluate_behavior_classifier, main


class _DummyDataset:
    def __init__(self) -> None:
        self.label_mapping = {"grooming": 0, "rearing": 1}
        self.indices = [0, 1]

    def get_num_classes(self) -> int:
        return 2

    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int):
        tensor = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
        label = 0 if idx == 0 else 1
        return tensor, label, f"video_{idx}.mpg"


class _DummyModel(torch.nn.Module):
    def forward(self, inputs):
        batch = int(inputs.shape[0])
        logits = torch.zeros((batch, 2), dtype=torch.float32, device=inputs.device)
        logits[:, 0] = 3.0
        logits[:, 1] = 1.0
        return logits


def test_evaluate_behavior_classifier_returns_metrics(monkeypatch) -> None:
    monkeypatch.setattr(
        "annolid.behavior.eval._build_dataset",
        lambda **kwargs: _DummyDataset(),
    )
    monkeypatch.setattr(
        "annolid.behavior.eval.load_classifier",
        lambda **kwargs: _DummyModel(),
    )

    payload = evaluate_behavior_classifier(
        video_folder="/tmp/videos",
        checkpoint_path="/tmp/best_model.pth",
        batch_size=2,
        split="all",
    )

    assert "test_metrics" in payload
    assert payload["test_metrics"]["accuracy"] >= 0.0
    assert "per_class" in payload["test_metrics"]
    assert len(payload["predictions"]) == 2
    assert "class_probabilities" in payload["predictions"][0]


def test_behavior_eval_main_writes_json(monkeypatch, tmp_path: Path) -> None:
    out = tmp_path / "metrics.json"
    monkeypatch.setattr(
        "annolid.behavior.eval.evaluate_behavior_classifier",
        lambda **kwargs: {"test_metrics": {"accuracy": 1.0}, "predictions": []},
    )
    rc = main(
        [
            "--video-folder",
            str(tmp_path),
            "--checkpoint-path",
            str(tmp_path / "best_model.pth"),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["test_metrics"]["accuracy"] == 1.0


def test_evaluate_behavior_classifier_writes_plot_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    plot_dir = tmp_path / "plots"
    monkeypatch.setattr(
        "annolid.behavior.eval._build_dataset",
        lambda **kwargs: _DummyDataset(),
    )
    monkeypatch.setattr(
        "annolid.behavior.eval.load_classifier",
        lambda **kwargs: _DummyModel(),
    )

    payload = evaluate_behavior_classifier(
        video_folder="/tmp/videos",
        checkpoint_path="/tmp/best_model.pth",
        batch_size=2,
        split="all",
        plot_dir=str(plot_dir),
    )

    artifacts = payload["artifacts"]
    assert "confusion_matrix" in artifacts
    assert Path(artifacts["confusion_matrix"]).exists()
