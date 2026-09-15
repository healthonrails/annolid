import numpy as np
import pandas as pd
import torch

from annolid.behavior.models import polygon_frame_classifier as p


def fixture_csv(path, gaps=False):
    rows = []
    for video in range(4):
        for i in range(8):
            rows.append(
                dict(
                    video=f"v{video}",
                    frame=f"f_{i + (10 if gaps and i > 3 else 0)}.json",
                    label="a" if i < (2 if video == 1 else 5) else "b",
                    intruder_features=f"[{video * 100 + i}, 2]",
                    resident_features="[3,4]",
                )
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_windows_do_not_cross_missing_frames(tmp_path):
    d = p.PolygonFrameDataset(
        fixture_csv(tmp_path / "data.csv", True),
        p.PolygonFeatureConfig(
            normalize_features=False, compute_dynamic_features=False
        ),
        3,
    )
    assert d._window_slice("v0", 3)[:, 0].tolist() == [2, 3, 3]
    assert d._window_slice("v0", 4)[:, 0].tolist() == [4, 4, 5]


def test_training_only_preprocessing_and_immutable_best(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(1)
    csv = fixture_csv(tmp_path / "data.csv")
    scores = iter([0.9, 0.1])
    monkeypatch.setattr(p, "_mean_average_precision", lambda *a, **k: next(scores))
    monkeypatch.setattr(p, "_plot_training_curves", lambda *a: None)
    f = p.PolygonFeatureConfig(polygon_pad_len=2, compute_dynamic_features=False)
    state = p.train_polygon_frame_classifier(
        csv,
        f,
        p.ModelConfig(
            window_size=3,
            hidden_dim=4,
            kernel_size=3,
            num_residual_blocks=1,
            use_attention=False,
        ),
        p.TrainingConfig(
            batch_size=8,
            num_epochs=2,
            num_workers=0,
            val_split_ratio=0.25,
            sampling_strategy="random",
        ),
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path,
    )
    raw = p.PolygonFrameDataset(
        csv,
        p.PolygonFeatureConfig(
            polygon_pad_len=2, normalize_features=False, compute_dynamic_features=False
        ),
        3,
    )
    x = np.stack([raw.video_features[v][i] for v, i in raw.indices])
    y = np.array([raw.video_labels[v][i] for v, i in raw.indices])
    idx = state["train_indices"]
    np.testing.assert_allclose(
        state["normalization"]["mean"], x[idx].mean(0), rtol=1e-5
    )
    weights = 1 / np.bincount(y[idx], minlength=2)
    np.testing.assert_allclose(
        state["class_weights"], weights / weights.mean(), rtol=1e-6
    )
    assert not set(state["train_videos"]) & set(state["val_videos"])
    saved = torch.load(
        tmp_path / "polygon_frame_classifier_best.pt", weights_only=False
    )
    assert state["best_epoch"] == 1
    assert all(
        torch.equal(v, saved["model_state"][k]) for k, v in state["model_state"].items()
    )
    assert any(
        not torch.equal(v, state["latest_model_state"][k])
        for k, v in state["model_state"].items()
    )
