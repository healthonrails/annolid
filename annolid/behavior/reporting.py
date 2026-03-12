from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn import metrics

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - plotting is optional
    plt = None


def plot_behavior_eval_artifacts(
    *,
    probs: Sequence[Sequence[float]],
    targets: Sequence[int],
    preds: Sequence[int],
    label_names: Sequence[str],
    plot_dir: Path,
) -> dict[str, str]:
    if plt is None:
        return {}
    if not probs:
        return {}

    plot_dir = Path(plot_dir).expanduser().resolve()
    plot_dir.mkdir(parents=True, exist_ok=True)

    y_score = np.asarray(probs, dtype=float)
    y_true_idx = np.asarray(targets, dtype=int)
    y_pred_idx = np.asarray(preds, dtype=int)
    labels = [str(name) for name in label_names]
    num_classes = len(labels)
    if y_score.ndim != 2 or num_classes <= 0:
        return {}

    artifacts: dict[str, str] = {}

    cm = metrics.confusion_matrix(
        y_true_idx, y_pred_idx, labels=list(range(num_classes))
    )
    fig_cm, ax_cm = plt.subplots(1, 1, figsize=(6, 5))
    im = ax_cm.imshow(cm, interpolation="nearest", cmap="Blues")
    fig_cm.colorbar(im, ax=ax_cm)
    ax_cm.set_xticks(range(num_classes))
    ax_cm.set_yticks(range(num_classes))
    ax_cm.set_xticklabels(labels, rotation=45, ha="right")
    ax_cm.set_yticklabels(labels)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    for i in range(num_classes):
        for j in range(num_classes):
            ax_cm.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
    fig_cm.tight_layout()
    cm_path = plot_dir / "confusion_matrix.png"
    fig_cm.savefig(cm_path)
    plt.close(fig_cm)
    artifacts["confusion_matrix"] = str(cm_path)

    y_true = np.zeros((len(y_true_idx), num_classes), dtype=int)
    for i, target in enumerate(y_true_idx):
        if 0 <= int(target) < num_classes:
            y_true[i, int(target)] = 1

    fig_pr, ax_pr = plt.subplots(1, 1, figsize=(6, 5))
    wrote_curve = False
    for idx, name in enumerate(labels):
        if y_true[:, idx].sum() == 0:
            continue
        precision, recall, _ = metrics.precision_recall_curve(
            y_true[:, idx], y_score[:, idx]
        )
        ap = metrics.average_precision_score(y_true[:, idx], y_score[:, idx])
        ax_pr.step(
            recall,
            precision,
            where="post",
            label=f"{name} (AP={ap:.3f})",
        )
        wrote_curve = True
    if wrote_curve:
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_xlim([0.0, 1.0])
        ax_pr.set_ylim([0.0, 1.05])
        ax_pr.grid(True, alpha=0.3)
        ax_pr.legend(fontsize=8)
        fig_pr.tight_layout()
        pr_path = plot_dir / "pr_curves.png"
        fig_pr.savefig(pr_path)
        artifacts["pr_curves"] = str(pr_path)
    plt.close(fig_pr)
    return artifacts


__all__ = ["plot_behavior_eval_artifacts"]
