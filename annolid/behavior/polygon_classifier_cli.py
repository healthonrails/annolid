"""CLI helpers for polygon behavior classifier training."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from annolid.behavior.polygon_classifier_workflow import (
    generate_polygon_train_test_csvs,
    train_polygon_classifier,
)
from annolid.utils.runs import shared_runs_root


_TCN_DEFAULTS = {
    "num_epochs": 500,
    "learning_rate": 1e-4,
    "batch_size": 8,
    "window_size": 1000,
    "hidden_dim": 32,
    "num_residual_blocks": 2,
    "kernel_size": 9,
    "dropout": 0.1,
}
_CONVNET_DEFAULTS = {
    "num_epochs": 30,
    "learning_rate": 4e-3,
    "batch_size": 64,
    "window_size": 11,
    "hidden_dim": 128,
    "num_residual_blocks": 6,
    "kernel_size": 9,
    "dropout": 0.3,
}


def _add_optional_default(
    parser: argparse.ArgumentParser,
    *flags: str,
    option_type: type,
    help_text: str,
) -> None:
    parser.add_argument(
        *flags,
        dest=flags[-1].lstrip("-").replace("-", "_"),
        type=option_type,
        default=None,
        help=help_text,
    )


def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--train-video",
        metavar=("ANNOTATION_DIR", "LABEL_CSV"),
        nargs=2,
        action="append",
        default=[],
        help="Training video annotation folder and manual behavior label CSV.",
    )
    parser.add_argument(
        "--test-video",
        metavar=("ANNOTATION_DIR", "LABEL_CSV"),
        nargs=2,
        action="append",
        default=[],
        help="Test video annotation folder and manual behavior label CSV.",
    )
    parser.add_argument(
        "--train-csv", default=None, help="Existing training feature CSV."
    )
    parser.add_argument("--test-csv", default=None, help="Existing test feature CSV.")
    parser.add_argument(
        "--csv-output-dir",
        default=None,
        help="Where generated train/test polygon point CSVs are written.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(shared_runs_root() / "polygon_classifier" / "train"),
        help="Directory where training run folders are written.",
    )
    parser.add_argument(
        "--model-type",
        choices=("tcn", "convnet"),
        default="tcn",
        help="Classifier architecture. Defaults to TCN.",
    )
    parser.add_argument("--run-name", default="exp", help="Run directory prefix.")
    parser.add_argument(
        "--num-points",
        type=int,
        default=50,
        help="Number of resampled points per polygon when generating CSVs.",
    )
    parser.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="Include polygon frames without manual labels when generating CSVs.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, for example cpu, cuda, or mps. Defaults to auto.",
    )
    _add_optional_default(
        parser,
        "--num-epochs",
        option_type=int,
        help_text="Training epochs. Defaults depend on --model-type.",
    )
    _add_optional_default(
        parser,
        "--batch-size",
        option_type=int,
        help_text="Batch size. Defaults depend on --model-type.",
    )
    _add_optional_default(
        parser,
        "--learning-rate",
        option_type=float,
        help_text="Learning rate. Defaults depend on --model-type.",
    )
    _add_optional_default(
        parser,
        "--window-size",
        option_type=int,
        help_text="Temporal sequence length for TCN, or sliding window size for ConvNet. Defaults depend on --model-type.",
    )
    _add_optional_default(
        parser,
        "--hidden-dim",
        option_type=int,
        help_text="Hidden feature dimension. Defaults depend on --model-type.",
    )
    _add_optional_default(
        parser,
        "--num-residual-blocks",
        option_type=int,
        help_text="Number of residual blocks. Defaults depend on --model-type.",
    )
    _add_optional_default(
        parser,
        "--kernel-size",
        option_type=int,
        help_text="TCN convolution kernel size. Defaults to DAART n_lags=4, kernel size 9.",
    )
    _add_optional_default(
        parser,
        "--dropout",
        option_type=float,
        help_text="Dropout rate. Defaults depend on --model-type.",
    )


def _model_defaults(model_type: str) -> dict[str, int | float]:
    return dict(_TCN_DEFAULTS if model_type == "tcn" else _CONVNET_DEFAULTS)


def _resolve_training_csvs(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    has_csvs = bool(args.train_csv) or bool(args.test_csv)
    has_videos = bool(args.train_video) or bool(args.test_video)
    if has_csvs and has_videos:
        raise ValueError(
            "Use either --train-csv/--test-csv or --train-video/--test-video, not both."
        )
    if has_csvs:
        if not args.train_csv or not args.test_csv:
            raise ValueError("--train-csv and --test-csv must be provided together.")
        return (
            str(Path(args.train_csv).expanduser()),
            str(Path(args.test_csv).expanduser()),
            {},
        )
    if not args.train_video or not args.test_video:
        raise ValueError(
            "Provide training data with either --train-csv/--test-csv or repeated "
            "--train-video and --test-video pairs."
        )

    csv_output_dir = args.csv_output_dir
    if not csv_output_dir:
        csv_output_dir = Path(args.output_dir).expanduser() / "generated_csvs"
    dataset = generate_polygon_train_test_csvs(
        train_assignments=args.train_video,
        test_assignments=args.test_video,
        output_dir=csv_output_dir,
        num_points=int(args.num_points),
        include_unlabeled=bool(args.include_unlabeled),
    )
    return dataset.train.csv, dataset.test.csv, {"dataset": asdict(dataset)}


def run_train_command(args: argparse.Namespace) -> int:
    defaults = _model_defaults(args.model_type)
    params = {
        name: getattr(args, name) if getattr(args, name) is not None else value
        for name, value in defaults.items()
    }
    train_csv, test_csv, payload = _resolve_training_csvs(args)
    outcome = train_polygon_classifier(
        train_csv=train_csv,
        test_csv=test_csv,
        output_dir=args.output_dir,
        model_type=args.model_type,
        run_name=args.run_name,
        num_epochs=int(params["num_epochs"]),
        batch_size=int(params["batch_size"]),
        learning_rate=float(params["learning_rate"]),
        window_size=int(params["window_size"]),
        hidden_dim=int(params["hidden_dim"]),
        num_residual_blocks=int(params["num_residual_blocks"]),
        kernel_size=int(params["kernel_size"]),
        dropout=float(params["dropout"]),
        device=args.device,
    )
    payload.update(
        {
            "training": asdict(outcome),
            "parameters": {
                "model_type": args.model_type,
                **params,
                "device": args.device or "auto",
            },
        }
    )
    print(json.dumps(payload, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m annolid.behavior.polygon_classifier_cli",
        description=(
            "Train a polygon classifier. Prefer "
            "`annolid-run train polygon_classifier` for installed environments."
        ),
    )
    add_train_args(parser)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(run_train_command(args))
    except Exception as exc:
        parser.exit(2, f"polygon classifier: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
