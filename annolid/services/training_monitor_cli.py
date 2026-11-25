"""Headless CLI for discovering, inspecting, diffing, and diagnosing training runs.

Wraps :mod:`annolid.services.training_monitor` with zero Qt or heavy ML top-level
imports so it works cleanly in lean installs, remote SSH sessions, and CI jobs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from annolid.services.training_monitor import (
    DiscoveredRun,
    analyze_run_health,
    compute_hparam_diff,
    discover_training_runs,
    inspect_run_dir,
    tail_log_text,
)


def _serialize_run(
    run: DiscoveredRun, *, include_hparams: bool = False
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run_dir": str(run.run_dir),
        "display_name": run.display_name,
        "alias": run.metadata.alias,
        "tags": list(run.metadata.tags),
        "starred": bool(run.metadata.starred),
        "task": run.task,
        "model": run.model,
        "status": run.status,
        "step_unit": run.step_unit,
        "last_step": run.last_step,
        "total_steps": run.total_steps,
        "train_loss": run.train_loss,
        "val_loss": run.val_loss,
        "min_train_loss": run.min_train_loss,
        "min_val_loss": run.min_val_loss,
        "primary_metric_name": run.primary_metric_name,
        "primary_metric_value": run.primary_metric_value,
        "best_metric_value": run.best_metric_value,
        "best_metric_step": run.best_metric_step,
        "results_csv": str(run.results_csv) if run.results_csv else None,
        "tensorboard_dir": str(run.tensorboard_dir) if run.tensorboard_dir else None,
        "log_path": str(run.log_path) if run.log_path else None,
        "config_path": str(run.config_path) if run.config_path else None,
        "checkpoints": [str(p) for p in run.checkpoints],
        "health_findings": [
            {
                "severity": f.severity,
                "code": f.code,
                "message": f.message,
            }
            for f in run.health_findings
        ],
    }
    if include_hparams:
        payload["hparams"] = dict(run.hparams)
    return payload


def _fmt_metric(val: Optional[float]) -> str:
    if val is None:
        return "—"
    return f"{val:.4f}"


def _cmd_list(args: argparse.Namespace) -> int:
    runs_root = Path(args.runs_root).expanduser().resolve()
    if not runs_root.exists():
        if args.json:
            print(json.dumps({"runs_root": str(runs_root), "runs": []}, indent=2))
            return 0
        print(f"No runs directory found at {runs_root}")
        return 0

    runs = discover_training_runs(
        runs_root,
        max_runs=max(1, int(args.max_runs)),
        max_depth=max(1, int(args.max_depth)),
        include_tb_scalars=bool(args.include_tb),
    )
    if args.json:
        print(
            json.dumps(
                {
                    "runs_root": str(runs_root),
                    "count": len(runs),
                    "runs": [_serialize_run(r) for r in runs],
                },
                indent=2,
            )
        )
        return 0

    if not runs:
        print(f"No training runs discovered under {runs_root}")
        return 0

    print(f"Discovered {len(runs)} training run(s) under {runs_root}:")
    for run in runs:
        step_text = (
            f"{run.last_step}/{run.total_steps} {run.step_unit}s"
            if run.last_step is not None and run.total_steps is not None
            else (
                f"{run.last_step} {run.step_unit}s"
                if run.last_step is not None
                else "n/a"
            )
        )
        metric_part = (
            f"{run.primary_metric_name.split('/')[-1]}={_fmt_metric(run.primary_metric_value)}"
            if run.primary_metric_name
            else "metric=—"
        )
        crit = sum(1 for f in run.health_findings if f.severity == "critical")
        warn = sum(1 for f in run.health_findings if f.severity == "warning")
        health_tag = (
            f" [CRITICAL:{crit}]" if crit else (f" [WARN:{warn}]" if warn else " [OK]")
        )
        print(
            f"  - {run.display_name} ({run.task}/{run.model}) | "
            f"step={step_text} | train_loss={_fmt_metric(run.train_loss)} | "
            f"{metric_part}{health_tag}"
        )
        print(f"    path: {run.run_dir}")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        print(f"Run directory does not exist: {run_dir}", file=sys.stderr)
        return 2

    run = inspect_run_dir(
        run_dir,
        include_tb_scalars=bool(args.include_tb),
    )
    tail_text = ""
    if int(args.tail) > 0 and run.log_path is not None and run.log_path.is_file():
        tail_text = tail_log_text(run.log_path, max_lines=int(args.tail))

    if args.json:
        payload = _serialize_run(run, include_hparams=True)
        if int(args.tail) > 0:
            payload["log_tail"] = tail_text
        print(json.dumps(payload, indent=2))
        return 0

    print(f"Run: {run.display_name}")
    print(f"  Path:         {run.run_dir}")
    print(f"  Task / Model: {run.task} / {run.model}")
    print(
        f"  Steps:        {run.last_step or '—'} / {run.total_steps or '—'} ({run.step_unit})"
    )
    print(
        f"  Train Loss:   {_fmt_metric(run.train_loss)} "
        f"(min={_fmt_metric(run.min_train_loss)})"
    )
    print(
        f"  Val Loss:     {_fmt_metric(run.val_loss)} "
        f"(min={_fmt_metric(run.min_val_loss)})"
    )
    if run.primary_metric_name:
        best_note = (
            f" (best={_fmt_metric(run.best_metric_value)} @ {run.step_unit} {run.best_metric_step})"
            if run.best_metric_value is not None
            else ""
        )
        print(
            f"  Metric:       {run.primary_metric_name}="
            f"{_fmt_metric(run.primary_metric_value)}{best_note}"
        )
    print(f"  Checkpoints:  {len(run.checkpoints)}")
    for ckpt in run.checkpoints[:5]:
        print(f"    - {ckpt}")
    if run.health_findings:
        print("  Health:")
        for finding in run.health_findings:
            print(f"    [{finding.severity.upper()}] {finding.code}: {finding.message}")
    else:
        print("  Health:       OK (no anomalies detected)")
    if tail_text:
        print(f"\n--- Log tail ({run.log_path}) ---")
        print(tail_text)
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    run_paths = [Path(p).expanduser().resolve() for p in args.run_dirs]
    missing = [str(p) for p in run_paths if not p.exists()]
    if missing:
        print(f"Missing run directory(s): {', '.join(missing)}", file=sys.stderr)
        return 2

    runs = [inspect_run_dir(p) for p in run_paths]
    rows = compute_hparam_diff(runs, only_diffs=bool(args.only_diffs))

    if args.json:
        print(
            json.dumps(
                {
                    "runs": [
                        {"display_name": r.display_name, "run_dir": str(r.run_dir)}
                        for r in runs
                    ],
                    "only_diffs": bool(args.only_diffs),
                    "rows": [
                        {
                            "key": row.key,
                            "values": list(row.values),
                            "differs": bool(row.differs),
                        }
                        for row in rows
                    ],
                },
                indent=2,
            )
        )
        return 0

    if not rows:
        print(
            "No hyperparameter differences found."
            if args.only_diffs
            else "No hyperparameters found."
        )
        return 0

    headers = ["key"] + [r.display_name for r in runs]
    col_widths = [len(h) for h in headers]
    for row in rows:
        col_widths[0] = max(col_widths[0], len(row.key))
        for i, val in enumerate(row.values, start=1):
            col_widths[i] = max(col_widths[i], len(val))

    def _fmt_row(cols: Sequence[str], marker: str = " ") -> str:
        padded = [c.ljust(col_widths[i]) for i, c in enumerate(cols)]
        return f"{marker} " + " | ".join(padded)

    print(_fmt_row(headers, " "))
    print("  " + "-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(_fmt_row([row.key, *row.values], "*" if row.differs else " "))
    return 0


def _cmd_health(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        print(f"Run directory does not exist: {run_dir}", file=sys.stderr)
        return 2

    run = inspect_run_dir(run_dir, include_tb_scalars=bool(args.include_tb))
    disk_free_gb: Optional[float] = None
    if bool(args.check_disk):
        try:
            usage = shutil.disk_usage(run.run_dir)
            disk_free_gb = float(usage.free) / float(1024**3)
        except OSError:
            disk_free_gb = None

    findings = analyze_run_health(
        run,
        disk_free_gb=disk_free_gb,
        spike_factor=float(args.spike_factor),
        overfit_ratio=float(args.overfit_ratio),
        plateau_steps=int(args.plateau_steps),
        low_disk_gb=float(args.low_disk_gb),
    )
    has_critical = any(f.severity == "critical" for f in findings)

    if args.json:
        print(
            json.dumps(
                {
                    "run_dir": str(run.run_dir),
                    "display_name": run.display_name,
                    "ok": not has_critical,
                    "findings": [
                        {
                            "severity": f.severity,
                            "code": f.code,
                            "message": f.message,
                        }
                        for f in findings
                    ],
                },
                indent=2,
            )
        )
    elif findings:
        print(f"Health findings for {run.display_name} ({run.run_dir}):")
        for f in findings:
            print(f"  [{f.severity.upper()}] {f.code}: {f.message}")
    else:
        print(
            f"Health OK for {run.display_name} ({run.run_dir}): no anomalies detected."
        )

    if bool(args.fail_on_critical) and has_critical:
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="annolid-run runs",
        description="Discover, inspect, compare, and diagnose Annolid training runs.",
    )
    sub = parser.add_subparsers(dest="runs_cmd", required=True)

    list_p = sub.add_parser("list", help="Discover training runs under a root folder.")
    list_p.add_argument(
        "--runs-root",
        default="runs",
        help="Root directory containing training runs (default: runs).",
    )
    list_p.add_argument(
        "--max-runs", type=int, default=50, help="Maximum runs to return."
    )
    list_p.add_argument(
        "--max-depth", type=int, default=6, help="Maximum directory traversal depth."
    )
    list_p.add_argument(
        "--include-tb",
        action="store_true",
        help="Also read TensorBoard scalar event files when available.",
    )
    list_p.add_argument("--json", action="store_true", help="Print JSON output.")
    list_p.set_defaults(_handler=_cmd_list)

    inspect_p = sub.add_parser(
        "inspect", help="Inspect metrics and artifacts of a run."
    )
    inspect_p.add_argument("run_dir", help="Path to the training run directory.")
    inspect_p.add_argument(
        "--include-tb",
        action="store_true",
        help="Also read TensorBoard scalar event files when available.",
    )
    inspect_p.add_argument(
        "--tail",
        type=int,
        default=0,
        help="Include the last N lines of train.log if present.",
    )
    inspect_p.add_argument("--json", action="store_true", help="Print JSON output.")
    inspect_p.set_defaults(_handler=_cmd_inspect)

    diff_p = sub.add_parser(
        "diff", help="Diff hyperparameters across two or more runs."
    )
    diff_p.add_argument(
        "run_dirs",
        nargs="+",
        help="Two or more training run directories to compare.",
    )
    diff_p.add_argument(
        "--only-diffs",
        action="store_true",
        help="Show only hyperparameters that differ across runs.",
    )
    diff_p.add_argument("--json", action="store_true", help="Print JSON output.")
    diff_p.set_defaults(_handler=_cmd_diff)

    health_p = sub.add_parser(
        "health", help="Run automated health anomaly checks on a training run."
    )
    health_p.add_argument("run_dir", help="Path to the training run directory.")
    health_p.add_argument(
        "--include-tb",
        action="store_true",
        help="Also read TensorBoard scalar event files when available.",
    )
    health_p.add_argument(
        "--spike-factor",
        type=float,
        default=4.0,
        help="Loss spike threshold multiplier vs recent median.",
    )
    health_p.add_argument(
        "--overfit-ratio",
        type=float,
        default=1.15,
        help="Val-loss rise ratio over best step to flag overfitting.",
    )
    health_p.add_argument(
        "--plateau-steps",
        type=int,
        default=15,
        help="Stale steps after peak primary metric before flagging plateau.",
    )
    health_p.add_argument(
        "--check-disk",
        action="store_true",
        help="Also check free disk space on the run directory's volume.",
    )
    health_p.add_argument(
        "--low-disk-gb",
        type=float,
        default=2.0,
        help="Minimum free disk space in GB when --check-disk is set.",
    )
    health_p.add_argument(
        "--fail-on-critical",
        action="store_true",
        help="Exit with code 2 when any critical anomaly is detected.",
    )
    health_p.add_argument("--json", action="store_true", help="Print JSON output.")
    health_p.set_defaults(_handler=_cmd_health)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    return int(args._handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
