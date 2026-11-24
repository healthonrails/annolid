"""Opt-in live integration checks against real Google Colab / Google Drive endpoints.

Skipped by default so CI and offline runs stay deterministic.
Run manually once signed in via `annolid-run colab auth login`:

    ANNOLID_COLAB_LIVE_TESTS=1 pytest tests/test_colab_live_integration.py -v
"""

from __future__ import annotations

import os

import pytest

from annolid.colab.service import ColabWorkflowService

_LIVE_ENABLED = str(
    os.environ.get("ANNOLID_COLAB_LIVE_TESTS") or ""
).strip().lower() in {
    "1",
    "true",
    "yes",
}

pytestmark = pytest.mark.skipif(
    not _LIVE_ENABLED,
    reason="Opt-in live Colab/Drive test (set ANNOLID_COLAB_LIVE_TESTS=1 to run).",
)


def test_live_colab_doctor_probes():
    svc = ColabWorkflowService()
    report = svc.run_diagnostics(live=True)
    probes = report["live_probes"]
    any_succeeded = any(
        isinstance(info, dict) and info.get("ok") is True for info in probes.values()
    )
    assert any_succeeded, (
        "No live Colab/Drive probe succeeded. Sign in first with `annolid-run colab auth login`. "
        f"Details: {probes}"
    )
