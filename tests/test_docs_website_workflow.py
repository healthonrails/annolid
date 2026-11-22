"""Deterministic tests for the docs/website workflow, showcase helpers, and site contracts."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(script_name: str):
    path = REPO_ROOT / "scripts" / script_name
    mod_name = script_name.replace(".py", "")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def test_site_workflow_check_passes_on_repository():
    site_workflow = _load_script_module("site_workflow.py")
    res = site_workflow.check_site(repo_root=REPO_ROOT, skip_mkdocs=False)
    assert res.ok, "site_workflow.check_site failed:\n" + "\n".join(res.errors)
    assert res.stats["website_refs_checked"] >= 10
    assert res.stats["portal_links_verified"] >= 10
    assert res.stats["docs_images_checked"] >= 20


def test_site_workflow_stage_produces_combined_layout(tmp_path: Path):
    site_workflow = _load_script_module("site_workflow.py")
    staged = site_workflow.stage_combined_site(
        tmp_path / "preview", repo_root=REPO_ROOT, strict=True
    )

    assert (staged / "index.html").is_file()
    assert site_workflow.REQUIRED_LANDING_TITLE in (staged / "index.html").read_text(
        encoding="utf-8"
    )
    assert (staged / "website-assets" / "css" / "main.css").is_file()
    assert (staged / "website-assets" / "js" / "main.js").is_file()
    assert (
        staged / "website-assets" / "images" / "showcase" / "mouse083_tracking_demo.gif"
    ).is_file()
    assert (staged / "portal" / "index.html").is_file()
    assert (
        staged
        / "portal"
        / "tutorials"
        / "mouse083_resident_intruder_showcase"
        / "index.html"
    ).is_file()
    assert (staged / "portal" / "assets" / "reports" / "mouse083_report.html").is_file()
    assert (staged / "workflows" / "index.html").is_file()


def test_machine_path_hygiene_allows_placeholders_and_flags_real_users():
    site_workflow = _load_script_module("site_workflow.py")
    assert (
        site_workflow._find_non_placeholder_machine_paths("/Users/you/video.mp4") == []
    )
    assert (
        site_workflow._find_non_placeholder_machine_paths(
            "/Users/name/Desktop/demo.mp4"
        )
        == []
    )
    assert (
        site_workflow._find_non_placeholder_machine_paths("/home/user/data.csv") == []
    )

    flagged = site_workflow._find_non_placeholder_machine_paths(
        "Source: /Users/jdoe_lab/Downloads/session.mp4"
    )
    assert flagged == ["/Users/jdoe_lab/"]


def test_site_check_catches_missing_title_and_broken_website_asset(tmp_path: Path):
    site_workflow = _load_script_module("site_workflow.py")
    (tmp_path / "website" / "assets").mkdir(parents=True)
    (tmp_path / "docs").mkdir(parents=True)
    (tmp_path / "website" / "index.html").write_text(
        '<html><title>Wrong</title><img src="website-assets/missing.png" /></html>',
        encoding="utf-8",
    )
    (tmp_path / "docs" / "index.md").write_text("# Docs\n", encoding="utf-8")
    (tmp_path / "docs" / "requirements_mkdocs.txt").write_text(
        "mkdocs\n", encoding="utf-8"
    )
    (tmp_path / "mkdocs.yml").write_text("site_name: Test\n", encoding="utf-8")

    res = site_workflow.check_site(repo_root=tmp_path, skip_mkdocs=True)
    assert not res.ok
    joined = "\n".join(res.errors)
    assert site_workflow.REQUIRED_LANDING_TITLE in joined
    assert "missing.png" in joined


def test_showcase_helpers_sanitize_and_render(tmp_path: Path):
    showcase = _load_script_module("generate_docs_showcase.py")

    raw_html = "<p>Source: /Users/example_dev/Downloads/annolid_sfn_videos/mouse083_tracked.csv</p>"
    cleaned = showcase.sanitize_html_paths(
        raw_html, ["/Users/example_dev/Downloads/annolid_sfn_videos"]
    )
    assert "/Users/" not in cleaned
    assert "mouse083_tracked.csv" in cleaned

    events_csv = showcase.write_canonical_events_csv(
        [("investigation", 0, 29, 30), ("mount", 30, 59, 30)],
        tmp_path / "events.csv",
        fps=30.0,
    )
    with events_csv.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
    assert rows[0]["Behavior"] == "investigation"
    assert rows[0]["Event"] == "start"
    assert rows[1]["Event"] == "stop"

    zones_json = showcase.write_demo_zones_json(
        tmp_path / "zones.json", width=1024, height=570
    )
    zdata = json.loads(zones_json.read_text(encoding="utf-8"))
    assert len(zdata["shapes"]) == 3
    assert all(s["flags"]["semantic_type"] == "zone" for s in zdata["shapes"])

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame_data = {
        "flags": {"investigation": True},
        "shapes": [
            {
                "label": "resident",
                "shape_type": "polygon",
                "points": [[40, 40], [110, 40], [110, 100], [40, 100]],
            },
            {"label": "resident_nose", "shape_type": "point", "points": [[108, 70]]},
            {"label": "resident_neck", "shape_type": "point", "points": [[80, 70]]},
        ],
    }
    rendered = showcase.draw_annotated_frame(
        frame,
        frame_data,
        frame_idx=42,
        fps=30.0,
        trails={"resident": [(60.0, 60.0), (75.0, 70.0)]},
        pair_distance_px=85.0,
        alert_text="QC FLAG: sample alert",
    )
    assert rendered.shape == frame.shape
    assert rendered.sum() > 0
