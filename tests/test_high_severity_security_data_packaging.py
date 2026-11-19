"""Focused regression tests for High-severity data integrity, security, and packaging items (D-1, D-2, S-1, S-2, S-4, P-1, P-2)."""

from __future__ import annotations

import asyncio
import io
import json
import pickle
import stat
import subprocess
import sys
import textwrap
import zipfile
from importlib import metadata
from pathlib import Path

import numpy as np
import pytest
import yaml

from annolid.annotation.batch_relabel import collect_label_counts, run_batch_relabel
from annolid.core.agent.tools import clawhub
from annolid.core.agent.tools.annolid_run import AnnolidRunTool
from annolid.gui import launcher as gui_launcher
from annolid.segmentation.maskrcnn.utils import _safe_pickle_loads
from annolid.utils.annotation_store import AnnotationStore, AnnotationStoreError
from annolid.utils.config import get_config


def test_d1_annotation_store_path_containment_lru_and_shim_opt_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame_path = tmp_path / "session" / "session_000000001.json"
    frame_path.parent.mkdir(parents=True, exist_ok=True)

    for unsafe_name in (
        "../../outside.ndjson",
        "/tmp/evil.ndjson",
        "sub/nested.ndjson",
        "..",
    ):
        with pytest.raises(AnnotationStoreError):
            AnnotationStore.for_frame_path(frame_path, store_name=unsafe_name)

    # Bounded LRU eviction on _CACHE
    monkeypatch.setattr(AnnotationStore, "_MAX_CACHE_ENTRIES", 4)
    AnnotationStore.invalidate_cache()
    paths = []
    for idx in range(6):
        sp = tmp_path / f"s_{idx}_annotations.ndjson"
        sp.write_text('{"frame":0,"shapes":[]}\n', encoding="utf-8")
        st = AnnotationStore(sp)
        assert st.get_frame(0) is not None
        paths.append(sp)
    assert len(AnnotationStore._CACHE) <= 4
    assert paths[0] not in AnnotationStore._CACHE
    assert paths[-1] in AnnotationStore._CACHE

    # Env opt-out skips stdlib json.load stub rewriting
    stub_file = frame_path.parent / "session_000000000.json"
    stub_file.write_text(
        json.dumps({"annotation_store": "session_annotations.ndjson", "frame": 0}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANNOLID_DISABLE_JSON_LOAD_SHIM", "1")
    with stub_file.open("r", encoding="utf-8") as fh:
        raw_stub = json.load(fh)
    assert raw_stub == {"annotation_store": "session_annotations.ndjson", "frame": 0}


def test_d2_batch_relabel_updates_body_parts_keypoints_and_skips_zone_files(
    tmp_path: Path,
) -> None:
    frame_json = tmp_path / "trial_000000000.json"
    frame_json.write_text(
        json.dumps(
            {
                "version": "5.0.0",
                "shapes": [
                    {
                        "label": "mouse_1",
                        "shape_type": "polygon",
                        "points": [[0, 0], [10, 0], [10, 10]],
                        "flags": {},
                    },
                    {
                        "label": "mouse_1:nose",
                        "shape_type": "polygon",
                        "parent_instance": "mouse_1",
                        "points": [[1, 1], [3, 1], [3, 3]],
                        "flags": {"parent_instance": "mouse_1"},
                    },
                    {
                        "label": "left_ear",
                        "shape_type": "point",
                        "instance_label": "mouse_1",
                        "points": [[2, 2]],
                        "flags": {"instance_label": "mouse_1"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    zones_json = tmp_path / "trial_zones.json"
    zones_json.write_text(
        json.dumps(
            {
                "shapes": [
                    {
                        "label": "mouse_1",
                        "shape_type": "polygon",
                        "points": [[0, 0], [50, 0], [50, 50]],
                        "flags": {"semantic_type": "zone"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    proj_json = tmp_path / "project.annolid.json"
    proj_json.write_text(
        json.dumps({"shapes": [{"label": "mouse_1"}]}), encoding="utf-8"
    )

    counts_before = collect_label_counts(root=tmp_path, include_annotation_stores=False)
    assert counts_before["mouse_1"] == 1  # excludes zones & project.annolid.json

    res = run_batch_relabel(
        root=tmp_path,
        old_label="mouse_1",
        new_label="resident",
        include_json_files=True,
        include_annotation_stores=False,
    )
    assert res.json_files_updated == 1
    assert res.shapes_renamed == 3

    updated = json.loads(frame_json.read_text(encoding="utf-8"))["shapes"]
    assert updated[0]["label"] == "resident"
    assert updated[1]["label"] == "resident:nose"
    assert updated[1]["parent_instance"] == "resident"
    assert updated[1]["flags"]["parent_instance"] == "resident"
    assert updated[2]["instance_label"] == "resident"
    assert updated[2]["flags"]["instance_label"] == "resident"

    # Zone and project metadata files remain untouched
    assert (
        json.loads(zones_json.read_text(encoding="utf-8"))["shapes"][0]["label"]
        == "mouse_1"
    )
    assert (
        json.loads(proj_json.read_text(encoding="utf-8"))["shapes"][0]["label"]
        == "mouse_1"
    )


def test_s1_safe_deserialization_ipc_yaml_and_torch_load_audit(tmp_path: Path) -> None:
    # 1. Restricted distributed IPC unpickler allows numpy/dict/list and blocks RCE globals
    safe_payload = pickle.dumps({"scores": np.array([0.9, 0.8]), "ids": [1, 2]})
    decoded = _safe_pickle_loads(safe_payload)
    assert decoded["ids"] == [1, 2]
    assert np.allclose(decoded["scores"], [0.9, 0.8])

    class _Exploit:
        def __reduce__(self):
            import os

            return (os.system, ("echo pwned",))

    with pytest.raises(pickle.UnpicklingError):
        _safe_pickle_loads(pickle.dumps(_Exploit()))

    # 2. YAML config loader rejects arbitrary python object tags
    bad_yaml = tmp_path / "evil.yaml"
    bad_yaml.write_text(
        "!!python/object/apply:os.system ['echo pwned']\n", encoding="utf-8"
    )
    with pytest.raises(yaml.YAMLError):
        get_config(str(bad_yaml))

    # 3. Verify audited first-party modules no longer call raw torch.load or yaml.FullLoader
    repo_root = Path(__file__).resolve().parents[1]
    audited = [
        "annolid/depth/run.py",
        "annolid/motion/deformation.py",
        "annolid/motion/main.py",
        "annolid/segmentation/dino_kpseg/data.py",
        "annolid/segmentation/dino_kpseg/eval.py",
        "annolid/segmentation/dino_kpseg/predictor.py",
        "annolid/segmentation/maskrcnn/train.py",
        "annolid/segmentation/maskrcnn/inference.py",
        "annolid/utils/config.py",
    ]
    for rel in audited:
        src = (repo_root / rel).read_text(encoding="utf-8")
        assert "torch.load(" not in src, f"Raw torch.load remains in {rel}"
        assert "FullLoader" not in src, f"yaml.FullLoader remains in {rel}"


def test_s2_annolid_run_tool_catches_system_exit_and_enforces_allowed_dir(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "workspace"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_csv = outside / "secret_tracking.csv"
    outside_csv.write_text(
        "frame_number,instance_name,cx,cy\n0,m1,1,1\n", encoding="utf-8"
    )

    tool = AnnolidRunTool(allowed_dir=allowed)

    # Subcommand --help raises SystemExit(0) inside argparse -> must return ok=True JSON
    help_res = json.loads(
        asyncio.run(tool.execute(command="annolid-run help kinematics"))
    )
    assert help_res["ok"] is True
    assert help_res["exit_code"] == 0
    assert "kinematics" in help_res["stdout"].lower()

    # Leading flags cannot bypass _is_mutating
    assert AnnolidRunTool._is_mutating(["--verbose", "train", "yolo"]) is True
    assert AnnolidRunTool._is_mutating(["--verbose", "update", "run"]) is True

    # Out-of-sandbox read operand and write flag are blocked
    blocked_read = json.loads(
        asyncio.run(tool.execute(command=f"annolid-run track-qc {outside_csv}"))
    )
    assert blocked_read["ok"] is False
    assert "outside allowed" in blocked_read["error"].lower()

    inside_ev = allowed / "events.csv"
    inside_ev.write_text("Recording time,Subject,Behavior,Event\n", encoding="utf-8")
    blocked_write = json.loads(
        asyncio.run(
            tool.execute(
                command=f"annolid-run time-budget {inside_ev} -o {outside / 'out.csv'}",
                allow_mutation=True,
            )
        )
    )
    assert blocked_write["ok"] is False
    assert "outside allowed" in blocked_write["error"].lower()


def test_s4_clawhub_rejects_zip_symlinks_workdir_escape_and_oversized_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    outside = tmp_path / "outside_ws"
    outside.mkdir()

    # 1. --workdir outside workspace is rejected
    esc = asyncio.run(
        clawhub.run_clawhub_command(
            ["install", "demo-skill", "--workdir", str(outside)], workspace=ws
        )
    )
    assert esc["ok"] is False
    assert "outside allowed workspace" in esc["error"]

    # 2. Zip archive containing a symlink entry is rejected by _safe_extract_zip
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("evil-skill/SKILL.md", "---\nname: evil-skill\n---\n")
        sym_info = zipfile.ZipInfo("evil-skill/leak.txt")
        sym_info.create_system = 3
        sym_info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(sym_info, "/etc/hosts")
    symlink_zip = tmp_path / "symlink_skill.zip"
    symlink_zip.write_bytes(buf.getvalue())

    with pytest.raises(ValueError, match="symlink"):
        clawhub._safe_extract_zip(symlink_zip, tmp_path / "extract_sym")

    # 3. Oversized archive download is rejected
    monkeypatch.setattr(clawhub, "_MAX_SKILL_DOWNLOAD_BYTES", 32)
    monkeypatch.setattr(clawhub, "_http_get_bytes", lambda _u, _t=60: b"x" * 64)
    over = asyncio.run(clawhub.clawhub_install_skill("demo-skill", workspace=ws))
    assert over["ok"] is False
    assert "limit" in over["error"].lower()


def test_p1_p2_packaging_manifest_spec_version_fallback_and_pandas_free_studio(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    manifest_txt = (repo_root / "MANIFEST.in").read_text(encoding="utf-8")
    for token in ("*.html", "*.css", "*.sh", "*.txt.gz", "*.swp"):
        assert token in manifest_txt

    pyproject_txt = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    assert "**/*.txt.gz" in pyproject_txt
    assert (repo_root / "annolid" / "tracking" / "__init__.py").is_file()
    assert not (
        repo_root / "annolid" / "detector" / "countgd" / "datasets" / ".odvg.py.swp"
    ).exists()

    for icon_name in ("record.png", "stop_record.png"):
        mode = (repo_root / "annolid" / "gui" / "icons" / icon_name).stat().st_mode
        assert bool(mode & stat.S_IRGRP) and bool(mode & stat.S_IROTH)

    spec_txt = (repo_root / "annolid.spec").read_text(encoding="utf-8")
    assert "annolid/gui/icons/*" in spec_txt
    assert 'copy_metadata("annolid")' in spec_txt

    # Version fallback when .dist-info is unavailable
    def _missing_pkg(_name: str) -> str:
        raise metadata.PackageNotFoundError("annolid")

    monkeypatch.setattr(gui_launcher.metadata, "version", _missing_pkg)
    gui_launcher._print_version()
    printed = capsys.readouterr().out.strip()
    assert printed != "unknown" and len(printed) >= 5

    # AnalysisStudioDialog imports cleanly even when pandas is absent (frozen desktop bundle)
    script = textwrap.dedent(
        """
        import importlib.abc
        import os
        import sys

        os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

        class PandasBlocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", 1)[0] == "pandas":
                    raise ModuleNotFoundError("blocked pandas")
                return None

        sys.meta_path.insert(0, PandasBlocker())
        from annolid.gui.widgets.analysis_studio_dialog import AnalysisStudioDialog
        assert AnalysisStudioDialog is not None
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
