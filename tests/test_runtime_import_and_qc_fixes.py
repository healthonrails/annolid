from __future__ import annotations

import ast
import http.server
import json
import subprocess
import sys
import threading
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
import pytest

from annolid.annotation.labelme2coco import _build_dataset_skeleton
from annolid.gui import cli as gui_cli
from annolid.gui import tensorboard as tb_mod
from annolid.gui import workers as workers_mod
from annolid.postprocessing.quality_control import TracksResults, pred_dict_to_labelme


def test_realtime_worker_handles_missing_pyzmq_cleanly(monkeypatch) -> None:
    monkeypatch.setattr(workers_mod, "zmq", None)
    worker = workers_mod.RealtimeSubscriberWorker("tcp://127.0.0.1:5555")
    errors: list[str] = []
    worker.error.connect(errors.append)

    worker.run()

    assert len(errors) == 1
    assert "pyzmq" in errors[0]


def test_tensorboard_http_helpers_work_without_requests() -> None:
    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/data/plugins_listing":
                body = json.dumps({"projector": {"enabled": True}}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

        def log_message(self, format, *args):  # noqa: A002
            return

    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{port}"
        assert tb_mod._can_connect(url, timeout=1.0) is True
        assert tb_mod._plugin_enabled(url, "projector", timeout=1.0) is True
        assert tb_mod._plugin_enabled(url, "scalars", timeout=1.0) is None
    finally:
        server.shutdown()
        server.server_close()


def test_gui_entrypoint_has_no_unguarded_non_gui_third_party_imports() -> None:
    allowed_top_modules = {
        "numpy",
        "scipy",
        "skimage",
        "sklearn",
        "easydict",
        "gdown",
        "cv2",
        "yaml",
        "pandas",
        "imgviz",
        "imageio",
        "tqdm",
        "shapely",
        "imageio_ffmpeg",
        "qimage2ndarray",
        "qtpy",
        "h5py",
        "psutil",
        "tree_config",
        "PIL",
        "chardet",
        "jsonschema",
        "pycocotools",
        "PySide6",
    }
    repo_root = Path(__file__).resolve().parent.parent
    visited: set[Path] = set()
    unguarded: list[tuple[str, str]] = []

    def _resolve(mod_name: str) -> Path | None:
        parts = mod_name.split(".")
        cand_py = repo_root.joinpath(*parts).with_suffix(".py")
        cand_pkg = repo_root.joinpath(*parts, "__init__.py")
        if cand_py.exists():
            return cand_py
        if cand_pkg.exists():
            return cand_pkg
        return None

    def _walk(py_file: Path) -> None:
        if py_file in visited:
            return
        visited.add(py_file)
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in tree.body:
            targets: list[tuple[str, int]] = []
            if isinstance(node, ast.Import):
                targets = [(alias.name, node.lineno) for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    pkg_parts = list(py_file.relative_to(repo_root).parts[:-1])
                    up = node.level - 1
                    base = pkg_parts[: len(pkg_parts) - up] if up else pkg_parts
                    full = ".".join(base + ([node.module] if node.module else []))
                else:
                    full = node.module or ""
                if full:
                    targets.append((full, node.lineno))
                    for alias in node.names:
                        targets.append((f"{full}.{alias.name}", node.lineno))
            for mod_name, lineno in targets:
                top = mod_name.split(".")[0]
                if top in sys.stdlib_module_names:
                    continue
                if top == "annolid":
                    resolved = _resolve(mod_name)
                    if resolved is not None:
                        _walk(resolved)
                elif top not in allowed_top_modules:
                    rel = py_file.relative_to(repo_root)
                    unguarded.append((top, f"{rel}:{lineno}"))

    _walk(repo_root / "annolid" / "gui" / "app.py")
    assert unguarded == []


def test_coco2yolo_has_no_syntax_warnings() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src_path = repo_root / "annolid" / "annotation" / "coco2yolo.py"
    source = src_path.read_text(encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        ast.parse(source, filename=str(src_path))


def test_tracked_python_sources_have_no_syntax_warnings() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    tracked = subprocess.check_output(
        ["git", "ls-files", "*.py"],
        cwd=repo_root,
        text=True,
    ).splitlines()
    assert tracked
    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        for rel in tracked:
            src_path = repo_root / rel
            if not src_path.is_file():
                continue
            source = src_path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(src_path))


def test_quality_control_tracks_results_and_border_mask_handling(
    tmp_path: Path,
) -> None:
    empty_res = TracksResults()
    assert empty_res.df is None
    assert empty_res.cap is None
    empty_res.clean_up()

    missing_res = TracksResults(
        video_file=str(tmp_path / "missing.mp4"),
        tracking_csv=str(tmp_path / "missing.csv"),
    )
    assert missing_res.df is None
    assert missing_res.cap is None

    assert pred_dict_to_labelme({"frame_number": 0}) == []

    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[0:10, 0:10] = 1
    rle = mask_util.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("utf-8")
    shapes = pred_dict_to_labelme(
        {
            "frame_number": 0,
            "x1": 0.0,
            "y1": 0.0,
            "x2": 10.0,
            "y2": 10.0,
            "instance_name": "mouse",
            "class_score": 0.95,
            "segmentation": str(rle),
            "tracking_id": 0,
        },
        keypoint_area_threshold=10,
    )
    assert len(shapes) == 1
    assert shapes[0].label == "mouse_0"
    xs = [float(pt[0]) for pt in shapes[0].points]
    ys = [float(pt[1]) for pt in shapes[0].points]
    assert min(xs) == pytest.approx(0.5)
    assert min(ys) == pytest.approx(0.5)

    csv_path = tmp_path / "tracks.csv"
    pd.DataFrame(
        [
            {"frame_number": 1, "instance_name": "mouse_2", "cx": 10.0, "cy": 10.0},
            {"frame_number": 3, "instance_name": "mouse_2", "cx": 20.0, "cy": 20.0},
        ]
    ).to_csv(csv_path, index=False)
    tr = TracksResults(tracking_csv=str(csv_path))
    filled = tr.fill_missing_instances(instance_name="mouse_2")
    assert set(filled["frame_number"].tolist()) == {0, 1, 2, 3}


def test_cli_utf8_files_and_coco_skeleton_emit_no_deprecation_warnings(
    tmp_path: Path,
) -> None:
    labels_file = tmp_path / "labels.txt"
    flags_file = tmp_path / "flags.txt"
    config_file = tmp_path / "config.yaml"
    labels_file.write_bytes("mouse_α\r\n\r\nrat_β\r\n".encode("utf-8"))
    flags_file.write_bytes("grooming\r\nrearing_γ\r\n".encode("utf-8"))
    config_file.write_text("{}\n", encoding="utf-8")

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _cfg, namespace, version_requested = gui_cli.parse_cli(
            [
                "--config",
                str(config_file),
                "--labels",
                str(labels_file),
                "--flags",
                str(flags_file),
            ]
        )
        skeleton = _build_dataset_skeleton([{"id": 1, "name": "mouse"}])

    assert version_requested is False
    assert namespace.labels == ["mouse_α", "rat_β"]
    assert namespace.flags == ["grooming", "rearing_γ"]
    info = skeleton["info"]
    assert isinstance(info["year"], int)
    assert str(info["date_created"]).endswith("Z")
    assert "+00:00Z" not in str(info["date_created"])


def test_mask_and_skeleton_helpers_work_without_scikit_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    import annolid.annotation.labelme2binarymasks as l2bm
    import annolid.postprocessing.skeletonization as skel_mod
    from annolid.gui.shape import _find_mask_contours

    real_import = builtins.__import__

    def _block_skimage(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "skimage" or name.startswith("skimage."):
            raise ImportError("Blocked skimage for Python 3.15+ fallback test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _block_skimage)
    monkeypatch.setattr(l2bm, "_skimage_polygon", None)
    monkeypatch.setattr(skel_mod, "_skimage_skeletonize", None)
    monkeypatch.setattr(skel_mod, "_skimage_medial_axis", None)

    mask = np.zeros((24, 32), dtype=bool)
    mask[6:18, 5:27] = True
    contours = _find_mask_contours(mask)
    assert len(contours) >= 1
    assert contours[0].ndim == 2 and contours[0].shape[1] == 2
    assert np.allclose(contours[0][0], contours[0][-1])

    lm_path = tmp_path / "frame_00000001.json"
    lm_path.write_text(
        json.dumps(
            {
                "version": "5.0.1",
                "flags": {},
                "shapes": [
                    {
                        "label": "mouse_1",
                        "points": [[5.0, 6.0], [26.0, 6.0], [26.0, 17.0], [5.0, 17.0]],
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {},
                    }
                ],
                "imagePath": "frame_00000001.png",
                "imageData": None,
                "imageHeight": 24,
                "imageWidth": 32,
            }
        ),
        encoding="utf-8",
    )
    processor = l2bm.LabelMeProcessor(str(lm_path))
    masks = processor.get_all_masks()
    assert len(masks) == 1
    binary_mask, label = masks[0]
    assert label == "mouse_1"
    assert int(binary_mask.sum()) > 100

    sp = skel_mod.ShapeProcessor(binary_mask.astype(bool))
    assert sp.skeleton.shape == (24, 32)
    assert int(sp.skeleton.sum()) > 0
    assert sp.dist_transform.shape == (24, 32)
