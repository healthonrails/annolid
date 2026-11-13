from __future__ import annotations

import ast
import http.server
import json
import sys
import threading
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
import pytest

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
        "numba",
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
