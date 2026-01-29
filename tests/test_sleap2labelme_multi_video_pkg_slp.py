import json
from io import BytesIO

import numpy as np
import pytest


def _png_bytes(size=(2, 3), color=(10, 20, 30)):
    pytest.importorskip("PIL")
    from PIL import Image

    im = Image.new("RGB", size, color=color)
    buf = BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue(), im.size


def test_sleap2labelme_pkg_slp_multi_video_outputs_per_video_dir(tmp_path):
    h5py = pytest.importorskip("h5py")

    from annolid.annotation.sleap2labelme import convert_sleap_h5_to_labelme

    img0_bytes, (w0, h0) = _png_bytes(size=(4, 5), color=(1, 2, 3))
    img1_bytes, (w1, h1) = _png_bytes(size=(6, 7), color=(4, 5, 6))

    in_path = tmp_path / "in.pkg.slp"
    with h5py.File(in_path, "w") as f:
        meta = {
            "nodes": [{"name": "p0"}, {"name": "p1"}],
            "skeletons": [
                {
                    "nodes": [{"id": 0}, {"id": 1}],
                    "links": [],
                    "graph": {"name": "sk0"},
                }
            ],
        }
        f.create_dataset("metadata", data=json.dumps(meta).encode("utf-8"))

        frames_dt = np.dtype(
            [
                ("frame_id", "<u8"),
                ("video", "<u4"),
                ("frame_idx", "<u8"),
                ("instance_id_start", "<u8"),
                ("instance_id_end", "<u8"),
            ]
        )
        frames = np.array(
            [
                (0, 0, 5, 0, 1),
                (1, 1, 5, 1, 2),
            ],
            dtype=frames_dt,
        )
        f.create_dataset("frames", data=frames)

        inst_dt = np.dtype(
            [
                ("instance_id", "<i8"),
                ("instance_type", "u1"),
                ("frame_id", "<u8"),
                ("skeleton", "<u4"),
                ("track", "<i4"),
                ("point_id_start", "<u8"),
                ("point_id_end", "<u8"),
            ]
        )
        instances = np.array(
            [
                (0, 0, 0, 0, 0, 0, 2),
                (1, 0, 1, 0, 0, 2, 4),
            ],
            dtype=inst_dt,
        )
        f.create_dataset("instances", data=instances)

        pts_dt = np.dtype([("x", "<f8"), ("y", "<f8"), ("visible", "?")])
        points = np.array(
            [
                (1.0, 2.0, True),
                (3.0, 4.0, True),
                (5.0, 6.0, True),
                (7.0, 8.0, True),
            ],
            dtype=pts_dt,
        )
        f.create_dataset("points", data=points)

        vlen_u8 = h5py.vlen_dtype(np.dtype("uint8"))
        v0 = f.create_group("video0")
        v0.create_dataset("frame_numbers", data=np.array([5], dtype=np.int32))
        v0_video = v0.create_dataset("video", (1,), dtype=vlen_u8)
        v0_video[0] = np.frombuffer(img0_bytes, dtype=np.uint8)

        v1 = f.create_group("video1")
        v1.create_dataset("frame_numbers", data=np.array([5], dtype=np.int32))
        v1_video = v1.create_dataset("video", (1,), dtype=vlen_u8)
        v1_video[0] = np.frombuffer(img1_bytes, dtype=np.uint8)

        videos_json = np.array(
            [
                b'{"backend":{"filename":".","dataset":"video0/video"}}',
                b'{"backend":{"filename":".","dataset":"video1/video"}}',
            ],
            dtype="|S64",
        )
        f.create_dataset("videos_json", data=videos_json)

    out_dir = tmp_path / "out"
    convert_sleap_h5_to_labelme(in_path, out_dir, save_frames=True, video_index=None)

    assert (out_dir / "pose_schema.json").exists()

    assert (out_dir / "video0_000000005.png").exists()
    assert (out_dir / "video1_000000005.png").exists()
    assert (out_dir / "video0_000000005.json").exists()
    assert (out_dir / "video1_000000005.json").exists()

    j0 = json.loads((out_dir / "video0_000000005.json").read_text(encoding="utf-8"))
    j1 = json.loads((out_dir / "video1_000000005.json").read_text(encoding="utf-8"))

    assert j0["imagePath"] == "video0_000000005.png"
    assert (j0["imageWidth"], j0["imageHeight"]) == (w0, h0)
    assert j1["imagePath"] == "video1_000000005.png"
    assert (j1["imageWidth"], j1["imageHeight"]) == (w1, h1)
