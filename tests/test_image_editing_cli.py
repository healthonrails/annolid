from __future__ import annotations

from pathlib import Path

from PIL import Image

from annolid.engine.cli import main as annolid_run


def test_annolid_run_image_edit_pillow_backend(tmp_path: Path) -> None:
    out_path = tmp_path / "out.png"
    rc = annolid_run(
        [
            "predict",
            "image-edit",
            "--backend",
            "pillow",
            "--prompt",
            "invert grayscale",
            "--width",
            "64",
            "--height",
            "48",
            "--num-images",
            "1",
            "--output",
            str(out_path),
        ]
    )
    assert rc == 0
    assert out_path.exists()
    img = Image.open(out_path)
    assert img.size == (64, 48)
