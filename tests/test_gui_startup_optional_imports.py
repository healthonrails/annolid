from __future__ import annotations

import subprocess
import sys
import textwrap


def test_gui_startup_does_not_require_optional_model_or_analysis_runtimes() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        blocked = {
            "anthropic",
            "ftfy",
            "h5py",
            "iopath",
            "matplotlib",
            "mcp",
            "numba",
            "onnxruntime",
            "openai",
            "pandas",
            "scipy",
            "skimage",
            "sklearn",
            "timm",
            "tokenizers",
            "torch",
            "torchvision",
            "transformers",
            "ultralytics",
        }

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                root = fullname.split(".", 1)[0]
                if root in blocked:
                    raise ModuleNotFoundError(
                        f"blocked optional startup import: {fullname}"
                    )
                return None

        sys.meta_path.insert(0, Blocker())
        import annolid.gui.launcher  # noqa: F401
        import annolid.gui.app  # noqa: F401
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_quality_control_labelme_helpers_do_not_require_pandas() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class PandasBlocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", 1)[0] == "pandas":
                    raise ModuleNotFoundError(
                        f"blocked optional startup import: {fullname}"
                    )
                return None

        sys.meta_path.insert(0, PandasBlocker())
        from annolid.postprocessing.quality_control import (
            TracksResults,
            pred_dict_to_labelme,
        )

        shapes = pred_dict_to_labelme(
            {
                "frame_number": 0,
                "x1": 10.0,
                "y1": 20.0,
                "x2": 30.0,
                "y2": 40.0,
                "instance_name": "mouse",
                "class_score": 0.9,
                "segmentation": None,
                "tracking_id": 1,
            }
        )
        assert len(shapes) == 1
        assert shapes[0].label == "mouse_1"
        assert (
            TracksResults.switch_left_right(
                {"instance_name": "Left_obj", "cx": 600.0}, width=800
            )
            == "Right_obj"
        )
        assert (
            TracksResults.switch_left_right(
                {"instance_name": "Left_obj", "cx": float("nan")}, width=800
            )
            == "Left_obj"
        )
        assert TracksResults().df is None
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
