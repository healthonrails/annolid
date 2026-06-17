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
