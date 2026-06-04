from __future__ import annotations

import subprocess
import sys


def test_realtime_perception_imports_without_ultralytics() -> None:
    script = """
import importlib.abc
import sys


class BlockUltralytics(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "ultralytics" or fullname.startswith("ultralytics."):
            raise ModuleNotFoundError("No module named 'ultralytics'")
        return None


sys.meta_path.insert(0, BlockUltralytics())
import annolid.realtime.perception
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout.splitlines()
