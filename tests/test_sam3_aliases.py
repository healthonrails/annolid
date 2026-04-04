from __future__ import annotations

import sys
import types

from annolid.segmentation.SAM.sam3.aliases import ensure_sam3_aliases


def test_ensure_sam3_aliases_does_not_mix_external_namespace(monkeypatch) -> None:
    fake_external = types.ModuleType("sam3")
    fake_external.__path__ = ["/tmp/external-sam3"]
    fake_external.__file__ = "/tmp/external-sam3/__init__.py"
    monkeypatch.setitem(sys.modules, "sam3", fake_external)

    ensure_sam3_aliases()

    pkg = sys.modules.get("sam3")
    assert pkg is fake_external
    assert "/tmp/external-sam3" in list(getattr(pkg, "__path__", []))
