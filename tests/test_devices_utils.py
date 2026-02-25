from __future__ import annotations

from annolid.utils.devices import clear_device_cache


def test_clear_device_cache_cuda_only() -> None:
    calls = {"cuda": 0, "mps": 0}

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def empty_cache() -> None:
            calls["cuda"] += 1

    class _Mps:
        @staticmethod
        def empty_cache() -> None:
            calls["mps"] += 1

    class _MpsBackend:
        @staticmethod
        def is_available() -> bool:
            return True

    class _Backends:
        mps = _MpsBackend()

    class _Torch:
        cuda = _Cuda()
        mps = _Mps()
        backends = _Backends()

    clear_device_cache(torch_module=_Torch(), device="cuda")

    assert calls["cuda"] == 1
    assert calls["mps"] == 0


def test_clear_device_cache_mps_requires_backend_available() -> None:
    calls = {"cuda": 0, "mps": 0}

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def empty_cache() -> None:
            calls["cuda"] += 1

    class _Mps:
        @staticmethod
        def empty_cache() -> None:
            calls["mps"] += 1

    class _MpsBackend:
        @staticmethod
        def is_available() -> bool:
            return False

    class _Backends:
        mps = _MpsBackend()

    class _Torch:
        cuda = _Cuda()
        mps = _Mps()
        backends = _Backends()

    clear_device_cache(torch_module=_Torch(), device="mps")

    assert calls["cuda"] == 0
    assert calls["mps"] == 0
