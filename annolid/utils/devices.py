import subprocess


def has_gpu():
    try:
        subprocess.check_output(["nvidia-smi"])
        return True
    except Exception:
        return False


def get_device():
    try:
        import torch
    except Exception:
        return "cpu"
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


def clear_device_cache(torch_module=None, device=None) -> None:
    """Best-effort cache cleanup for CUDA/MPS backends."""
    try:
        torch = torch_module
        if torch is None:
            import torch as torch  # type: ignore
    except Exception:
        return

    device_type = ""
    if device is not None:
        device_type = str(getattr(device, "type", device)).lower()

    if not device_type or device_type.startswith("cuda"):
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    if not device_type or device_type.startswith("mps"):
        try:
            mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
            mps = getattr(torch, "mps", None)
            if mps_backend is not None and mps_backend.is_available():
                if mps is not None and hasattr(mps, "empty_cache"):
                    mps.empty_cache()
        except Exception:
            pass
