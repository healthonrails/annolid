import subprocess


def has_gpu():
    try:
        output = subprocess.check_output(['nvidia-smi'])
        return True
    except:
        return False


def get_device():
    try:
        import torch
    except Exception:
        return "cpu"
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device
