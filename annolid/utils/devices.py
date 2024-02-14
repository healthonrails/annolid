import subprocess


def has_gpu():
    try:
        output = subprocess.check_output(['nvidia-smi'])
        return True
    except:
        return False


def get_device():
    import torch
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Using device: {device}')
    return device
