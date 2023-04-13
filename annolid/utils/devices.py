import subprocess


def has_gpu():
    try:
        output = subprocess.check_output(['nvidia-smi'])
        return True
    except:
        return False
