import os
import subprocess
import platform


def open_or_start_file(file_name):
    # macOS
    if platform.system() == 'Darwin':
        subprocess.call(('open', file_name))
    # Windows
    elif platform.system() == 'Windows':
        os.startfile(file_name)
    # linux
    else:
        subprocess.call(('xdg-open', file_name))
