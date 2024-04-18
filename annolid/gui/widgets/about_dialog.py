import sys
import psutil
import torch
import subprocess
from qtpy.QtWidgets import (QApplication, QDialog,
                            QVBoxLayout, QLabel, QPushButton)
from annolid.utils.devices import get_device
from annolid.gui import app


def get_annolid_version():
    return app.__version__


def get_conda_version():
    # Retrieve Conda version using subprocess
    try:
        result = subprocess.check_output(
            ["conda", "--version"], universal_newlines=True)
        return result.strip()
    except subprocess.CalledProcessError:
        return "N/A"


def get_conda_env():
    # Retrieve Conda environment name using subprocess
    try:
        result = subprocess.check_output(
            ["conda", "info", "--envs"], universal_newlines=True)
        env_lines = result.strip().split('\n')
        current_env_line = [line for line in env_lines if '*' in line][0]
        return current_env_line.split()[1]
    except (subprocess.CalledProcessError, IndexError):
        return "N/A"


class SystemInfoDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("System Information")
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.init_ui()

    def init_ui(self):
        # Application-specific information
        annolid_version_label = QLabel(
            f"AnnoLid Version: {get_annolid_version()}")
        device_label = QLabel(f"Device: {get_device()}")
        pytorch_version_label = QLabel(f"PyTorch Version: {torch.__version__}")

        # Python, Conda, and environment information
        python_version_label = QLabel(
            f"Python Version: {sys.version.split()[0]}")
        conda_version_label = QLabel(f"Conda Version: {get_conda_version()}")
        conda_env_label = QLabel(f"Conda Environment: {get_conda_env()}")

        # System information
        self.os_label = QLabel()
        self.cpu_label = QLabel()
        self.memory_label = QLabel()
        self.disk_label = QLabel()

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_info)

        self.layout.addWidget(annolid_version_label)
        self.layout.addWidget(device_label)
        self.layout.addWidget(pytorch_version_label)
        self.layout.addWidget(python_version_label)
        self.layout.addWidget(conda_version_label)
        self.layout.addWidget(conda_env_label)
        self.layout.addWidget(QLabel("Operating System:"))
        self.layout.addWidget(self.os_label)
        self.layout.addWidget(QLabel("CPU:"))
        self.layout.addWidget(self.cpu_label)
        self.layout.addWidget(QLabel("Memory:"))
        self.layout.addWidget(self.memory_label)
        self.layout.addWidget(QLabel("Disk Space:"))
        self.layout.addWidget(self.disk_label)
        self.layout.addWidget(self.refresh_button)

        self.refresh_info()

    def refresh_info(self):
        self.os_label.setText(sys.platform)
        self.cpu_label.setText(f"{psutil.cpu_count()} cores")
        self.memory_label.setText(
            f"{psutil.virtual_memory().total / (1024 ** 3):.2f} GB")
        partitions = psutil.disk_partitions()
        disk_info = ""
        for partition in partitions:
            usage = psutil.disk_usage(partition.mountpoint)
            disk_info += f"{partition.device} - Total: {usage.total / (1024 ** 3):.2f} GB"
            disk_info += f"Used: {usage.used / (1024 ** 3):.2f} GB\n"
        self.disk_label.setText(disk_info)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = SystemInfoDialog()
    dialog.exec_()
