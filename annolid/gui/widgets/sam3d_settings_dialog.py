from qtpy import QtWidgets


class Sam3DSettingsDialog(QtWidgets.QDialog):
    """Simple settings panel for SAM 3D integration."""

    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("SAM 3D Settings"))
        self.setModal(True)
        cfg = config or {}
        layout = QtWidgets.QFormLayout(self)

        self.repo_edit = QtWidgets.QLineEdit(cfg.get("repo_path", "sam-3d-objects"))
        self.repo_btn = QtWidgets.QPushButton(self.tr("Browse…"))
        self.repo_btn.clicked.connect(self._browse_repo)
        layout.addRow(self.tr("SAM3D repo"), self._wrap(self.repo_edit, self.repo_btn))

        self.checkpoints_edit = QtWidgets.QLineEdit(cfg.get("checkpoints_dir", ""))
        self.checkpoints_btn = QtWidgets.QPushButton(self.tr("Browse…"))
        self.checkpoints_btn.clicked.connect(self._browse_checkpoints)
        layout.addRow(
            self.tr("Checkpoints dir"),
            self._wrap(self.checkpoints_edit, self.checkpoints_btn),
        )

        self.tag_edit = QtWidgets.QLineEdit(cfg.get("checkpoint_tag", "hf"))
        layout.addRow(self.tr("Checkpoint tag"), self.tag_edit)

        self.python_edit = QtWidgets.QLineEdit(cfg.get("python_executable", ""))
        self.python_btn = QtWidgets.QPushButton(self.tr("Browse…"))
        self.python_btn.clicked.connect(self._browse_python)
        layout.addRow(
            self.tr("Python (SAM3D env)"),
            self._wrap(self.python_edit, self.python_btn),
        )

        self.output_edit = QtWidgets.QLineEdit(cfg.get("output_dir", ""))
        self.output_btn = QtWidgets.QPushButton(self.tr("Browse…"))
        self.output_btn.clicked.connect(self._browse_output)
        layout.addRow(
            self.tr("Default output dir"),
            self._wrap(self.output_edit, self.output_btn),
        )

        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(-1, 2_147_483_647)
        self.seed_spin.setValue(
            int(cfg.get("seed")) if cfg.get("seed") is not None else -1
        )
        layout.addRow(self.tr("Seed (-1 = random)"), self.seed_spin)

        self.compile_chk = QtWidgets.QCheckBox(self.tr("Enable torch.compile"))
        self.compile_chk.setChecked(bool(cfg.get("compile", False)))
        layout.addRow(self.compile_chk)

        self.timeout_spin = QtWidgets.QSpinBox()
        self.timeout_spin.setRange(5, 24 * 3600)
        self.timeout_spin.setValue(int(cfg.get("timeout_s", 3600)))
        layout.addRow(self.tr("Timeout (s)"), self.timeout_spin)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _wrap(self, widget, button):
        box = QtWidgets.QHBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        container = QtWidgets.QWidget()
        box.addWidget(widget)
        box.addWidget(button)
        container.setLayout(box)
        return container

    def _browse_repo(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, self.tr("Select SAM3D repository")
        )
        if path:
            self.repo_edit.setText(path)

    def _browse_checkpoints(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, self.tr("Select checkpoints directory")
        )
        if path:
            self.checkpoints_edit.setText(path)

    def _browse_python(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, self.tr("Select Python executable")
        )
        if path:
            self.python_edit.setText(path)

    def _browse_output(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, self.tr("Select output directory")
        )
        if path:
            self.output_edit.setText(path)

    def values(self):
        seed_val = self.seed_spin.value()
        return {
            "repo_path": self.repo_edit.text().strip(),
            "checkpoints_dir": self.checkpoints_edit.text().strip(),
            "checkpoint_tag": self.tag_edit.text().strip() or "hf",
            "python_executable": self.python_edit.text().strip(),
            "output_dir": self.output_edit.text().strip(),
            "seed": None if seed_val < 0 else seed_val,
            "compile": self.compile_chk.isChecked(),
            "timeout_s": self.timeout_spin.value(),
        }
