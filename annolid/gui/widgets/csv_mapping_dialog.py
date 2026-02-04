from qtpy import QtWidgets


class CSVPointCloudMappingDialog(QtWidgets.QDialog):
    def __init__(self, columns, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Map CSV Columns to Point Cloud")
        self.resize(480, 380)
        cols = list(columns)

        form = QtWidgets.QFormLayout()

        def combo(with_none=False):
            cb = QtWidgets.QComboBox()
            if with_none:
                cb.addItem("<None>")
            cb.addItems(cols)
            return cb

        self.x_cb = combo()
        self.y_cb = combo()
        self.z_cb = combo()
        self.intensity_cb = combo(with_none=True)
        self.color_cb = combo(with_none=True)
        self.label_cb = combo(with_none=True)

        # Best-effort autopick
        lower = {c.lower(): c for c in cols}
        for key, cb in (("x", self.x_cb), ("y", self.y_cb), ("z", self.z_cb)):
            if key in lower:
                cb.setCurrentText(lower[key])
        for key, cb in (("intensity", self.intensity_cb), ("i", self.intensity_cb)):
            if key in lower:
                cb.setCurrentText(lower[key])
        for key, cb in (
            ("label", self.color_cb),
            ("class", self.color_cb),
            ("color", self.color_cb),
        ):
            if key in lower:
                cb.setCurrentText(lower[key])
        # autopick for region labels
        for key, cb in (
            ("region", self.label_cb),
            ("name", self.label_cb),
            ("area", self.label_cb),
            ("label", self.label_cb),
        ):
            if (
                key in lower and cb.currentText() == "<None>"
            ):  # only if not already picked
                cb.setCurrentText(lower[key])

        form.addRow("X column", self.x_cb)
        form.addRow("Y column", self.y_cb)
        form.addRow("Z column", self.z_cb)
        form.addRow("Intensity (optional)", self.intensity_cb)
        form.addRow("Color by (optional)", self.color_cb)
        form.addRow("Region label (optional)", self.label_cb)

        # Scale controls (apply spacing before loading)
        scale_row = QtWidgets.QHBoxLayout()
        self.sx = QtWidgets.QDoubleSpinBox()
        self.sy = QtWidgets.QDoubleSpinBox()
        self.sz = QtWidgets.QDoubleSpinBox()
        for sb in (self.sx, self.sy, self.sz):
            sb.setDecimals(6)
            sb.setRange(0.000001, 1e9)
            sb.setValue(1.0)
        scale_row.addWidget(self.sx)
        scale_row.addWidget(self.sy)
        scale_row.addWidget(self.sz)
        scale_widget = QtWidgets.QWidget()
        scale_widget.setLayout(scale_row)
        form.addRow("Scale X/Y/Z", scale_widget)

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Color mode:"))
        self.mode_cont = QtWidgets.QRadioButton("Continuous")
        self.mode_cat = QtWidgets.QRadioButton("Categorical")
        self.mode_cont.setChecked(True)
        mode_row.addWidget(self.mode_cont)
        mode_row.addWidget(self.mode_cat)
        mode_row.addStretch(1)

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.addLayout(form)
        vbox.addLayout(mode_row)
        btns = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        db = QtWidgets.QDialogButtonBox(btns)
        db.accepted.connect(self.accept)
        db.rejected.connect(self.reject)
        vbox.addWidget(db)

    def mapping(self):
        def val(cb):
            t = cb.currentText()
            return None if t == "<None>" else t

        return {
            "x": self.x_cb.currentText(),
            "y": self.y_cb.currentText(),
            "z": self.z_cb.currentText(),
            "intensity": val(self.intensity_cb),
            "color_by": val(self.color_cb),
            "label_by": val(self.label_cb),
            "color_mode": "categorical" if self.mode_cat.isChecked() else "continuous",
            "sx": float(self.sx.value()),
            "sy": float(self.sy.value()),
            "sz": float(self.sz.value()),
        }
