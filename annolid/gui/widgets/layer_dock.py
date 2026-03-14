from __future__ import annotations

from qtpy import QtCore, QtWidgets


class ViewerLayerDockWidget(QtWidgets.QDockWidget):
    layerVisibilityChanged = QtCore.Signal(str, bool)
    layerOpacityChanged = QtCore.Signal(str, float)
    layerSelected = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__("Layers", parent)
        self.setObjectName("viewerLayerDock")
        self._layer_map: dict[str, dict] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        container = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.layer_list = QtWidgets.QListWidget(container)
        self.layer_list.itemChanged.connect(self._on_item_changed)
        self.layer_list.currentItemChanged.connect(self._on_current_item_changed)
        layout.addWidget(self.layer_list)

        self.details_label = QtWidgets.QLabel("", container)
        self.details_label.setWordWrap(True)
        layout.addWidget(self.details_label)

        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, container)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        layout.addWidget(self.opacity_slider)

        self.setWidget(container)
        self._set_opacity_enabled(False)

    def _set_opacity_enabled(self, enabled: bool) -> None:
        self.opacity_slider.setEnabled(bool(enabled))

    def _current_layer_id(self) -> str | None:
        item = self.layer_list.currentItem()
        if item is None:
            return None
        return str(item.data(QtCore.Qt.UserRole) or "") or None

    def _on_current_item_changed(self, current, _previous) -> None:
        layer_id = (
            str(current.data(QtCore.Qt.UserRole) or "") if current is not None else ""
        )
        layer = self._layer_map.get(layer_id, {})
        supports_opacity = bool(layer.get("supports_opacity", False))
        with QtCore.QSignalBlocker(self.opacity_slider):
            self.opacity_slider.setValue(
                int(round(float(layer.get("opacity", 1.0) or 1.0) * 100.0))
            )
        self._set_opacity_enabled(supports_opacity)
        self.details_label.setText(str(layer.get("details", "") or ""))
        if layer_id:
            self.layerSelected.emit(layer_id)

    def _on_item_changed(self, item) -> None:
        if item is None:
            return
        layer_id = str(item.data(QtCore.Qt.UserRole) or "")
        if not layer_id:
            return
        layer = self._layer_map.get(layer_id, {})
        if not bool(layer.get("checkable", True)):
            return
        visible = item.checkState() == QtCore.Qt.Checked
        self.layerVisibilityChanged.emit(layer_id, visible)

    def _on_opacity_changed(self, value: int) -> None:
        layer_id = self._current_layer_id()
        if not layer_id:
            return
        layer = self._layer_map.get(layer_id, {})
        if not bool(layer.get("supports_opacity", False)):
            return
        self.layerOpacityChanged.emit(layer_id, float(value) / 100.0)

    def set_layers(self, layers: list[dict]) -> None:
        self._layer_map = {
            str(layer.get("id") or ""): dict(layer or {})
            for layer in list(layers or [])
            if str(layer.get("id") or "")
        }
        selected_layer_id = self._current_layer_id()
        with QtCore.QSignalBlocker(self.layer_list):
            self.layer_list.clear()
            for layer in list(layers or []):
                layer_id = str(layer.get("id") or "")
                if not layer_id:
                    continue
                item = QtWidgets.QListWidgetItem(str(layer.get("name") or layer_id))
                item.setData(QtCore.Qt.UserRole, layer_id)
                flags = item.flags()
                if bool(layer.get("checkable", True)):
                    item.setFlags(
                        flags
                        | QtCore.Qt.ItemIsUserCheckable
                        | QtCore.Qt.ItemIsSelectable
                        | QtCore.Qt.ItemIsEnabled
                    )
                    item.setCheckState(
                        QtCore.Qt.Checked
                        if bool(layer.get("visible", True))
                        else QtCore.Qt.Unchecked
                    )
                else:
                    item.setFlags(flags & ~QtCore.Qt.ItemIsUserCheckable)
                self.layer_list.addItem(item)
            if selected_layer_id:
                for index in range(self.layer_list.count()):
                    item = self.layer_list.item(index)
                    if str(item.data(QtCore.Qt.UserRole) or "") == selected_layer_id:
                        self.layer_list.setCurrentItem(item)
                        break
            if self.layer_list.currentItem() is None and self.layer_list.count() > 0:
                self.layer_list.setCurrentRow(0)
        if self.layer_list.currentItem() is None:
            self.details_label.setText("")
            self._set_opacity_enabled(False)
