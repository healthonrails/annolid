import json
from pathlib import Path
from typing import Callable, List, Optional

from labelme.ai import MODELS
from qtpy import QtCore, QtWidgets

from annolid.gui.models_registry import MODEL_REGISTRY, ModelConfig
from annolid.utils.logger import logger


class AIModelManager(QtCore.QObject):
    """Manage the AI model selector, including user-supplied YOLO weights."""

    custom_models_changed = QtCore.Signal()

    def __init__(
        self,
        *,
        parent: QtWidgets.QWidget,
        combo: QtWidgets.QComboBox,
        settings: QtCore.QSettings,
        base_config: dict,
        canvas_getter: Callable[[], Optional[object]],
    ) -> None:
        super().__init__(parent)
        self._combo = combo
        self._settings = settings
        self._config = base_config or {}
        self._canvas_getter = canvas_getter

        self._browse_custom_label = parent.tr("Browse Custom YOLO…")
        self._custom_model_configs: List[ModelConfig] = self._load_custom_models(
        )
        self._last_selection: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def initialize(self, default_selection: Optional[str] = None) -> None:
        """Populate the combo box and attach change handling."""
        target = default_selection or self._config.get("ai", {}).get("default")
        self._refresh_combo(target)
        self._combo.currentIndexChanged.connect(self._handle_index_changed)

    def get_current_model(self) -> Optional[ModelConfig]:
        """Return the ModelConfig associated with the current combo selection."""
        current_text = self._combo.currentText()
        return next(
            (cfg for cfg in self.all_model_configs if cfg.display_name == current_text),
            None,
        )

    def get_current_weight(self) -> str:
        model = self.get_current_model()
        return model.weight_file if model else "Segment-Anything (Edge)"

    @property
    def custom_model_names(self) -> List[str]:
        return [cfg.display_name for cfg in self._custom_model_configs]

    @property
    def all_model_configs(self) -> List[ModelConfig]:
        return [*MODEL_REGISTRY, *self._custom_model_configs]

    def refresh(self, target_selection: Optional[str] = None) -> None:
        """Rebuild the combo box while preserving or applying a selection."""
        self._refresh_combo(target_selection)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _refresh_combo(self, target_selection: Optional[str]) -> None:
        combo = self._combo
        preserve = (
            target_selection
            or self._last_selection
            or self._config.get("ai", {}).get("default")
        )

        combo.blockSignals(True)
        combo.clear()

        for model in MODELS:
            combo.addItem(model.name)

        registry_names = [cfg.display_name for cfg in MODEL_REGISTRY]
        for name in registry_names:
            if combo.findText(name) == -1:
                combo.addItem(name)

        for name in self.custom_model_names:
            if combo.findText(name) == -1:
                combo.addItem(name)

        if combo.findText(self._browse_custom_label) == -1:
            combo.addItem(self._browse_custom_label)

        desired = preserve
        if desired and combo.findText(desired) != -1:
            index = combo.findText(desired)
        else:
            default_text = self._config.get("ai", {}).get("default")
            if default_text and combo.findText(default_text) != -1:
                index = combo.findText(default_text)
            else:
                index = 0 if combo.count() else -1

        if index >= 0:
            combo.setCurrentIndex(index)
            self._last_selection = combo.currentText()

        combo.blockSignals(False)

    def _handle_index_changed(self) -> None:
        current_text = self._combo.currentText()
        if current_text == self._browse_custom_label:
            previous = self._last_selection
            self._combo.blockSignals(True)
            if previous and self._combo.findText(previous) != -1:
                self._combo.setCurrentIndex(self._combo.findText(previous))
            elif self._combo.count() > 0:
                self._combo.setCurrentIndex(0)
            self._combo.blockSignals(False)
            self._prompt_for_custom_model()
            return

        self._last_selection = current_text

        canvas = self._canvas_getter()
        if not canvas:
            return
        if getattr(canvas, "createMode", None) in ["ai_polygon", "ai_mask"]:
            canvas.initializeAiModel(
                name=current_text,
                _custom_ai_models=self.custom_model_names,
            )

    def _prompt_for_custom_model(self) -> None:
        parent_obj = self.parent()
        if isinstance(parent_obj, QtWidgets.QWidget):
            parent_widget: Optional[QtWidgets.QWidget] = parent_obj
        else:
            # Fallback to the combo box's top-level window if the manager's parent
            # is not a QWidget instance (should not happen, but keeps it robust).
            window = self._combo.window()
            parent_widget = window if isinstance(
                window, QtWidgets.QWidget) else None

        dialog = QtWidgets.QFileDialog(parent_widget)
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        dialog.setNameFilters(
            [
                self.tr("YOLO Weights (*.pt *.pth)"),
                self.tr("All Files (*)"),
            ]
        )
        dialog.setWindowTitle(self.tr("Select YOLO Weights"))

        if not dialog.exec_():
            return

        selected_files = dialog.selectedFiles()
        if not selected_files:
            return

        weight_path = Path(selected_files[0]).expanduser().resolve()
        if not weight_path.is_file():
            QtWidgets.QMessageBox.warning(
                parent_widget,
                self.tr("Invalid weights"),
                self.tr("Selected file is not a valid YOLO weight file."),
            )
            return

        existing = next(
            (cfg for cfg in self._custom_model_configs
             if Path(cfg.weight_file) == weight_path),
            None,
        )
        if existing:
            target_display = existing.display_name
        else:
            display_base = weight_path.stem
            existing_names = {
                cfg.display_name for cfg in MODEL_REGISTRY
            } | {cfg.display_name for cfg in self._custom_model_configs}
            candidate = display_base
            counter = 2
            while candidate in existing_names:
                candidate = f"{display_base} ({counter})"
                counter += 1

            new_config = ModelConfig(
                display_name=candidate,
                identifier=str(weight_path),
                weight_file=str(weight_path),
            )
            self._custom_model_configs.append(new_config)
            self._save_custom_models()
            target_display = candidate
            self.custom_models_changed.emit()

        self._refresh_combo(target_display)

    def _load_custom_models(self) -> List[ModelConfig]:
        stored = self._settings.value("ai/custom_yolo_models", [])
        entries: List[ModelConfig] = []

        if isinstance(stored, str):
            try:
                raw_list = json.loads(stored)
            except json.JSONDecodeError:
                raw_list = []
        elif isinstance(stored, (list, tuple)):
            raw_list = list(stored)
        else:
            raw_list = []

        for item in raw_list:
            if isinstance(item, str):
                try:
                    item = json.loads(item)
                except json.JSONDecodeError:
                    continue
            if not isinstance(item, dict):
                continue
            weight = item.get("weight") or item.get("path")
            if not weight:
                continue
            resolved = Path(weight).expanduser().resolve()
            if not resolved.exists():
                logger.warning(
                    "Skipping missing custom YOLO weights: %s", resolved)
                continue
            display = item.get("display") or resolved.stem
            identifier = item.get("identifier") or str(resolved)
            entries.append(
                ModelConfig(
                    display_name=display,
                    identifier=str(identifier),
                    weight_file=str(resolved),
                )
            )
        return entries

    def _save_custom_models(self) -> None:
        payload = [
            {
                "display": cfg.display_name,
                "identifier": cfg.identifier,
                "weight": cfg.weight_file,
            }
            for cfg in self._custom_model_configs
        ]
        self._settings.setValue("ai/custom_yolo_models", json.dumps(payload))
