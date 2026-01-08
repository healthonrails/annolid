import json
from pathlib import Path
from typing import Callable, List, Optional, Set

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
        self._missing_custom_weights_logged: Set[str] = set()
        self._custom_model_configs: List[ModelConfig] = self._load_custom_models(
        )
        self._last_selection: Optional[str] = None
        self._refresh_in_progress = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def initialize(self, default_selection: Optional[str] = None) -> None:
        """Populate the combo box and attach change handling."""
        target = default_selection or self._config.get("ai", {}).get("default")
        self._refresh_combo(target)
        self._combo.currentIndexChanged.connect(self._handle_index_changed)
        self._combo.installEventFilter(self)

    def get_current_model(self) -> Optional[ModelConfig]:
        """Return the ModelConfig associated with the current combo selection."""
        current_text = self._combo.currentText()
        selected = self._find_model_config(current_text)
        if selected and self._is_custom_model(selected):
            if not self._is_model_available(selected):
                self._warn_missing_custom_weight(selected)
                fallback = self._restore_previous_selection(
                    exclude=selected.display_name)
                return fallback
        return selected

    def get_current_weight(self) -> str:
        model = self.get_current_model()
        if not model:
            return "Segment-Anything (Edge)"
        if self._is_custom_model(model) and not self._is_model_available(model):
            self._warn_missing_custom_weight(model)
            fallback = self._restore_previous_selection(
                exclude=model.display_name)
            if fallback:
                return fallback.weight_file
            return "Segment-Anything (Edge)"
        return model.weight_file

    @property
    def custom_model_names(self) -> List[str]:
        return [cfg.display_name for cfg in self._available_custom_models]

    @property
    def all_model_configs(self) -> List[ModelConfig]:
        return [*MODEL_REGISTRY, *self._available_custom_models]

    def refresh(self, target_selection: Optional[str] = None) -> None:
        """Rebuild the combo box while preserving or applying a selection."""
        self._refresh_combo(target_selection)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        """Keep the model list in sync with on-disk weight files."""
        if obj is self._combo and event.type() in (
            QtCore.QEvent.FocusIn,
            QtCore.QEvent.MouseButtonPress,
        ):
            # If the user deleted/moved weights while the app is open, ensure
            # the next dropdown open reflects current disk state.
            current = self._combo.currentText()
            self._refresh_combo(current)
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _refresh_combo(self, target_selection: Optional[str]) -> None:
        if self._refresh_in_progress:
            return
        self._refresh_in_progress = True
        combo = self._combo
        self._prune_missing_custom_models()
        preserve = (
            target_selection
            or self._last_selection
            or self._config.get("ai", {}).get("default")
        )

        combo.blockSignals(True)
        try:
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
        finally:
            combo.blockSignals(False)
            self._refresh_in_progress = False

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

        selected_model = self._find_model_config(current_text)
        if selected_model and self._is_custom_model(selected_model):
            if not self._is_model_available(selected_model):
                self._warn_missing_custom_weight(selected_model)
                self._restore_previous_selection(
                    exclude=selected_model.display_name)
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
        parent_widget = self._get_parent_widget()

        dialog = QtWidgets.QFileDialog(parent_widget)
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        dialog.setNameFilters(
            [
                self.tr(
                    "YOLO Models (*.pt *.pth *.onnx *.engine *.mlpackage)"),
                self.tr("All Files (*)"),
            ]
        )
        dialog.setWindowTitle(self.tr("Select YOLO Model"))

        if not dialog.exec_():
            return

        selected_files = dialog.selectedFiles()
        if not selected_files:
            return

        weight_path = Path(selected_files[0]).expanduser().resolve()
        is_model_path = weight_path.is_file() or (
            weight_path.is_dir() and weight_path.suffix == ".mlpackage"
        )
        if not is_model_path:
            QtWidgets.QMessageBox.warning(
                parent_widget,
                self.tr("Invalid model selection"),
                self.tr(
                    "Selected path is not a compatible YOLO model export."),
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

    def _get_parent_widget(self) -> Optional[QtWidgets.QWidget]:
        parent_obj = self.parent()
        if isinstance(parent_obj, QtWidgets.QWidget):
            return parent_obj
        window = self._combo.window()
        return window if isinstance(window, QtWidgets.QWidget) else None

    def _find_model_config(self, display_name: str) -> Optional[ModelConfig]:
        return next(
            (
                cfg
                for cfg in self.all_model_configs
                if cfg.display_name == display_name
            ),
            None,
        )

    def _is_custom_model(self, model: ModelConfig) -> bool:
        return model in self._custom_model_configs

    def _is_model_available(self, model: ModelConfig) -> bool:
        if not self._is_custom_model(model):
            return True
        resolved = Path(model.weight_file).expanduser()
        exists = resolved.exists()
        if exists:
            resolved_str = str(resolved.resolve())
            self._missing_custom_weights_logged.discard(resolved_str)
        return exists

    @property
    def _available_custom_models(self) -> List[ModelConfig]:
        """Return custom models whose weight path currently exists on disk."""
        available: List[ModelConfig] = []
        for cfg in self._custom_model_configs:
            if self._is_model_available(cfg):
                available.append(cfg)
        return available

    def _prune_missing_custom_models(self) -> None:
        """Remove custom model entries whose weights are missing from disk."""
        kept: List[ModelConfig] = []
        removed: List[ModelConfig] = []

        for cfg in self._custom_model_configs:
            if self._is_model_available(cfg):
                kept.append(cfg)
            else:
                removed.append(cfg)

        if not removed:
            return

        self._custom_model_configs = kept
        self._save_custom_models()
        self.custom_models_changed.emit()

    def _warn_missing_custom_weight(self, model: ModelConfig) -> None:
        resolved = Path(model.weight_file).expanduser().resolve()
        resolved_str = str(resolved)
        first_time = resolved_str not in self._missing_custom_weights_logged
        if first_time:
            logger.warning("Custom YOLO weights not found: %s", resolved_str)
            self._missing_custom_weights_logged.add(resolved_str)

        parent_widget = self._get_parent_widget()
        if parent_widget and first_time:
            QtWidgets.QMessageBox.warning(
                parent_widget,
                self.tr("Custom model weights not found"),
                self.tr(
                    "The custom YOLO weights for \"{name}\" were not found at:\n"
                    "{path}\n"
                    "Please reconnect the storage location or choose a different model."
                ).format(name=model.display_name, path=resolved_str),
            )

    def _restore_previous_selection(
        self, *, exclude: Optional[str] = None
    ) -> Optional[ModelConfig]:
        candidates = []
        if self._last_selection:
            candidates.append(self._last_selection)
        default_text = self._config.get("ai", {}).get("default")
        if default_text and default_text not in candidates:
            candidates.append(default_text)
        candidates.extend(
            cfg.display_name for cfg in self.all_model_configs
            if cfg.display_name not in candidates
        )

        for name in candidates:
            if exclude and name == exclude:
                continue
            cfg = self._find_model_config(name)
            if cfg and self._is_model_available(cfg):
                index = self._combo.findText(cfg.display_name)
                if index != -1:
                    self._combo.blockSignals(True)
                    self._combo.setCurrentIndex(index)
                    self._combo.blockSignals(False)
                    self._last_selection = self._combo.currentText()
                return cfg
        return None

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
