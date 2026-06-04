import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

from annolid.utils.annotation_compat import AI_MODELS as MODELS
from qtpy import QtCore, QtWidgets

from annolid.gui.models_registry import (
    ModelConfig,
    get_model_unavailable_reason,
    get_runtime_model_registry,
)
from annolid.utils.logger import logger


RECOMMENDED_AI_MODEL_NAMES: Tuple[str, ...] = (
    "SegmentAnything (Edge)",
    "Cutie",
    "SAM3",
    "Cutie + DINOv3 Keypoint Segmentation",
    "DINOv3 Keypoint Segmentation",
    "TAPNext (ONNX)",
    "YOLO11n",
    "MediaPipe Pose",
)


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

        self._browse_custom_label = parent.tr("Browse Custom Model…")
        self._more_models_label = parent.tr("More models...")
        self._missing_custom_weights_logged: Set[str] = set()
        self._custom_model_configs: List[ModelConfig] = self._load_custom_models()
        self._auto_discovered_configs: List[ModelConfig] = []
        self._runtime_registry: List[ModelConfig] = get_runtime_model_registry(
            config=self._config,
            settings=self._settings,
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
                    exclude=selected.display_name
                )
                return fallback
        return selected

    def get_current_weight(self) -> str:
        model = self.get_current_model()
        if not model:
            return "Segment-Anything (Edge)"
        if self._is_custom_model(model) and not self._is_model_available(model):
            self._warn_missing_custom_weight(model)
            fallback = self._restore_previous_selection(exclude=model.display_name)
            if fallback:
                return fallback.weight_file
            return "Segment-Anything (Edge)"
        return model.weight_file

    @property
    def custom_model_names(self) -> List[str]:
        return [cfg.display_name for cfg in self._available_custom_models] + [
            cfg.display_name for cfg in self._auto_discovered_configs
        ]

    @property
    def all_model_configs(self) -> List[ModelConfig]:
        return [
            *self._runtime_registry,
            *self._auto_discovered_configs,
            *self._available_custom_models,
        ]

    def refresh(self, target_selection: Optional[str] = None) -> None:
        """Rebuild the combo box while preserving or applying a selection."""
        self._refresh_combo(target_selection)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        """Keep the model list in sync with on-disk weight files."""
        if obj is self._combo and event.type() == QtCore.QEvent.MouseButtonPress:
            # If the user deleted/moved weights while the app is open, ensure
            # the next dropdown open reflects current disk state.
            current = self._combo.currentText()
            QtCore.QTimer.singleShot(0, lambda: self._refresh_combo(current))
            return False
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
        self._discover_recent_runs()
        preserve = (
            target_selection
            or self._last_selection
            or self._config.get("ai", {}).get("default")
        )

        combo.blockSignals(True)
        try:
            combo.clear()
            self._runtime_registry = get_runtime_model_registry(
                config=self._config,
                settings=self._settings,
            )

            existing_weights = set()
            for cfg in self._runtime_registry:
                if cfg.weight_file:
                    try:
                        existing_weights.add(str(Path(cfg.weight_file).resolve()))
                    except Exception:
                        existing_weights.add(str(cfg.weight_file))

            for name in self._visible_model_names(preserve):
                self._add_model_combo_item(name)

            for cfg in self._auto_discovered_configs:
                try:
                    resolved_weight = str(Path(cfg.weight_file).resolve())
                except Exception:
                    resolved_weight = str(cfg.weight_file)

                if resolved_weight in existing_weights:
                    continue

                name = cfg.display_name
                if combo.findText(name) == -1:
                    self._add_model_combo_item(name)

            for name in self.custom_model_names:
                if combo.findText(name) == -1:
                    self._add_model_combo_item(name)

            if combo.findText(self._more_models_label) == -1:
                combo.addItem(self._more_models_label)

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
                model_obj = combo.model()
                item = model_obj.item(index) if hasattr(model_obj, "item") else None
                if item is not None and not item.isEnabled():
                    index = -1
                    for candidate_index in range(combo.count()):
                        candidate_item = (
                            model_obj.item(candidate_index)
                            if hasattr(model_obj, "item")
                            else None
                        )
                        if candidate_item is None or candidate_item.isEnabled():
                            index = candidate_index
                            break

            if index >= 0:
                combo.setCurrentIndex(index)
                self._last_selection = combo.currentText()
        finally:
            combo.blockSignals(False)
            self._refresh_in_progress = False

    def _handle_index_changed(self) -> None:
        current_text = self._combo.currentText()
        if current_text == self._more_models_label:
            previous = self._last_selection
            self._restore_combo_text(previous)
            self._prompt_for_model_from_catalog()
            return

        if current_text == self._browse_custom_label:
            previous = self._last_selection
            self._restore_combo_text(previous)
            self._prompt_for_custom_model()
            return

        selected_model = self._find_model_config(current_text)
        if selected_model and not self._is_model_available(selected_model):
            if self._is_custom_model(selected_model):
                self._warn_missing_custom_weight(selected_model)
            else:
                reason = get_model_unavailable_reason(selected_model)
                if reason:
                    logger.warning(
                        "Model '%s' unavailable: %s",
                        selected_model.display_name,
                        reason,
                    )
            self._restore_previous_selection(exclude=selected_model.display_name)
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

    def _prompt_for_model_from_catalog(self) -> None:
        selected = self._select_model_from_catalog()
        if selected:
            self._refresh_combo(selected)
            self._handle_index_changed()

    def _prompt_for_custom_model(self) -> None:
        parent_widget = self._get_parent_widget()

        dialog = QtWidgets.QFileDialog(parent_widget)
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        dialog.setNameFilters(
            [
                self.tr("Models (*.pt *.pth *.onnx *.engine *.mlpackage)"),
                self.tr("All Files (*)"),
            ]
        )
        dialog.setWindowTitle(self.tr("Select Model"))

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
                self.tr("Selected path is not a compatible YOLO model export."),
            )
            return

        existing = next(
            (
                cfg
                for cfg in self._custom_model_configs
                if Path(cfg.weight_file) == weight_path
            ),
            None,
        )
        if existing:
            target_display = existing.display_name
        else:
            display_base = weight_path.stem
            existing_names = {cfg.display_name for cfg in self._runtime_registry} | {
                cfg.display_name for cfg in self._custom_model_configs
            }
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
            (cfg for cfg in self.all_model_configs if cfg.display_name == display_name),
            None,
        )

    def _legacy_model_names(self) -> List[str]:
        return [
            str(getattr(model, "name", "")).strip()
            for model in MODELS
            if str(getattr(model, "name", "")).strip()
        ]

    def _all_model_names(self) -> List[str]:
        seen: Set[str] = set()
        names: List[str] = []
        for name in self._legacy_model_names():
            if name not in seen:
                seen.add(name)
                names.append(name)
        for cfg in self.all_model_configs:
            if cfg.display_name not in seen:
                seen.add(cfg.display_name)
                names.append(cfg.display_name)
        return names

    def _visible_model_names(self, preserve: Optional[str]) -> List[str]:
        """Return a short toolbar list while keeping current/default selections."""
        all_names = set(self._all_model_names())
        visible: List[str] = []

        def append(name: Optional[str]) -> None:
            if not name:
                return
            if name not in all_names:
                return
            if name not in visible:
                visible.append(name)

        append(preserve)
        append(self._config.get("ai", {}).get("default"))
        for name in RECOMMENDED_AI_MODEL_NAMES:
            append(name)
        return visible

    def _add_model_combo_item(self, name: str) -> None:
        if self._combo.findText(name) != -1:
            return
        self._combo.addItem(name)
        idx = self._combo.findText(name)
        cfg = self._find_model_config(name)
        if cfg is not None:
            self._set_model_item_availability(idx, cfg)

    def _set_model_item_availability(self, idx: int, cfg: ModelConfig) -> None:
        if idx < 0:
            return
        reason = get_model_unavailable_reason(cfg)
        model_obj = self._combo.model()
        item = model_obj.item(idx) if hasattr(model_obj, "item") else None
        if reason:
            self._combo.setItemData(idx, reason, QtCore.Qt.ToolTipRole)
            if item is not None:
                item.setEnabled(False)
        else:
            self._combo.setItemData(idx, "", QtCore.Qt.ToolTipRole)
            if item is not None:
                item.setEnabled(True)

    def _restore_combo_text(self, text: Optional[str]) -> None:
        self._combo.blockSignals(True)
        try:
            if text and self._combo.findText(text) != -1:
                self._combo.setCurrentIndex(self._combo.findText(text))
            elif self._combo.count() > 0:
                self._combo.setCurrentIndex(0)
        finally:
            self._combo.blockSignals(False)

    def _select_model_from_catalog(self) -> Optional[str]:
        parent_widget = self._get_parent_widget()
        dialog = QtWidgets.QDialog(parent_widget)
        dialog.setWindowTitle(self.tr("Choose AI Model"))
        dialog.resize(560, 520)

        layout = QtWidgets.QVBoxLayout(dialog)
        search = QtWidgets.QLineEdit(dialog)
        search.setPlaceholderText(self.tr("Search models"))
        layout.addWidget(search)

        tree = QtWidgets.QTreeWidget(dialog)
        tree.setColumnCount(2)
        tree.setHeaderLabels([self.tr("Model"), self.tr("Type")])
        tree.setRootIsDecorated(True)
        tree.setAlternatingRowColors(True)
        layout.addWidget(tree)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=dialog,
        )
        select_button = buttons.button(QtWidgets.QDialogButtonBox.Ok)
        if select_button is not None:
            select_button.setText(self.tr("Select"))
            select_button.setEnabled(False)
        layout.addWidget(buttons)

        entries = self._catalog_entries()

        def selected_model_name() -> Optional[str]:
            item = tree.currentItem()
            if item is None or item.isDisabled():
                return None
            name = item.data(0, QtCore.Qt.UserRole)
            return str(name) if name else None

        def update_select_button() -> None:
            if select_button is not None:
                select_button.setEnabled(selected_model_name() is not None)

        def rebuild(filter_text: str = "") -> None:
            normalized_filter = filter_text.strip().lower()
            tree.clear()
            groups: Dict[str, QtWidgets.QTreeWidgetItem] = {}
            current_name = self._last_selection or self._combo.currentText()

            for name, group, cfg in entries:
                haystack = f"{name} {group}".lower()
                if normalized_filter and normalized_filter not in haystack:
                    continue
                parent_item = groups.get(group)
                if parent_item is None:
                    parent_item = QtWidgets.QTreeWidgetItem([group, ""])
                    parent_item.setFlags(
                        parent_item.flags() & ~QtCore.Qt.ItemIsSelectable
                    )
                    tree.addTopLevelItem(parent_item)
                    groups[group] = parent_item

                item = QtWidgets.QTreeWidgetItem([name, group])
                item.setData(0, QtCore.Qt.UserRole, name)
                if cfg is not None:
                    reason = get_model_unavailable_reason(cfg)
                    if reason:
                        item.setDisabled(True)
                        item.setToolTip(0, reason)
                parent_item.addChild(item)
                if name == current_name:
                    tree.setCurrentItem(item)

            tree.expandAll()
            tree.resizeColumnToContents(0)
            update_select_button()

        def accept_current() -> None:
            if selected_model_name() is not None:
                dialog.accept()

        search.textChanged.connect(rebuild)
        tree.currentItemChanged.connect(lambda *_: update_select_button())
        tree.itemDoubleClicked.connect(lambda *_: accept_current())
        buttons.accepted.connect(accept_current)
        buttons.rejected.connect(dialog.reject)

        rebuild()
        search.setFocus()

        if not dialog.exec_():
            return None
        return selected_model_name()

    def _catalog_entries(self) -> List[Tuple[str, str, Optional[ModelConfig]]]:
        entries: List[Tuple[str, str, Optional[ModelConfig]]] = []
        seen: Set[str] = set()

        def append(name: str, group: str, cfg: Optional[ModelConfig]) -> None:
            if not name or name in seen:
                return
            seen.add(name)
            entries.append((name, group, cfg))

        for name in self._legacy_model_names():
            append(name, self.tr("Point prompts"), None)

        custom_names = {cfg.display_name for cfg in self._available_custom_models}
        discovered_names = {cfg.display_name for cfg in self._auto_discovered_configs}
        for cfg in self.all_model_configs:
            if cfg.display_name in custom_names:
                group = self.tr("Custom models")
            elif cfg.display_name in discovered_names:
                group = self.tr("Recent training runs")
            else:
                group = self._model_group_label(cfg)
            append(cfg.display_name, group, cfg)

        return entries

    def _model_group_label(self, cfg: ModelConfig) -> str:
        text = f"{cfg.display_name} {cfg.identifier} {cfg.weight_file}".lower()
        if "yolo" in text:
            return self.tr("YOLO / prompted detection")
        if "dino" in text or "cutie" in text or "tracker" in text or "videomt" in text:
            return self.tr("Tracking and video segmentation")
        if "mediapipe" in text:
            return self.tr("Pose and realtime")
        if "sam" in text or "efficient" in text:
            return self.tr("Segmentation")
        return self.tr("Other models")

    def _is_custom_model(self, model: ModelConfig) -> bool:
        return model in self._custom_model_configs

    def _is_model_available(self, model: ModelConfig) -> bool:
        if not self._is_custom_model(model):
            return get_model_unavailable_reason(model) is None
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
                    'The custom weights for "{name}" were not found at:\n'
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
            cfg.display_name
            for cfg in self.all_model_configs
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
        stored = self._settings.value("ai/custom_models", None)
        if stored is None:
            # Fallback to old key for backward compatibility
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
        self._settings.setValue("ai/custom_models", json.dumps(payload))

    def _discover_recent_runs(self) -> None:
        """Scan the runs directory for recently trained YOLO and DINO KPSEG models."""
        from annolid.utils.runs import shared_runs_root

        root = shared_runs_root()
        if not root.exists():
            self._auto_discovered_configs = []
            return

        candidates: List[Tuple[float, ModelConfig]] = []

        def _scan_dir(parent_dir: Path, prefix: str):
            if not parent_dir.exists():
                return
            for run_dir in parent_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                weight_file = run_dir / "weights" / "best.pt"
                if weight_file.exists():
                    mtime = weight_file.stat().st_mtime
                    cfg = ModelConfig(
                        display_name=f"{prefix}: {run_dir.name}",
                        identifier=str(weight_file),
                        weight_file=str(weight_file),
                    )
                    candidates.append((mtime, cfg))

        # YOLO runs are typically in root / "train"
        _scan_dir(root / "train", "YOLO")
        # DINO runs are typically in root / "dino_kpseg" / "train"
        _scan_dir(root / "dino_kpseg" / "train", "DINO KPSEG")

        # Sort by modification time descending and keep top 5
        candidates.sort(key=lambda x: x[0], reverse=True)
        self._auto_discovered_configs = [cfg for _, cfg in candidates[:5]]
