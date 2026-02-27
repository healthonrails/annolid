from __future__ import annotations

from pathlib import Path

import pandas as pd
from qtpy import QtCore, QtWidgets

from annolid.core.behavior.spec import (
    DEFAULT_SCHEMA_FILENAME,
    default_behavior_spec,
    save_behavior_spec,
)
from annolid.segmentation.dino_kpseg import defaults as dino_defaults
from annolid.gui.widgets import (
    LabelingProgressDashboardDialog,
    TrainingDashboardDialog,
)
from annolid.utils.logger import logger


class ProjectWorkflowMixin:
    """Project schema, wizard workflow, and DINO tool proxy helpers."""

    def open_project_schema_dialog(self) -> None:
        """Open the schema editor dialog and persist changes."""
        from annolid.gui.widgets.project_dialog import ProjectDialog

        schema = self.project_schema or default_behavior_spec()
        dialog = ProjectDialog(schema, parent=self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            new_schema = dialog.get_schema()
            self.project_schema = new_schema
            self.behavior_controller.configure_from_schema(new_schema)
            self._populate_behavior_controls_from_schema(new_schema)
            self._sync_behavior_flags_from_schema(new_schema)
            self._update_modifier_controls_for_behavior(self.event_type)

            target_path = self.project_schema_path
            if target_path is None:
                default_dir = (
                    Path(self.video_file).with_suffix("")
                    if self.video_file
                    else Path.cwd()
                )
                default_dir.mkdir(parents=True, exist_ok=True)
                default_path = default_dir / DEFAULT_SCHEMA_FILENAME
                path_str, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    self.tr("Save Project Schema"),
                    str(default_path),
                    self.tr("Schema Files (*.json *.yaml *.yml)"),
                )
                if not path_str:
                    return
                target_path = Path(path_str)
            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                save_behavior_spec(new_schema, target_path)
                self.project_schema_path = target_path
            except OSError as exc:
                QtWidgets.QMessageBox.critical(
                    self,
                    self.tr("Project Schema"),
                    self.tr("Failed to save schema:\n%s") % exc,
                )
            else:
                self.statusBar().showMessage(
                    self.tr("Project schema saved to %s") % target_path.name,
                    4000,
                )

    def open_new_project_wizard(self) -> None:
        """Open the new project wizard for guided project creation."""
        from annolid.gui.widgets.project_wizard import ProjectWizard

        wizard = ProjectWizard(parent=self)
        wizard.project_created.connect(self._on_project_created)
        wizard.exec_()

    def _on_project_created(self, project_path: Path, schema, videos: list) -> None:
        """Handle project creation from the wizard."""
        self.project_schema = schema
        self.project_schema_path = project_path / "project.annolid.json"
        self.behavior_controller.configure_from_schema(schema)
        self._populate_behavior_controls_from_schema(schema)
        self._sync_behavior_flags_from_schema(schema)

        if videos:
            self.openVideo(from_video_list=True, video_path=videos[0])

        self.statusBar().showMessage(
            self.tr("Project '%s' created successfully") % project_path.name,
            5000,
        )

    def open_export_dataset_wizard(self) -> None:
        """Open the dataset export wizard."""
        from annolid.gui.widgets.dataset_wizard import DatasetExportWizard

        source_dir = None
        if self.video_file:
            potential_dir = Path(self.video_file).with_suffix("")
            if potential_dir.is_dir():
                source_dir = str(potential_dir)

        wizard = DatasetExportWizard(source_dir=source_dir, parent=self)
        wizard.export_complete.connect(self._on_dataset_exported)
        wizard.exec_()

    def _on_dataset_exported(self, output_path: Path, format_type: str) -> None:
        """Handle dataset export completion."""
        self.statusBar().showMessage(
            self.tr("Dataset exported to %s (%s format)")
            % (output_path.name, format_type.upper()),
            5000,
        )

    def open_training_wizard(self) -> None:
        """Open the training wizard for guided model training."""
        from annolid.gui.widgets.training_wizard import TrainingWizard

        dataset_path = None
        if self.video_file:
            potential_yaml = Path(self.video_file).with_suffix("") / "data.yaml"
            if potential_yaml.exists():
                dataset_path = str(potential_yaml)

        wizard = TrainingWizard(dataset_path=dataset_path, parent=self)
        wizard.training_requested.connect(self._on_training_requested)
        wizard.exec_()

    def _on_training_requested(self, config: dict) -> None:
        """Handle training request from the wizard."""
        backend = config.get("backend", "yolo")

        if backend == "yolo":
            self._start_yolo_training_from_wizard(config)
        elif backend == "dino_kpseg":
            self._start_dino_training_from_wizard(config)
        else:
            self._start_maskrcnn_training_from_wizard(config)

        if config.get("open_dashboard"):
            self._open_training_dashboard()

    def _start_yolo_training_from_wizard(self, config: dict) -> None:
        """Start YOLO training with wizard configuration."""
        try:
            self.yolo_training_manager.start_training(
                yolo_model_file=config.get("model", "yolo11n-seg.pt"),
                model_path=None,
                data_config_path=config.get("dataset_path"),
                epochs=config.get("epochs", 100),
                image_size=config.get("imgsz", 640),
                batch_size=config.get("batch", 8),
                device=config.get("device", ""),
                plots=True,
                train_overrides=None,
                out_dir=config.get("output_dir"),
            )
            self.statusBar().showMessage(self.tr("YOLO training started"), 3000)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                self.tr("Training Error"),
                self.tr("Failed to start training: %s") % str(e),
            )

    def _start_dino_training_from_wizard(self, config: dict) -> None:
        """Start DINO KPSEG training with wizard configuration."""
        try:
            self.dino_kpseg_training_manager.start_training(
                data_config_path=config.get("dataset_path"),
                data_format=str(config.get("data_format", "auto") or "auto"),
                out_dir=config.get("output_dir"),
                model_name=config.get("model"),
                short_side=config.get("short_side", dino_defaults.SHORT_SIDE),
                layers=config.get("layers", dino_defaults.LAYERS),
                radius_px=config.get("radius_px", dino_defaults.RADIUS_PX),
                hidden_dim=config.get("hidden_dim", dino_defaults.HIDDEN_DIM),
                lr=config.get("lr", dino_defaults.LR),
                epochs=config.get("epochs", dino_defaults.EPOCHS),
                batch_size=config.get("batch", dino_defaults.BATCH),
                threshold=config.get("threshold", dino_defaults.THRESHOLD),
                device=config.get("device", ""),
                cache_features=config.get("cache_features", True),
            )
            self.statusBar().showMessage(self.tr("DINO KPSEG training started"), 3000)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                self.tr("Training Error"),
                self.tr("Failed to start training: %s") % str(e),
            )

    def _start_maskrcnn_training_from_wizard(self, config: dict) -> None:
        """Start MaskRCNN training with wizard configuration."""
        self.statusBar().showMessage(
            self.tr("MaskRCNN training: use the legacy Train Models dialog"), 5000
        )

    def _open_training_dashboard(self) -> None:
        """Open the training dashboard for monitoring."""
        try:
            dialog = getattr(self, "_training_dashboard_dialog", None)
            if dialog is None:
                dialog = TrainingDashboardDialog(settings=self.settings, parent=self)
                dialog.dashboard.register_training_manager(self.yolo_training_manager)
                dialog.dashboard.register_training_manager(
                    self.dino_kpseg_training_manager
                )
                dialog.finished.connect(
                    lambda *_: setattr(self, "_training_dashboard_dialog", None)
                )
                dialog.destroyed.connect(
                    lambda *_: setattr(self, "_training_dashboard_dialog", None)
                )
                self._training_dashboard_dialog = dialog

            dialog.show()
            dialog.setWindowState(dialog.windowState() & ~QtCore.Qt.WindowMinimized)
            dialog.raise_()
            dialog.activateWindow()
            QtWidgets.QApplication.processEvents()
        except Exception as e:
            logger.debug(f"Could not open training dashboard: {e}")

    def open_labeling_progress_dashboard(self) -> None:
        """Open the labeling progress dashboard (project stats + gamification)."""
        try:
            dialog = getattr(self, "_labeling_progress_dashboard_dialog", None)
            if dialog is None:
                project_root = self.project_controller.get_current_project_path()
                annotation_root = None
                try:
                    annotation_root = self.project_controller.get_project_directory(
                        "annotations"
                    )
                except Exception:
                    annotation_root = None
                if project_root is None and getattr(self, "video_results_folder", None):
                    project_root = Path(self.video_results_folder).resolve()
                if annotation_root is None and getattr(self, "annotation_dir", None):
                    annotation_root = Path(self.annotation_dir).resolve()
                if project_root is None:
                    project_root = Path.cwd()
                if annotation_root is None:
                    annotation_root = project_root

                dialog = LabelingProgressDashboardDialog(
                    initial_project_root=project_root,
                    initial_annotation_root=annotation_root,
                    parent=self,
                )
                dialog.finished.connect(
                    lambda *_: setattr(
                        self, "_labeling_progress_dashboard_dialog", None
                    )
                )
                dialog.destroyed.connect(
                    lambda *_: setattr(
                        self, "_labeling_progress_dashboard_dialog", None
                    )
                )
                self._labeling_progress_dashboard_dialog = dialog

            dialog.show()
            dialog.setWindowState(dialog.windowState() & ~QtCore.Qt.WindowMinimized)
            dialog.raise_()
            dialog.activateWindow()
            QtWidgets.QApplication.processEvents()
        except Exception as exc:
            logger.debug("Could not open labeling progress dashboard: %s", exc)

    def open_inference_wizard(self) -> None:
        """Open the inference wizard for running model predictions."""
        from annolid.gui.widgets.inference_wizard import InferenceWizard

        video_path = self.video_file if self.video_file else None

        wizard = InferenceWizard(video_path=video_path, parent=self)
        wizard.exec_()

    def _load_labels(self, labels_csv_file):
        """Load labels from the given CSV file."""
        self._df = pd.read_csv(labels_csv_file)
        self._df.rename(columns={"Unnamed: 0": "frame_number"}, inplace=True)

    def _toggle_patch_similarity_tool(self, checked=False):
        self.dino_controller.toggle_patch_similarity(checked)

    def _toggle_pca_map_tool(self, checked=False):
        self.dino_controller.toggle_pca_map(checked)

    def _deactivate_pca_map(self):
        self.dino_controller.deactivate_pca_map()

    def _request_pca_map(self) -> None:
        self.dino_controller.request_pca_map()

    def _open_patch_similarity_settings(self):
        self.dino_controller.open_patch_similarity_settings()
