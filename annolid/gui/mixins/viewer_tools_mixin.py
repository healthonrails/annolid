from __future__ import annotations

import tempfile
from pathlib import Path

from qtpy import QtWidgets

from annolid.gui.models_registry import PATCH_SIMILARITY_MODELS
from annolid.gui.threejs_examples import generate_threejs_example


class ViewerToolsMixin:
    """3D viewer and PCA map UI helpers."""

    def open_3d_viewer(self):
        """Open Annolid's built-in 3D stack viewer."""
        tiff_path = None
        try:
            from annolid.data import videos as _videos_mod

            if (
                isinstance(self.video_loader, _videos_mod.TiffStackVideo)
                and self.video_file
            ):
                tiff_path = str(self.video_file)
        except Exception:
            pass

        if not tiff_path:
            start_dir = (
                str(Path(self.filename).parent)
                if getattr(self, "filename", None)
                else "."
            )
            filters = self.tr(
                "3D sources (*.tif *.tiff *.ome.tif *.ome.tiff *.nii *.nii.gz *.dcm *.dicom *.ima *.IMA *.ply *.csv *.xyz *.stl *.STL *.obj *.OBJ *.zarr *.zarr.json *.zgroup);;All files (*.*)"
            )
            dialog = QtWidgets.QFileDialog(
                self, self.tr("Choose 3D Volume (TIFF/NIfTI/DICOM/Zarr)")
            )
            dialog.setDirectory(start_dir)
            dialog.setNameFilter(filters)
            dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
            dialog.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
            dialog.setOption(QtWidgets.QFileDialog.ReadOnly, True)
            paths: list[str] = []
            if dialog.exec_() == QtWidgets.QDialog.Accepted:
                paths = dialog.selectedFiles()
            if not paths:
                folder = QtWidgets.QFileDialog.getExistingDirectory(
                    self,
                    self.tr("Choose Volume Folder (DICOM/Zarr)"),
                    start_dir,
                    QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.ReadOnly,
                )
                if folder:
                    paths = [folder]
                else:
                    return
            if paths:

                def _normalize_volume_selection(raw: str) -> str:
                    try:
                        p = Path(raw)
                        if p.is_file():
                            if p.name.lower() == "zarr.json":
                                return str(p.parent)
                            if (p.parent / ".zarray").exists():
                                return str(p.parent)
                        cur = p
                        for _ in range(3):
                            if (
                                cur.name.lower().endswith(".zarr")
                                or (cur / ".zarray").exists()
                                or (cur / "zarr.json").exists()
                            ):
                                return str(cur)
                            cur = cur.parent
                    except Exception:
                        pass
                    return raw

                tiff_path = _normalize_volume_selection(paths[0])

        vtk_missing = False
        vtk_error = None
        try:
            from annolid.gui.widgets.vtk_volume_viewer import VTKVolumeViewerDialog  # type: ignore

            dlg = VTKVolumeViewerDialog(tiff_path, parent=self)
            dlg.setModal(False)
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
            return
        except ModuleNotFoundError as exc:
            vtk_error = exc
            vtk_missing = True
        except ImportError as exc:
            vtk_error = exc
            vtk_missing = True
        except Exception as exc:
            vtk_error = exc

        try:
            suffix = Path(tiff_path).suffix.lower() if tiff_path else ""
        except Exception:
            suffix = ""

        point_cloud_suffixes = {".ply", ".csv", ".xyz"}
        mesh_suffixes = {".stl", ".obj"}
        threejs_suffixes = point_cloud_suffixes | mesh_suffixes

        if suffix in threejs_suffixes:
            manager = getattr(self, "threejs_manager", None)
            if manager is not None:
                try:
                    if manager.show_model_in_viewer(tiff_path):
                        return
                except Exception:
                    pass

        requires_vtk = suffix in point_cloud_suffixes or suffix in mesh_suffixes

        def _vtk_available() -> tuple[bool, str | None]:
            try:
                try:
                    import vtkmodules  # noqa: F401
                except Exception:
                    import vtk  # noqa: F401

                return True, None
            except Exception as exc:
                return False, str(exc)

        _ok, _probe = _vtk_available()
        vtk_missing = not _ok

        if requires_vtk:
            if vtk_missing:
                QtWidgets.QMessageBox.information(
                    self,
                    self.tr("Mesh/Point Cloud Viewer Requires VTK"),
                    self.tr(
                        "PLY/CSV/XYZ point clouds and STL/OBJ meshes require VTK with Qt support.\n\n"
                        f"Details: {_probe or 'Unknown import error'}\n\n"
                        "Conda:  conda install -c conda-forge vtk\n"
                        "Pip:    pip install vtk"
                    ),
                )
            else:
                QtWidgets.QMessageBox.warning(
                    self,
                    self.tr("Mesh/Point Cloud Viewer"),
                    self.tr("Failed to open the VTK mesh/point cloud viewer.\n%s")
                    % (str(vtk_error) if vtk_error else ""),
                )
            return

        try:
            from annolid.gui.widgets.volume_viewer import VolumeViewerDialog

            dlg = VolumeViewerDialog(tiff_path, parent=self)
            dlg.setModal(False)
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("3D Viewer"),
                self.tr(f"Unable to open 3D viewer: {e}"),
            )
            return

        if vtk_missing:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("True 3D Rendering (Optional)"),
                self.tr(
                    "For interactive 3D volume rendering, install VTK with Qt support.\n\n"
                    "Conda:  conda install -c conda-forge vtk\n"
                    "Pip:    pip install vtk\n\n"
                    "You are currently using the built-in slice/MIP viewer."
                ),
            )

    def open_threejs_example(self, example_id: str):
        manager = getattr(self, "threejs_manager", None)
        if manager is None:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Three.js Examples"),
                self.tr("Three.js canvas is not available in this session."),
            )
            return
        try:
            base_dir = Path(tempfile.gettempdir()) / "annolid_threejs_examples"
            example_path = generate_threejs_example(example_id, base_dir)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Three.js Examples"),
                self.tr("Failed to generate example:\n%1").replace("%1", str(exc)),
            )
            return
        if not manager.show_model_in_viewer(example_path):
            # If it's an HTML file, we can try to open it directly if the manager supports it
            # or use the system browser.
            if example_path.suffix.lower() == ".html":
                from annolid.gui.widgets.threejs_viewer_server import (
                    _ensure_threejs_http_server,
                )

                base_url = _ensure_threejs_http_server()
                url = f"{base_url}/threejs/{example_path.name}"
                if hasattr(
                    manager, "show_url_in_viewer"
                ) and manager.show_url_in_viewer(url):
                    return
                import webbrowser

                webbrowser.open(url)
                return

            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Three.js Examples"),
                self.tr("Unable to open generated example in Three.js viewer."),
            )

    def _on_pca_map_started(self):
        self.statusBar().showMessage(self.tr("Computing PCA feature map…"))

    def _on_pca_map_finished(self, payload: dict) -> None:
        if not self.pca_map_action.isChecked():
            return
        overlay = payload.get("overlay_rgba")
        self.canvas.setPCAMapOverlay(overlay)
        cluster_labels = payload.get("cluster_labels") or []
        if cluster_labels:
            labels_text = ", ".join(cluster_labels)
            message = self.tr("PCA clustering ready (%s)") % labels_text
        else:
            message = self.tr("PCA feature map ready.")
        self.statusBar().showMessage(message, 4000)

    def _on_pca_map_error(self, message: str) -> None:
        self.canvas.setPCAMapOverlay(None)
        QtWidgets.QMessageBox.warning(
            self,
            self.tr("PCA Feature Map"),
            message,
        )
        self._deactivate_pca_map()

    def _open_pca_map_settings(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(self.tr("PCA Feature Map Settings"))
        layout = QtWidgets.QFormLayout(dialog)

        model_combo = QtWidgets.QComboBox(dialog)
        for cfg in PATCH_SIMILARITY_MODELS:
            model_combo.addItem(cfg.display_name, cfg.identifier)

        current_index = model_combo.findData(self.pca_map_model)
        if current_index >= 0:
            model_combo.setCurrentIndex(current_index)

        alpha_spin = QtWidgets.QDoubleSpinBox(dialog)
        alpha_spin.setRange(0.05, 1.0)
        alpha_spin.setSingleStep(0.05)
        alpha_spin.setValue(self.pca_map_alpha)

        cluster_spin = QtWidgets.QSpinBox(dialog)
        cluster_spin.setRange(0, 32)
        cluster_spin.setValue(max(0, int(getattr(self, "pca_map_clusters", 0))))

        layout.addRow(self.tr("Model"), model_combo)
        layout.addRow(self.tr("Overlay opacity"), alpha_spin)
        layout.addRow(self.tr("Cluster count"), cluster_spin)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=dialog,
        )
        layout.addRow(buttons)

        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.pca_map_model = model_combo.currentData()
            self.pca_map_alpha = alpha_spin.value()
            self.pca_map_clusters = cluster_spin.value()
            self.settings.setValue("pca_map/model", self.pca_map_model)
            self.settings.setValue("pca_map/alpha", self.pca_map_alpha)
            self.settings.setValue("pca_map/clusters", self.pca_map_clusters)
            self.statusBar().showMessage(
                self.tr("PCA feature map preferences updated."),
                3000,
            )
            if self.pca_map_action.isChecked():
                self._request_pca_map()
