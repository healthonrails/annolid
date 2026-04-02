from __future__ import annotations

import io
import json
import os
import os.path as osp
import re
from pathlib import Path
from typing import Set

import numpy as np
from PIL import Image
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt

from annolid.annotation.timestamps import convert_frame_number_to_time
from annolid.gui.label_file import LabelFile, LabelFileError
from annolid.gui.window_base import PY2, utils
from annolid.infrastructure import AnnotationStore
from annolid.infrastructure.filesystem import (
    get_frame_number_from_json,
)
from annolid.utils.logger import logger


class PersistenceLifecycleMixin:
    """Label persistence, title/state updates, and prediction cleanup helpers."""

    _autosave_timer: QtCore.QTimer | None = None

    def _is_auto_save_enabled(self) -> bool:
        action = getattr(getattr(self, "actions", None), "saveAuto", None)
        if action is not None:
            try:
                return bool(action.isChecked())
            except Exception:
                pass
        return bool((self._config or {}).get("auto_save", False))

    def _ensure_autosave_timer(self) -> QtCore.QTimer:
        timer = getattr(self, "_autosave_timer", None)
        if timer is None:
            timer = QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._run_auto_save)
            self._autosave_timer = timer
        return timer

    def _schedule_auto_save(self) -> None:
        if not self.filename:
            return
        timer = self._ensure_autosave_timer()
        # Debounce shape edits so drag/move operations do not spam disk writes.
        timer.start(700)

    def _run_auto_save(self) -> None:
        if not self._is_auto_save_enabled() or not self.filename or not self.dirty:
            return
        try:
            self.saveFile()
        except Exception as exc:
            logger.debug("Auto-save failed: %s", exc)

    def _cancel_auto_save(self) -> None:
        timer = getattr(self, "_autosave_timer", None)
        if timer is not None:
            timer.stop()

    def _get_pil_image_from_state(self) -> Image.Image | None:
        if self.imageData is None:
            return None

        if isinstance(self.imageData, bytes):
            try:
                pil_image = Image.open(io.BytesIO(self.imageData))
            except Exception as e:
                logger.error(f"Failed to load PIL Image from bytes: {e}")
                return None
        elif isinstance(self.imageData, Image.Image):
            pil_image = self.imageData
        elif isinstance(self.imageData, QtGui.QImage):
            qimage = QtGui.QImage(self.imageData)
            image_bytes = self._qimage_to_bytes(qimage)
            if image_bytes is None:
                logger.error("Failed to serialize QImage to bytes for saving.")
                return None
            try:
                with io.BytesIO(image_bytes) as buffer:
                    pil_image = Image.open(buffer)
                    pil_image.load()
            except Exception as e:
                logger.error(f"Failed to convert QImage to PIL Image: {e}")
                return None
        else:
            # Some callers may store QImage-like wrappers. Try Qt conversion
            # without importing additional Qt binding helper modules.
            try:
                qimage = QtGui.QImage(self.imageData)
            except Exception:
                qimage = QtGui.QImage()
            if not qimage.isNull():
                image_bytes = self._qimage_to_bytes(qimage)
                if image_bytes is None:
                    logger.error("Failed to serialize QImage-like object for saving.")
                    return None
                try:
                    with io.BytesIO(image_bytes) as buffer:
                        pil_image = Image.open(buffer)
                        pil_image.load()
                except Exception as e:
                    logger.error(
                        f"Failed to convert QImage-like object to PIL Image: {e}"
                    )
                    return None
                if pil_image.mode != "RGB":
                    return pil_image.convert("RGB")
                return pil_image
            logger.warning(
                f"self.imageData is of an unexpected type ({type(self.imageData)}). "
                "Cannot convert to PIL.Image for saving."
            )
            return None

        if pil_image.mode != "RGB":
            return pil_image.convert("RGB")

        return pil_image

    def saveLabels(self, filename, save_image_data=True):
        lf = LabelFile()
        has_zone_shapes = False

        def with_manual_tag(description: str | None) -> str:
            text = (description or "").strip()
            if not text:
                return "manul"
            lowered_tokens = {token.strip().lower() for token in text.split(",")}
            if "manul" in lowered_tokens:
                return text
            return f"{text}, manul"

        def format_shape(s):
            data = s.other_data.copy()
            if s.description and "zone" in s.description.lower():
                pass
            if len(s.points) <= 1:
                s.shape_type = "point"
            data.update(
                dict(
                    label=s.label.encode("utf-8") if PY2 else s.label,
                    points=[(p.x(), p.y()) for p in s.points],
                    group_id=s.group_id,
                    shape_type=s.shape_type,
                    flags=s.flags,
                    mask=None
                    if s.mask is None
                    else utils.img_arr_to_b64(s.mask.astype(np.uint8)),
                    visible=s.visible,
                    description=with_manual_tag(s.description),
                )
            )
            return data

        shapes = [format_shape(item.shape()) for item in self.labelList]
        flags = {}
        if self.flag_widget:
            flags = {
                _flag: True
                for _flag in self.flag_widget._get_existing_flag_names()
                if self.is_behavior_active(self.frame_number, _flag)
            }

        if self.canvas.current_behavior_text is not None:
            behaviors = self.canvas.current_behavior_text.split(",")
            for behavior in behaviors:
                if len(behavior) > 0:
                    flags[behavior] = True
        try:
            imagePath = osp.relpath(self.imagePath, osp.dirname(filename))
            imageData = None
            save_embedded_image_data = bool(
                save_image_data and self._config["store_data"]
            )
            if hasattr(self, "_has_large_image_page_navigation") and bool(
                self._has_large_image_page_navigation()
            ):
                save_embedded_image_data = False
            if save_embedded_image_data:
                pil_image_to_save = self._get_pil_image_from_state()
                if pil_image_to_save:
                    imageData = utils.img_pil_to_data(pil_image_to_save)

            image_height = self.image.height()
            image_width = self.image.width()
            large_backend = getattr(self, "large_image_backend", None)
            if large_backend is not None:
                try:
                    image_width, image_height = large_backend.get_level_shape(0)
                except Exception:
                    pass

            other_data = dict(self.otherData or {})
            if hasattr(self, "_has_large_image_page_navigation") and bool(
                self._has_large_image_page_navigation()
            ):
                other_data["large_image_page"] = {
                    "page_index": int(getattr(self, "frame_number", 0) or 0),
                    "page_count": int(getattr(self, "num_frames", 1) or 1),
                    "label_path": str(filename),
                    "source_path": str(getattr(self, "imagePath", "") or ""),
                }

            if osp.dirname(filename) and not osp.exists(osp.dirname(filename)):
                os.makedirs(osp.dirname(filename))
            lf.save(
                filename=filename,
                shapes=shapes,
                imagePath=Path(imagePath).name,
                imageData=imageData,
                imageHeight=image_height,
                imageWidth=image_width,
                otherData=other_data,
                flags=flags,
                caption=self.canvas.getCaption(),
            )
            if has_zone_shapes:
                self.zone_path = filename

            self.labelFile = lf
            items = self.fileListWidget.findItems(self.imagePath, Qt.MatchExactly)
            if len(items) > 0:
                if len(items) != 1:
                    raise RuntimeError("There are duplicate files.")
                items[0].setCheckState(Qt.Checked)
                items[0].setData(Qt.UserRole, True)
                items[0].setForeground(QtGui.QBrush())
            if shapes and self.video_file:
                self.video_manager_widget.json_saved.emit(self.video_file, filename)
                logger.debug(
                    f"Emitted VideoManagerWidget.json_saved for video: {self.video_file}, JSON: {filename}"
                )
            return True
        except LabelFileError as e:
            self.errorMessage(
                self.tr("Error saving label data"), self.tr("<b>%s</b>") % e
            )
            return False

    def _saveFile(self, filename):
        json_saved = self.saveLabels(filename)
        if filename and json_saved:
            self.changed_json_stats[filename] = (
                self.changed_json_stats.get(filename, 0) + 1
            )
            if (
                self.changed_json_stats[filename] >= 1
                and self._pred_res_folder_suffix in filename
            ):
                changed_folder = str(Path(filename).parent)
                if "_edited" not in changed_folder:
                    changed_folder = Path(changed_folder + "_edited")
                else:
                    changed_folder = Path(changed_folder)
                if not changed_folder.exists():
                    changed_folder.mkdir(exist_ok=True, parents=True)
                changed_filename = changed_folder / Path(filename).name
                _ = self.saveLabels(str(changed_filename))
                _ = self._saveImageFile(str(changed_filename).replace(".json", ".png"))
            image_filename = self._saveImageFile(filename)
            index_file = self._auto_collect_labelme_pair(filename, image_filename)
            self._update_label_stats_from_index(index_file=index_file)
            self._save_ai_mask_renders(image_filename)
            self.imageList.append(image_filename)
            self.addRecentFile(filename)
            label_file = self._getLabelFile(filename)
            self._addItem(image_filename, label_file)

            if self.caption_widget is not None:
                self.caption_widget.set_image_path(image_filename)

            if self.video_results_folder:
                try:
                    self._refresh_manual_seed_slider_marks(self.video_results_folder)
                except Exception:
                    logger.debug(
                        "Failed to refresh manual seed marks after save.",
                        exc_info=True,
                    )

            self.setClean()

    def _resolve_label_index_file(self) -> str:
        index_file = os.environ.get("ANNOLID_LABEL_INDEX_FILE", "").strip()
        if not index_file:
            index_file = (self.config or {}).get(
                "label_index_file"
            ) or self.settings.value("dataset/label_index_file", "", type=str)
        dataset_root = (
            os.environ.get("ANNOLID_LABEL_COLLECTION_DIR")
            or (self.config or {}).get("label_collection_dir")
            or self.settings.value("dataset/label_collection_dir", "", type=str)
        )
        dataset_root_path = Path(dataset_root).expanduser() if dataset_root else None
        try:
            from annolid.datasets.labelme_collection import (
                default_label_index_path,
                resolve_label_index_path,
            )
        except Exception:
            if not index_file:
                if dataset_root_path is not None:
                    index_file = str(
                        (
                            dataset_root_path.resolve()
                            / "logs"
                            / "label_index"
                            / "annolid_dataset.jsonl"
                        )
                    )
                else:
                    index_file = str(
                        (
                            Path.home()
                            / ".annolid"
                            / "logs"
                            / "label_index"
                            / "annolid_dataset.jsonl"
                        )
                        .expanduser()
                        .resolve()
                    )
            elif (
                dataset_root_path is not None
                and not Path(index_file).expanduser().is_absolute()
            ):
                index_file = str((dataset_root_path.resolve() / index_file).resolve())
        else:
            if not index_file:
                index_file = str(default_label_index_path(dataset_root_path))
            else:
                index_file = str(
                    resolve_label_index_path(Path(index_file), dataset_root_path)
                )
        return index_file

    def _auto_collect_labelme_pair(self, json_path: str, image_path: str) -> str | None:
        manual_only_env = (
            os.environ.get("ANNOLID_LABEL_INDEX_MANUAL_ONLY", "1").strip().lower()
        )
        manual_only = manual_only_env not in {"0", "false", "no", "off"}
        if manual_only and not self._is_manual_label_json(Path(json_path)):
            logger.debug(
                "Skipping auto label indexing for non-manual JSON: %s", json_path
            )
            return None

        index_file = self._resolve_label_index_file()

        include_empty_value = (
            os.environ.get("ANNOLID_LABEL_INDEX_INCLUDE_EMPTY", "0").strip().lower()
        )
        include_empty = include_empty_value in {"1", "true", "yes", "on"}

        try:
            from annolid.datasets.labelme_collection import index_labelme_pair

            index_labelme_pair(
                json_path=Path(json_path),
                index_file=Path(index_file),
                image_path=Path(image_path) if image_path else None,
                include_empty=include_empty,
                source="annolid_gui",
            )
            return index_file
        except Exception as exc:
            logger.warning("Auto label indexing failed for %s: %s", json_path, exc)
            return None

    def _is_manual_label_json(self, json_path: Path) -> bool:
        """Return True when a LabelMe JSON includes explicit manual-label markers."""
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        shapes = payload.get("shapes", [])
        if not isinstance(shapes, list) or not shapes:
            return False
        for shape in shapes:
            if not isinstance(shape, dict):
                continue
            description = str(shape.get("description") or "").lower()
            # Keep backward compatibility with existing "manul" marker typo.
            if "manul" in description or "manual" in description:
                return True
        return False

    def _update_label_stats_from_index(self, *, index_file: str | None) -> None:
        if not index_file:
            return

        project_root = None
        try:
            current_project = self.project_controller.get_current_project_path()
            if current_project:
                project_root = Path(current_project).expanduser().resolve()
        except Exception:
            project_root = None

        try:
            from annolid.datasets.label_index_stats import update_label_stats_snapshot

            stats = update_label_stats_snapshot(
                index_file=Path(index_file),
                project_root=project_root,
            )
            self.label_stats[str(Path(index_file).expanduser().resolve())] = stats
        except Exception as exc:
            logger.debug("Failed to update label stats snapshot from index: %s", exc)
            stats = None

        try:
            dialog = getattr(self, "_labeling_progress_dashboard_dialog", None)
            dashboard = getattr(dialog, "dashboard", None)
            if dashboard is not None:
                if isinstance(stats, dict) and hasattr(
                    dashboard, "apply_index_stats_snapshot"
                ):
                    dashboard.apply_index_stats_snapshot(stats)
                else:
                    dashboard.refresh_stats()
        except Exception:
            logger.debug(
                "Failed to refresh labeling dashboard after save.", exc_info=True
            )

    def getLabelFile(self):
        if str(self.filename).lower().endswith(".json"):
            label_file = str(self.filename)
        else:
            label_file = osp.splitext(str(self.filename))[0] + ".json"

        return label_file

    def popLabelListMenu(self, point):
        try:
            self.menus.labelList.exec_(self.labelList.mapToGlobal(point))
        except AttributeError:
            return

    def setDirty(self):
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)
        self.dirty = True
        self.actions.save.setEnabled(True)
        title = "Annolid"
        if self.filename is not None:
            title = self.getTitle(clean=False)
        self.setWindowTitle(title)
        if self._is_auto_save_enabled():
            self._schedule_auto_save()

    def getTitle(self, clean=True):
        title = "Annolid"
        if self.caption_widget is not None:
            self.caption_widget.set_image_path(self.filename)
            if getattr(self.caption_widget, "behavior_widget", None) is not None:
                try:
                    self.caption_widget.behavior_widget.set_current_frame(
                        self.frame_number
                    )
                except Exception:
                    pass
        _filename = os.path.basename(self.filename)
        if self.video_loader:
            if self.frame_number:
                self._time_stamp = convert_frame_number_to_time(
                    self.frame_number, self.fps
                )
                if clean:
                    title = f"{title}-Video Timestamp:{self._time_stamp}|Events:{self.behavior_controller.events_count}"
                    title = f"{title}|Frame_number:{self.frame_number}"
                else:
                    title = f"{title}|Video Timestamp:{self._time_stamp}"
                    title = f"{title}|Frame_number:{self.frame_number}*"
            else:
                if clean:
                    title = "{} - {}".format(title, _filename)
                else:
                    title = "{} - {}*".format(title, _filename)
        return title

    def deleteAllFuturePredictions(self):
        if not self.video_loader or not self.video_results_folder:
            return

        prediction_folder = Path(self.video_results_folder)
        if not prediction_folder.exists():
            return

        try:
            # Deleting future predictions is an explicit "resume from here" action.
            # Keep a one-shot hint so the next prediction run starts at N+1 and
            # treats frame N as the latest seed context.
            self._prediction_forced_start_frame = max(0, int(self.frame_number) + 1)
        except Exception:
            self._prediction_forced_start_frame = None

        deleted_files = 0
        seed_frames = sorted(self._collect_seed_frames(prediction_folder))
        protected_frames: Set[int] = set(seed_frames)
        next_seed = next(
            (int(seed) for seed in seed_frames if int(seed) > int(self.frame_number)),
            None,
        )
        delete_start = int(self.frame_number) + 1
        delete_end = int(next_seed) - 1 if next_seed is not None else None

        logger.info(f"Scanning for future predictions in: {prediction_folder}")
        if next_seed is not None:
            logger.info(
                "Future manual seed detected at frame %s; deleting predicted frames in [%s, %s].",
                next_seed,
                delete_start,
                delete_end,
            )
        else:
            logger.info(
                "No future manual seed detected; deleting all predicted frames after frame %s.",
                self.frame_number,
            )

        for prediction_path in prediction_folder.iterdir():
            if not prediction_path.is_file():
                continue
            if prediction_path.suffix.lower() != ".json":
                continue

            match = re.search(r"(\d+)(?=\.json$)", prediction_path.name)
            if not match:
                logger.debug(
                    "Skipping file with unexpected name format: %s",
                    prediction_path.name,
                )
                continue

            try:
                frame_number = int(float(match.group(1)))
            except (ValueError, IndexError):
                logger.warning(
                    "Could not parse frame number from file: %s",
                    prediction_path.name,
                )
                continue

            if frame_number <= int(self.frame_number):
                continue
            if next_seed is not None and frame_number >= int(next_seed):
                continue
            if frame_number in protected_frames:
                continue
            if frame_number < int(delete_start):
                continue
            if delete_end is not None and frame_number > int(delete_end):
                continue

            try:
                prediction_path.unlink()
                deleted_files += 1
            except OSError as e:
                logger.error("Failed to delete file %s: %s", prediction_path, e)

        store_removed = 0
        try:
            store = AnnotationStore.for_frame_path(
                prediction_folder / f"{prediction_folder.name}_000000000.json"
            )
            if delete_end is None:
                store_removed = store.remove_frames_after(
                    int(self.frame_number), protected_frames=protected_frames
                )
            else:
                store_removed = store.remove_frames_in_range(
                    int(delete_start),
                    int(delete_end),
                    protected_frames=protected_frames,
                )
        except Exception as exc:
            logger.error(
                "Failed to prune annotation store in %s: %s",
                prediction_folder,
                exc,
            )

        if deleted_files or store_removed:
            logger.info(
                "%s future prediction JSON(s) removed and %s store record(s) pruned.",
                deleted_files,
                store_removed,
            )
            if self.seekbar:
                self.seekbar.removeMarksByType("predicted")
                self.seekbar.removeMarksByType("predicted_existing")
                self.seekbar.removeMarksByType("prediction_progress")
            self.last_known_predicted_frame = -1
            self.prediction_start_timestamp = 0.0
            if hasattr(self, "_update_progress_bar"):
                self._update_progress_bar(0)
            try:
                self._scan_prediction_folder(str(prediction_folder))
            except Exception as exc:
                logger.debug(
                    "Failed to rescan prediction folder after deletion: %s", exc
                )
            try:
                if delete_end is None:
                    self._prune_tracking_stats_frames(
                        prediction_folder,
                        start_frame=int(delete_start),
                        end_frame=None,
                        protected_frames=protected_frames,
                    )
                else:
                    self._prune_tracking_stats_frames(
                        prediction_folder,
                        start_frame=int(delete_start),
                        end_frame=int(delete_end),
                        protected_frames=protected_frames,
                    )
            except Exception:
                logger.debug(
                    "Failed to prune tracking stats after deleting future predictions.",
                    exc_info=True,
                )
            try:
                refresh_missing = getattr(
                    self,
                    "_refresh_missing_instance_slider_marks_from_tracking_stats",
                    None,
                )
                if callable(refresh_missing):
                    refresh_missing(prediction_folder)
            except Exception:
                logger.debug(
                    "Failed to refresh missing-instance slider marks after deletion.",
                    exc_info=True,
                )
        else:
            logger.info("No future prediction files or store records required removal.")

    def _collect_seed_frames(self, prediction_folder: Path) -> Set[int]:
        seed_frames: Set[int] = set()
        for path in prediction_folder.glob("*.json"):
            if not path.is_file():
                continue
            has_sidecar = any(
                path.with_suffix(ext).exists() for ext in (".png", ".jpg", ".jpeg")
            )
            if not has_sidecar:
                continue
            try:
                seed_frames.add(int(get_frame_number_from_json(path.name)))
            except Exception:
                continue
        return seed_frames

    def deletePredictionsFromSeedToNext(self):
        if not self.video_loader or not self.video_results_folder:
            return False, None, None

        prediction_folder = Path(self.video_results_folder)
        if not prediction_folder.exists():
            return False, None, None

        seed_frames = sorted(self._collect_seed_frames(prediction_folder))
        protected_frames: Set[int] = set(seed_frames)

        current_seed = None
        if seed_frames:
            for seed in seed_frames:
                if seed <= self.frame_number:
                    current_seed = seed
                else:
                    break
        if current_seed is None:
            current_seed = self.frame_number

        next_seed = None
        for seed in seed_frames:
            if seed > current_seed:
                next_seed = seed
                break

        if next_seed is not None and next_seed - 1 < current_seed:
            return False, current_seed, next_seed

        deleted_files = 0
        logger.info(
            "Deleting predictions from seed frame %s to %s.",
            current_seed,
            next_seed if next_seed is not None else "end",
        )

        for prediction_path in prediction_folder.iterdir():
            if not prediction_path.is_file():
                continue
            if prediction_path.suffix.lower() != ".json":
                continue

            match = re.search(r"(\d+)(?=\.json$)", prediction_path.name)
            if not match:
                continue

            try:
                frame_number = int(float(match.group(1)))
            except (ValueError, IndexError):
                continue

            if frame_number < current_seed:
                continue
            if next_seed is not None and frame_number >= next_seed:
                continue
            if frame_number in protected_frames:
                continue

            try:
                prediction_path.unlink()
                deleted_files += 1
            except OSError as e:
                logger.error("Failed to delete file %s: %s", prediction_path, e)

        store_removed = 0
        try:
            store = AnnotationStore.for_frame_path(
                prediction_folder / f"{prediction_folder.name}_000000000.json"
            )
            store_removed = store.remove_frames_in_range(
                current_seed,
                next_seed - 1 if next_seed is not None else None,
                protected_frames=protected_frames,
            )
        except Exception as exc:
            logger.error(
                "Failed to prune annotation store in %s: %s",
                prediction_folder,
                exc,
            )

        if deleted_files or store_removed:
            logger.info(
                "%s prediction JSON(s) removed and %s store record(s) pruned.",
                deleted_files,
                store_removed,
            )
            if self.seekbar:
                self.seekbar.removeMarksByType("predicted")
                self.seekbar.removeMarksByType("predicted_existing")
                self.seekbar.removeMarksByType("prediction_progress")
            self.last_known_predicted_frame = -1
            self.prediction_start_timestamp = 0.0
            if hasattr(self, "_update_progress_bar"):
                self._update_progress_bar(0)
            try:
                self._scan_prediction_folder(str(prediction_folder))
            except Exception as exc:
                logger.debug(
                    "Failed to rescan prediction folder after deletion: %s", exc
                )
            try:
                self._prune_tracking_stats_frames(
                    prediction_folder,
                    start_frame=int(current_seed),
                    end_frame=(int(next_seed) - 1) if next_seed is not None else None,
                    protected_frames=protected_frames,
                )
            except Exception:
                logger.debug(
                    "Failed to prune tracking stats after deleting prediction range.",
                    exc_info=True,
                )
            try:
                refresh_missing = getattr(
                    self,
                    "_refresh_missing_instance_slider_marks_from_tracking_stats",
                    None,
                )
                if callable(refresh_missing):
                    refresh_missing(prediction_folder)
            except Exception:
                logger.debug(
                    "Failed to refresh missing-instance slider marks after range deletion.",
                    exc_info=True,
                )
            # Hint the next prediction run to restart from the frame after this
            # seed so tracking uses the updated seed as context.
            try:
                self._prediction_forced_start_frame = max(0, int(current_seed) + 1)
            except Exception:
                self._prediction_forced_start_frame = None
        else:
            logger.info(
                "No predicted files required removal for the current seed range."
            )

        return bool(deleted_files or store_removed), current_seed, next_seed

    @staticmethod
    def _frame_in_prune_window(
        frame: int,
        *,
        start_frame: int,
        end_frame: int | None,
        protected_frames: Set[int],
    ) -> bool:
        if frame in protected_frames:
            return False
        if frame < int(start_frame):
            return False
        if end_frame is not None and frame > int(end_frame):
            return False
        return True

    def _prune_tracking_stats_frames(
        self,
        prediction_folder: Path,
        *,
        start_frame: int,
        end_frame: int | None,
        protected_frames: Set[int],
    ) -> None:
        stats_path = prediction_folder / f"{prediction_folder.name}_tracking_stats.json"
        if not stats_path.exists():
            return
        try:
            with stats_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh) or {}
        except Exception:
            logger.debug(
                "Failed to load tracking stats for prune: %s",
                stats_path,
                exc_info=True,
            )
            return

        changed = False
        frame_stats = payload.get("frame_stats", {})
        if isinstance(frame_stats, dict):
            pruned: dict[str, dict] = {}
            for frame_key, entry in frame_stats.items():
                if not isinstance(entry, dict):
                    continue
                try:
                    frame_idx = int(frame_key)
                except (TypeError, ValueError):
                    pruned[str(frame_key)] = entry
                    continue
                if self._frame_in_prune_window(
                    frame_idx,
                    start_frame=int(start_frame),
                    end_frame=end_frame,
                    protected_frames=protected_frames,
                ):
                    changed = True
                    continue
                pruned[str(frame_key)] = entry
            payload["frame_stats"] = pruned

        bad_shape_events = payload.get("bad_shape_events", [])
        if isinstance(bad_shape_events, list):
            kept_events = []
            for event in bad_shape_events:
                if not isinstance(event, dict):
                    continue
                frame = event.get("frame")
                try:
                    frame_idx = int(frame)
                except (TypeError, ValueError):
                    kept_events.append(event)
                    continue
                if self._frame_in_prune_window(
                    frame_idx,
                    start_frame=int(start_frame),
                    end_frame=end_frame,
                    protected_frames=protected_frames,
                ):
                    changed = True
                    continue
                kept_events.append(event)
            payload["bad_shape_events"] = kept_events

        prediction_segments = payload.get("prediction_segments", [])
        if isinstance(prediction_segments, list):
            kept_segments = []
            range_end = int(end_frame) if end_frame is not None else None
            for segment in prediction_segments:
                if not isinstance(segment, dict):
                    continue
                try:
                    seg_start = int(segment.get("start_frame"))
                    seg_end = int(segment.get("end_frame"))
                except (TypeError, ValueError):
                    kept_segments.append(segment)
                    continue
                overlap = seg_end >= int(start_frame) and (
                    range_end is None or seg_start <= range_end
                )
                if overlap:
                    changed = True
                    continue
                kept_segments.append(segment)
            payload["prediction_segments"] = kept_segments

        if not changed:
            return

        self._recompute_pruned_tracking_stats_summary(payload)
        payload["updated_at"] = QtCore.QDateTime.currentDateTimeUtc().toString(
            QtCore.Qt.ISODate
        )
        try:
            with stats_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
        except Exception:
            logger.debug(
                "Failed to persist pruned tracking stats: %s",
                stats_path,
                exc_info=True,
            )

    @staticmethod
    def _recompute_pruned_tracking_stats_summary(payload: dict) -> None:
        frame_stats = payload.get("frame_stats", {})
        manual_frames: Set[int] = set()
        bad_shape_frames: Set[int] = set()
        bad_shape_failed_frames: Set[int] = set()
        missing_instance_frames: Set[int] = set()
        if isinstance(frame_stats, dict):
            for frame_key, entry in frame_stats.items():
                if not isinstance(entry, dict):
                    continue
                try:
                    frame_idx = int(frame_key)
                except (TypeError, ValueError):
                    continue
                sources = entry.get("sources", [])
                source_set = (
                    {str(source) for source in sources}
                    if isinstance(sources, list)
                    else set()
                )
                if "manual_seed" in source_set or "json" in source_set:
                    manual_frames.add(frame_idx)
                if int(entry.get("bad_shape_count", 0) or 0) > 0:
                    bad_shape_frames.add(frame_idx)
                if int(entry.get("bad_shape_failed_count", 0) or 0) > 0:
                    bad_shape_failed_frames.add(frame_idx)
                if int(entry.get("missing_instance_count", 0) or 0) > 0:
                    missing_instance_frames.add(frame_idx)
        abnormal_segment_events = 0
        prediction_segments = payload.get("prediction_segments", [])
        if isinstance(prediction_segments, list):
            abnormal_segment_events = len(
                [
                    seg
                    for seg in prediction_segments
                    if isinstance(seg, dict)
                    and str(seg.get("status", "")) != "processed"
                ]
            )
        payload["summary"] = {
            "manual_frames": int(len(manual_frames)),
            "manual_segments": [],
            "bad_shape_frames": int(len(bad_shape_frames)),
            "bad_shape_failed_frames": int(len(bad_shape_failed_frames)),
            "missing_instance_frames": int(len(missing_instance_frames)),
            "abnormal_segment_events": int(abnormal_segment_events),
        }

    def deleteFile(self):
        mb = QtWidgets.QMessageBox
        msg = self.tr(
            "You are about to permanently delete this label file, "
            "or delete predicted labels after the current frame. "
            "What would you like to do?"
        )
        msg_box = mb(self)
        msg_box.setIcon(mb.Warning)
        msg_box.setText(msg)
        msg_box.setInformativeText(
            self.tr(
                "Yes: delete the current label file. "
                "Yes to All: delete predicted frames from the next frame onward."
            )
        )
        msg_box.setStandardButtons(mb.No | mb.Yes | mb.YesToAll)
        msg_box.setDefaultButton(mb.No)
        answer = msg_box.exec_()

        if answer == mb.No:
            return
        elif answer == mb.YesToAll:
            self.deleteAllFuturePredictions()
        else:
            label_file = self.getLabelFile()
            if osp.exists(label_file):
                os.remove(label_file)
                img_file = label_file.replace(".json", ".png")
                if osp.exists(img_file):
                    os.remove(img_file)
                logger.info("Label file is removed: {}".format(label_file))

                item = self.fileListWidget.currentItem()
                if item:
                    item.setData(Qt.UserRole, False)
                    item.setForeground(QtGui.QBrush(QtGui.QColor(160, 160, 160)))

                self.resetState()

    def setClean(self):
        self._cancel_auto_save()
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.createMode.setEnabled(True)
        self.actions.createRectangleMode.setEnabled(True)
        self.actions.createCircleMode.setEnabled(True)
        self.actions.createLineMode.setEnabled(True)
        self.actions.createPointMode.setEnabled(True)
        self.actions.createLineStripMode.setEnabled(True)
        title = "Annolid"
        if self.filename is not None:
            title = self.getTitle()
        self.setWindowTitle(title)

        if self.hasLabelFile():
            self.actions.deleteFile.setEnabled(True)
        else:
            self.actions.deleteFile.setEnabled(False)

    def _on_auto_save_toggled(self, enabled: bool) -> None:
        self._config["auto_save"] = bool(enabled)
        try:
            self.settings.setValue("app/auto_save", bool(enabled))
        except Exception:
            pass
        if bool(enabled):
            self._schedule_auto_save()
            self.statusBar().showMessage("Auto Save enabled", 2500)
        else:
            self._cancel_auto_save()
            self.statusBar().showMessage("Auto Save disabled", 2500)

    def save_labels(self):
        file_name, extension = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save labels file",
            str(self.here.parent / "annotation"),
            filter="*.txt",
        )
        _ = extension
        if len(file_name) < 1:
            return

        if Path(file_name).is_file() or Path(file_name).parent.is_dir():
            labels_text_list = ["__ignore__", "_background_"]
            for i in range(self.uniqLabelList.count()):
                label_name = self.uniqLabelList.item(i).data(QtCore.Qt.UserRole)
                labels_text_list.append(label_name)

            with open(file_name, "w") as lt:
                for ltl in labels_text_list:
                    lt.writelines(str(ltl) + "\n")
