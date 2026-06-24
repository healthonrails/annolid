from __future__ import annotations

import base64
import io
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from qtpy import QtCore

from annolid.gui.label_file import LabelFile
from annolid.gui.mixins.ai_mask_prompt_mixin import AiMaskPromptMixin
from annolid.gui.mixins.persistence_lifecycle_mixin import PersistenceLifecycleMixin
from annolid.gui.mixins.prediction_progress_mixin import PredictionProgressMixin
from annolid.gui.widgets.video_slider import VideoSliderMark


class _EmbedPolicyProbe(PersistenceLifecycleMixin):
    def __init__(self, config, *, video_file=None):
        self._config = config
        self.video_file = video_file


def test_video_saves_do_not_embed_duplicate_frame_data_by_default() -> None:
    window = _EmbedPolicyProbe({"store_data": True}, video_file="clip.mp4")

    assert window._should_embed_image_data(save_image_data=True) is False


def test_video_frame_embedding_can_be_explicitly_enabled() -> None:
    window = _EmbedPolicyProbe(
        {
            "store_data": True,
            "store_video_frame_data": True,
        },
        video_file="clip.mp4",
    )

    assert window._should_embed_image_data(save_image_data=True) is True


def test_existing_video_seed_image_is_not_reencoded(tmp_path: Path) -> None:
    image_path = tmp_path / "clip_000000001.png"
    image_path.write_bytes(b"existing seed")

    class _ImageData:
        def save(self, *_args, **_kwargs):
            raise AssertionError("existing video seed image must not be rewritten")

    window = SimpleNamespace(video_file="clip.mp4", imageData=_ImageData())

    result = AiMaskPromptMixin._saveImageFile(
        window,
        str(image_path.with_suffix(".json")),
    )

    assert result == str(image_path)
    assert image_path.read_bytes() == b"existing seed"


def test_empty_video_seed_image_is_repaired(tmp_path: Path) -> None:
    image_path = tmp_path / "clip_000000001.png"
    image_path.write_bytes(b"")

    class _ImageData:
        def save(self, filename):
            Path(filename).write_bytes(b"repaired seed")
            return True

    window = SimpleNamespace(video_file="clip.mp4", imageData=_ImageData())

    AiMaskPromptMixin._saveImageFile(
        window,
        str(image_path.with_suffix(".json")),
    )

    assert image_path.read_bytes() == b"repaired seed"


def test_label_file_dimension_check_reads_png_header_without_full_decode(
    monkeypatch,
) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (37, 23)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    monkeypatch.setattr(
        "annolid.gui.label_file.utils.img_b64_to_arr",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("full image decode should not be required")
        ),
    )

    assert LabelFile._check_image_height_and_width(encoded, 1, 1) == (23, 37)


class _StatsWindow(QtCore.QObject, PersistenceLifecycleMixin):
    def __init__(self) -> None:
        super().__init__()
        self.project_controller = SimpleNamespace(get_current_project_path=lambda: None)
        self.label_stats = {}


def test_label_stats_refresh_is_coalesced_in_background(
    monkeypatch,
    tmp_path: Path,
) -> None:
    started = []
    pool = SimpleNamespace(start=lambda task: started.append(task))
    monkeypatch.setattr(
        QtCore.QThreadPool,
        "globalInstance",
        staticmethod(lambda: pool),
    )
    window = _StatsWindow()
    index_file = str(tmp_path / "index.jsonl")
    index_key = str(Path(index_file).resolve())

    window._schedule_label_stats_update(index_file=index_file)
    window._schedule_label_stats_update(index_file=index_file)

    assert len(started) == 1
    assert index_key in window._label_stats_pending

    window._on_label_stats_task_finished(index_key, {"records_total": 1})

    assert len(started) == 2
    assert window.label_stats[index_key] == {"records_total": 1}


class _Seekbar:
    def __init__(self) -> None:
        self.marks = []
        self.updates = 0

    def getMarks(self, *, mark_type):
        return [mark for mark in self.marks if mark.mark_type == mark_type]

    def addMark(self, mark):
        self.marks.append(mark)

    def update(self):
        self.updates += 1


def test_manual_seed_mark_is_added_without_folder_rescan() -> None:
    seekbar = _Seekbar()
    window = SimpleNamespace(seekbar=seekbar, num_frames=100)

    PredictionProgressMixin._add_manual_seed_slider_mark(window, 42)
    PredictionProgressMixin._add_manual_seed_slider_mark(window, 42)

    assert seekbar.marks == [VideoSliderMark(mark_type="manual_seed", val=42)]
    assert seekbar.updates == 1


def test_register_saved_file_does_not_duplicate_long_video_entries() -> None:
    image_path = "/tmp/clip_000000042.png"
    added = []
    window = SimpleNamespace(
        imageList=[image_path],
        _known_file_paths={image_path},
        _addItem=lambda image, label: added.append((image, label)),
    )

    PersistenceLifecycleMixin._register_saved_file(
        window,
        image_path,
        "/tmp/clip_000000042.json",
    )

    assert window.imageList == [image_path]
    assert added == [(image_path, "/tmp/clip_000000042.json")]
