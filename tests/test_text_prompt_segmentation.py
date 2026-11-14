from __future__ import annotations

from collections import OrderedDict
import os
import sys
import threading
import time
import types
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from qtpy import QtCore, QtGui, QtWidgets
from shapely.geometry import Polygon

from annolid.core.agent.gui_backend.tool_handlers_chat_controls import (
    set_ai_text_prompt_tool,
)
from annolid.detector.grounding_dino import GroundingDINO
from annolid.gui.mixins.core_interaction_mixin import CoreInteractionMixin
from annolid.gui.shape import Shape
from annolid.gui.widgets.canvas import Canvas
from annolid.gui.widgets.sam3_manager import Sam3Manager
from annolid.gui.widgets.text_prompt import AiRectangleWidget
from annolid.segmentation.SAM.sam3.prompt_builder import (
    normalize_text_prompt,
    split_text_prompts,
)
from annolid.segmentation.SAM.sam_hq.sam_hq import SamHQSegmenter
from annolid.segmentation.SAM.segment_anything import (
    _select_best_mask_index,
    _select_best_polygon,
)
from annolid.utils.prompts import (
    DetectionBoxCandidate,
    assign_instance_labels,
    format_grounding_dino_caption,
    is_sliver_mask_remnant,
    match_predicted_phrase_to_label,
    parse_every_prompt,
    parse_text_prompt,
    sanitize_and_nms_box_candidates,
)
from annolid.vision.florence_2 import Florence2Predictor, create_shapes_from_mask_dict


def _ensure_qapp() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_parse_text_prompt_supports_multiclass_aliases_and_case_dedup() -> None:
    prompt = (
        "black mouse -> MouseBlack, white mouse = MouseWhite; "
        "BLACK MOUSE -> duplicate\nwater spout . 0.5g pellet=Pellet"
    )
    parsed = parse_text_prompt(prompt)

    assert parsed.queries == (
        "black mouse",
        "white mouse",
        "water spout",
        "0.5g pellet",
    )
    assert parsed.labels == (
        "MouseBlack",
        "MouseWhite",
        "water spout",
        "Pellet",
    )
    # SAM3 receives clean visual queries without alias suffixes
    assert split_text_prompts(prompt) == [
        "black mouse",
        "white mouse",
        "water spout",
        "0.5g pellet",
    ]
    assert normalize_text_prompt("Mouse, mouse; pup=baby_mouse") == "Mouse, pup"


def test_parse_every_prompt_preserves_digits_in_target_label() -> None:
    assert parse_every_prompt("every 48 mouse_1") == (48, "mouse_1")
    assert parse_every_prompt("24 every cell_group2") == (24, "cell_group2")
    assert parse_every_prompt("every 999 nuclei") == (32, "nuclei")
    assert parse_every_prompt("every = 16 -> droplet_3") == (16, "droplet_3")
    assert parse_every_prompt("every") == (32, "object")
    assert parse_every_prompt("everyday mouse") is None


def test_grounding_dino_caption_and_phrase_matching() -> None:
    prompt = "Black Mouse -> MouseBlack, White Mouse = MouseWhite, food pellet"
    assert (
        format_grounding_dino_caption(prompt)
        == "black mouse . white mouse . food pellet ."
    )

    parsed = parse_text_prompt(prompt)
    assert match_predicted_phrase_to_label("black mouse", parsed) == "MouseBlack"
    assert match_predicted_phrase_to_label("white", parsed) == "MouseWhite"
    assert match_predicted_phrase_to_label("pellet", parsed) == "food pellet"

    single = parse_text_prompt("spotted rodent -> TargetAnimal")
    assert match_predicted_phrase_to_label("rodent", single) == "TargetAnimal"


def test_sanitize_and_nms_box_candidates_deduplicates_and_clamps() -> None:
    candidates = [
        # Duplicate lower-confidence box for same class -> suppressed
        DetectionBoxCandidate(bbox=(10, 10, 50, 50), label="mouse", score=0.55),
        # Higher-confidence primary box with out-of-bounds / inverted coords -> clamped & kept
        DetectionBoxCandidate(bbox=(52, 52, 8, -5), label="mouse", score=0.91),
        # Near-identical cross-label hallucination (IoU > 0.88) -> suppressed
        DetectionBoxCandidate(bbox=(9, 2, 51, 51), label="animal", score=0.60),
        # Moderately overlapping distinct class (e.g. mouse on wheel, IoU < 0.88) -> kept
        DetectionBoxCandidate(bbox=(25, 25, 80, 80), label="wheel", score=0.80),
        # Degenerate tiny box -> dropped
        DetectionBoxCandidate(bbox=(70, 70, 71, 71), label="mouse", score=0.99),
    ]

    kept = sanitize_and_nms_box_candidates(
        candidates,
        image_shape=(100, 100),
        same_label_iou_threshold=0.50,
        cross_label_iou_threshold=0.88,
    )

    assert [(c.label, c.score, c.bbox) for c in kept] == [
        ("mouse", 0.91, (8.0, 0.0, 52.0, 52.0)),
        ("wheel", 0.80, (25.0, 25.0, 80.0, 80.0)),
    ]


def test_assign_instance_labels_and_sliver_filter() -> None:
    assert assign_instance_labels(["Mouse", "Pup", "Mouse"], use_group_id=False) == [
        ("Mouse_0", None),
        ("Pup_0", None),
        ("Mouse_1", None),
    ]
    assert assign_instance_labels(["Mouse", "Pup", "Mouse"], use_group_id=True) == [
        ("Mouse", 1),
        ("Pup", 1),
        ("Mouse", 2),
    ]

    orig = np.zeros((40, 40), dtype=bool)
    orig[5:35, 5:35] = True  # 900 px
    sliver = np.zeros((40, 40), dtype=bool)
    sliver[5:6, 5:15] = True  # 10 px (< 12% of 900 px)
    assert is_sliver_mask_remnant(sliver, orig) is True

    small_valid = np.zeros((40, 40), dtype=bool)
    small_valid[10:14, 10:14] = True  # 16 px unclipped small object
    assert is_sliver_mask_remnant(small_valid, small_valid) is False


def test_grounding_dino_caches_tokenized_captions_and_maps_aliases() -> None:
    gd = GroundingDINO.__new__(GroundingDINO)
    gd.box_threshold = 0.30
    gd.text_threshold = 0.20
    gd.target_size = (64, 64)
    gd._text_cache_max_entries = 8
    from collections import OrderedDict

    gd._text_inputs_cache = OrderedDict()

    encode_calls = {"count": 0}

    class _FakeTokenizer:
        def encode(self, text: str):
            encode_calls["count"] += 1
            # Simulate [CLS] black mouse . white mouse . [SEP]
            ids = [101, 2001, 2002, 1012, 2003, 2002, 1012, 102]
            return SimpleNamespace(
                ids=ids,
                type_ids=[0] * len(ids),
                attention_mask=[1] * len(ids),
            )

        def decode(self, token_ids):
            if 2001 in token_ids:
                return "black mouse"
            if 2003 in token_ids:
                return "white mouse"
            return "mouse"

    logits = np.full((1, 3, 8), -10.0, dtype=np.float32)
    logits[0, 0, [1, 2]] = 2.0  # sigmoid ~ 0.88 -> black mouse
    logits[0, 1, [1, 2]] = 1.2  # sigmoid ~ 0.77 -> duplicate black mouse
    logits[0, 2, [4, 5]] = (
        0.0  # sigmoid = 0.50 -> white mouse (passes 0.45, fails 0.60)
    )
    boxes = np.array(
        [
            [
                [0.25, 0.25, 0.20, 0.20],
                [0.26, 0.26, 0.20, 0.20],
                [0.75, 0.75, 0.20, 0.20],
            ]
        ],
        dtype=np.float32,
    )

    gd.net = SimpleNamespace(
        max_text_len=256,
        tokenizer=_FakeTokenizer(),
        get_onnx_inference=lambda *_args, **_kwargs: (logits, boxes),
    )

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    prompt = "black mouse -> MouseBlack, white mouse -> MouseWhite"

    high_thresh = gd.predict_detections(image, prompt, box_threshold=0.60)
    assert [c.label for c in high_thresh] == ["MouseBlack"]
    assert gd.box_threshold == 0.30  # default not mutated

    low_thresh = gd.predict_detections(image, prompt, box_threshold=0.45)
    assert [c.label for c in low_thresh] == ["MouseBlack", "MouseWhite"]
    # Tokenizer ran only once across both frames because caption was cached!
    assert encode_calls["count"] == 1

    # Empty filtering safety
    none_found = gd.predict_detections(image, prompt, box_threshold=0.99)
    assert none_found == []


def test_sam_hq_segmenter_caches_image_embeddings_and_batches_boxes() -> None:
    seg = SamHQSegmenter.__new__(SamHQSegmenter)
    seg._DEFAULT_BOX_BATCH_SIZE = 3
    seg._cached_image_signature = None

    calls = {"set_image": 0, "predict_batches": []}

    class _FakeTransform:
        @staticmethod
        def apply_boxes_torch(boxes_tensor, _shape):
            return boxes_tensor

    class _FakePredictor:
        device = "cpu"
        is_image_set = True
        transform = _FakeTransform()

        def set_image(self, _image):
            calls["set_image"] += 1

        def predict_torch(self, *, boxes, **_kwargs):
            import torch

            n = int(boxes.shape[0])
            calls["predict_batches"].append(n)
            masks = torch.ones((n, 1, 16, 16), dtype=torch.bool)
            scores = torch.linspace(0.9, 0.5, steps=n).unsqueeze(1)
            return masks, scores, None

    seg.predictor = _FakePredictor()
    frame_a = np.zeros((16, 16, 3), dtype=np.uint8)
    boxes = [[1, 1, 8, 8] for _ in range(7)]

    masks1, scores1, out_boxes1 = seg.segment_objects(frame_a, boxes)
    assert masks1.shape == (7, 16, 16)
    assert scores1.shape == (7,)
    assert out_boxes1.shape == (7, 4)
    assert calls["set_image"] == 1
    assert calls["predict_batches"] == [3, 3, 1]

    # Second prompt on identical frame skips ViT image encoding!
    seg.segment_objects(frame_a, boxes[:2])
    assert calls["set_image"] == 1

    # Changing pixel content triggers fresh encoding
    frame_b = frame_a.copy()
    frame_b[0, 0, 0] = 255
    seg.segment_objects(frame_b, boxes[:1])
    assert calls["set_image"] == 2


def test_sam_onnx_box_prompt_selection_prefers_tight_object_over_background() -> None:
    # Candidate 0: bloated background mask covering the whole frame (including box corners)
    # Candidate 1: tight circular object mask inside box [20, 20, 60, 60] (corners are False)
    masks = np.zeros((1, 2, 80, 80), dtype=np.float32) - 1.0
    masks[0, 0, :, :] = 1.0
    yy, xx = np.ogrid[:80, :80]
    circle = (xx - 40) ** 2 + (yy - 40) ** 2 <= 15**2
    masks[0, 1, circle] = 1.0

    chosen = _select_best_mask_index(
        masks=masks,
        scores=np.array([[0.85, 0.88]], dtype=np.float32),
        points=np.array([[20.0, 20.0], [60.0, 60.0]], dtype=np.float32),
        point_labels=np.array([2, 3], dtype=np.int32),
    )
    assert chosen == 1

    tiny_speckle = [22, 22, 24, 22, 24, 24, 22, 24]
    main_animal = [25, 25, 55, 25, 55, 55, 25, 55]
    best_poly = _select_best_polygon(
        [tiny_speckle, main_animal],
        points=[[20.0, 20.0], [60.0, 60.0]],
        point_labels=[2, 3],
    )
    assert best_poly is not None
    assert len(best_poly) == 8
    assert float(best_poly[2]) == 55.0


def test_florence2_multiclass_alias_prompt_and_disjoint_polygons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import annolid.vision.florence_2 as fl_mod

    seen_queries: list[str] = []

    def _fake_florence2(_proc, _model, task, _image, text_input=None):
        seen_queries.append(str(text_input))
        if text_input == "black mouse":
            poly = [[10, 10, 50, 10, 50, 50, 10, 50]]
        else:
            poly = [[40, 40, 80, 40, 80, 80, 40, 80]]
        return {task: {"polygons": [poly], "labels": [text_input]}}

    monkeypatch.setattr(fl_mod, "florence2", _fake_florence2)
    predictor = Florence2Predictor.__new__(Florence2Predictor)
    predictor.model_name = "microsoft/Florence-2-large"
    predictor._model = SimpleNamespace()
    predictor._processor = SimpleNamespace()
    predictor._device = "cpu"
    predictor._dtype = None
    predictor.ensure_loaded = lambda: None  # type: ignore[method-assign]

    res = predictor.predict(
        image=SimpleNamespace(),
        text_input="black mouse -> MouseBlack, white mouse -> MouseWhite",
        include_caption=False,
    )
    assert seen_queries == ["black mouse", "white mouse"]
    assert res.mask_dict["labels"] == ["MouseBlack", "MouseWhite"]

    shapes = create_shapes_from_mask_dict(res.mask_dict)
    assert [s.label for s in shapes] == ["MouseBlack", "MouseWhite"]
    g0 = Polygon([(p.x(), p.y()) for p in shapes[0].points])
    g1 = Polygon([(p.x(), p.y()) for p in shapes[1].points])
    assert g0.disjoint(g1)


def test_ai_rectangle_widget_controls_history_and_signals() -> None:
    _ensure_qapp()
    widget = AiRectangleWidget()
    try:
        long_prompt = (
            "black mouse -> MouseBlack, white mouse -> MouseWhite, "
            "food pellet -> Pellet, water spout -> Spout"
        )
        assert len(long_prompt) > 50
        widget.set_prompt_text(long_prompt)
        assert widget.prompt_text() == long_prompt

        widget.set_output_mode("box")
        assert widget.output_mode() == "rectangle"
        widget.set_output_mode("mask")
        assert widget.output_mode() == "mask"
        widget.set_box_threshold(0.42)
        assert widget.box_threshold() == pytest.approx(0.42)
        widget.set_replace_existing(True)
        assert widget.replace_existing() is True
        widget.set_use_countgd(True)
        assert widget.use_countgd() is True

        widget.set_completion_suggestions(["MouseBlack", "MouseWhite", "mouseblack"])
        widget.record_prompt("black mouse -> MouseBlack")
        widget.record_prompt("food pellet")
        completions = widget._completer_model.stringList()
        assert completions[:2] == ["food pellet", "black mouse -> MouseBlack"]
        assert "MouseWhite" in completions

        fired = {"count": 0}
        stopped = {"count": 0}
        widget.runRequested.connect(
            lambda: fired.__setitem__("count", fired["count"] + 1)
        )
        widget.stopRequested.connect(
            lambda: stopped.__setitem__("count", stopped["count"] + 1)
        )
        widget._runPromptButton.click()
        widget._aiRectanglePrompt.returnPressed.emit()
        assert fired["count"] == 2
        assert stopped["count"] == 0

        widget.set_running(True)
        assert widget.is_running() is True
        assert widget._runPromptButton.text() == "Stop"
        widget._runPromptButton.click()
        assert stopped["count"] == 1
        assert fired["count"] == 2

        widget.set_running(False)
        assert widget.is_running() is False
        assert widget._runPromptButton.text() == "Go"
    finally:
        widget.close()


def test_canvas_predict_ai_rectangle_multiclass_modes_fallback_replace_and_atomic_undo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ensure_qapp()
    import sys

    class _FakeMultiDetector:
        def predict_detections(
            self, _image, prompt, *, box_threshold=None, nms_iou_threshold=0.50
        ):
            _ = (prompt, box_threshold, nms_iou_threshold)
            return [
                DetectionBoxCandidate(
                    bbox=(10, 10, 46, 46),
                    label="MouseBlack",
                    score=0.92,
                    source="grounding_dino",
                ),
                DetectionBoxCandidate(
                    bbox=(12, 12, 45, 45),
                    label="MouseBlack",
                    score=0.61,
                    source="grounding_dino",
                ),
                DetectionBoxCandidate(
                    bbox=(54, 54, 90, 90),
                    label="MouseWhite",
                    score=0.87,
                    source="grounding_dino",
                ),
            ]

    canvas = Canvas()
    try:
        image = QtGui.QImage(100, 100, QtGui.QImage.Format_RGB32)
        image.fill(QtGui.QColor(30, 40, 50))
        canvas.loadPixmap(QtGui.QPixmap.fromImage(image), clear_shapes=False)
        canvas.createMode = "grounding_sam"
        canvas._ai_model_rect = _FakeMultiDetector()

        # Add one manual polygon that must NEVER be removed by replace_existing=True
        manual = Shape(label="arena_zone", shape_type="polygon", description="manual")
        manual.points = [
            QtCore.QPointF(1, 1),
            QtCore.QPointF(98, 1),
            QtCore.QPointF(98, 98),
            QtCore.QPointF(1, 98),
        ]
        manual.close()
        canvas.loadShapes([manual], replace=True)

        # Simulate SAM-HQ optional package not installed -> built-in AI model fallback!
        monkeypatch.setitem(sys.modules, "annolid.segmentation.SAM.sam_hq", None)

        class _FakeBuiltinBoxSAM:
            name = "FakeBuiltinSAM"

            def set_image(self, image):
                _ = image
                return None

            def predict_mask_from_points(self, points, point_labels):
                assert point_labels == [2, 3]
                (x1, y1), (x2, y2) = points
                m = np.zeros((100, 100), dtype=bool)
                m[int(y1) : int(y2), int(x1) : int(x2)] = True
                return m

        canvas._ai_model = _FakeBuiltinBoxSAM()

        created = canvas.predictAiRectangle(
            "black mouse -> MouseBlack, white mouse -> MouseWhite",
            use_group_id=True,
        )
        assert created == 2
        assert len(canvas.shapes) == 3  # 1 manual + 2 deduplicated prompt polygons
        prompt_shapes = [s for s in canvas.shapes if s.description == "grounding_sam"]
        assert [(s.label, s.group_id, s.shape_type) for s in prompt_shapes] == [
            ("MouseBlack", 1, "polygon"),
            ("MouseWhite", 1, "polygon"),
        ]

        # Single atomic Undo removes BOTH generated shapes in one step while keeping manual!
        assert canvas.isShapeRestorable is True
        canvas.restoreShape()
        assert [s.label for s in canvas.shapes] == ["arena_zone"]

        # Output mode = rectangle skips SAM completely, and replace_existing swaps cleanly
        created_boxes = canvas.predictAiRectangle(
            "black mouse -> MouseBlack, white mouse -> MouseWhite",
            output_mode="rectangle",
        )
        assert created_boxes == 2
        assert [
            (s.label, s.shape_type)
            for s in canvas.shapes
            if s.description == "grounding_sam"
        ] == [
            ("MouseBlack_0", "rectangle"),
            ("MouseWhite_0", "rectangle"),
        ]

        # Re-run with output_mode="mask" and replace_existing=True -> replaces prior boxes!
        created_masks = canvas.predictAiRectangle(
            "black mouse -> MouseBlack, white mouse -> MouseWhite",
            output_mode="mask",
            replace_existing=True,
        )
        assert created_masks == 2
        assert len(canvas.shapes) == 3
        assert [
            (s.label, s.shape_type)
            for s in canvas.shapes
            if s.description == "grounding_sam"
        ] == [
            ("MouseBlack_0", "mask"),
            ("MouseWhite_0", "mask"),
        ]
    finally:
        canvas.close()


def test_core_interaction_grounding_sam_preserves_alias_case_and_handles_flags() -> (
    None
):
    _ensure_qapp()
    captured: dict[str, object] = {}
    applied_flags: list[dict[str, bool]] = []
    cleared = {"count": 0}
    messages: list[str] = []

    class _Harness(CoreInteractionMixin):
        def __init__(self) -> None:
            self.aiRectangle = AiRectangleWidget()
            self.uniqLabelList = QtWidgets.QListWidget()
            item = QtWidgets.QListWidgetItem("MouseBlack")
            item.setData(QtCore.Qt.UserRole, "MouseBlack")
            self.uniqLabelList.addItem(item)
            self.flags_controller = SimpleNamespace(
                apply_prompt_flags=lambda f: applied_flags.append(dict(f)),
                clear_flags=lambda: cleared.__setitem__("count", cleared["count"] + 1),
            )
            self.canvas = SimpleNamespace(
                predictAiRectangle=lambda prompt, **kwargs: (
                    captured.update({"prompt": prompt, **kwargs}) or 2
                )
            )

        def toggleDrawMode(self, _edit: bool, createMode: str = "") -> None:
            captured["mode"] = createMode

        def tr(self, text: str) -> str:
            return text

        def status(self, msg: str, _timeout: int = 5000) -> None:
            messages.append(msg)

    harness = _Harness()
    try:
        harness.aiRectangle.set_prompt_text("Black Mouse -> MouseBlack")
        harness.aiRectangle.set_box_threshold(0.35)
        harness.aiRectangle.set_output_mode("box")
        harness.aiRectangle.set_replace_existing(True)

        count = harness._grounding_sam()
        assert count == 2
        assert captured["prompt"] == "Black Mouse -> MouseBlack"
        assert captured["box_threshold"] == pytest.approx(0.35)
        assert captured["output_mode"] == "rectangle"
        assert captured["replace_existing"] is True
        assert any("Segmented 2 shape(s)" in m for m in messages)

        # Case-insensitive FLAGS: handling
        harness.aiRectangle.set_prompt_text("FLAGS: grooming, rearing")
        harness._grounding_sam()
        assert applied_flags == [{"grooming": False, "rearing": False}]

        harness.aiRectangle.set_prompt_text("flags:   ")
        harness._grounding_sam()
        assert cleared["count"] == 1
    finally:
        harness.aiRectangle.close()


def test_set_ai_text_prompt_tool_supports_extended_options_and_validates() -> None:
    calls: list[tuple[str, object]] = []

    legacy = set_ai_text_prompt_tool(
        "mouse",
        use_countgd=True,
        invoke_set_ai_text_prompt=lambda text, flag: calls.append(
            ("base", (text, flag))
        )
        is None,
    )
    assert legacy == {
        "ok": True,
        "queued": True,
        "text": "mouse",
        "use_countgd": True,
    }

    extended = set_ai_text_prompt_tool(
        "black mouse=MouseBlack, white mouse=MouseWhite",
        use_countgd=False,
        box_threshold=0.25,
        output_mode="box",
        replace_existing=True,
        invoke_set_ai_text_prompt=lambda text, flag: calls.append(
            ("base", (text, flag))
        )
        is None,
        invoke_configure_ai_text_prompt=lambda opts: calls.append(("opts", dict(opts)))
        is None,
    )
    assert extended["ok"] is True
    assert extended["output_mode"] == "rectangle"
    assert extended["box_threshold"] == pytest.approx(0.25)
    assert extended["replace_existing"] is True

    bad_thresh = set_ai_text_prompt_tool(
        "mouse",
        use_countgd=False,
        box_threshold=1.5,
        invoke_set_ai_text_prompt=lambda *_a: True,
    )
    assert bad_thresh["ok"] is False
    assert "box_threshold" in str(bad_thresh["error"])


def test_canvas_caches_countgd_counter_and_deduplicates_with_grounding_dino(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ensure_qapp()
    import sys
    import types

    init_calls = {"count": 0}
    count_queries: list[str] = []

    class _FakeObjectCounter:
        def __init__(self) -> None:
            init_calls["count"] += 1

        def count_objects(
            self,
            _pil_img,
            *,
            text_prompt,
            exemplar_image=None,
            exemplar_boxes=None,
            confidence_threshold=0.23,
        ):
            _ = (exemplar_image, exemplar_boxes, confidence_threshold)
            count_queries.append(str(text_prompt))
            # Overlaps heavily with GroundingDINO's first box + adds a second distinct object
            return [[10, 10, 40, 40], [55, 55, 85, 85]]

    fake_cgd_mod = types.ModuleType("annolid.detector.countgd.predict")
    fake_cgd_mod.ObjectCounter = _FakeObjectCounter  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "annolid.detector.countgd.predict", fake_cgd_mod)

    class _FakeGD:
        def predict_detections(self, _img, _prompt, **_kwargs):
            return [
                DetectionBoxCandidate(
                    bbox=(11, 11, 41, 41),
                    label="Cell",
                    score=0.90,
                    source="grounding_dino",
                )
            ]

    canvas = Canvas()
    try:
        image = QtGui.QImage(100, 100, QtGui.QImage.Format_RGB32)
        image.fill(QtGui.QColor(20, 20, 20))
        canvas.loadPixmap(QtGui.QPixmap.fromImage(image), clear_shapes=False)
        canvas.createMode = "grounding_sam"
        canvas._ai_model_rect = _FakeGD()

        c1 = canvas.predictAiRectangle(
            "round cell -> Cell",
            use_countgd=True,
            output_mode="rectangle",
        )
        c2 = canvas.predictAiRectangle(
            "round cell -> Cell",
            use_countgd=True,
            output_mode="rectangle",
            replace_existing=True,
        )
        # Duplicate [10,10,40,40] + [11,11,41,41] merged into 1 box + [55,55,85,85] = 2 shapes!
        assert c1 == 2
        assert c2 == 2
        # ObjectCounter initialized only ONCE across repeated CountGD runs!
        assert init_calls["count"] == 1
        assert count_queries == ["round cell", "round cell"]
        assert [s.label for s in canvas.shapes] == ["Cell_0", "Cell_1"]
    finally:
        canvas.close()
        assert canvas._countgd_counter is None


def _spin_until(predicate, timeout_s: float = 3.0) -> bool:
    app = _ensure_qapp()
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        app.processEvents()
        if predicate():
            app.processEvents()
            return True
        time.sleep(0.005)
    app.processEvents()
    return bool(predicate())


def test_interactive_async_text_prompt_runs_off_ui_thread_and_updates_shapes() -> None:
    _ensure_qapp()
    main_tid = threading.get_ident()
    entered = threading.Event()
    release = threading.Event()
    worker_tids: list[int] = []
    messages: list[str] = []

    class _GateDetector:
        def predict_detections(self, _image, _prompt, **_kwargs):
            worker_tids.append(threading.get_ident())
            entered.set()
            release.wait(timeout=2.0)
            return [
                DetectionBoxCandidate(
                    bbox=(15, 15, 65, 65),
                    label="MouseBlack",
                    score=0.93,
                    source="grounding_dino",
                )
            ]

    class _FastSegmenter:
        def segment_objects(self, _image, bboxes, *, stop_event=None):
            _ = stop_event
            m = np.zeros((len(bboxes), 80, 80), dtype=bool)
            m[0, 15:65, 15:65] = True
            return m, np.array([0.95], dtype=np.float32), np.asarray(bboxes)

    class _InteractiveWindow(CoreInteractionMixin):
        def __init__(self) -> None:
            self.aiRectangle = AiRectangleWidget()
            self.uniqLabelList = QtWidgets.QListWidget()
            self.flags_controller = SimpleNamespace(
                apply_prompt_flags=lambda _f: None,
                clear_flags=lambda: None,
            )
            self.canvas = Canvas()
            img = QtGui.QImage(80, 80, QtGui.QImage.Format_RGB32)
            img.fill(QtGui.QColor(25, 35, 45))
            self.canvas.loadPixmap(QtGui.QPixmap.fromImage(img), clear_shapes=False)
            self.canvas._ai_model_rect = _GateDetector()
            self.canvas.sam_hq_model = _FastSegmenter()
            self.aiRectangle.runRequested.connect(self._grounding_sam_interactive)
            self.aiRectangle.stopRequested.connect(self._cancel_grounding_sam)

        def toggleDrawMode(self, _edit: bool, createMode: str = "") -> None:
            self.canvas.createMode = createMode

        def tr(self, text: str) -> str:
            return text

        def status(self, msg: str, _timeout: int = 5000) -> None:
            messages.append(msg)

    win = _InteractiveWindow()
    try:
        win.aiRectangle.set_prompt_text("black mouse -> MouseBlack")
        win.aiRectangle._runPromptButton.click()

        assert entered.wait(timeout=2.0) is True
        assert worker_tids and worker_tids[0] != main_tid
        assert win.aiRectangle.is_running() is True
        assert win.aiRectangle._runPromptButton.text() == "Stop"
        # Widget-scoped busy cursor is used; global QApplication override stack stays clean
        assert QtWidgets.QApplication.overrideCursor() is None
        win.canvas.restoreCursor()
        assert win.canvas.cursor().shape() == QtCore.Qt.WaitCursor

        release.set()
        assert _spin_until(lambda: not win.canvas.is_text_prompt_running()) is True
        assert _spin_until(lambda: not win.canvas._inflight_prompt_workers) is True

        assert win.aiRectangle.is_running() is False
        assert win.aiRectangle._runPromptButton.text() == "Go"
        assert [s.label for s in win.canvas.shapes] == ["MouseBlack_0"]
        assert win.canvas.isShapeRestorable is True
        assert any("Segmented 1 shape(s)" in m for m in messages)
    finally:
        release.set()
        win.canvas.close()
        win.aiRectangle.close()


def test_interactive_async_text_prompt_stop_button_cancels_cleanly() -> None:
    _ensure_qapp()
    entered = threading.Event()
    release = threading.Event()
    messages: list[str] = []

    class _SlowDetector:
        def predict_detections(self, _image, _prompt, **_kwargs):
            entered.set()
            release.wait(timeout=2.0)
            return [
                DetectionBoxCandidate(
                    bbox=(10, 10, 50, 50),
                    label="Mouse",
                    score=0.90,
                    source="grounding_dino",
                )
            ]

    class _InteractiveWindow(CoreInteractionMixin):
        def __init__(self) -> None:
            self.aiRectangle = AiRectangleWidget()
            self.uniqLabelList = QtWidgets.QListWidget()
            self.flags_controller = SimpleNamespace()
            self.canvas = Canvas()
            img = QtGui.QImage(80, 80, QtGui.QImage.Format_RGB32)
            img.fill(QtGui.QColor(40, 40, 40))
            self.canvas.loadPixmap(QtGui.QPixmap.fromImage(img), clear_shapes=False)
            self.canvas._ai_model_rect = _SlowDetector()
            self.aiRectangle.runRequested.connect(self._grounding_sam_interactive)
            self.aiRectangle.stopRequested.connect(self._cancel_grounding_sam)

        def toggleDrawMode(self, _edit: bool, createMode: str = "") -> None:
            self.canvas.createMode = createMode

        def tr(self, text: str) -> str:
            return text

        def status(self, msg: str, _timeout: int = 5000) -> None:
            messages.append(msg)

    win = _InteractiveWindow()
    try:
        win.aiRectangle.set_prompt_text("mouse")
        win.aiRectangle.set_output_mode("box")
        win.aiRectangle._runPromptButton.click()
        assert entered.wait(timeout=2.0) is True
        assert win.aiRectangle._runPromptButton.text() == "Stop"

        # Clicking the same button while running triggers cooperative cancellation
        win.aiRectangle._runPromptButton.click()
        assert win.aiRectangle.is_running() is False
        assert win.aiRectangle._runPromptButton.text() == "Go"

        release.set()
        assert _spin_until(lambda: not win.canvas._inflight_prompt_workers) is True
        assert win.canvas.shapes == []
        assert any("Cancelled text-prompt segmentation" in m for m in messages)
    finally:
        release.set()
        win.canvas.close()
        win.aiRectangle.close()


def test_async_text_prompt_discards_stale_results_on_frame_switch() -> None:
    _ensure_qapp()
    entered = threading.Event()
    release = threading.Event()
    outcomes: list[tuple[str, int, str | None]] = []

    class _SlowDetector:
        def predict_detections(self, _image, _prompt, **_kwargs):
            entered.set()
            release.wait(timeout=2.0)
            return [
                DetectionBoxCandidate(
                    bbox=(10, 10, 50, 50),
                    label="Mouse",
                    score=0.95,
                    source="grounding_dino",
                )
            ]

    canvas = Canvas()
    try:
        frame_a = QtGui.QImage(80, 80, QtGui.QImage.Format_RGB32)
        frame_a.fill(QtGui.QColor(10, 20, 30))
        canvas.filename = "video.mp4"
        canvas.frame_number = 0
        canvas.loadPixmap(QtGui.QPixmap.fromImage(frame_a), clear_shapes=True)
        canvas._ai_model_rect = _SlowDetector()

        started = canvas.start_async_text_prompt(
            "mouse",
            output_mode="rectangle",
            on_finished=lambda kind, cnt, _el, detail: outcomes.append(
                (kind, cnt, detail)
            ),
        )
        assert started is True
        assert entered.wait(timeout=2.0) is True

        # Switch to Frame B while Frame A inference is still in flight
        frame_b = QtGui.QImage(80, 80, QtGui.QImage.Format_RGB32)
        frame_b.fill(QtGui.QColor(200, 180, 160))
        canvas.frame_number = 1
        canvas.loadPixmap(QtGui.QPixmap.fromImage(frame_b), clear_shapes=True)

        release.set()
        assert _spin_until(lambda: not canvas._inflight_prompt_workers) is True
        # Frame B must never receive Frame A's shapes!
        assert canvas.shapes == []
        assert outcomes == [("cancelled", 0, "frame_changed")]

        # Also test late frame-number drift AFTER worker finishes computing before queued slot runs
        outcomes.clear()
        done_computing = threading.Event()

        class _FastDetector:
            def predict_detections(self, _image, _prompt, **_kwargs):
                res = [
                    DetectionBoxCandidate(
                        bbox=(12, 12, 48, 48),
                        label="Mouse",
                        score=0.91,
                        source="grounding_dino",
                    )
                ]
                done_computing.set()
                return res

        canvas._ai_model_rect = _FastDetector()
        assert (
            canvas.start_async_text_prompt(
                "mouse",
                output_mode="rectangle",
                on_finished=lambda kind, cnt, _el, detail: outcomes.append(
                    (kind, cnt, detail)
                ),
            )
            is True
        )
        assert done_computing.wait(timeout=2.0) is True
        # Mutate frame identity before processing the queued UI-thread completion signal
        canvas.frame_number = 2
        canvas._invalidate_ai_model_image_signature_cache()

        assert _spin_until(lambda: len(outcomes) == 1) is True
        assert _spin_until(lambda: not canvas._inflight_prompt_workers) is True
        assert outcomes == [("stale", 0, "frame_changed")]
        assert canvas.shapes == []
    finally:
        release.set()
        canvas.close()


def test_async_text_prompt_error_recovery_and_close_event_waits_for_worker() -> None:
    _ensure_qapp()
    messages: list[str] = []

    class _BrokenDetector:
        def predict_detections(self, _image, _prompt, **_kwargs):
            raise RuntimeError("ONNX session corrupted")

    class _InteractiveWindow(CoreInteractionMixin):
        def __init__(self) -> None:
            self.aiRectangle = AiRectangleWidget()
            self.uniqLabelList = QtWidgets.QListWidget()
            self.flags_controller = SimpleNamespace()
            self.canvas = Canvas()
            img = QtGui.QImage(64, 64, QtGui.QImage.Format_RGB32)
            img.fill(QtGui.QColor(50, 50, 50))
            self.canvas.loadPixmap(QtGui.QPixmap.fromImage(img), clear_shapes=False)
            self.canvas._ai_model_rect = _BrokenDetector()
            self.aiRectangle.runRequested.connect(self._grounding_sam_interactive)
            self.aiRectangle.stopRequested.connect(self._cancel_grounding_sam)

        def toggleDrawMode(self, _edit: bool, createMode: str = "") -> None:
            self.canvas.createMode = createMode

        def tr(self, text: str) -> str:
            return text

        def status(self, msg: str, _timeout: int = 5000) -> None:
            messages.append(msg)

    win = _InteractiveWindow()
    try:
        win.aiRectangle.set_prompt_text("mouse")
        win.aiRectangle._runPromptButton.click()
        assert _spin_until(lambda: not win.canvas.is_text_prompt_running()) is True
        assert _spin_until(lambda: not win.canvas._inflight_prompt_workers) is True
        assert win.aiRectangle.is_running() is False
        assert win.aiRectangle._runPromptButton.text() == "Go"
        assert any("ONNX session corrupted" in m for m in messages)
    finally:
        win.canvas.close()
        win.aiRectangle.close()

    # Verify closeEvent waits for an active worker before releasing models
    order: list[str] = []
    in_worker = threading.Event()

    class _ClosableDetector:
        def predict_detections(self, _image, _prompt, **_kwargs):
            in_worker.set()
            time.sleep(0.05)
            order.append("worker_done")
            return []

        def close(self) -> None:
            order.append("detector_closed")

    canvas2 = Canvas()
    img2 = QtGui.QImage(64, 64, QtGui.QImage.Format_RGB32)
    img2.fill(QtGui.QColor(30, 30, 30))
    canvas2.loadPixmap(QtGui.QPixmap.fromImage(img2), clear_shapes=False)
    canvas2._ai_model_rect = _ClosableDetector()
    assert canvas2.start_async_text_prompt("mouse", output_mode="rectangle") is True
    assert in_worker.wait(timeout=2.0) is True

    canvas2.close()
    assert order == ["worker_done", "detector_closed"]
    assert canvas2._ai_model_rect is None


def test_onnx_cancellable_run_triggers_run_options_terminate_immediately() -> None:
    from annolid.utils.cancellation import (
        InferenceCancelledError,
        ObservableStopEvent,
        onnx_cancellable_run,
    )

    entered = threading.Event()
    terminated = threading.Event()

    class _BlockingOrtSession:
        def run(self, _output_names, _input_feed, run_options=None):
            assert run_options is not None
            entered.set()
            deadline = time.perf_counter() + 2.0
            while time.perf_counter() < deadline:
                if getattr(run_options, "terminate", False):
                    terminated.set()
                    raise RuntimeError(
                        "[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Exiting due to terminate flag"
                    )
                time.sleep(0.002)
            return [np.zeros((1, 4), dtype=np.float32)]

    stop_event = ObservableStopEvent()
    session = _BlockingOrtSession()
    caught: list[BaseException] = []

    def _worker() -> None:
        try:
            onnx_cancellable_run(
                session,
                ["boxes"],
                {"img": np.zeros((1, 3, 8, 8), dtype=np.float32)},
                stop_event=stop_event,
            )
        except BaseException as exc:
            caught.append(exc)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    assert entered.wait(timeout=1.0) is True
    stop_event.set()
    t.join(timeout=1.0)
    assert not t.is_alive()
    assert terminated.is_set() is True
    assert len(caught) == 1
    assert isinstance(caught[0], InferenceCancelledError)
    # Listener must be unsubscribed after run exits
    assert stop_event._listeners == []

    # GroundingDINO forwards stop_event and returns [] cleanly when cancelled mid-inference
    gd = GroundingDINO.__new__(GroundingDINO)
    gd.box_threshold = 0.3
    gd.text_threshold = 0.25
    gd.max_Prompt_len = 256
    gd.input_shape = (16, 16)
    gd._text_cache = OrderedDict()
    gd._text_cache_lock = threading.Lock()
    gd._text_cache_max_entries = 8
    gd.preprocess = lambda _img, cap: (
        np.zeros((1, 3, 16, 16), dtype=np.float32),
        {"img": np.zeros((1, 3, 16, 16), dtype=np.float32)},
        str(cap),
    )

    seen_stop_events: list[object] = []

    class _CancellingNet:
        def get_onnx_inference(
            self, _blob, *, input_data=None, extract=False, stop_event=None
        ):
            _ = (input_data, extract)
            seen_stop_events.append(stop_event)
            raise InferenceCancelledError("aborted mid-graph")

    gd.net = _CancellingNet()
    ev = ObservableStopEvent()
    out = gd.predict_detections(
        np.zeros((16, 16, 3), dtype=np.uint8),
        "mouse",
        stop_event=ev,
    )
    assert out == []
    assert seen_stop_events == [ev]


def test_torch_cancellable_forward_aborts_mid_module_and_cleans_up_hooks() -> None:
    import torch
    from annolid.segmentation.SAM.sam_hq.sam_hq import SamHQSegmenter
    from annolid.utils.cancellation import (
        InferenceCancelledError,
        ObservableStopEvent,
        torch_cancellable_forward,
    )

    stop_event = ObservableStopEvent()
    block2_calls = {"count": 0}

    class _Block1(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            stop_event.set()
            return x + 1.0

    class _Block2(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            block2_calls["count"] += 1
            return x * 2.0

    net = torch.nn.Sequential(_Block1(), _Block2())
    with pytest.raises(InferenceCancelledError):
        with torch.inference_mode(), torch_cancellable_forward(net, stop_event):
            _ = net(torch.zeros((1, 4), dtype=torch.float32))

    assert block2_calls["count"] == 0
    for mod in net.modules():
        assert len(mod._forward_pre_hooks) == 0

    # Verify SamHQSegmenter._ensure_image_encoded rolls back partial state on mid-encode cancel
    reset_calls = {"count": 0}
    cancel_ev = ObservableStopEvent()

    class _EncodeModule(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            cancel_ev.set()
            return x

    enc_mod = _EncodeModule()
    sam_root = torch.nn.Sequential(enc_mod, torch.nn.Identity())

    class _Predictor:
        def set_image(self, _img: np.ndarray) -> None:
            _ = sam_root(torch.zeros((1, 2), dtype=torch.float32))

        def reset_image(self) -> None:
            reset_calls["count"] += 1

    seg = SamHQSegmenter.__new__(SamHQSegmenter)
    seg.sam = sam_root
    seg.predictor = _Predictor()
    seg._state_lock = threading.RLock()
    seg._cached_image_signature = None

    frame = np.zeros((24, 24, 3), dtype=np.uint8)
    masks, scores, boxes = seg.segment_objects(
        frame, [[2, 2, 18, 18]], stop_event=cancel_ev
    )
    assert len(masks) == 0 and len(scores) == 0 and len(boxes) == 0
    assert seg._cached_image_signature is None
    assert reset_calls["count"] == 1


def test_segment_anything_compact_cache_key_supersede_cancel_and_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from annolid.segmentation.SAM import segment_anything as sam_mod
    from annolid.segmentation.SAM.segment_anything import SegmentAnythingModel

    frame1 = np.zeros((32, 48, 3), dtype=np.uint8)
    frame2 = np.ones((32, 48, 3), dtype=np.uint8) * 127

    key1 = SegmentAnythingModel._compute_image_key(frame1)
    key2 = SegmentAnythingModel._compute_image_key(frame2)
    assert key1 != key2
    assert isinstance(key1, tuple) and len(key1) == 3
    assert not any(isinstance(part, (bytes, bytearray, np.ndarray)) for part in key1)

    entered_first = threading.Event()
    first_stop_seen = threading.Event()

    model = SegmentAnythingModel.__new__(SegmentAnythingModel)
    model._image_size = 64
    model._encoder_session = object()
    model._decoder_session = object()
    model._lock = threading.Lock()
    model._image_embedding_cache = OrderedDict()
    model._max_cached_images = 4
    model._thread = None
    model._active_embed_stop_event = None
    model._pending_image_key = None
    model._pending_image = None

    def _fake_compute(image_size, encoder_session, image, *, stop_event=None):
        _ = (image_size, encoder_session)
        if int(image[0, 0, 0]) == 0:
            entered_first.set()
            deadline = time.perf_counter() + 2.0
            while time.perf_counter() < deadline:
                if stop_event is not None and stop_event.is_set():
                    first_stop_seen.set()
                    from annolid.utils.cancellation import InferenceCancelledError

                    raise InferenceCancelledError("superseded")
                time.sleep(0.002)
        return np.ones((1, 4), dtype=np.float32)

    monkeypatch.setattr(sam_mod, "_compute_image_embedding", _fake_compute)
    model.set_image(frame1)
    assert entered_first.wait(timeout=1.0) is True

    # Superseding with frame2 immediately signals frame1's ObservableStopEvent
    model.set_image(frame2)
    assert first_stop_seen.wait(timeout=1.0) is True
    emb2 = model._get_image_embedding()
    assert emb2.shape == (1, 4)
    assert key1 not in model._image_embedding_cache
    assert key2 in model._image_embedding_cache

    model.close()
    assert model._image_embedding_cache == OrderedDict()
    assert model._encoder_session is None
    assert model._decoder_session is None


def test_canvas_mid_inference_cancel_propagates_observable_event_and_deferred_close_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ensure_qapp()
    from annolid.utils.cancellation import InferenceCancelledError

    entered = threading.Event()
    aborted_via_listener = threading.Event()

    class _OrtLikeDetector:
        def predict_detections(self, _image, _prompt, *, stop_event=None, **_kwargs):
            assert stop_event is not None
            abort_flag = {"terminate": False}
            unsub = stop_event.add_listener(
                lambda: abort_flag.__setitem__("terminate", True)
            )
            try:
                entered.set()
                deadline = time.perf_counter() + 3.0
                while time.perf_counter() < deadline:
                    if abort_flag["terminate"]:
                        aborted_via_listener.set()
                        raise InferenceCancelledError("ORT terminated")
                    time.sleep(0.002)
                return []
            finally:
                unsub()

    canvas = Canvas()
    try:
        img = QtGui.QImage(64, 64, QtGui.QImage.Format_RGB32)
        img.fill(QtGui.QColor(20, 30, 40))
        canvas.loadPixmap(QtGui.QPixmap.fromImage(img), clear_shapes=False)
        canvas._ai_model_rect = _OrtLikeDetector()

        outcomes: list[str] = []
        assert (
            canvas.start_async_text_prompt(
                "mouse",
                output_mode="rectangle",
                on_finished=lambda kind, _c, _e, _d: outcomes.append(kind),
            )
            is True
        )
        assert entered.wait(timeout=1.0) is True
        canvas.cancel_async_text_prompt(wait_ms=500, reason="user_stop")
        assert aborted_via_listener.is_set() is True
        assert _spin_until(lambda: not canvas._inflight_prompt_workers) is True
        assert outcomes == ["cancelled"]
    finally:
        canvas.close()

    # Verify deferred model release when closeEvent times out waiting on an uninterruptible lock holder
    worker_in_lock = threading.Event()
    unblock_worker = threading.Event()
    closed_events: list[str] = []

    class _StubbornCustomDetector:
        def predict_detections(self, _image, _prompt, **_kwargs):
            worker_in_lock.set()
            unblock_worker.wait(timeout=2.0)
            return []

        def close(self) -> None:
            closed_events.append("closed")

    canvas_deferred = Canvas()
    img2 = QtGui.QImage(64, 64, QtGui.QImage.Format_RGB32)
    img2.fill(QtGui.QColor(15, 15, 15))
    canvas_deferred.loadPixmap(QtGui.QPixmap.fromImage(img2), clear_shapes=False)
    canvas_deferred._ai_model_rect = _StubbornCustomDetector()

    orig_cancel = canvas_deferred.cancel_async_text_prompt
    monkeypatch.setattr(
        canvas_deferred,
        "cancel_async_text_prompt",
        lambda **_kw: orig_cancel(wait_ms=10, reason="closing", notify_callback=False),
    )

    class _ShortTimeoutLockProxy:
        def __init__(self, real_lock: threading.RLock) -> None:
            self._real = real_lock

        def acquire(self, blocking: bool = True, timeout: float = -1):
            if timeout is not None and timeout > 0:
                return self._real.acquire(timeout=0.01)
            return self._real.acquire(blocking=blocking)

        def release(self) -> None:
            self._real.release()

        def __enter__(self):
            return self._real.__enter__()

        def __exit__(self, exc_type, exc, tb):
            return self._real.__exit__(exc_type, exc, tb)

    canvas_deferred._prompt_inference_lock = _ShortTimeoutLockProxy(  # type: ignore[assignment]
        canvas_deferred._prompt_inference_lock
    )

    assert (
        canvas_deferred.start_async_text_prompt("mouse", output_mode="rectangle")
        is True
    )
    assert worker_in_lock.wait(timeout=1.0) is True

    # closeEvent returns promptly with deferred flag armed while worker still holds lock
    canvas_deferred.close()
    assert canvas_deferred._pending_close_model_release is True
    assert closed_events == []

    # Once worker finishes, its finally block deterministically closes & clears models
    unblock_worker.set()
    assert _spin_until(lambda: closed_events == ["closed"]) is True
    assert canvas_deferred._ai_model_rect is None
    assert canvas_deferred._pending_close_model_release is False


def test_countgd_object_counter_ignores_host_cli_argv_and_supports_stop_and_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib
    import sys
    import types
    import torch
    from PIL import Image
    from annolid.utils.cancellation import ObservableStopEvent

    stop_event = ObservableStopEvent()
    second_stage_ran = {"ran": False}

    class _Stage1(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            stop_event.set()
            return x

    class _Stage2(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            second_stage_ran["ran"] = True
            return x

    class _DummyCountGDNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stage1 = _Stage1()
            self.stage2 = _Stage2()

        def forward(self, *_args, **_kwargs):
            t = self.stage1(torch.zeros((1, 2), dtype=torch.float32))
            _ = self.stage2(t)
            return {}

    fake_slconfig_mod = types.ModuleType("annolid.detector.countgd.util.slconfig")
    fake_slconfig_mod.SLConfig = SimpleNamespace(  # type: ignore[attr-defined]
        fromfile=lambda _p: SimpleNamespace(
            merge_from_dict=lambda _d: None,
            _cfg_dict=SimpleNamespace(to_dict=lambda: {"modelname": "_unit_test_cgd"}),
        )
    )
    reg_dict = {"_unit_test_cgd": lambda _args: (_DummyCountGDNet(), None, None)}
    fake_models_pkg = types.ModuleType("annolid.detector.countgd.models")
    fake_models_pkg.__path__ = []  # type: ignore[attr-defined]
    fake_reg_mod = types.ModuleType("annolid.detector.countgd.models.registry")
    fake_reg_mod.MODULE_BUILD_FUNCS = SimpleNamespace(  # type: ignore[attr-defined]
        _module_dict=reg_dict,
        get=lambda name: reg_dict[name],
    )

    monkeypatch.setitem(
        sys.modules, "gdown", SimpleNamespace(cached_download=lambda *_a, **_k: None)
    )
    monkeypatch.setitem(
        sys.modules, "annolid.detector.countgd.util.slconfig", fake_slconfig_mod
    )
    monkeypatch.setitem(sys.modules, "annolid.detector.countgd.models", fake_models_pkg)
    monkeypatch.setitem(
        sys.modules, "annolid.detector.countgd.models.registry", fake_reg_mod
    )
    sys.modules.pop("annolid.detector.countgd.predict", None)
    cgd_predict = importlib.import_module("annolid.detector.countgd.predict")

    monkeypatch.setattr(
        sys,
        "argv",
        ["pytest", "-k", "countgd", "--unrecognized-host-flag=123"],
    )
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_a, **_kw: {"model": {}},
    )

    counter = cgd_predict.ObjectCounter.__new__(cgd_predict.ObjectCounter)
    counter.here = "/tmp"
    counter.device = "cpu"
    counter.model = counter._load_model("dummy.pth", "dummy_cfg.py", "cpu")
    counter.transform = lambda _pil, target: (torch.zeros((3, 16, 16)), target)

    pil_img = Image.new("RGB", (32, 32))
    boxes = counter.count_objects(pil_img, "mouse", stop_event=stop_event)
    assert boxes == []
    assert second_stage_ran["ran"] is False

    counter.close()
    assert counter.model is None
    assert counter.transform is None
    sys.modules.pop("annolid.detector.countgd.predict", None)


def test_cancellable_download_file_aborts_mid_stream_cleans_part_and_falls_back_on_readonly(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib
    import urllib.request
    from annolid.utils.cancellation import (
        InferenceCancelledError,
        ObservableStopEvent,
        cancellable_download_file,
    )

    # 1. Mid-stream abort via ObservableStopEvent closes active stream immediately and removes .part file
    entered_second_read = threading.Event()
    stream_closed = threading.Event()
    stop_event = ObservableStopEvent()

    class _StreamingResponse:
        def __init__(self) -> None:
            self.headers = {"Content-Length": "131072"}
            self._calls = 0

        def read(self, _n: int = 65536) -> bytes:
            self._calls += 1
            if self._calls == 1:
                return b"A" * 65536
            entered_second_read.set()
            deadline = time.perf_counter() + 2.0
            while time.perf_counter() < deadline:
                if stream_closed.is_set():
                    raise OSError("socket closed due to cancellation")
                time.sleep(0.002)
            return b"B" * 65536

        def close(self) -> None:
            stream_closed.set()

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda _req, timeout=30.0: _StreamingResponse(),
    )

    dest1 = tmp_path / "weights" / "model.onnx"
    caught: list[BaseException] = []

    def _dl_worker() -> None:
        try:
            cancellable_download_file(
                "https://example.invalid/model.onnx",
                dest1,
                stop_event=stop_event,
            )
        except BaseException as exc:
            caught.append(exc)

    dl_thread = threading.Thread(target=_dl_worker, daemon=True)
    dl_thread.start()
    assert entered_second_read.wait(timeout=1.0) is True
    stop_event.set()
    dl_thread.join(timeout=1.0)
    assert not dl_thread.is_alive()
    assert stream_closed.is_set() is True
    assert len(caught) == 1 and isinstance(caught[0], InferenceCancelledError)
    assert not dest1.exists()
    assert list(dest1.parent.glob("*.part")) == []

    # 2. Checksum mismatch deletes .part and never creates destination file
    payload = b"valid-model-weights-payload"
    valid_md5 = hashlib.md5(payload).hexdigest()

    class _StaticResponse:
        def __init__(self, data: bytes, declared_len: int | None = None) -> None:
            self._buf = memoryview(data)
            self._pos = 0
            self.headers = {
                "Content-Length": str(
                    declared_len if declared_len is not None else len(data)
                )
            }

        def read(self, n: int = 65536) -> bytes:
            if self._pos >= len(self._buf):
                return b""
            end = min(len(self._buf), self._pos + n)
            out = bytes(self._buf[self._pos : end])
            self._pos = end
            return out

        def close(self) -> None:
            pass

    urlopen_calls = {"count": 0}

    def _mock_static_urlopen(_req, timeout=30.0):
        _ = timeout
        urlopen_calls["count"] += 1
        return _StaticResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", _mock_static_urlopen)
    dest_bad = tmp_path / "weights" / "bad.onnx"
    with pytest.raises(RuntimeError, match="MD5 checksum mismatch"):
        cancellable_download_file(
            "https://example.invalid/bad.onnx",
            dest_bad,
            md5="0" * 32,
        )
    assert not dest_bad.exists()
    assert list(dest_bad.parent.glob("*.part")) == []

    # 3. Read-only preferred directory transparently falls back to ANNOLID_CACHE_DIR and caches across calls
    readonly_dir = tmp_path / "readonly_site_packages"
    readonly_dir.mkdir(parents=True, exist_ok=True)
    readonly_dir.chmod(0o555)
    cache_root = tmp_path / "user_cache"
    monkeypatch.setenv("ANNOLID_CACHE_DIR", str(cache_root))
    urlopen_calls["count"] = 0
    try:
        resolved = cancellable_download_file(
            "https://example.invalid/gd.onnx",
            readonly_dir / "gd.onnx",
            md5=valid_md5,
            cache_subdir="grounding_dino",
        )
        expected_cached = cache_root / "grounding_dino" / "gd.onnx"
        assert resolved == str(expected_cached)
        assert expected_cached.read_bytes() == payload
        assert urlopen_calls["count"] == 1

        # Second call hits fallback cache with zero network requests
        resolved_again = cancellable_download_file(
            "https://example.invalid/gd.onnx",
            readonly_dir / "gd.onnx",
            md5=valid_md5,
            cache_subdir="grounding_dino",
        )
        assert resolved_again == str(expected_cached)
        assert urlopen_calls["count"] == 1
    finally:
        readonly_dir.chmod(0o755)


def test_lock_free_model_init_allows_instant_canvas_close_and_closes_late_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ensure_qapp()
    import gc
    import annolid.detector.grounding_dino as gd_mod
    from annolid.gui.widgets import canvas as canvas_mod

    in_ctor = threading.Event()
    release_ctor = threading.Event()
    closed_instances: list[str] = []

    class _SlowInitGroundingDINO:
        def __init__(self, *, stop_event=None) -> None:
            self.seen_stop = stop_event
            in_ctor.set()
            release_ctor.wait(timeout=2.0)

        def predict_detections(self, *_a, **_kw):
            return []

        def close(self) -> None:
            closed_instances.append("closed")

    monkeypatch.setattr(gd_mod, "GroundingDINO", _SlowInitGroundingDINO)

    canvas = Canvas()
    img = QtGui.QImage(64, 64, QtGui.QImage.Format_RGB32)
    img.fill(QtGui.QColor(10, 20, 30))
    canvas.loadPixmap(QtGui.QPixmap.fromImage(img), clear_shapes=False)

    orig_cancel = canvas.cancel_async_text_prompt
    monkeypatch.setattr(
        canvas,
        "cancel_async_text_prompt",
        lambda **_kw: orig_cancel(wait_ms=15, reason="closing", notify_callback=False),
    )

    assert canvas.start_async_text_prompt("mouse", output_mode="rectangle") is True
    assert in_ctor.wait(timeout=1.0) is True

    # _prompt_inference_lock MUST NOT be held while GroundingDINO.__init__ is executing!
    acquired_immediately = canvas._prompt_inference_lock.acquire(blocking=False)
    assert acquired_immediately is True
    canvas._prompt_inference_lock.release()

    # Closing the canvas completes promptly without stalling on constructor completion
    close_start = time.perf_counter()
    canvas.close()
    assert (time.perf_counter() - close_start) < 0.4
    gc.collect()

    # Once the constructor returns, the late model is immediately closed and never published
    release_ctor.set()
    assert _spin_until(lambda: closed_instances == ["closed"]) is True
    assert canvas._ai_model_rect is None
    assert _spin_until(lambda: not canvas_mod._ANCHORED_QT_WORKERS) is True


def test_onnx_base_model_provider_fallback_and_init_stop_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import annolid.inference.onnx_model as onnx_mod
    from annolid.utils.cancellation import InferenceCancelledError, ObservableStopEvent

    seen_providers: list[list[str]] = []
    stop_ev = ObservableStopEvent()

    monkeypatch.setattr(onnx_mod, "has_gpu", lambda: True)
    monkeypatch.setattr(
        onnx_mod.ort,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )

    class _FakeSession:
        def __init__(self, _path, *, providers=None, sess_options=None) -> None:
            _ = sess_options
            seen_providers.append(list(providers or []))

    monkeypatch.setattr(onnx_mod.ort, "InferenceSession", _FakeSession)

    # Even when nvidia-smi reports a GPU, CPU-only onnxruntime selects CPUExecutionProvider cleanly
    m = onnx_mod.ONNXBaseModel("dummy.onnx")
    assert seen_providers == [["CPUExecutionProvider"]]
    m.close()
    assert m.ort_session is None

    # Setting stop_event during session creation closes session and raises InferenceCancelledError
    class _SlowSession:
        def __init__(self, _path, *, providers=None, sess_options=None) -> None:
            _ = (providers, sess_options)
            stop_ev.set()

    monkeypatch.setattr(onnx_mod.ort, "InferenceSession", _SlowSession)
    with pytest.raises(InferenceCancelledError):
        onnx_mod.ONNXBaseModel("dummy.onnx", stop_event=stop_ev)


def test_florence2_mid_generation_cancel_video_stop_and_predictor_close(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    from PIL import Image
    import annolid.vision.florence_2 as f2_mod
    from annolid.utils.cancellation import InferenceCancelledError, ObservableStopEvent

    # 1. Mid-generation token decoding aborts on the next forward step and cleans up hooks
    stop_ev = ObservableStopEvent()
    step_calls = {"count": 0}

    class _AutoregressiveDecoderStep(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            step_calls["count"] += 1
            if step_calls["count"] == 1:
                stop_ev.set()
            return x + 1.0

    class _FakeFlorenceModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder_step = _AutoregressiveDecoderStep()
            self.device = torch.device("cpu")

        def generate(self, *, input_ids, pixel_values, **_kwargs):
            _ = pixel_values
            x = input_ids.float()
            for _ in range(4):
                x = self.decoder_step(x)
            return input_ids

    class _FakeProcessor:
        def __call__(self, *, text, images, return_tensors="pt"):
            _ = (text, images, return_tensors)
            return {
                "input_ids": torch.ones((1, 3), dtype=torch.long),
                "pixel_values": torch.zeros((1, 3, 8, 8), dtype=torch.float32),
            }

        def batch_decode(self, ids, skip_special_tokens=False):
            _ = (ids, skip_special_tokens)
            return ["decoded"]

        def post_process_generation(self, text, *, task, image_size):
            _ = (text, image_size)
            return {task: "caption"}

    fake_model = _FakeFlorenceModel()
    fake_proc = _FakeProcessor()
    with pytest.raises(InferenceCancelledError):
        f2_mod.florence2(
            fake_proc,
            fake_model,
            "<MORE_DETAILED_CAPTION>",
            Image.new("RGB", (16, 16)),
            stop_event=stop_ev,
        )
    assert step_calls["count"] == 1
    for submodule in fake_model.modules():
        assert len(submodule._forward_pre_hooks) == 0

    # 2. process_nth_frame_from_video halts cleanly on stop_event without writing partial frame JSONs
    vid_stop = ObservableStopEvent()
    cap_released = {"called": False}

    class _FakeCap:
        def release(self) -> None:
            cap_released["called"] = True

    class _FakeVideoReader:
        def __init__(self) -> None:
            self.cap = _FakeCap()

        def total_frames(self) -> int:
            return 30

        def load_frame(self, idx: int) -> np.ndarray:
            return np.full((20, 20, 3), idx, dtype=np.uint8)

    monkeypatch.setattr(f2_mod, "video_loader", lambda _p: _FakeVideoReader())

    predict_frames: list[int] = []

    class _FakePredictor:
        def predict(self, frame_rgb, *, stop_event=None, **_kwargs):
            frame_val = int(frame_rgb[0, 0, 0])
            predict_frames.append(frame_val)
            if frame_val >= 10 and stop_event is not None:
                stop_event.set()
            return f2_mod.Florence2Result(
                mask_dict={
                    "polygons": [[[(2.0, 2.0), (12.0, 2.0), (12.0, 12.0)]]],
                    "labels": ["mouse"],
                },
                caption="mouse frame",
                raw_outputs={},
            )

    fake_video_path = str(tmp_path / "sample_run.mp4")
    with pytest.raises(InferenceCancelledError):
        f2_mod.process_nth_frame_from_video(
            fake_video_path,
            10,
            _FakePredictor(),  # type: ignore[arg-type]
            segmentation_task="<REFERRING_EXPRESSION_SEGMENTATION>",
            text_input="mouse",
            stop_event=vid_stop,
        )
    assert predict_frames == [0, 10]
    assert cap_released["called"] is True
    out_dir = tmp_path / "sample_run"
    assert (out_dir / f"{0:09d}.json").is_file()
    # Frame 10 was cancelled before completion and must NOT be written
    assert not (out_dir / f"{10:09d}.json").exists()

    # 3. Predictor.close() releases cached weights and processor
    pred = f2_mod.Florence2Predictor(model_name="dummy/florence")
    pred._model = fake_model
    pred._processor = fake_proc
    pred._device = "cpu"
    pred._dtype = torch.float32
    pred.close()
    assert pred._model is None
    assert pred._processor is None
    assert pred.device is None
    assert pred.dtype is None


def test_florence2_dock_stop_button_stale_frame_discard_and_replace_dirty_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ensure_qapp()
    from PIL import Image
    from annolid.gui.shape import Shape
    from annolid.gui.widgets import florence2_widget as f2w_mod
    from annolid.vision.florence_2 import Florence2Result

    critical_dialogs: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "critical",
        lambda *_args: critical_dialogs.append(str(_args)),
    )

    class _HostWindow(QtWidgets.QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.filename = "video.mp4"
            self.frame_number = 0
            self.video_file = "video.mp4"
            self.dirty_calls = 0
            self.caption_widget = None
            self.status_messages: list[str] = []
            self.statusBar().messageChanged.connect(
                lambda msg: self.status_messages.append(str(msg)) if msg else None
            )
            self.canvas = SimpleNamespace(
                shapes=[],
                setCaption=lambda c: setattr(self, "_last_caption", c),
            )
            self._last_caption = None

        def _get_pil_image_from_state(self):
            return Image.new("RGB", (32, 32))

        def loadShapes(self, shapes, replace: bool = True) -> None:
            if replace:
                self.canvas.shapes = list(shapes)
            else:
                self.canvas.shapes.extend(shapes)

        def setDirty(self) -> None:
            self.dirty_calls += 1

        def openCaption(self) -> None:
            self.caption_widget = SimpleNamespace(
                set_caption=lambda _c: None,
                set_image_path=lambda _p: None,
            )

    win = _HostWindow()
    dock = f2w_mod.Florence2DockWidget(win)
    win.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
    closed_predictors: list[str] = []

    try:
        # 1. Stop button cancels an active frame job without showing an error dialog
        entered = threading.Event()
        release = threading.Event()

        class _SlowPredictor:
            def predict(self, _img, *, stop_event=None, **_kwargs):
                entered.set()
                deadline = time.perf_counter() + 2.0
                while time.perf_counter() < deadline:
                    if stop_event is not None and stop_event.is_set():
                        from annolid.utils.cancellation import InferenceCancelledError

                        raise InferenceCancelledError("stopped")
                    if release.is_set():
                        break
                    time.sleep(0.005)
                return Florence2Result(
                    mask_dict={"polygons": [], "labels": []},
                    caption="late caption",
                    raw_outputs={},
                )

            def close(self) -> None:
                closed_predictors.append("slow")

        dock._florence_predictors["microsoft/Florence-2-large"] = _SlowPredictor()  # type: ignore[assignment]
        dock.florence_widget.prompt_edit.setText("mouse")
        dock.florence_widget.run_frame_button.click()

        assert entered.wait(timeout=1.5) is True
        assert dock.florence_widget.is_running() is True
        assert dock.florence_widget.stop_button.isEnabled() is True
        assert dock.florence_widget.run_frame_button.isEnabled() is False

        dock.florence_widget.stop_button.click()
        assert dock.florence_widget.is_running() is False
        assert dock.florence_widget.stop_button.isEnabled() is False
        assert _spin_until(lambda: not dock.is_job_running()) is True
        assert critical_dialogs == []
        assert win.dirty_calls == 0
        assert any("cancelled" in m.lower() for m in win.status_messages)

        # 2. Stale-frame protection: switching frame mid-run never mutates the new frame
        done_predict = threading.Event()
        allow_return = threading.Event()

        class _FrameZeroPredictor:
            def predict(self, _img, **_kwargs):
                done_predict.set()
                allow_return.wait(timeout=2.0)
                return Florence2Result(
                    mask_dict={
                        "polygons": [[[(4.0, 4.0), (20.0, 4.0), (20.0, 20.0)]]],
                        "labels": ["mouse"],
                    },
                    caption="frame 0 caption",
                    raw_outputs={},
                )

            def close(self) -> None:
                closed_predictors.append("frame0")

        dock._florence_predictors["microsoft/Florence-2-large"] = _FrameZeroPredictor()  # type: ignore[assignment]
        win.frame_number = 0
        dock.florence_widget.run_frame_button.click()
        assert done_predict.wait(timeout=1.5) is True

        # Navigate to Frame 1 before Frame 0's Florence-2 result arrives
        win.frame_number = 1
        manual_shape = Shape(label="manual_frame1", shape_type="polygon", flags={})
        manual_shape.points = [
            QtCore.QPointF(1.0, 1.0),
            QtCore.QPointF(8.0, 1.0),
            QtCore.QPointF(8.0, 8.0),
        ]
        win.canvas.shapes = [manual_shape]

        allow_return.set()
        assert _spin_until(lambda: not dock.is_job_running()) is True
        assert [s.label for s in win.canvas.shapes] == ["manual_frame1"]
        assert win._last_caption is None
        assert win.dirty_calls == 0
        assert any("active frame changed" in m.lower() for m in win.status_messages)

        # 3. replace_existing=True marks window dirty even when 0 new Florence shapes are returned
        old_florence_shape = Shape(
            label="old_mouse",
            shape_type="polygon",
            description="florence",
            flags={},
        )
        old_florence_shape.points = [
            QtCore.QPointF(2.0, 2.0),
            QtCore.QPointF(10.0, 2.0),
            QtCore.QPointF(10.0, 10.0),
        ]
        win.canvas.shapes = [manual_shape, old_florence_shape]
        req = f2w_mod.Florence2Request(
            target="frame",
            model_name="microsoft/Florence-2-large",
            text_input="mouse",
            segmentation_task="<REFERRING_EXPRESSION_SEGMENTATION>",
            include_caption=False,
            caption_task=None,
            replace_existing=True,
        )
        dock._running_frame_token = dock._capture_window_frame_token()
        dock._handle_florence_result(
            (
                req,
                Florence2Result(
                    mask_dict={"polygons": [], "labels": []},
                    caption=None,
                    raw_outputs={},
                ),
            )
        )
        assert [s.label for s in win.canvas.shapes] == ["manual_frame1"]
        assert win.dirty_calls == 1

        # 4. close_predictors releases cached predictor instances
        dock.close_predictors()
        assert "frame0" in closed_predictors
    finally:
        release.set()
        dock.close()
        win.close()
        assert _spin_until(lambda: not f2w_mod._ANCHORED_FLORENCE_WORKERS) is True


def test_concurrent_same_destination_download_deduplicates_and_supports_waiter_cancel(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import urllib.request
    from annolid.utils.cancellation import (
        InferenceCancelledError,
        ObservableStopEvent,
        cancellable_download_file,
    )

    in_first_download = threading.Event()
    release_first_download = threading.Event()
    urlopen_calls = {"count": 0}
    payload = b"shared-checkpoint-binary-data"

    class _GateResponse:
        def __init__(self) -> None:
            self.headers = {"Content-Length": str(len(payload))}
            self._done = False

        def read(self, _n: int = 65536) -> bytes:
            if self._done:
                return b""
            in_first_download.set()
            release_first_download.wait(timeout=2.0)
            self._done = True
            return payload

        def close(self) -> None:
            pass

    def _mock_urlopen(_req, timeout=30.0):
        _ = timeout
        urlopen_calls["count"] += 1
        return _GateResponse()

    monkeypatch.setattr(urllib.request, "urlopen", _mock_urlopen)

    dest = tmp_path / "shared" / "sam_hq.pth"
    results: dict[str, object] = {}

    def _downloader(key: str, stop_ev=None) -> None:
        try:
            results[key] = cancellable_download_file(
                "https://example.invalid/sam_hq.pth",
                dest,
                stop_event=stop_ev,
            )
        except BaseException as exc:
            results[key] = exc

    t1 = threading.Thread(target=_downloader, args=("t1",), daemon=True)
    t2 = threading.Thread(target=_downloader, args=("t2",), daemon=True)
    cancel_waiter_ev = ObservableStopEvent()
    t3 = threading.Thread(
        target=_downloader, args=("t3", cancel_waiter_ev), daemon=True
    )

    t1.start()
    assert in_first_download.wait(timeout=1.0) is True
    t2.start()
    t3.start()
    time.sleep(0.08)

    # Cancelling waiter t3 aborts t3 promptly while t1 continues undisturbed
    cancel_waiter_ev.set()
    t3.join(timeout=1.0)
    assert not t3.is_alive()
    assert isinstance(results["t3"], InferenceCancelledError)

    release_first_download.set()
    t1.join(timeout=1.0)
    t2.join(timeout=1.0)
    assert results["t1"] == str(dest)
    assert results["t2"] == str(dest)
    assert urlopen_calls["count"] == 1
    assert dest.read_bytes() == payload


def test_reserved_toolbar_commands_yoloe_alias_mapping_and_sam3_runner_stop_propagation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import types
    from annolid.gui.widgets.sam3_manager import Sam3Manager
    from annolid.gui.workers import FlexibleWorker
    from annolid.segmentation.yolos import InferenceProcessor
    from annolid.utils.prompts import parse_text_prompt

    # 1. Reserved toolbar commands (`flags:...` and `every N ...`) never leak into _current_text_prompt
    class _PromptHost(CoreInteractionMixin):
        def __init__(self, text: str) -> None:
            self.aiRectangle = SimpleNamespace(
                _aiRectanglePrompt=SimpleNamespace(text=lambda: text)
            )

    assert _PromptHost("flags: grooming, rearing")._current_text_prompt() is None
    assert _PromptHost("every 32 cell")._current_text_prompt() is None
    assert _PromptHost("EVERY 16")._current_text_prompt() is None
    assert (
        _PromptHost("black mouse -> MouseBlack")._current_text_prompt()
        == "black mouse -> MouseBlack"
    )

    # 2. YOLOE text-prompt alias mapping preserves CLIP queries while assigning canonical labels
    parsed = parse_text_prompt("black mouse -> MouseBlack, white mouse = MouseWhite")
    assert list(parsed.queries) == ["black mouse", "white mouse"]
    assert list(parsed.labels) == ["MouseBlack", "MouseWhite"]

    proc = InferenceProcessor.__new__(InferenceProcessor)
    proc.model_name = "yoloe-11l-seg.pt"
    proc.prompt_class_names = list(parsed.labels)
    proc.model = SimpleNamespace(names={0: "black mouse", 1: "white mouse"}, model=None)
    proc._apply_prompt_names_to_model()
    assert proc.model.names == {0: "MouseBlack", 1: "MouseWhite"}

    # 3. Sam3Manager video runner exposes stop_event + request_stop to FlexibleWorker and does not fall back on user stop
    agent_stop_seen: list[bool] = []
    standard_built = {"called": False}

    class _DummyStopExc(RuntimeError):
        pass

    def _fake_agent_video(**kwargs):
        ev = kwargs.get("stop_event")
        assert ev is not None
        # Simulate user clicking Stop mid-agent pass
        ev.set()
        agent_stop_seen.append(ev.is_set())
        raise _DummyStopExc("user stopped")

    class _FakeSam3Session:
        def __init__(self, **_kwargs) -> None:
            standard_built["called"] = True

        def run(self, *, stop_event=None):
            _ = stop_event
            return "SAM3 done#5"

        def request_stop(self) -> None:
            pass

    fake_cap_mod = types.ModuleType("annolid.infrastructure.capabilities")
    fake_cap_mod.ensure_capability = lambda _name: None  # type: ignore[attr-defined]
    fake_adapter_mod = types.ModuleType("annolid.segmentation.SAM.sam3.adapter")
    fake_adapter_mod.Sam3StopRequested = _DummyStopExc  # type: ignore[attr-defined]
    fake_adapter_mod.process_video_with_agent = _fake_agent_video  # type: ignore[attr-defined]
    fake_adapter_mod.SAM3VideoProcessor = _FakeSam3Session  # type: ignore[attr-defined]

    monkeypatch.setitem(
        sys.modules, "annolid.infrastructure.capabilities", fake_cap_mod
    )
    monkeypatch.setitem(
        sys.modules, "annolid.segmentation.SAM.sam3.adapter", fake_adapter_mod
    )
    import annolid.segmentation.SAM.sam3 as sam3_pkg

    monkeypatch.setattr(sam3_pkg, "adapter", fake_adapter_mod, raising=False)
    import annolid.segmentation.SAM.sam_v2 as sam_v2_mod

    monkeypatch.setattr(
        sam_v2_mod,
        "load_manual_seed_annotations_from_video",
        lambda _v: ([], {}),
    )

    host = SimpleNamespace(
        video_file="mouse.mp4",
        frame_number=0,
        canvas=SimpleNamespace(shapes=[]),
        epsilon_for_polygon=2.0,
        _config={},
        _current_text_prompt=lambda: "black mouse -> MouseBlack",
    )
    mgr = Sam3Manager(host)  # type: ignore[arg-type]
    runner = mgr.build_video_processor("SAM3", "sam3.pt", None)
    assert callable(runner)
    assert callable(getattr(runner, "request_stop", None))

    # FlexibleWorker must inject both pred_worker and stop_event into runner
    fw = FlexibleWorker(task_function=runner)
    injected = fw._inject_cancellation_kwargs({})
    assert "pred_worker" in injected and "stop_event" in injected

    result = runner(pred_worker=fw, stop_event=fw.stop_event)
    assert result == "Stopped by user."
    assert agent_stop_seen == [True]
    # Crucial: user cancellation must NOT fall back to launching standard full-video SAM3!
    assert standard_built["called"] is False


def test_qt_to_numpy_image_conversions_own_memory_and_preserve_rgb_on_temp_rvalues() -> (
    None
):
    _ensure_qapp()
    import gc
    from annolid.utils import annotation_compat
    from annolid.utils.qt2cv import (
        convert_cv_image_to_qt_image,
        convert_qt_image_to_cv_image,
        convert_qt_image_to_rgb_cv_image,
    )

    formats = [
        QtGui.QImage.Format_RGB32,
        QtGui.QImage.Format_ARGB32,
        QtGui.QImage.Format_RGBA8888,
        QtGui.QImage.Format_RGBX8888,
        QtGui.QImage.Format_RGB888,
    ]
    # Odd width (641) ensures Format_RGB888 has row padding (bytesPerLine != width * 3).
    width, height = 641, 73
    expected_rgb = [212, 97, 34]

    for fmt in formats:
        src = QtGui.QImage(width, height, fmt)
        src.fill(QtGui.QColor(*expected_rgb))
        pixmap = QtGui.QPixmap.fromImage(src)
        del src

        for converter in (
            convert_qt_image_to_rgb_cv_image,
            convert_qt_image_to_cv_image,
            annotation_compat.img_qt_to_arr,
        ):
            # Pass temporary rvalue QImage with zero caller reference retained.
            arr = converter(pixmap.toImage())
            gc.collect()
            # Allocate multi-MB buffer to exercise allocator page reuse before OpenCV read.
            _pressure = np.full((1024, 1024, 4), 0xAA, dtype=np.uint8)
            assert arr.flags["OWNDATA"] is True
            assert arr.flags["C_CONTIGUOUS"] is True
            assert arr.dtype == np.uint8
            assert arr.shape == (height, width, 3)
            assert arr[0, 0].tolist() == expected_rgb
            assert arr[height - 1, width - 1].tolist() == expected_rgb
            resized = cv2.resize(arr, (320, 240))
            assert resized.shape == (240, 320, 3)
            del _pressure

    gray16_fmt = getattr(QtGui.QImage, "Format_Grayscale16", None)
    if gray16_fmt is not None:
        raw16 = np.linspace(1000, 60000, num=65 * 49, dtype=np.uint16).reshape(49, 65)
        raw16_c = np.ascontiguousarray(raw16)
        qgray = QtGui.QImage(
            raw16_c.data,
            65,
            49,
            65 * 2,
            gray16_fmt,
        ).copy()
        del raw16, raw16_c
        gray_rgb = convert_qt_image_to_rgb_cv_image(qgray)
        del qgray
        gc.collect()
        assert gray_rgb.flags["OWNDATA"] is True
        assert gray_rgb.flags["C_CONTIGUOUS"] is True
        assert gray_rgb.dtype == np.uint8
        assert gray_rgb.shape == (49, 65, 3)
        assert int(gray_rgb.min()) == 0
        assert int(gray_rgb.max()) == 255

    # Also verify numpy -> QImage detaches from caller buffer.
    src_np = np.full((32, 48, 3), [15, 125, 235], dtype=np.uint8)
    qt_out = convert_cv_image_to_qt_image(src_np)
    src_np.fill(0)
    del src_np
    gc.collect()
    roundtrip = convert_qt_image_to_rgb_cv_image(qt_out)
    assert roundtrip[10, 10].tolist() == [15, 125, 235]


def test_canvas_async_text_prompt_sets_worker_stack_size_and_copies_frame_buffer() -> (
    None
):
    _ensure_qapp()
    import gc

    captured: dict[str, object] = {}
    entered = threading.Event()
    proceed = threading.Event()

    class _InspectingDetector:
        def predict_detections(self, image, _prompt, **_kwargs):
            captured["owns_data"] = bool(image.flags["OWNDATA"])
            captured["c_contiguous"] = bool(image.flags["C_CONTIGUOUS"])
            captured["dtype"] = image.dtype
            captured["shape"] = image.shape
            entered.set()
            proceed.wait(timeout=3.0)
            gc.collect()
            resized = cv2.resize(image, (800, 800))
            captured["pixel"] = image[20, 20].tolist()
            captured["resized_shape"] = resized.shape
            return [
                DetectionBoxCandidate(
                    bbox=(10, 10, 60, 60),
                    label="fish",
                    score=0.92,
                    source="grounding_dino",
                )
            ]

    canvas = Canvas()
    try:
        qimg = QtGui.QImage(321, 181, QtGui.QImage.Format_ARGB32)
        qimg.fill(QtGui.QColor(30, 90, 210))
        canvas.loadPixmap(QtGui.QPixmap.fromImage(qimg), clear_shapes=True)
        del qimg
        canvas._ai_model_rect = _InspectingDetector()

        assert canvas.start_async_text_prompt("fish", output_mode="rectangle") is True
        req_id = canvas._active_prompt_request_id
        assert req_id is not None
        worker_thread = canvas._inflight_prompt_workers[req_id][0]
        assert worker_thread.stackSize() >= 8 * 1024 * 1024

        assert entered.wait(timeout=3.0) is True
        # Replace canvas pixmap while worker thread has not yet called cv2.resize.
        replacement = QtGui.QImage(16, 16, QtGui.QImage.Format_RGB32)
        replacement.fill(QtGui.QColor(0, 0, 0))
        canvas.pixmap = QtGui.QPixmap.fromImage(replacement)
        gc.collect()
        proceed.set()

        assert _spin_until(lambda: not canvas._inflight_prompt_workers) is True
        assert captured["owns_data"] is True
        assert captured["c_contiguous"] is True
        assert captured["dtype"] == np.uint8
        assert captured["shape"] == (181, 321, 3)
        assert captured["pixel"] == [30, 90, 210]
        assert captured["resized_shape"] == (800, 800, 3)

        # Verify Canvas.samEmbedding also converts cleanly via RGB helper on PySide6.
        sam_seen: list[np.ndarray] = []
        sam_img = QtGui.QImage(65, 45, QtGui.QImage.Format_ARGB32)
        sam_img.fill(QtGui.QColor(180, 70, 25))
        canvas.loadPixmap(QtGui.QPixmap.fromImage(sam_img), clear_shapes=True)
        canvas.sam_predictor = SimpleNamespace(
            set_image=lambda arr: sam_seen.append(arr)
        )
        canvas.samEmbedding()
        assert len(sam_seen) == 1
        assert sam_seen[0][0, 0].tolist() == [180, 70, 25]
    finally:
        proceed.set()
        canvas.close()


def test_ai_rectangle_widget_defers_completer_reset_while_popup_is_visible() -> None:
    app = _ensure_qapp()
    widget = AiRectangleWidget()
    real_completer = widget._completer
    try:
        widget.set_completion_suggestions(["fin", "tail"])
        assert widget._completer_model.stringList() == ["fin", "tail"]

        popup_visible = {"value": True}
        fake_popup = SimpleNamespace(isVisible=lambda: popup_visible["value"])
        widget._completer = SimpleNamespace(popup=lambda: fake_popup)  # type: ignore[assignment]

        # Simulate Enter pressed while QCompleter popup is open: synchronous modelReset
        # must NOT happen mid-event, and multiple updates must coalesce without stale overwrite.
        widget.record_prompt("fish")
        widget.set_completion_suggestions(["fin", "tail", "gill"])
        assert widget._completer_model.stringList() == ["fin", "tail"]

        popup_visible["value"] = False
        assert _spin_until(
            lambda: (
                widget._completer_model.stringList() == ["fish", "fin", "tail", "gill"]
            ),
            timeout_s=1.0,
        )
        app.processEvents()
        assert widget._completer_model.stringList() == ["fish", "fin", "tail", "gill"]
    finally:
        widget._completer = real_completer
        widget.close()


def test_sam3_agent_orchestrator_aborts_mid_round_on_stop_event_and_disables_force_terminate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from annolid.segmentation.SAM.sam3 import agent_video_orchestrator as orch
    from annolid.segmentation.SAM.sam3.adapter import Sam3StopRequested
    from annolid.segmentation.yolos import InferenceProcessor
    from annolid.utils.cancellation import ObservableStopEvent

    assert (
        getattr(InferenceProcessor.run_inference, "allow_force_thread_terminate", None)
        is False
    )

    # 1. Verify _build_send_generate_request checks stop_event before & after LLM call.
    monkeypatch.setattr(
        orch,
        "resolve_llm_config",
        lambda **_kw: SimpleNamespace(providers={}, bot={}),
    )
    monkeypatch.setattr(
        orch,
        "resolve_openai_compat",
        lambda _cfg: SimpleNamespace(
            base_url="http://localhost:11434/v1",
            model="test-vlm",
            api_key="test-key",
        ),
    )

    stop_ev = ObservableStopEvent()
    llm_calls = {"count": 0}

    def _fake_send_generate_request(messages, **_kwargs):
        _ = messages
        llm_calls["count"] += 1
        # User clicks Stop while MLLM request is in flight!
        stop_ev.set()
        return {"role": "assistant", "content": "continuing tool loop"}

    monkeypatch.setattr(orch, "_send_generate_request", _fake_send_generate_request)
    send_fn, _resolved = orch._build_send_generate_request(
        orch.AgentConfig(prompt="fish"),
        stop_event=stop_ev,
    )
    with pytest.raises(Sam3StopRequested):
        send_fn([{"role": "user", "content": "find fish"}])
    assert llm_calls["count"] == 1

    # Calling again when already stopped must fail fast before touching network/MLLM.
    with pytest.raises(Sam3StopRequested):
        send_fn([{"role": "user", "content": "round 2"}])
    assert llm_calls["count"] == 1

    # 2. Verify _run_agent_on_frame propagates Sam3StopRequested directly (not wrapped in RuntimeError).
    frame_file = tmp_path / "frame0.jpg"
    cv2.imwrite(str(frame_file), np.zeros((32, 32, 3), dtype=np.uint8))

    fake_agent_core = types.ModuleType(
        "annolid.segmentation.SAM.sam3.sam3.agent.agent_core"
    )
    fake_agent_core.agent_inference = (  # type: ignore[attr-defined]
        lambda *, send_generate_request, **_kw: send_generate_request([])
    )
    fake_sam3_inner = types.ModuleType("annolid.segmentation.SAM.sam3.sam3")
    fake_agent_pkg = types.ModuleType("annolid.segmentation.SAM.sam3.sam3.agent")
    monkeypatch.setitem(
        sys.modules, "annolid.segmentation.SAM.sam3.sam3", fake_sam3_inner
    )
    monkeypatch.setitem(
        sys.modules, "annolid.segmentation.SAM.sam3.sam3.agent", fake_agent_pkg
    )
    monkeypatch.setitem(
        sys.modules,
        "annolid.segmentation.SAM.sam3.sam3.agent.agent_core",
        fake_agent_core,
    )

    with pytest.raises(Sam3StopRequested):
        orch._run_agent_on_frame(
            frame_file,
            orch.AgentConfig(prompt="fish", output_dir=str(tmp_path)),
            send_fn,
        )

    # 3. Verify Sam3Manager runners disable unsafe QThread.terminate().
    fake_cap_mod = types.ModuleType("annolid.infrastructure.capabilities")
    fake_cap_mod.ensure_capability = lambda _name: None  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules, "annolid.infrastructure.capabilities", fake_cap_mod
    )
    import annolid.segmentation.SAM.sam_v2 as sam_v2_mod

    monkeypatch.setattr(
        sam_v2_mod, "load_manual_seed_annotations_from_video", lambda _v: ([], {})
    )
    host = SimpleNamespace(
        video_file="fish.mp4",
        frame_number=0,
        canvas=SimpleNamespace(shapes=[]),
        epsilon_for_polygon=2.0,
        _config={},
        _current_text_prompt=lambda: "fish",
    )
    mgr = Sam3Manager(host)  # type: ignore[arg-type]
    for mode_name in ("SAM3", "SAM3 Agent"):
        runner = mgr.build_video_processor(mode_name, "sam3.pt", "fish")
        assert callable(runner)
        assert getattr(runner, "allow_force_thread_terminate", None) is False


def test_sam3_agent_client_llm_closes_active_http_client_mid_flight_on_stop_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from annolid.segmentation.SAM.sam3.sam3.agent import client_llm
    from annolid.utils.cancellation import InferenceCancelledError, ObservableStopEvent

    seen_timeouts: list[object] = []

    for event_factory in (ObservableStopEvent, threading.Event):
        entered_http = threading.Event()
        socket_closed = threading.Event()
        close_calls = {"count": 0}

        class _BlockingOpenAIClient:
            def __init__(self, **_kwargs) -> None:
                seen_timeouts.append(_kwargs.get("timeout"))
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=self._create)
                )

            def _create(self, **_kwargs):
                entered_http.set()
                if socket_closed.wait(timeout=5.0):
                    raise RuntimeError("httpx socket closed on cancel")
                raise AssertionError("HTTP client was not closed on stop_event.set()")

            def close(self) -> None:
                close_calls["count"] += 1
                socket_closed.set()

        monkeypatch.setattr(client_llm, "OpenAI", _BlockingOpenAIClient)
        stop_ev = event_factory()
        outcome: list[BaseException] = []

        def _worker() -> None:
            try:
                client_llm.send_generate_request(
                    [{"role": "user", "content": "segment fish"}],
                    server_url="http://127.0.0.1:11434/v1",
                    model="qwen-vl",
                    api_key="none",
                    timeout=45.0,
                    stop_event=stop_ev,
                )
            except BaseException as exc:
                outcome.append(exc)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        assert entered_http.wait(timeout=2.0) is True
        started_cancel = time.perf_counter()
        stop_ev.set()
        t.join(timeout=1.5)
        elapsed = time.perf_counter() - started_cancel

        assert not t.is_alive()
        assert elapsed < 1.0
        assert close_calls["count"] >= 1
        assert len(outcome) == 1 and isinstance(outcome[0], InferenceCancelledError)

    assert seen_timeouts == [45.0, 45.0]


def test_flexible_worker_observable_stop_event_prestart_cancel_and_partial_introspection() -> (
    None
):
    _ensure_qapp()
    import functools
    from annolid.gui.workers import FlexibleWorker
    from annolid.segmentation.SAM.edge_sam_bg import (
        VideoProcessor as EdgeSamVideoProcessor,
    )
    from annolid.segmentation.SAM.sam_v2 import (
        process_video,
        process_video_efficienttam,
    )
    from annolid.segmentation.cutie_vos.video_processor import CutieVideoProcessor
    from annolid.segmentation.insid3_video import Insid3VideoProcessor
    from annolid.segmentation.videomt_onnx import VideoMTOnnxVideoProcessor
    from annolid.tracker.base import BasePointTracker
    from annolid.utils.cancellation import ObservableStopEvent

    # 1. ObservableStopEvent fires listeners synchronously on request_stop(), and
    #    request_stop() BEFORE run() starts is honored rather than cleared by run().
    calls: list[str] = []
    listener_fired: list[bool] = []
    results: list[object] = []

    def _task(*, stop_event=None):
        calls.append(f"ran:{stop_event.is_set() if stop_event else None}")
        return "Completed"

    fw = FlexibleWorker(_task)
    assert isinstance(fw.stop_event, ObservableStopEvent)
    fw.stop_event.add_listener(lambda: listener_fired.append(True))
    fw.finished_signal.connect(results.append)

    fw.request_stop()
    assert listener_fired == [True]
    fw.run()
    assert calls == []
    assert results == ["Stopped"]

    # Subsequent reuse without a new stop request resets cleanly and runs the task.
    fw.run()
    assert calls == ["ran:False"]
    assert results == ["Stopped", "Completed"]

    # 2. functools.partial-wrapped SAM2 / EfficientTAM and native video processors
    #    report allows_force_terminate() is False and inject cancellation kwargs.
    for fn in (
        functools.partial(process_video, video_path="fish.mp4"),
        functools.partial(process_video_efficienttam, video_path="fish.mp4"),
    ):
        partial_worker = FlexibleWorker(fn)
        assert partial_worker.allows_force_terminate() is False
        injected = partial_worker._inject_cancellation_kwargs({})
        assert injected["pred_worker"] is partial_worker
        assert injected["stop_event"] is partial_worker.stop_event

    assert CutieVideoProcessor.allow_force_thread_terminate is False
    assert BasePointTracker.allow_force_thread_terminate is False
    for method in (
        VideoMTOnnxVideoProcessor.process_video_frames,
        Insid3VideoProcessor.process_video_frames,
        EdgeSamVideoProcessor.process_video_frames,
    ):
        assert getattr(method, "allow_force_thread_terminate", None) is False


def test_sam2_propagation_stops_cooperatively_and_recording_widget_writes_bgr(
    tmp_path,
) -> None:
    _ensure_qapp()
    import torch

    from annolid.gui.widgets.video_recording import RecordingWidget
    from annolid.segmentation.SAM.sam_v2 import SAM2VideoProcessor
    from annolid.utils.cancellation import ObservableStopEvent

    # 1. SAM2VideoProcessor._propagate halts immediately when stop_event is set mid-stream.
    stop_ev = ObservableStopEvent()
    saved_frames: list[int] = []

    proc = SAM2VideoProcessor.__new__(SAM2VideoProcessor)
    proc.video_dir = str(tmp_path)
    proc.frame_shape = (32, 32, 3)
    proc._stop_event = stop_ev
    proc._pred_worker = None
    proc._stop_requested = False

    def _fake_propagate(_state):
        for idx in range(5):
            if idx == 1:
                stop_ev.set()
            yield idx, [1], [torch.ones((1, 8, 8), dtype=torch.float32)]

    proc.predictor = SimpleNamespace(propagate_in_video=_fake_propagate)
    proc._save_annotations = (  # type: ignore[method-assign]
        lambda _fn, _md, _fs, frame_idx=None, **_kw: saved_frames.append(int(frame_idx))
    )
    proc._propagate(inference_state={})
    assert saved_frames == [0]

    # 2. RecordingWidget converts RGB QImage frames to BGR for cv2.VideoWriter.
    qimg = QtGui.QImage(24, 16, QtGui.QImage.Format_RGB32)
    qimg.fill(QtGui.QColor(220, 100, 30))
    pix = QtGui.QPixmap.fromImage(qimg)
    written_frames: list[np.ndarray] = []

    rec = RecordingWidget(canvas=lambda: SimpleNamespace())
    try:
        rec.is_recording = True
        rec._grab_canvas_pixmap = lambda _c: pix  # type: ignore[method-assign]
        rec.video_writer = SimpleNamespace(write=written_frames.append)
        rec.capture_frame()
        assert len(written_frames) == 1
        assert written_frames[0].shape == (16, 24, 3)
        # OpenCV BGR order: [B=30, G=100, R=220]
        assert written_frames[0][0, 0].tolist() == [30, 100, 220]
    finally:
        rec.close()
