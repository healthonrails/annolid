from __future__ import annotations

import os

from pathlib import Path

from qtpy import QtWidgets
from qtpy import QtCore

from annolid.annotation.pose_schema import PoseSchema
from annolid.gui.widgets.keypoint_sequencer import KeypointSequencerWidget


os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

_QAPP = None


def _ensure_qapp():
    global _QAPP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _QAPP = app
    return _QAPP


def test_keypoint_sequencer_cycles_and_wraps_with_instance_prefix():
    _ensure_qapp()
    widget = KeypointSequencerWidget()
    try:
        schema = PoseSchema.from_dict(
            {
                "keypoints": ["nose", "left_ear", "right_ear"],
                "edges": [["nose", "left_ear"], ["nose", "right_ear"]],
                "instances": ["resident"],
            }
        )
        widget.set_pose_schema(schema)
        widget.enable_checkbox.setChecked(True)
        widget.instance_combo.setCurrentIndex(
            widget.instance_combo.findText("resident")
        )
        widget.set_active_keypoints(["nose", "right_ear"])
        for name in ["nose", "right_ear"]:
            row = widget.keypoint_order().index(name)
            widget.keypoint_list.item(row).setSelected(True)
        widget._lock_selected_keypoints()  # noqa: SLF001 - targeted behavior

        assert widget.consume_next_label() == "resident_nose"
        assert widget.consume_next_label() == "resident_right_ear"
        assert widget.consume_next_label() == "resident_nose"
    finally:
        widget.close()


def test_keypoint_sequencer_disabled_without_active_keypoints():
    _ensure_qapp()
    widget = KeypointSequencerWidget()
    try:
        schema = PoseSchema.from_dict({"keypoints": ["nose", "tail_base"]})
        widget.set_pose_schema(schema)
        widget.enable_checkbox.setChecked(True)
        widget.set_active_keypoints([])

        assert widget.is_sequence_enabled() is False
        assert widget.consume_next_label() is None
    finally:
        widget.close()


def test_keypoint_sequencer_disables_toggle_when_schema_has_no_keypoints():
    _ensure_qapp()
    widget = KeypointSequencerWidget()
    try:
        widget.enable_checkbox.setChecked(True)
        schema = PoseSchema.from_dict({"keypoints": []})
        widget.set_pose_schema(schema)

        assert widget.enable_checkbox.isEnabled() is False
        assert widget.enable_checkbox.isChecked() is False
    finally:
        widget.close()


def test_keypoint_sequencer_allows_manual_keypoint_add_without_schema():
    _ensure_qapp()
    widget = KeypointSequencerWidget()
    try:
        widget.set_pose_schema(None)
        widget.load_keypoints_from_labels(["nose", "left_ear"])
        widget.enable_checkbox.setChecked(True)
        assert widget.is_sequence_enabled() is False

        for name in ["nose", "left_ear"]:
            row = widget.keypoint_order().index(name)
            widget.keypoint_list.item(row).setSelected(True)
        widget._lock_selected_keypoints()  # noqa: SLF001 - targeted behavior

        assert widget.is_sequence_enabled() is True
        assert widget.consume_next_label() == "nose"
        assert widget.consume_next_label() == "left_ear"
    finally:
        widget.close()


def test_keypoint_sequencer_loads_keypoints_from_labels_dock():
    _ensure_qapp()

    host = QtWidgets.QMainWindow()
    host.uniqLabelList = QtWidgets.QListWidget(host)
    host.uniqLabelList.addItem("nose")
    host.uniqLabelList.addItem("tail_base")

    widget = KeypointSequencerWidget(parent=host)
    try:
        widget.set_pose_schema(None)
        widget._load_from_labels_dock()  # noqa: SLF001 - targeted widget behavior

        assert widget.active_keypoints() == ["nose", "tail_base"]
    finally:
        widget.close()
        host.close()


def test_keypoint_sequencer_loads_labels_from_userrole_not_display_text():
    _ensure_qapp()

    host = QtWidgets.QMainWindow()
    host.uniqLabelList = QtWidgets.QListWidget(host)
    item = QtWidgets.QListWidgetItem("nose [12]")
    item.setData(QtCore.Qt.UserRole, "nose")
    host.uniqLabelList.addItem(item)

    widget = KeypointSequencerWidget(parent=host)
    try:
        widget.set_pose_schema(None)
        widget._load_from_labels_dock()  # noqa: SLF001 - targeted widget behavior
        assert widget.keypoint_order() == ["nose"]
    finally:
        widget.close()
        host.close()


def test_keypoint_sequencer_remove_and_reorder_keypoints():
    _ensure_qapp()
    widget = KeypointSequencerWidget()
    try:
        schema = PoseSchema.from_dict({"keypoints": ["nose", "left_ear", "right_ear"]})
        widget.set_pose_schema(schema)
        widget.set_active_keypoints(["nose", "left_ear", "right_ear"])

        # Move right_ear up once
        row = widget.keypoint_order().index("right_ear")
        widget.keypoint_list.setCurrentRow(row)
        widget._move_selected_keypoints(-1)  # noqa: SLF001 - targeted behavior
        assert widget.keypoint_order() == ["nose", "right_ear", "left_ear"]

        # Remove middle keypoint
        widget.keypoint_list.setCurrentRow(1)
        widget._remove_selected_keypoints()  # noqa: SLF001 - targeted behavior
        assert widget.keypoint_order() == ["nose", "left_ear"]
        assert widget._schema.keypoints == ["nose", "left_ear"]  # noqa: SLF001
    finally:
        widget.close()


def test_keypoint_sequencer_keeps_prefixed_schema_edges_when_base_exists():
    _ensure_qapp()
    widget = KeypointSequencerWidget()
    try:
        schema = PoseSchema.from_dict(
            {
                "keypoints": ["nose", "tail_base"],
                "instances": ["resident"],
                "edges": [["resident_nose", "resident_tail_base"]],
            }
        )
        widget.set_pose_schema(schema)
        assert widget._schema.edges == [("resident_nose", "resident_tail_base")]  # noqa: SLF001
    finally:
        widget.close()


def test_keypoint_sequencer_loads_and_saves_pose_schema(tmp_path: Path):
    _ensure_qapp()
    widget = KeypointSequencerWidget()
    try:
        schema_path = tmp_path / "pose_schema.json"
        schema_path.write_text(
            '{"keypoints":["nose","tail_base"],"edges":[["nose","tail_base"]]}',
            encoding="utf-8",
        )

        assert widget.load_schema_from_path(str(schema_path), quiet=True) is True
        assert widget.keypoint_order() == ["nose", "tail_base"]
        assert widget.schema_path() == str(schema_path)

        out_path = tmp_path / "saved_pose_schema.json"
        assert widget.save_schema_to_path(str(out_path), quiet=True) is True
        assert out_path.exists()
    finally:
        widget.close()


def test_keypoint_sequencer_preview_clicks_toggle_edges():
    _ensure_qapp()
    widget = KeypointSequencerWidget()
    try:
        schema = PoseSchema.from_dict({"keypoints": ["nose", "tail_base"], "edges": []})
        widget.set_pose_schema(schema)

        widget._on_preview_keypoint_clicked("nose")  # noqa: SLF001
        widget._on_preview_keypoint_clicked("tail_base")  # noqa: SLF001
        assert widget._schema.edges == [("nose", "tail_base")]  # noqa: SLF001
        assert widget.edges_list.count() == 1

        widget._on_preview_keypoint_clicked("nose")  # noqa: SLF001
        widget._on_preview_keypoint_clicked("tail_base")  # noqa: SLF001
        assert widget._schema.edges == []  # noqa: SLF001
        assert widget.edges_list.count() == 0
    finally:
        widget.close()


def test_keypoint_sequencer_auto_symmetry_builds_pairs_and_flip_preview():
    _ensure_qapp()
    widget = KeypointSequencerWidget()
    try:
        schema = PoseSchema.from_dict(
            {"keypoints": ["nose", "left_ear", "right_ear", "tail_base"]}
        )
        widget.set_pose_schema(schema)

        widget._auto_fill_symmetry_pairs()  # noqa: SLF001 - targeted behavior

        assert ("left_ear", "right_ear") in widget._symmetry_pairs_as_pairs()  # noqa: SLF001
        assert widget.flip_preview_label.text().startswith("[")
    finally:
        widget.close()


def test_keypoint_sequencer_instance_prefix_config_updates_schema():
    _ensure_qapp()
    widget = KeypointSequencerWidget()
    try:
        schema = PoseSchema.from_dict({"keypoints": ["nose"]})
        widget.set_pose_schema(schema)

        widget.instance_prefixes_edit.setText("resident,intruder")
        widget.instance_separator_edit.setText("-")
        widget._on_instance_config_changed()  # noqa: SLF001 - targeted behavior

        assert widget._schema.instances == ["resident", "intruder"]  # noqa: SLF001
        assert widget._schema.instance_separator == "-"  # noqa: SLF001
    finally:
        widget.close()


def test_keypoint_sequencer_normalizes_prefixed_schema():
    _ensure_qapp()
    widget = KeypointSequencerWidget()
    try:
        schema = PoseSchema.from_dict(
            {
                "keypoints": [
                    "resident_nose",
                    "resident_tail_base",
                    "intruder_nose",
                    "intruder_tail_base",
                ],
                "edges": [["resident_nose", "resident_tail_base"]],
            }
        )
        widget.set_pose_schema(schema)

        widget._normalize_prefixed_schema()  # noqa: SLF001 - targeted behavior

        assert widget._schema.instances == ["resident", "intruder"]  # noqa: SLF001
        assert widget._schema.keypoints == ["nose", "tail_base"]  # noqa: SLF001
    finally:
        widget.close()


def test_keypoint_sequencer_locked_order_stays_stable_with_new_active_keypoints():
    _ensure_qapp()
    widget = KeypointSequencerWidget()
    try:
        schema = PoseSchema.from_dict(
            {"keypoints": ["nose", "left_ear", "right_ear", "tail_base", "paw"]}
        )
        widget.set_pose_schema(schema)
        widget.enable_checkbox.setChecked(True)
        widget.set_active_keypoints(["nose", "left_ear", "right_ear", "tail_base"])

        for name in ["nose", "left_ear", "right_ear", "tail_base"]:
            row = widget.keypoint_order().index(name)
            widget.keypoint_list.item(row).setSelected(True)
        widget._lock_selected_keypoints()  # noqa: SLF001 - targeted behavior
        widget.keypoint_list.clearSelection()

        # New keypoint becomes active later in another frame.
        row = widget.keypoint_order().index("paw")
        widget.keypoint_list.item(row).setCheckState(QtCore.Qt.Checked)

        assert widget.sequence_active_keypoints() == [  # noqa: SLF001
            "nose",
            "left_ear",
            "right_ear",
            "tail_base",
        ]

        got = [widget.consume_next_label() for _ in range(5)]
        assert got == ["nose", "left_ear", "right_ear", "tail_base", "nose"]
    finally:
        widget.close()


def test_keypoint_sequencer_locked_items_can_be_reordered():
    _ensure_qapp()
    widget = KeypointSequencerWidget()
    try:
        schema = PoseSchema.from_dict({"keypoints": ["nose", "left_ear", "right_ear"]})
        widget.set_pose_schema(schema)
        widget.set_active_keypoints(["nose", "left_ear", "right_ear"])
        widget.enable_checkbox.setChecked(True)

        for name in ["nose", "left_ear", "right_ear"]:
            row = widget.keypoint_order().index(name)
            widget.keypoint_list.item(row).setSelected(True)
        widget._lock_selected_keypoints()  # noqa: SLF001 - targeted behavior

        widget.keypoint_list.clearSelection()
        row = widget.keypoint_order().index("right_ear")
        widget.keypoint_list.setCurrentRow(row)
        widget._move_selected_keypoints(-1)  # noqa: SLF001 - targeted behavior

        assert widget.locked_keypoints() == ["nose", "right_ear", "left_ear"]  # noqa: SLF001
    finally:
        widget.close()


def test_keypoint_sequencer_toggle_lock_for_item():
    _ensure_qapp()
    widget = KeypointSequencerWidget()
    try:
        schema = PoseSchema.from_dict({"keypoints": ["nose", "left_ear"]})
        widget.set_pose_schema(schema)
        item = widget.keypoint_list.item(0)
        widget._toggle_lock_for_item(item)  # noqa: SLF001 - targeted behavior
        assert widget.locked_keypoints() == ["nose"]
        widget._toggle_lock_for_item(item)  # noqa: SLF001 - targeted behavior
        assert widget.locked_keypoints() == []
    finally:
        widget.close()


def test_keypoint_sequencer_manual_add_syncs_pose_schema():
    _ensure_qapp()
    widget = KeypointSequencerWidget()
    try:
        widget.set_pose_schema(None)
        widget.add_keypoint_edit.setText("nose,left_ear")
        widget._add_keypoint_from_input()  # noqa: SLF001 - targeted behavior
        assert widget.current_schema() is not None
        assert widget.current_schema().keypoints == ["nose", "left_ear"]
    finally:
        widget.close()


def test_keypoint_sequencer_auto_normalizes_prefixed_schema_on_load(tmp_path: Path):
    _ensure_qapp()
    widget = KeypointSequencerWidget()
    try:
        schema_path = tmp_path / "pose_schema.json"
        schema_path.write_text(
            (
                '{"keypoints":["resident_nose","resident_tail_base",'
                '"intruder_nose","intruder_tail_base"],'
                '"edges":[["resident_nose","resident_tail_base"]]}'
            ),
            encoding="utf-8",
        )
        assert widget.load_schema_from_path(str(schema_path), quiet=True) is True
        assert widget.current_schema() is not None
        assert widget.current_schema().instances == ["resident", "intruder"]
        assert widget.current_schema().keypoints == ["nose", "tail_base"]
    finally:
        widget.close()


def test_keypoint_sequencer_locked_iteration_ignores_active_check_state():
    _ensure_qapp()
    widget = KeypointSequencerWidget()
    try:
        schema = PoseSchema.from_dict({"keypoints": ["nose", "left_ear", "right_ear"]})
        widget.set_pose_schema(schema)
        widget.enable_checkbox.setChecked(True)
        widget.set_active_keypoints(["nose", "left_ear", "right_ear"])

        for name in ["nose", "left_ear"]:
            row = widget.keypoint_order().index(name)
            widget.keypoint_list.item(row).setSelected(True)
        widget._lock_selected_keypoints()  # noqa: SLF001 - targeted behavior

        # Uncheck all active boxes; lock sequence should still drive labeling.
        for i in range(widget.keypoint_list.count()):
            widget.keypoint_list.item(i).setCheckState(QtCore.Qt.Unchecked)

        assert widget.is_sequence_enabled() is True
        got = [widget.consume_next_label() for _ in range(3)]
        assert got == ["nose", "left_ear", "nose"]
    finally:
        widget.close()


def test_keypoint_sequencer_locked_order_persists_across_schema_switch():
    _ensure_qapp()
    widget = KeypointSequencerWidget()
    try:
        schema_a = PoseSchema.from_dict(
            {"keypoints": ["nose", "left_ear", "right_ear", "tail_base"]}
        )
        widget.set_pose_schema(schema_a)
        widget.enable_checkbox.setChecked(True)
        widget.set_active_keypoints(["nose", "left_ear", "right_ear", "tail_base"])

        for name in ["nose", "left_ear", "tail_base"]:
            row = widget.keypoint_order().index(name)
            widget.keypoint_list.item(row).setSelected(True)
        widget._lock_selected_keypoints()  # noqa: SLF001 - targeted behavior
        assert widget.locked_keypoints() == ["nose", "left_ear", "tail_base"]  # noqa: SLF001

        # Simulate switching to another image/schema with different visible keypoints.
        schema_b = PoseSchema.from_dict({"keypoints": ["paw", "ear_tip"]})
        widget.set_pose_schema(schema_b)

        assert widget.locked_keypoints() == ["nose", "left_ear", "tail_base"]  # noqa: SLF001
        assert widget.consume_next_label() == "nose"
        assert widget.consume_next_label() == "left_ear"
        assert widget.consume_next_label() == "tail_base"
    finally:
        widget.close()
