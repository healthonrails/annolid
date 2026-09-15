import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from qtpy import QtWidgets

from annolid.gui.behavior_controller import BehaviorEvent
from annolid.gui.widgets.behavior_log import BehaviorEventLogWidget


@pytest.fixture
def widget():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = BehaviorEventLogWidget()
    yield panel
    panel.close()
    panel.deleteLater()
    app.processEvents()


def event(frame, boundary, subject="Mouse A", **kwargs):
    return BehaviorEvent(frame, "grooming", boundary, subject=subject, **kwargs)


def test_duration_pairs_subjects_and_modifiers_before_filtering(widget):
    events = [
        event(0, "start"),
        event(10, "start", "Mouse B"),
        event(15, "start", modifiers=("left",)),
        event(20, "end", confirmed=False),
        event(40, "end", "Mouse B"),
        event(45, "end", modifiers=("left",)),
    ]
    widget.set_events(events, fps=10)
    assert [widget._table.item(row, 8).text() for row in (3, 4, 5)] == [
        "2.00",
        "3.00",
        "3.00",
    ]
    widget._review_filter.setCurrentIndex(1)
    assert widget._display_events == [events[3]]
    assert widget._table.item(0, 8).text() == "2.00"
    assert not widget._pairing_issues


def test_pairing_issues_do_not_guess_or_mutate_events(widget):
    events = [
        event(0, "end"),
        event(1, "start"),
        event(2, "start"),
        event(3, "end"),
        event(4, "start", "Mouse B"),
    ]
    widget.set_events(events, fps=10)
    widget._review_filter.setCurrentIndex(2)
    assert len(widget._display_events) == 5
    assert widget._table.item(0, 9).text().endswith("Missing start")
    assert widget._table.item(3, 8).text() == "—"
    assert widget._table.item(4, 9).text().endswith("Missing end")
    assert widget._events == events


def test_search_selection_and_visible_actions(widget):
    first = event(0, "start", category="Maintenance")
    second = event(10, "end", category="Maintenance", confirmed=False)
    widget.set_events([first, second], fps=10)
    assert not widget._jump_button.isEnabled()
    widget._table.selectRow(1)
    jumps, reviews, edits = [], [], []
    widget.jumpToFrame.connect(jumps.append)
    widget.confirmRequested.connect(reviews.append)
    widget.editRequested.connect(edits.append)
    widget._search.setText("MAINTENANCE")
    assert widget.current_event() == second
    widget._jump_button.click()
    widget._review_button.click()
    widget._edit_button.click()
    assert jumps == [10]
    assert reviews == [second]
    assert edits == [second]
    widget._search.setText("no match")
    assert widget.current_event() is None
    assert not widget._review_button.isEnabled()
    assert "No events match" in widget._hint.text()


def test_clear_resets_filters_selection_and_fps(widget):
    widget.set_events([event(0, "start")], fps=30)
    widget._search.setText("grooming")
    widget._review_filter.setCurrentIndex(2)
    widget._table.selectRow(0)
    widget.clear()
    assert widget._search.text() == ""
    assert widget._review_filter.currentData() == "all"
    assert widget._fps is None
    assert widget.current_event() is None
    assert not widget._undo_button.isEnabled()
    widget.set_events([event(0, "start"), event(30, "end")])
    assert widget._table.item(1, 8).text() == "—"


def test_duration_uses_timestamps_and_reports_backwards_time(widget):
    widget.set_events([event(0, "start", timestamp=5), event(10, "end", timestamp=4)])
    assert len(widget._pairing_issues) == 2
    assert widget._table.item(1, 8).text() == "—"
