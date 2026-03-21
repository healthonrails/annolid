from __future__ import annotations

from types import SimpleNamespace

from annolid.gui.app import AnnolidWindow


class _FakeDock:
    def __init__(self, on_show=None) -> None:
        self._on_show = on_show
        self.raised = False

    def show(self) -> None:
        if callable(self._on_show):
            self._on_show()

    def raise_(self) -> None:
        self.raised = True


def test_set_unrelated_docks_visible_restore_handles_reentrant_mutation() -> None:
    holder = SimpleNamespace(_other_docks_states={})

    def _reentrant_mutation() -> None:
        holder._other_docks_states[_FakeDock()] = True

    dock = _FakeDock(on_show=_reentrant_mutation)
    holder._other_docks_states[dock] = True

    AnnolidWindow.set_unrelated_docks_visible(holder, True)

    # Re-entrant mutation should not crash and should be preserved for the
    # next visibility cycle.
    assert len(holder._other_docks_states) == 1
    assert dock.raised is True
