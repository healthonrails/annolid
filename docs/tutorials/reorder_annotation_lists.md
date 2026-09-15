# Reorder annotation lists

Drag a row up or down in either the **Labels** summary or **Label Instances**
list and drop it at the insertion indicator. In Label Instances, select several
rows with Ctrl/Cmd-click or Shift-click to move them together. Drops are limited
to the same list.

- **Label Instances:** changes the shape order on the standard canvas and in the
  saved annotation JSON. Later rows are painted over earlier rows on the standard
  canvas. The annotation becomes unsaved; save normally to keep the order.
  **Undo** restores the previous shape order. Selection, checkboxes, labels,
  group IDs, and geometry are preserved.
- **Labels:** changes only the summary display order. Counts and colors still
  update normally. The custom order survives list refreshes and file/frame
  switches within the current window; new labels appear after the custom order.
  This display preference does not change shape order or mark annotations as
  unsaved. It resets when the window is closed.

Reordering does not rename labels or change any tracking identities. Tiled-image
rendering retains its existing overlay depth rules.
