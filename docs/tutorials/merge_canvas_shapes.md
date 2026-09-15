# Merge canvas shapes

Use **Edit Polygons** mode on the standard image/video canvas. Select two or
more polygons or rectangles in the Label Instances list (use Ctrl/Cmd-click for
multiple selection), then choose **Edit → Merge Selected Shapes**, or right-click
on the selected shapes and choose **Merge Selected Shapes**.

The selected shapes become one polygon covering their exact combined area.
Overlapping shapes, contained shapes, and shapes sharing an edge are supported.
The merged shape stays selected; **Undo** restores the original shapes. Save the
annotation normally to keep the result.

Shapes must have matching labels, group IDs, flags, descriptions, and custom
metadata. Edit conflicting values to match before merging. Hidden shapes,
attached masks, custom vertex labels, invalid polygons, disconnected unions,
and unions containing holes are rejected with an explanation; the originals
remain unchanged. Points, lines, circles, and the tiled large-image editor are
not supported by this action.
