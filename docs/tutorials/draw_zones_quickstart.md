# Draw Zones Quickstart (GUI)

Use this guide when you want to draw and manage zones quickly on a loaded frame.

## Before You Start

1. Open Annolid.
2. Load your video (or image) and move to a representative seed frame (commonly frame `0`).
3. Optional: open **Video Tools → Zones** if you want zone inventory/defaults/presets while editing.

If no frame is loaded, the zone panel will stay read-only.

## Fast Drawing Flow

Use Annolid's existing canvas drawing tools:

1. Choose a draw mode from the main Annolid toolbar or canvas context menu.
2. Draw any shape(s) on the live frame.
3. Name the shape in the label dialog and enable zone classification:
   - mark **Zone type** in the label dialog, or
   - use a zone label name such as `left_zone`, `right_chamber`, `tube_zone`.
4. Save the shape.
5. Optional: in **Video Tools → Zones**, review/edit zone semantics from **Zone Inventory** and **Zone Details**.

The Zone Dock no longer duplicates draw controls; it focuses on classification and zone metadata management.
If your label/description includes zone words (for example `zone`, `chamber`, `doorway`, `tube`, `passage`), Annolid will recognize it as a zone candidate (`zone (keyword)`) in the inventory.
Defined zones are displayed across all video frames so you can navigate and annotate without redrawing zone boundaries on each frame.

## Show Zones Across Frames

Use **View → Show Zones On All Frames**:

- enabled: zone overlays remain visible on every frame.
- disabled: zone overlays are hidden on non-local frames unless explicitly saved on that frame.

This toggle only affects zone overlays. It does not duplicate non-zone instance polygons across frames.

## Set Good Defaults First

In **Zone Details → Defaults for New Zones**, set:

- `zone_kind`
- `phase`
- `occupant_role`
- `access_state`
- `zone_group` when zones are mutually exclusive within one scoring concept
- optional tags

Every new zone you draw will inherit these values, which avoids repetitive editing.
When you mark a selected shape as a zone, these defaults are applied unless you override them in the selected-zone form.

Use `zone_group` to separate overlapping concepts. For example, chamber zones can
use `zone_group=chamber`, while tether-reach zones can use
`zone_group=tether`. This lets one frame count as both a chamber occupancy and a
tether occupancy when that is the intended analysis.

## Manage Zones Efficiently

In **Zone Inventory**:

1. Filter by label/kind/phase/role/tags.
2. Sort by label, kind, or area.
3. Select a zone to edit it in **Zone Details**.

Useful actions for the selected zone:

- **Use as Zone**: convert an existing non-zone annotation into a zone.
- **Apply Zone Details**: save label/metadata edits on an already classified zone.
- **Use Selected as Defaults**: copy current zone semantics for future drawings.
- **Duplicate**: clone zone + semantics (offset slightly for immediate editing).
- **Recolor**: visual differentiation only.
- **Delete Selected**: remove zone from current canvas.

## Save Zone JSON

Use **Save Zone JSON** (top row in the panel).

Recommended naming:

- `video_stem_zones.json`

The file stays LabelMe-compatible and can be reloaded with **Load Zone JSON**.

## Define Zone Occupancy Policies

Use **Zone Policies** in the Zone Dock when a known prior should correct zone
membership for a tracked instance without moving centroids. A typical example is
a stimulus animal that is known to remain in `chamber_D` even when transparent
walls make its centroid overlap another chamber boundary.

1. Enter the tracked instance name, such as `stim_D`.
2. Choose the zone group, such as `chamber`.
3. Choose a mode:
   - `force_one`: exactly one listed zone is active in the group.
   - `preserve_if_inside`: keep listed zones only when the raw membership is already inside.
   - `allow_only`: clear all group zones except the listed zones.
   - `prefer`: if multiple group zones are active, keep the first listed active zone.
   - `force_all`: set listed zones active.
   - `deny`: set listed zones inactive.
4. Enter one or more zone labels.
5. Click **Add Rule**, then **Save Policy JSON**.

The saved policy JSON can be selected in **Video Tools → Zone Analysis** for
**Zone-Corrected Tracked CSV** or other policy-aware zone exports. The correction
changes only zone columns and writes an audit CSV; it does not rewrite the
manual label JSON files.

## Practical Patterns

### Three-chamber assays

- Chambers: `zone_kind=chamber`
- Neutral connecting tubes: `zone_kind=connector_tube` or `occupant_role=neutral`
- Optional hard marker: `neutral_zone=true` in flags

### Social-zone scoring

Neutral connector/tube zones are excluded from social summaries by default.
If you want one included in social metrics, add social tags or set `include_in_social=true`.

## Common Mistakes

- Drawing before setting defaults, then having to retag every zone manually.
- Saving zones from the wrong frame/layout.
- Forgetting to enable **Zone type** in the label dialog when creating zone shapes.
- Reusing one zone JSON across sessions with different crops or geometry.

## Next Step

After drawing and saving zones, run:

- **Video Tools → Zone Analysis** for zone metrics and assay/social summaries.
- **Convert → Save CSV** with `Generate *_tracked.csv` enabled to include one zone-occupancy column per zone label (`1` if center is inside, `0` if outside).
