# Draw Zones Quickstart (GUI)

Use this guide when you want to draw and manage zones quickly on a loaded frame.

## Before You Start

1. Open Annolid.
2. Load your video (or image) and move to a representative frame.
3. Open **Video Tools → Zones**.

If no frame is loaded, the zone panel will stay read-only.

## Fast Drawing Flow

Use Annolid's existing canvas drawing tools:

1. Choose a draw mode from the main Annolid toolbar or canvas context menu.
2. Draw any shape(s) on the live frame.
3. In **Video Tools → Zones**, select a shape from **Zone Inventory**.
4. In **Zone Details**, click **Use as Zone**.

The Zone Dock no longer duplicates draw controls; it focuses on classification and zone metadata management.
If your label/description includes zone words (for example `zone`, `chamber`, `doorway`, `tube`, `passage`), Annolid will recognize it as a zone candidate (`zone (keyword)`) in the inventory.
Defined zones are displayed across all video frames so you can navigate and annotate without redrawing zone boundaries on each frame.

## Set Good Defaults First

In **Zone Details → Defaults for New Zones**, set:

- `zone_kind`
- `phase`
- `occupant_role`
- `access_state`
- optional tags

Every new zone you draw will inherit these values, which avoids repetitive editing.
When you mark a selected shape as a zone, these defaults are applied unless you override them in the selected-zone form.

## Manage Zones Efficiently

In **Zone Inventory**:

1. Filter by label/kind/phase/role/tags.
2. Sort by label, kind, or area.
3. Select a zone to edit it in **Zone Details**.

Useful actions for the selected zone:

- **Use as Zone**: convert the selected annotation shape into a zone.
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
- Forgetting to click **Use as Zone** after drawing shapes.
- Reusing one zone JSON across sessions with different crops or geometry.

## Next Step

After drawing and saving zones, run:

- **Video Tools → Zone Analysis** for zone metrics and assay/social summaries.
