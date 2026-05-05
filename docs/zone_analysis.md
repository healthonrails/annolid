# Zone Analysis Reference

This page describes the current zone-analysis contract used by Annolid's GUI and post-processing tools.

## Saved Zone Contract

New zone shapes should include explicit metadata:

- `semantic_type=zone`
- `zone_kind`
- `zone_group`
- `phase`
- `occupant_role`
- `access_state`

For three-chamber social setups, mark connecting tubes as neutral transit with either:

- `zone_kind=connector_tube`, or
- `occupant_role=neutral`, or
- `flags.neutral_zone=true`.

Legacy zone JSON files are still accepted when they use bounded compatibility hints, but explicit metadata is the preferred contract for new projects.

## Supported Outputs

### Legacy Place-Preference CSV

This is the historical output. It keeps one column per zone and is useful when you need compatibility with older scripts.

### Generic Zone Metrics CSV

This output adds:

- occupancy frames and seconds
- dwell frames and seconds
- entry counts
- aggregate chamber occupancy/entries (`all_chambers`, `stim_chambers`)
- aggregate neutral-transit occupancy/entries (`neutral_transit_zones`)
- transition counts
- barrier-adjacent frames and seconds
- outside-zone counts

If a `*_tracked.csv` file already contains zone columns, Annolid uses those
explicit membership columns for occupancy scoring. Otherwise it falls back to
centroid-in-zone geometry.

### Zone-Corrected Tracked CSV

This output applies an optional zone occupancy policy and writes a corrected CSV
without changing frame JSON files or centroid coordinates. It is intended for
camera-projection cases where a known prior constrains an animal to a legal
zone even when the raw centroid crosses a simple boundary.

Outputs:

- `session_tracked_zone_corrected.csv`
- `session_tracked_zone_corrected_audit.csv`

The audit CSV records the frame, instance, rule name, raw membership, corrected
membership, and changed zone columns.

### Assay Summary Markdown + CSV

This output explains:

- which zones were included
- which zones were excluded
- which zones were accessible or blocked
- which phase profile was used
- which metrics were computed

### Social Summary Markdown + CSV

This output is intended for social-approach assays and adds:

- latency to first enter each social zone
- zone dwell and occupancy using the rover-side social zones
- door proximity based on the selected anchor point
- pairwise centroid proximity across tracked voles
- the anchor source used for each instance

Neutral connector/tube zones are excluded from social scoring by default unless you explicitly opt in with social tags or `include_in_social=true`.

Latency is measured from the first analyzed frame unless you supply an explicit reference frame in the Zone Analysis dialog.

## Available Assay Profiles

- **Generic**
  - Analyze every explicit zone without profile-specific access filtering.
- **Phase 1**
  - Intended for partially blocked chamber layouts.
- **Phase 2**
  - Intended for fully open chamber layouts.

The same saved zone file can produce different summaries by changing the selected profile.

## Arena Presets

The Zone Dock currently includes two one-click presets:

- **3x3 Chamber Layout**
- **3x3 Social Door Layout**

The chamber preset creates nine editable chamber zones on the live canvas. The social-door preset creates rover-side approach zones around the mesh doors.

You can still:

- rename chambers,
- recolor them,
- move or resize them,
- or delete shapes you do not need.

## Recommended File Layout

- Video: `session.mp4`
- Zone JSON: `session_zones.json`
- Zone occupancy policy JSON: `session_zones_policy.json`
- Generic metrics CSV: `session_zone_metrics.csv`
- Zone-corrected tracked CSV: `session_tracked_zone_corrected.csv`
- Zone-corrected audit CSV: `session_tracked_zone_corrected_audit.csv`
- Assay summary Markdown: `session_phase_1_assay_summary.md`
- Assay summary CSV: `session_phase_1_assay_summary.csv`
- Social summary Markdown: `session_generic_social_summary.md`
- Social summary CSV: `session_generic_social_summary.csv`
- Pairwise centroid CSV: `session_generic_social_summary_pairwise.csv`

## GUI Entry Points

- **Video Tools → Zones**
  - optionally open the zone dock to manage/edit zones on the current frame
  - zone shapes can also be created directly on canvas and marked in the label popup via **Zone type**
  - use the **Define Zones** tab to review frame context and the current zone inventory
  - use the **Zone Details** tab to edit the selected zone label, kind, group, phase, access state, and barrier-adjacent behavior
  - use the **Zone Policies** tab to build and save known-prior occupancy rules for tracked instances
  - use the **Metrics** tab to preview zone area plus the occupancy, dwell, entry, transition, and barrier metrics that the selected zone will influence
- **Video Tools → Zone Analysis**
  - select session inputs (video, zone JSON, FPS, optional latency reference)
  - optionally select a zone occupancy policy JSON for corrected zone scoring
  - choose an assay profile and output mode from one dialog
  - run one-click export for legacy CSV, generic metrics, corrected tracked CSV, assay summary, or social summary reports

## Zone Occupancy Policies

A zone occupancy policy changes only zone membership columns for analysis. It
does not move centroids and does not rewrite LabelMe JSON.

Example:

```json
{
  "instance_policies": [
    {
      "instance_name": "stim_D",
      "rules": [
        {
          "name": "stim_D_legal_chamber",
          "zone_group": "chamber",
          "mode": "force_one",
          "zone": "chamber_D"
        },
        {
          "name": "stim_D_tether_membership",
          "zone_group": "tether",
          "mode": "preserve_if_inside",
          "allowed_zones": ["tether_D"]
        }
      ]
    }
  ]
}
```

Supported rule modes:

- `force_one`: set exactly one zone in a group to `1` and the other group zones to `0`.
- `force_all`: set listed zones to `1`.
- `deny`: set listed zones to `0`.
- `allow_only`: clear all zones in the group except the listed zones.
- `preserve_if_inside`: keep raw membership for listed zones and clear other zones in the group.
- `prefer`: when several zones in a group are active, keep the first active preferred zone.

Use `zone_group` metadata on zone shapes when overlapping zones represent
different concepts. For example, chamber zones can use group `chamber`, while
tether-reach zones can use group `tether`; this allows the same frame to count
as both `chamber_D` and `tether_D` when that is the intended scoring rule.

You can author these policy files from **Video Tools → Zones → Zone Policies**.
The dock writes the same JSON structure shown above, so users do not need to
hand-edit policy files for common force, allow, prefer, and deny rules.
