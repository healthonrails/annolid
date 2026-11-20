"""Bidirectional interoperability with BORIS (.boris JSON and Tabular/Aggregated CSV/TSV).

Supports:
- Importing native BORIS project files (``.boris`` v7/v8/v9 JSON) into Annolid
  ``ProjectSchema`` (ethogram behaviors, point vs. state types, shortcut keys,
  categories, structured single/multi-select modifier sets, exclusion matrices,
  focal subjects, and independent variables) plus per-observation ``ParsedEventRow``
  streams and ready-to-run ``cohort_manifest.json`` files.
- Exporting Annolid ``ProjectSchema`` and observation event logs back to valid
  ``.boris`` project JSON files as well as BORIS Tabular Events and Aggregated
  Events CSV/TSV tables.
"""

from __future__ import annotations

from collections import defaultdict, deque
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from annolid.behavior.event_utils import (
    DEFAULT_SUBJECT_LABEL,
    ParsedEventRow,
    classify_event_label,
    parse_event_mapping_rows,
    parse_recording_seconds,
    read_parsed_event_rows,
)
from annolid.behavior.external_data import ExternalDataStreamSpec
from annolid.behavior.project_schema import (
    BehaviorDefinition,
    CategoryDefinition,
    IndependentVariableDefinition,
    ModifierDefinition,
    ProjectSchema,
    SubjectDefinition,
    save_schema,
)
from annolid.utils.atomic_io import atomic_write_json, atomic_write_text

__all__ = [
    "BorisIOError",
    "BorisObservation",
    "BorisProjectImport",
    "BorisMaterializedArtifacts",
    "import_boris_project",
    "export_boris_project",
    "write_boris_project",
    "materialize_boris_project",
    "write_annolid_events_csv",
    "write_boris_tabular_csv",
    "write_boris_aggregated_csv",
]

_MODIFIER_KEY_SUFFIX_RE = re.compile(r"^(.*?)\s*\(([A-Za-z0-9])\)\s*$")
_DEFAULT_CATEGORY_PALETTE = (
    "#2E7D32",
    "#1565C0",
    "#EF6C00",
    "#6A1B9A",
    "#00838F",
    "#C62828",
    "#455A64",
    "#AD1457",
)


class BorisIOError(Exception):
    """Raised when a BORIS project or export cannot be parsed or serialized."""


@dataclass(frozen=True)
class BorisObservation:
    """Single coded observation session extracted from or destined for a .boris file."""

    observation_id: str
    events: Tuple[ParsedEventRow, ...] = ()
    media_files: Tuple[str, ...] = ()
    fps: Optional[float] = None
    time_offset_s: float = 0.0
    description: str = ""
    observation_date: str = ""
    independent_variables: Dict[str, str] = field(default_factory=dict)
    media_by_player: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    media_offsets: Dict[str, float] = field(default_factory=dict)
    external_data: Tuple[ExternalDataStreamSpec, ...] = ()


@dataclass(frozen=True)
class BorisProjectImport:
    """Complete result of importing a ``.boris`` project file."""

    project_name: str
    project_description: str
    project_date: str
    format_version: str
    schema: ProjectSchema
    observations: Tuple[BorisObservation, ...]
    warnings: Tuple[str, ...] = ()
    source_path: Optional[Path] = None
    converters: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BorisMaterializedArtifacts:
    """Paths written when converting a ``.boris`` project into an Annolid directory."""

    output_dir: Path
    schema_path: Path
    observation_csvs: Dict[str, Path]
    cohort_manifest_path: Optional[Path]
    warnings: Tuple[str, ...] = ()


def _slugify_id(raw: object, fallback: str) -> str:
    text = str(raw or "").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return slug or fallback


def _ordered_boris_entries(container: object) -> List[Mapping[str, Any]]:
    """Normalize BORIS ``{"0": {...}, "1": {...}}`` dicts or lists into ordered mappings."""
    if isinstance(container, list):
        return [item for item in container if isinstance(item, Mapping)]
    if isinstance(container, Mapping):

        def _sort_key(kv: Tuple[Any, Any]) -> Tuple[int, int, str]:
            raw_k = str(kv[0]).strip()
            try:
                return (0, int(raw_k), raw_k)
            except ValueError:
                return (1, 0, raw_k)

        return [
            val
            for _k, val in sorted(container.items(), key=_sort_key)
            if isinstance(val, Mapping)
        ]
    return []


def _parse_boris_modifier_value(raw_val: object) -> Tuple[str, Optional[str]]:
    text = str(raw_val or "").strip()
    if not text:
        return "", None
    match = _MODIFIER_KEY_SUFFIX_RE.match(text)
    if match:
        return match.group(1).strip(), match.group(2)
    return text, None


def _boris_mod_type_to_mode(raw_type: object) -> str:
    if isinstance(raw_type, int):
        if raw_type == 0:
            return "single"
        if raw_type == 1:
            return "multi"
        if raw_type in (2, 3):
            return "numeric"
    txt = str(raw_type or "").strip().lower()
    if "single" in txt or txt == "0":
        return "single"
    if "num" in txt or txt == "2":
        return "numeric"
    if "text" in txt:
        return "text"
    return "multi"


def _mode_to_boris_mod_type(mode: str) -> int:
    norm = str(mode or "multi").strip().lower()
    if norm == "single":
        return 0
    if norm in {"numeric", "text"}:
        return 2
    return 1


def _boris_iv_type_to_schema(raw_type: object) -> str:
    txt = str(raw_type or "text").strip().lower()
    if "num" in txt:
        return "numeric"
    if "list" in txt or "set" in txt or "val" in txt:
        return "value_from_list"
    if "time" in txt or "date" in txt:
        return "timestamp"
    return "text"


def _split_excluded(raw_excluded: object) -> List[str]:
    if raw_excluded is None:
        return []
    if isinstance(raw_excluded, (list, tuple, set)):
        return [str(x).strip() for x in raw_excluded if str(x).strip()]
    text = str(raw_excluded).strip()
    if not text:
        return []
    return [part.strip() for part in re.split(r"[,;|]", text) if part.strip()]


def _extract_media_by_player(
    obs_dict: Mapping[str, Any],
) -> Dict[str, Tuple[str, ...]]:
    raw_file = obs_dict.get("file")
    by_player: Dict[str, Tuple[str, ...]] = {}
    if isinstance(raw_file, Mapping):
        for player_key, paths in sorted(raw_file.items(), key=lambda kv: str(kv[0])):
            p_str = str(player_key).strip()
            items: List[str] = []
            if isinstance(paths, (list, tuple)):
                for p in paths:
                    s = str(p or "").strip()
                    if s and s not in items:
                        items.append(s)
            elif isinstance(paths, str) and paths.strip():
                items.append(paths.strip())
            if items:
                by_player[p_str] = tuple(items)
    elif isinstance(raw_file, (list, tuple)):
        items = [str(p).strip() for p in raw_file if str(p or "").strip()]
        if items:
            by_player["1"] = tuple(items)
    elif isinstance(raw_file, str) and raw_file.strip():
        by_player["1"] = (raw_file.strip(),)
    return by_player


def _extract_media_offsets(obs_dict: Mapping[str, Any]) -> Dict[str, float]:
    offsets: Dict[str, float] = {}
    media_info = obs_dict.get("media_info")
    if isinstance(media_info, Mapping):
        raw_off = media_info.get("offset")
        if isinstance(raw_off, Mapping):
            for k, v in raw_off.items():
                parsed = parse_recording_seconds(v)
                if parsed is not None:
                    offsets[str(k).strip()] = float(parsed)
    return offsets


def _extract_external_data(
    obs_dict: Mapping[str, Any],
    *,
    project_converters: Optional[Mapping[str, Any]] = None,
) -> Tuple[ExternalDataStreamSpec, ...]:
    raw_plot = obs_dict.get("plot_data")
    streams: List[ExternalDataStreamSpec] = []
    if isinstance(raw_plot, Mapping):
        for _k, entry in sorted(raw_plot.items(), key=lambda kv: str(kv[0])):
            if isinstance(entry, Mapping):
                spec = ExternalDataStreamSpec.from_boris_plot_dict(
                    entry, project_converters=project_converters
                )
                if spec.file_path:
                    streams.append(spec)
    elif isinstance(raw_plot, (list, tuple)):
        for entry in raw_plot:
            if isinstance(entry, Mapping):
                spec = ExternalDataStreamSpec.from_boris_plot_dict(
                    entry, project_converters=project_converters
                )
                if spec.file_path:
                    streams.append(spec)
    return tuple(streams)


def _extract_media_files(obs_dict: Mapping[str, Any]) -> Tuple[str, ...]:
    raw_file = obs_dict.get("file")
    collected: List[str] = []
    if isinstance(raw_file, Mapping):
        for _player_key, paths in sorted(raw_file.items(), key=lambda kv: str(kv[0])):
            if isinstance(paths, (list, tuple)):
                for p in paths:
                    s = str(p or "").strip()
                    if s and s not in collected:
                        collected.append(s)
            elif isinstance(paths, str) and paths.strip():
                if paths.strip() not in collected:
                    collected.append(paths.strip())
    elif isinstance(raw_file, (list, tuple)):
        for p in raw_file:
            s = str(p or "").strip()
            if s and s not in collected:
                collected.append(s)
    elif isinstance(raw_file, str) and raw_file.strip():
        collected.append(raw_file.strip())
    return tuple(collected)


def _extract_obs_fps(obs_dict: Mapping[str, Any]) -> Optional[float]:
    for key in ("fps", "media_info"):
        val = obs_dict.get(key)
        if isinstance(val, (int, float)) and val > 0:
            return float(val)
        if isinstance(val, (list, tuple)) and val:
            first = parse_recording_seconds(val[0])
            if first and first > 0:
                return first
        if isinstance(val, Mapping):
            fps_map = val.get("fps") if "fps" in val else val
            if isinstance(fps_map, Mapping) and fps_map:
                first_val = next(iter(fps_map.values()))
                parsed = parse_recording_seconds(first_val)
                if parsed and parsed > 0:
                    return parsed
    return None


def import_boris_project(
    source: Path | str | Mapping[str, Any],
    *,
    source_path: Optional[Path | str] = None,
    apply_time_offset: bool = False,
) -> BorisProjectImport:
    """Parse a ``.boris`` JSON file or dictionary into :class:`BorisProjectImport`."""
    resolved_path: Optional[Path] = (
        Path(source_path) if source_path is not None else None
    )
    if isinstance(source, (str, Path)):
        resolved_path = Path(source)
        if not resolved_path.exists():
            raise FileNotFoundError(f"BORIS project file not found: {resolved_path}")
        try:
            payload = json.loads(resolved_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise BorisIOError(
                f"Invalid JSON in BORIS project '{resolved_path}': {exc}"
            ) from exc
    elif isinstance(source, Mapping):
        payload = dict(source)
    else:
        raise BorisIOError("BORIS project input must be a file path or mapping.")

    if not isinstance(payload, Mapping):
        raise BorisIOError("BORIS project root must be a JSON object.")

    warnings: List[str] = []
    project_name = str(
        payload.get("project_name")
        or (resolved_path.stem if resolved_path is not None else "boris_project")
    ).strip()
    project_desc = str(
        payload.get("project_description") or payload.get("description") or ""
    ).strip()
    project_date = str(payload.get("project_date") or payload.get("date") or "").strip()
    fmt_version = str(
        payload.get("project_format_version") or payload.get("boris_version") or "7.0"
    ).strip()

    # 1. Categories
    categories: List[CategoryDefinition] = []
    cat_name_to_id: Dict[str, str] = {}
    raw_cats = payload.get("behavioral_categories") or []
    raw_cats_conf = payload.get("behavioral_categories_conf") or {}
    conf_by_name: Dict[str, Mapping[str, Any]] = {}
    if isinstance(raw_cats_conf, Mapping):
        for entry in raw_cats_conf.values():
            if isinstance(entry, Mapping) and entry.get("name"):
                conf_by_name[str(entry["name"]).strip()] = entry

    if isinstance(raw_cats, list):
        for idx, raw_cat in enumerate(raw_cats):
            if isinstance(raw_cat, Mapping):
                c_name = str(raw_cat.get("name") or raw_cat.get("id") or "").strip()
                c_color = str(raw_cat.get("color") or "").strip() or None
                c_desc = str(raw_cat.get("description") or "").strip() or None
            else:
                c_name = str(raw_cat or "").strip()
                conf = conf_by_name.get(c_name, {})
                c_color = str(conf.get("color") or "").strip() or None
                c_desc = str(conf.get("description") or "").strip() or None
            if not c_name or c_name in cat_name_to_id:
                continue
            cid = _slugify_id(c_name, f"category_{idx + 1}")
            if c_color is None:
                c_color = _DEFAULT_CATEGORY_PALETTE[
                    len(categories) % len(_DEFAULT_CATEGORY_PALETTE)
                ]
            categories.append(
                CategoryDefinition(
                    id=cid,
                    name=c_name,
                    color=c_color,
                    description=c_desc,
                )
            )
            cat_name_to_id[c_name] = cid

    # 2. Subjects
    subjects: List[SubjectDefinition] = []
    raw_subjects = _ordered_boris_entries(
        payload.get("subjects_conf")
        if "subjects_conf" in payload
        else payload.get("subjects")
    )
    seen_subj_ids: set[str] = set()
    for idx, s_entry in enumerate(raw_subjects):
        s_name = str(
            s_entry.get("name") or s_entry.get("subject") or s_entry.get("id") or ""
        ).strip()
        if not s_name:
            continue
        sid = _slugify_id(s_entry.get("id") or s_name, f"subject_{idx + 1}")
        if sid in seen_subj_ids:
            sid = f"{sid}_{idx + 1}"
        seen_subj_ids.add(sid)
        s_key = (
            str(s_entry.get("key") or s_entry.get("key_binding") or "").strip() or None
        )
        s_desc = str(s_entry.get("description") or "").strip() or None
        s_color = str(s_entry.get("color") or "").strip() or None
        subjects.append(
            SubjectDefinition(
                id=sid,
                name=s_name,
                description=s_desc,
                key_binding=s_key,
                color=s_color,
            )
        )

    # 3. Independent Variables
    indep_vars: List[IndependentVariableDefinition] = []
    raw_ivs = _ordered_boris_entries(payload.get("independent_variables"))
    seen_iv_ids: set[str] = set()
    for idx, iv_entry in enumerate(raw_ivs):
        iv_label = str(
            iv_entry.get("label") or iv_entry.get("name") or iv_entry.get("id") or ""
        ).strip()
        if not iv_label:
            continue
        iv_id = _slugify_id(iv_entry.get("id") or iv_label, f"var_{idx + 1}")
        if iv_id in seen_iv_ids:
            iv_id = f"{iv_id}_{idx + 1}"
        seen_iv_ids.add(iv_id)
        var_type = _boris_iv_type_to_schema(iv_entry.get("type"))
        default_val = (
            str(
                iv_entry.get("default value")
                if "default value" in iv_entry
                else (iv_entry.get("default_value") or "")
            ).strip()
            or None
        )
        raw_possible = (
            iv_entry.get("possible values")
            if "possible values" in iv_entry
            else iv_entry.get("allowed_values")
        )
        if isinstance(raw_possible, (list, tuple)):
            allowed_vals = [str(v).strip() for v in raw_possible if str(v).strip()]
        elif isinstance(raw_possible, str) and raw_possible.strip():
            allowed_vals = [v.strip() for v in raw_possible.split(",") if v.strip()]
        else:
            allowed_vals = []
        iv_desc = str(iv_entry.get("description") or "").strip() or None
        indep_vars.append(
            IndependentVariableDefinition(
                id=iv_id,
                name=iv_label,
                var_type=var_type,
                default_value=default_val,
                allowed_values=allowed_vals,
                description=iv_desc,
            )
        )

    # 4. Ethogram Behaviors, Modifiers, and Exclusion Matrix
    modifiers: List[ModifierDefinition] = []
    mod_by_key: Dict[Tuple[str, str], str] = {}  # (group_lower, name_lower) -> mod.id
    behaviors: List[BehaviorDefinition] = []
    state_by_code: Dict[str, bool] = {}

    raw_behaviors = _ordered_boris_entries(
        payload.get("behaviors_conf")
        if "behaviors_conf" in payload
        else payload.get("ethogram")
    )
    exclusion_matrix = payload.get("exclusion_matrix")
    matrix_exclusions: Dict[str, set[str]] = defaultdict(set)
    if isinstance(exclusion_matrix, Mapping):
        for beh_a, row_map in exclusion_matrix.items():
            if isinstance(row_map, Mapping):
                for beh_b, is_excl in row_map.items():
                    if bool(is_excl) and str(beh_a).strip() != str(beh_b).strip():
                        matrix_exclusions[str(beh_a).strip()].add(str(beh_b).strip())
                        matrix_exclusions[str(beh_b).strip()].add(str(beh_a).strip())

    seen_codes: set[str] = set()
    for idx, b_entry in enumerate(raw_behaviors):
        raw_code = str(b_entry.get("code") or b_entry.get("behavior") or "").strip()
        if not raw_code:
            continue
        if raw_code in seen_codes:
            warnings.append(f"Skipping duplicate BORIS behavior code '{raw_code}'.")
            continue
        seen_codes.add(raw_code)

        b_name = str(b_entry.get("name") or b_entry.get("behavior") or raw_code).strip()
        b_desc = str(b_entry.get("description") or "").strip() or None
        b_key = (
            str(
                b_entry.get("key")
                or b_entry.get("key_combination")
                or b_entry.get("key_binding")
                or ""
            ).strip()
            or None
        )
        b_type_str = str(b_entry.get("type") or "State event").strip().lower()
        is_state = "point" not in b_type_str
        state_by_code[raw_code] = is_state
        state_by_code[b_name] = is_state

        raw_cat_name = str(b_entry.get("category") or "").strip()
        cat_id: Optional[str] = None
        if raw_cat_name:
            if raw_cat_name not in cat_name_to_id:
                cid = _slugify_id(raw_cat_name, f"category_{len(categories) + 1}")
                color_auto = _DEFAULT_CATEGORY_PALETTE[
                    len(categories) % len(_DEFAULT_CATEGORY_PALETTE)
                ]
                categories.append(
                    CategoryDefinition(id=cid, name=raw_cat_name, color=color_auto)
                )
                cat_name_to_id[raw_cat_name] = cid
            cat_id = cat_name_to_id[raw_cat_name]

        b_color = str(b_entry.get("color") or "").strip() or None

        # Parse BORIS modifier sets attached to this behavior
        beh_mod_ids: List[str] = []
        raw_mods = b_entry.get("modifiers")
        if isinstance(raw_mods, Mapping):
            mod_sets = _ordered_boris_entries(raw_mods)
            for set_idx, mset in enumerate(mod_sets):
                group_name = (
                    str(mset.get("name") or f"Set {set_idx + 1}").strip()
                    or f"Set {set_idx + 1}"
                )
                sel_mode = _boris_mod_type_to_mode(mset.get("type"))
                values = mset.get("values") or []
                if sel_mode in {"numeric", "text"} and not values:
                    values = [group_name]
                for val in values:
                    val_name, val_key = _parse_boris_modifier_value(val)
                    if not val_name:
                        continue
                    lookup_k = (group_name.lower(), val_name.lower())
                    mid = mod_by_key.get(lookup_k)
                    if mid is None:
                        base_mid = _slugify_id(
                            f"{group_name}_{val_name}"
                            if group_name.lower() != val_name.lower()
                            else val_name,
                            f"mod_{len(modifiers) + 1}",
                        )
                        existing_mids = {m.id for m in modifiers}
                        mid = base_mid
                        if mid in existing_mids:
                            mid = f"{base_mid}_{len(modifiers) + 1}"
                        modifiers.append(
                            ModifierDefinition(
                                id=mid,
                                name=val_name,
                                description=str(mset.get("description") or "").strip()
                                or None,
                                group=group_name,
                                selection_mode=sel_mode,
                                key_binding=val_key,
                            )
                        )
                        mod_by_key[lookup_k] = mid
                    if mid not in beh_mod_ids:
                        beh_mod_ids.append(mid)
        elif isinstance(raw_mods, (list, tuple)):
            for raw_m in raw_mods:
                if isinstance(raw_m, Mapping):
                    group_name = str(raw_m.get("name") or "Modifiers").strip()
                    sel_mode = _boris_mod_type_to_mode(raw_m.get("type"))
                    for val in raw_m.get("values") or [
                        raw_m.get("value") or group_name
                    ]:
                        val_name, val_key = _parse_boris_modifier_value(val)
                        if not val_name:
                            continue
                        lookup_k = (group_name.lower(), val_name.lower())
                        mid = mod_by_key.get(lookup_k)
                        if mid is None:
                            mid = _slugify_id(val_name, f"mod_{len(modifiers) + 1}")
                            modifiers.append(
                                ModifierDefinition(
                                    id=mid,
                                    name=val_name,
                                    group=group_name,
                                    selection_mode=sel_mode,
                                    key_binding=val_key,
                                )
                            )
                            mod_by_key[lookup_k] = mid
                        if mid not in beh_mod_ids:
                            beh_mod_ids.append(mid)
                elif isinstance(raw_m, str) and raw_m.strip():
                    val_name, val_key = _parse_boris_modifier_value(raw_m)
                    lookup_k = ("default", val_name.lower())
                    mid = mod_by_key.get(lookup_k)
                    if mid is None:
                        mid = _slugify_id(val_name, f"mod_{len(modifiers) + 1}")
                        modifiers.append(
                            ModifierDefinition(
                                id=mid,
                                name=val_name,
                                group=None,
                                selection_mode="multi",
                                key_binding=val_key,
                            )
                        )
                        mod_by_key[lookup_k] = mid
                    if mid not in beh_mod_ids:
                        beh_mod_ids.append(mid)

        excl = set(_split_excluded(b_entry.get("excluded")))
        excl.update(matrix_exclusions.get(raw_code, set()))
        excl.discard(raw_code)

        behaviors.append(
            BehaviorDefinition(
                code=raw_code,
                name=b_name,
                description=b_desc,
                category_id=cat_id,
                modifier_ids=beh_mod_ids,
                key_binding=b_key,
                is_state=is_state,
                exclusive_with=sorted(excl),
                color=b_color,
            )
        )

    # Ensure symmetric exclusive_with references across imported behaviors
    beh_codes = {b.code for b in behaviors}
    sym_excl: Dict[str, set[str]] = {b.code: set(b.exclusive_with) for b in behaviors}
    for b in behaviors:
        for other in list(b.exclusive_with):
            if other in beh_codes and other != b.code:
                sym_excl[other].add(b.code)
    for b in behaviors:
        b.exclusive_with = sorted(sym_excl[b.code])

    # 5. Observations & Events
    raw_proj_converters = payload.get("converters")
    project_converters: Dict[str, Any] = (
        {str(k): v for k, v in raw_proj_converters.items()}
        if isinstance(raw_proj_converters, Mapping)
        else {}
    )
    observations: List[BorisObservation] = []
    raw_observations = payload.get("observations") or {}
    if isinstance(raw_observations, Mapping):
        for obs_id, obs_val in raw_observations.items():
            if not isinstance(obs_val, Mapping):
                continue
            time_offset = (
                parse_recording_seconds(
                    obs_val.get("time_offset", obs_val.get("time offset"))
                )
                or 0.0
            )
            media_files = _extract_media_files(obs_val)
            media_by_player = _extract_media_by_player(obs_val)
            media_offsets = _extract_media_offsets(obs_val)
            external_data = _extract_external_data(
                obs_val, project_converters=project_converters
            )
            fps_val = _extract_obs_fps(obs_val)
            obs_desc = str(obs_val.get("description") or "").strip()
            obs_date = str(obs_val.get("date") or "").strip()

            raw_iv_vals = (
                obs_val.get("independent_variables")
                if isinstance(obs_val.get("independent_variables"), Mapping)
                else obs_val.get("independent_variables_values")
            )
            iv_values: Dict[str, str] = {}
            if isinstance(raw_iv_vals, Mapping):
                for k, v in raw_iv_vals.items():
                    if v is not None and str(v).strip():
                        iv_values[str(k).strip()] = str(v).strip()

            active_states: Dict[Tuple[str, str], bool] = {}
            parsed_events: List[ParsedEventRow] = []
            raw_events = obs_val.get("events") or []
            for ev_idx, ev in enumerate(raw_events, start=1):
                if isinstance(ev, (list, tuple)) and len(ev) >= 3:
                    raw_ts = ev[0]
                    raw_subj = str(ev[1] or "").strip()
                    beh_code = str(ev[2] or "").strip()
                    explicit_status: Optional[str] = None
                    raw_mod = ""
                    raw_comment = ""
                    if len(ev) >= 6 and classify_event_label(str(ev[3] or "")) in {
                        "start",
                        "end",
                        "point",
                    }:
                        explicit_status = str(ev[3] or "").strip()
                        raw_mod = str(ev[4] or "").strip()
                        raw_comment = str(ev[5] or "").strip()
                    else:
                        if len(ev) >= 4:
                            cand_3 = str(ev[3] or "").strip()
                            if len(ev) == 4 and classify_event_label(cand_3) in {
                                "start",
                                "end",
                                "point",
                            }:
                                explicit_status = cand_3
                            else:
                                raw_mod = cand_3
                        if len(ev) >= 5:
                            raw_comment = str(ev[4] or "").strip()
                elif isinstance(ev, Mapping):
                    raw_ts = ev.get("timestamp", ev.get("time"))
                    raw_subj = str(ev.get("subject") or "").strip()
                    beh_code = str(ev.get("behavior") or ev.get("code") or "").strip()
                    explicit_status = (
                        str(ev.get("status") or ev.get("event") or "").strip() or None
                    )
                    raw_mod = str(
                        ev.get("modifiers") or ev.get("modifier") or ""
                    ).strip()
                    raw_comment = str(ev.get("comment") or "").strip()
                else:
                    warnings.append(
                        f"Observation '{obs_id}' event #{ev_idx}: unsupported event record {ev!r}."
                    )
                    continue

                ts_val = parse_recording_seconds(raw_ts)
                if ts_val is None or not beh_code:
                    warnings.append(
                        f"Observation '{obs_id}' event #{ev_idx}: invalid timestamp or behavior."
                    )
                    continue

                adj_ts = ts_val + (time_offset if apply_time_offset else 0.0)
                has_subj = bool(raw_subj and raw_subj.lower() != "no focal subject")
                subj = raw_subj if has_subj else DEFAULT_SUBJECT_LABEL

                if explicit_status is not None:
                    ev_kind = classify_event_label(explicit_status) or "point"
                    raw_ev_label = explicit_status
                else:
                    is_state_beh = state_by_code.get(beh_code, True)
                    if not is_state_beh:
                        ev_kind = "point"
                        raw_ev_label = "POINT"
                    else:
                        state_key = (subj, beh_code)
                        currently_open = active_states.get(state_key, False)
                        if not currently_open:
                            active_states[state_key] = True
                            ev_kind = "start"
                            raw_ev_label = "START"
                        else:
                            active_states[state_key] = False
                            ev_kind = "end"
                            raw_ev_label = "STOP"

                mod_parts = tuple(
                    m.strip()
                    for m in re.split(r"[|,;]", raw_mod)
                    if m.strip() and m.strip().lower() != "none"
                )
                parsed_events.append(
                    ParsedEventRow(
                        line_number=ev_idx,
                        recording_time=adj_ts,
                        raw_recording_time=f"{adj_ts:.6f}",
                        trial_time=ts_val,
                        subject=subj,
                        has_explicit_subject=has_subj,
                        behavior=beh_code,
                        raw_event=raw_ev_label,
                        event_kind=ev_kind,
                        modifiers=mod_parts,
                        comment=raw_comment,
                    )
                )

            for (subj_open, beh_open), still_open in active_states.items():
                if still_open:
                    warnings.append(
                        f"Observation '{obs_id}': unclosed state event for subject '{subj_open}', behavior '{beh_open}'."
                    )

            observations.append(
                BorisObservation(
                    observation_id=str(obs_id),
                    events=tuple(parsed_events),
                    media_files=media_files,
                    fps=fps_val,
                    time_offset_s=time_offset,
                    description=obs_desc,
                    observation_date=obs_date,
                    independent_variables=iv_values,
                    media_by_player=media_by_player,
                    media_offsets=media_offsets,
                    external_data=external_data,
                )
            )

    first_obs_vars = (
        dict(observations[0].independent_variables) if len(observations) == 1 else {}
    )
    schema = ProjectSchema(
        behaviors=behaviors,
        categories=categories,
        modifiers=modifiers,
        subjects=subjects,
        independent_variables=indep_vars,
        observation_variables=first_obs_vars,
    )

    return BorisProjectImport(
        project_name=project_name,
        project_description=project_desc,
        project_date=project_date,
        format_version=fmt_version,
        schema=schema,
        observations=tuple(observations),
        warnings=tuple(warnings),
        source_path=resolved_path,
        converters=project_converters,
    )


def _normalize_event_stream(
    events: Iterable[ParsedEventRow | Mapping[str, object]],
    *,
    default_subject: str = DEFAULT_SUBJECT_LABEL,
) -> List[ParsedEventRow]:
    items = list(events)
    if not items:
        return []
    if all(isinstance(x, ParsedEventRow) for x in items):
        return list(items)  # type: ignore[return-value]
    mappings = [x if isinstance(x, Mapping) else {} for x in items]
    return parse_event_mapping_rows(mappings, default_subject=default_subject)


def export_boris_project(
    schema: ProjectSchema,
    *,
    observations: Optional[Sequence[BorisObservation]] = None,
    project_name: str = "Annolid Behavioral Project",
    project_description: str = "",
    project_date: Optional[str] = None,
    converters: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Serialize ``ProjectSchema`` and optional ``observations`` into BORIS v7.0 JSON."""
    date_str = project_date or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    converters_out: Dict[str, Any] = dict(converters) if converters else {}
    cat_map = schema.category_map()
    mod_map = schema.modifier_map()

    behavioral_categories = [cat.name or cat.id for cat in schema.categories]
    behavioral_categories_conf = {
        str(idx): {
            "name": cat.name or cat.id,
            "color": cat.color or "",
            "description": cat.description or "",
        }
        for idx, cat in enumerate(schema.categories)
    }

    subjects_conf = {
        str(idx): {
            "key": subj.key_binding or "",
            "name": subj.name or subj.id,
            "description": subj.description or "",
        }
        for idx, subj in enumerate(schema.subjects)
    }

    indep_vars_conf: Dict[str, Dict[str, Any]] = {}
    for idx, iv in enumerate(schema.independent_variables):
        boris_type = (
            "value from list" if iv.var_type == "value_from_list" else iv.var_type
        )
        indep_vars_conf[str(idx)] = {
            "label": iv.name or iv.id,
            "description": iv.description or "",
            "type": boris_type,
            "default value": iv.default_value or "",
            "possible values": ",".join(iv.allowed_values),
        }

    behaviors_conf: Dict[str, Dict[str, Any]] = {}
    for idx, beh in enumerate(schema.behaviors):
        cat_name = ""
        if beh.category_id and beh.category_id in cat_map:
            cat_obj = cat_map[beh.category_id]
            cat_name = cat_obj.name or cat_obj.id
        elif beh.category_id:
            cat_name = beh.category_id

        # Group attached modifiers by their group/set name
        grouped_mods: Dict[str, List[ModifierDefinition]] = defaultdict(list)
        for mid in beh.modifier_ids:
            m_obj = mod_map.get(mid)
            if m_obj is not None:
                grp = m_obj.group or "Modifiers"
                grouped_mods[grp].append(m_obj)

        boris_mods: Dict[str, Dict[str, Any]] = {}
        for g_idx, (grp_name, mod_list) in enumerate(grouped_mods.items()):
            mode = mod_list[0].selection_mode if mod_list else "multi"
            values = [
                f"{m.name} ({m.key_binding})" if m.key_binding else (m.name or m.id)
                for m in mod_list
            ]
            boris_mods[str(g_idx)] = {
                "name": grp_name,
                "type": _mode_to_boris_mod_type(mode),
                "description": mod_list[0].description or "",
                "values": values,
            }

        excl_codes = sorted(schema.symmetric_exclusions_for(beh.code))
        behaviors_conf[str(idx)] = {
            "type": "State event" if beh.is_state else "Point event",
            "key": beh.key_binding or "",
            "code": beh.code,
            "description": beh.description or beh.name or "",
            "color": beh.color or "",
            "category": cat_name,
            "modifiers": boris_mods if boris_mods else "",
            "excluded": ",".join(excl_codes),
            "coding map": "",
        }

    obs_out: Dict[str, Dict[str, Any]] = {}
    for obs in observations or ():
        boris_events: List[List[Any]] = []
        ordered_rows = sorted(
            obs.events,
            key=lambda r: (
                float(r.recording_time or 0.0),
                0 if r.event_kind == "end" else 1,
                r.line_number,
            ),
        )
        for row in ordered_rows:
            if row.recording_time is None or not row.behavior:
                continue
            subj_out = (
                row.subject
                if row.has_explicit_subject and row.subject
                else (row.subject or "No focal subject")
            )
            mod_str = "|".join(row.modifiers) if row.modifiers else ""
            boris_events.append(
                [
                    round(float(row.recording_time), 6),
                    subj_out,
                    row.behavior,
                    mod_str,
                    row.comment or "",
                ]
            )

        if obs.media_by_player:
            file_dict: Dict[str, List[str]] = {
                str(pk): list(paths) for pk, paths in obs.media_by_player.items()
            }
            file_dict.setdefault("1", [])
            file_dict.setdefault("2", [])
        else:
            file_dict = {"1": list(obs.media_files), "2": []}

        all_media = [p for paths in file_dict.values() for p in paths if p]
        obs_entry: Dict[str, Any] = {
            "file": file_dict,
            "type": "MEDIA" if all_media else "LIVE",
            "date": obs.observation_date or date_str,
            "description": obs.description,
            "time_offset": float(obs.time_offset_s),
            "events": boris_events,
            "independent_variables": dict(
                obs.independent_variables or schema.observation_variables
            ),
            "visualize_spectrogram": False,
            "close_behaviors_between_videos": False,
        }
        media_info_out: Dict[str, Any] = {}
        if obs.fps is not None and obs.fps > 0:
            media_info_out["fps"] = {
                (all_media[0] if all_media else "media_1"): float(obs.fps)
            }
        if obs.media_offsets:
            media_info_out["offset"] = {
                str(k): float(v) for k, v in obs.media_offsets.items()
            }
        if media_info_out:
            obs_entry["media_info"] = media_info_out
        if obs.external_data:
            obs_entry["plot_data"] = {
                str(idx): spec.to_boris_plot_dict()
                for idx, spec in enumerate(obs.external_data)
            }
            for spec in obs.external_data:
                for ck, cv in spec.converter_definitions.items():
                    converters_out.setdefault(str(ck), cv)
        obs_out[obs.observation_id] = obs_entry

    return {
        "time_format": "seconds",
        "project_date": date_str,
        "project_name": project_name,
        "project_description": project_description,
        "project_format_version": "7.0",
        "subjects_conf": subjects_conf,
        "behaviors_conf": behaviors_conf,
        "observations": obs_out,
        "behavioral_categories": behavioral_categories,
        "behavioral_categories_conf": behavioral_categories_conf,
        "independent_variables": indep_vars_conf,
        "coding_map": {},
        "behaviors_coding_map": [],
        "converters": converters_out,
    }


def write_boris_project(
    path: Path | str,
    schema: ProjectSchema,
    *,
    observations: Optional[Sequence[BorisObservation]] = None,
    project_name: str = "Annolid Behavioral Project",
    project_description: str = "",
    converters: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write a ``.boris`` JSON project file atomically."""
    out_path = Path(path)
    payload = export_boris_project(
        schema,
        observations=observations,
        project_name=project_name,
        project_description=project_description,
        converters=converters,
    )
    atomic_write_json(out_path, payload, indent=2, ensure_ascii=False)
    return out_path


def write_annolid_events_csv(
    rows: Iterable[ParsedEventRow | Mapping[str, object]],
    path: Path | str,
    *,
    default_subject: str = DEFAULT_SUBJECT_LABEL,
) -> Path:
    """Write canonical Annolid behavior events CSV preserving modifiers and comments."""
    parsed = _normalize_event_stream(rows, default_subject=default_subject)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "Trial time",
            "Recording time",
            "Subject",
            "Behavior",
            "Event",
            "Modifiers",
            "Comment",
        ]
    )
    for row in parsed:
        if row.recording_time is None or not row.behavior:
            continue
        rec_t = float(row.recording_time)
        trial_t = float(row.trial_time) if row.trial_time is not None else rec_t
        if row.event_kind == "start":
            ev_label = "state start"
        elif row.event_kind == "end":
            ev_label = "state stop"
        elif row.event_kind == "point":
            ev_label = "point"
        else:
            ev_label = row.raw_event or "point"
        writer.writerow(
            [
                f"{trial_t:.6f}",
                f"{rec_t:.6f}",
                row.subject or default_subject,
                row.behavior,
                ev_label,
                "|".join(row.modifiers) if row.modifiers else "",
                row.comment or "",
            ]
        )
    out = Path(path)
    atomic_write_text(out, buf.getvalue(), encoding="utf-8")
    return out


def write_boris_tabular_csv(
    rows: Iterable[ParsedEventRow | Mapping[str, object]],
    path: Path | str,
    *,
    observation_id: str = "Observation_1",
    media_file: str = "",
    schema: Optional[ProjectSchema] = None,
    delimiter: Optional[str] = None,
) -> Path:
    """Write BORIS Tabular Events CSV/TSV (``Status`` = START / STOP / POINT)."""
    out = Path(path)
    delim = delimiter or ("\t" if out.suffix.lower() == ".tsv" else ",")
    parsed = _normalize_event_stream(rows)
    beh_map = schema.behavior_map() if schema else {}
    cat_map = schema.category_map() if schema else {}

    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=delim)
    writer.writerow(
        [
            "Observation id",
            "Media file",
            "Subject",
            "Behavior",
            "Behavioral category",
            "Modifiers",
            "Behavior type",
            "Time",
            "Status",
            "Comment",
        ]
    )
    for row in parsed:
        if row.recording_time is None or not row.behavior:
            continue
        bdef = beh_map.get(row.behavior)
        cat_label = ""
        if bdef and bdef.category_id:
            cobj = cat_map.get(bdef.category_id)
            cat_label = (cobj.name or cobj.id) if cobj else bdef.category_id
        if row.event_kind == "point" or (bdef is not None and not bdef.is_state):
            btype = "POINT"
            status = "POINT"
        elif row.event_kind == "end":
            btype = "STATE"
            status = "STOP"
        else:
            btype = "STATE"
            status = "START"
        writer.writerow(
            [
                observation_id,
                media_file,
                row.subject,
                row.behavior,
                cat_label,
                "|".join(row.modifiers) if row.modifiers else "",
                btype,
                f"{float(row.recording_time):.6f}",
                status,
                row.comment or "",
            ]
        )
    atomic_write_text(out, buf.getvalue(), encoding="utf-8")
    return out


def write_boris_aggregated_csv(
    rows: Iterable[ParsedEventRow | Mapping[str, object]],
    path: Path | str,
    *,
    observation_id: str = "Observation_1",
    media_file: str = "",
    schema: Optional[ProjectSchema] = None,
    delimiter: Optional[str] = None,
) -> Path:
    """Write BORIS Aggregated Events CSV/TSV (``Start (s)``, ``Stop (s)``, ``Duration (s)``)."""
    out = Path(path)
    delim = delimiter or ("\t" if out.suffix.lower() == ".tsv" else ",")
    parsed = _normalize_event_stream(rows)
    beh_map = schema.behavior_map() if schema else {}
    cat_map = schema.category_map() if schema else {}

    agg_records: List[Tuple[float, List[str]]] = []
    open_queues: Dict[Tuple[str, str], Deque[ParsedEventRow]] = defaultdict(deque)

    ordered = sorted(
        parsed,
        key=lambda r: (
            float(r.recording_time or 0.0),
            0 if r.event_kind == "end" else 1,
            r.line_number,
        ),
    )

    def _cat_for(beh_code: str) -> str:
        bdef = beh_map.get(beh_code)
        if bdef and bdef.category_id:
            cobj = cat_map.get(bdef.category_id)
            return (cobj.name or cobj.id) if cobj else bdef.category_id
        return ""

    for row in ordered:
        if row.recording_time is None or not row.behavior:
            continue
        t_val = float(row.recording_time)
        bdef = beh_map.get(row.behavior)
        is_pt = row.event_kind == "point" or (bdef is not None and not bdef.is_state)
        if is_pt:
            agg_records.append(
                (
                    t_val,
                    [
                        observation_id,
                        media_file,
                        row.subject,
                        row.behavior,
                        _cat_for(row.behavior),
                        "|".join(row.modifiers) if row.modifiers else "",
                        "POINT",
                        f"{t_val:.6f}",
                        f"{t_val:.6f}",
                        "0.000000",
                        row.comment or "",
                        "",
                    ],
                )
            )
            continue

        series_key = (row.subject, row.behavior)
        if row.event_kind == "start":
            open_queues[series_key].append(row)
        elif row.event_kind == "end" and open_queues[series_key]:
            start_row = open_queues[series_key].popleft()
            s_t = float(start_row.recording_time or 0.0)
            dur = max(0.0, t_val - s_t)
            mods = start_row.modifiers or row.modifiers
            agg_records.append(
                (
                    s_t,
                    [
                        observation_id,
                        media_file,
                        start_row.subject,
                        start_row.behavior,
                        _cat_for(start_row.behavior),
                        "|".join(mods) if mods else "",
                        "STATE",
                        f"{s_t:.6f}",
                        f"{t_val:.6f}",
                        f"{dur:.6f}",
                        start_row.comment or "",
                        row.comment or "",
                    ],
                )
            )

    agg_records.sort(key=lambda item: item[0])
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=delim)
    writer.writerow(
        [
            "Observation id",
            "Media file",
            "Subject",
            "Behavior",
            "Behavioral category",
            "Modifiers",
            "Behavior type",
            "Start (s)",
            "Stop (s)",
            "Duration (s)",
            "Comment start",
            "Comment stop",
        ]
    )
    for _sort_t, rec in agg_records:
        writer.writerow(rec)
    atomic_write_text(out, buf.getvalue(), encoding="utf-8")
    return out


def materialize_boris_project(
    boris_source: Path | str | BorisProjectImport,
    output_dir: Path | str,
    *,
    observation_filter: Optional[Sequence[str]] = None,
    apply_time_offset: bool = False,
    group_variable: Optional[str] = None,
) -> BorisMaterializedArtifacts:
    """Convert a ``.boris`` file into ``project.annolid.json``, event CSVs, and ``cohort_manifest.json``."""
    imported = (
        boris_source
        if isinstance(boris_source, BorisProjectImport)
        else import_boris_project(boris_source, apply_time_offset=apply_time_offset)
    )
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    schema_path = out_dir / "project.annolid.json"
    save_schema(imported.schema, schema_path)

    allowed_obs = (
        {str(o).strip() for o in observation_filter if str(o).strip()}
        if observation_filter
        else None
    )

    obs_csvs: Dict[str, Path] = {}
    manifest_records: List[Dict[str, Any]] = []
    group_key_candidates = ("group", "condition", "treatment", "genotype", "cohort")

    for obs in imported.observations:
        if allowed_obs is not None and obs.observation_id not in allowed_obs:
            continue
        safe_stem = _slugify_id(obs.observation_id, "observation")
        csv_path = out_dir / f"{safe_stem}_events.csv"
        if csv_path in obs_csvs.values():
            csv_path = out_dir / f"{safe_stem}_{len(obs_csvs) + 1}_events.csv"
        write_annolid_events_csv(obs.events, csv_path)
        obs_csvs[obs.observation_id] = csv_path

        iv_lower = {k.lower(): v for k, v in obs.independent_variables.items()}
        group_val = "ungrouped"
        if group_variable and str(group_variable).strip().lower() in iv_lower:
            explicit_g = iv_lower[str(group_variable).strip().lower()]
            if explicit_g:
                group_val = explicit_g
        if group_val == "ungrouped":
            for cand in group_key_candidates:
                if cand in iv_lower and iv_lower[cand]:
                    group_val = iv_lower[cand]
                    break
        if group_val == "ungrouped" and len(obs.independent_variables) == 1:
            group_val = next(iter(obs.independent_variables.values())) or "ungrouped"

        rec_entry: Dict[str, Any] = {
            "recording_id": obs.observation_id,
            "events_csv": csv_path.name,
            "group": group_val,
        }
        if obs.media_files:
            rec_entry["media_file"] = obs.media_files[0]
        for k, v in obs.independent_variables.items():
            if k not in rec_entry:
                rec_entry[k] = v
        manifest_records.append(rec_entry)

    manifest_path: Optional[Path] = None
    if manifest_records:
        manifest_path = out_dir / "cohort_manifest.json"
        atomic_write_json(
            manifest_path,
            {
                "project_name": imported.project_name,
                "schema_path": schema_path.name,
                "recordings": manifest_records,
            },
            indent=2,
            ensure_ascii=False,
        )

    return BorisMaterializedArtifacts(
        output_dir=out_dir,
        schema_path=schema_path,
        observation_csvs=obs_csvs,
        cohort_manifest_path=manifest_path,
        warnings=imported.warnings,
    )


def load_boris_observation_from_csv(
    csv_path: Path | str,
    *,
    observation_id: Optional[str] = None,
    media_file: Optional[str] = None,
) -> BorisObservation:
    """Helper to wrap an Annolid or BORIS events CSV into a :class:`BorisObservation`."""
    resolved = Path(csv_path)
    rows = read_parsed_event_rows(resolved)
    obs_id = observation_id or resolved.stem.removesuffix("_events").removesuffix(
        "_timestamps"
    )
    media = (str(media_file),) if media_file else ()
    return BorisObservation(
        observation_id=obs_id,
        events=tuple(rows),
        media_files=media,
    )
