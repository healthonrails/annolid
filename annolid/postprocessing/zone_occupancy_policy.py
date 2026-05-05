from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from annolid.postprocessing.zone_schema import (
    ZoneShapeSpec,
    zone_shape_covers_point,
)


@dataclass(frozen=True)
class ZoneOccupancyPolicyResult:
    dataframe: pd.DataFrame
    audit: pd.DataFrame


def load_zone_occupancy_policy(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    policy_path = Path(path).expanduser()
    if not policy_path.exists() or not policy_path.is_file():
        raise FileNotFoundError(f"Zone occupancy policy not found: {policy_path}")
    with policy_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Zone occupancy policy must be a JSON object.")
    return payload


def zone_column_name_map(zone_specs: Sequence[ZoneShapeSpec]) -> dict[str, str]:
    seen: dict[str, int] = {}
    mapping: dict[str, str] = {}
    for spec in zone_specs:
        base = str(spec.display_label or spec.label or "zone").strip() or "zone"
        count = seen.get(base, 0) + 1
        seen[base] = count
        column = base if count == 1 else f"{base}_{count}"
        mapping[spec.display_label] = column
        mapping[base] = column
        mapping[column] = column
    return mapping


def zone_group_map(zone_specs: Sequence[ZoneShapeSpec]) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {}
    column_map = zone_column_name_map(zone_specs)
    for spec in zone_specs:
        column = column_map.get(spec.display_label)
        if not column:
            continue
        flags = dict(spec.flags or {})
        tokens = {
            str(flags.get("zone_group") or "").strip(),
            str(flags.get("occupancy_group") or "").strip(),
            str(spec.zone_kind or "").strip(),
        }
        for token in tokens:
            if not token:
                continue
            groups.setdefault(token, set()).add(column)
    return groups


def ensure_zone_columns(
    dataframe: pd.DataFrame,
    zone_specs: Sequence[ZoneShapeSpec],
) -> pd.DataFrame:
    output = dataframe.copy()
    column_map = zone_column_name_map(zone_specs)
    for column in dict.fromkeys(column_map.values()):
        if column not in output.columns:
            output[column] = 0

    if output.empty or not zone_specs:
        return output
    if "cx" not in output.columns or "cy" not in output.columns:
        return output

    for spec in zone_specs:
        column = column_map.get(spec.display_label)
        if not column:
            continue
        existing_values = output[column].fillna(0).astype(str)
        if existing_values.isin({"1", "1.0", "true", "True"}).any():
            continue
        values: list[int] = []
        for _, row in output.iterrows():
            try:
                point = (float(row["cx"]), float(row["cy"]))
                inside = zone_shape_covers_point(spec, point)
            except Exception:
                inside = False
            values.append(1 if inside else 0)
        output[column] = values
    return output


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    try:
        if isinstance(value, (int, float)):
            return float(value) != 0.0
    except Exception:
        return False
    return str(value or "").strip().lower() in {"1", "1.0", "true", "yes", "on"}


def _set_flag(dataframe: pd.DataFrame, index: Any, column: str, value: int) -> bool:
    old_value = 1 if _truthy(dataframe.at[index, column]) else 0
    new_value = 1 if int(value) else 0
    if old_value == new_value:
        return False
    dataframe.at[index, column] = new_value
    return True


def _row_matches_instance(row: pd.Series, rule: Mapping[str, Any]) -> bool:
    if "instance_name" not in row:
        return False
    instance_name = str(row.get("instance_name") or "")
    allowed = rule.get("instance_names", rule.get("instances", None))
    if allowed is None:
        single = rule.get("instance_name", rule.get("instance", None))
        allowed = [single] if single is not None else []
    if isinstance(allowed, str):
        allowed = [allowed]
    return instance_name in {str(item) for item in list(allowed or [])}


def _row_in_frame_range(row: pd.Series, rule: Mapping[str, Any]) -> bool:
    frame_value = row.get("frame_number", row.get("frame", row.get("frame_idx", None)))
    if frame_value is None:
        return True
    try:
        frame = int(float(frame_value))
    except Exception:
        return True
    start = rule.get("start_frame", rule.get("frame_start", None))
    end = rule.get("end_frame", rule.get("frame_end", None))
    if start is not None and frame < int(start):
        return False
    if end is not None and frame > int(end):
        return False
    return True


def _columns_for_rule(
    rule: Mapping[str, Any],
    *,
    column_map: Mapping[str, str],
    groups: Mapping[str, set[str]],
) -> tuple[set[str], list[str]]:
    group_name = str(rule.get("zone_group") or rule.get("group") or "").strip()
    group_columns = set(groups.get(group_name, set())) if group_name else set()
    names = rule.get("zones", rule.get("allowed_zones", None))
    if names is None:
        zone = rule.get("zone", None)
        names = [zone] if zone is not None else []
    if isinstance(names, str):
        names = [names]
    selected = [
        column_map.get(str(name), str(name)) for name in list(names or []) if str(name)
    ]
    selected = [name for name in selected if name in set(column_map.values())]
    if not group_columns:
        group_columns = set(selected)
    return group_columns, selected


def _membership(row: pd.Series, columns: Sequence[str]) -> list[str]:
    return [
        column for column in columns if column in row.index and _truthy(row[column])
    ]


def apply_zone_occupancy_policy(
    dataframe: pd.DataFrame,
    zone_specs: Sequence[ZoneShapeSpec],
    policy: Mapping[str, Any] | None,
) -> ZoneOccupancyPolicyResult:
    if not policy:
        return ZoneOccupancyPolicyResult(
            dataframe=dataframe.copy(),
            audit=pd.DataFrame(),
        )

    output = ensure_zone_columns(dataframe, zone_specs)
    column_map = zone_column_name_map(zone_specs)
    groups = zone_group_map(zone_specs)
    audit_rows: list[dict[str, Any]] = []
    rule_blocks = policy.get("instance_policies", policy.get("rules", []))
    if isinstance(rule_blocks, Mapping):
        rule_blocks = [rule_blocks]

    for block in list(rule_blocks or []):
        if not isinstance(block, Mapping):
            continue
        nested_rules = block.get("rules")
        rules = list(nested_rules or [block])
        for rule in rules:
            if not isinstance(rule, Mapping):
                continue
            effective_rule = dict(block)
            effective_rule.update(rule)
            mode = str(effective_rule.get("mode") or "").strip().lower()
            if not mode:
                continue
            group_columns, selected_columns = _columns_for_rule(
                effective_rule,
                column_map=column_map,
                groups=groups,
            )
            if not group_columns and not selected_columns:
                continue
            ordered_group_columns = sorted(group_columns)
            for index, row in output.iterrows():
                if not _row_matches_instance(row, effective_rule):
                    continue
                if not _row_in_frame_range(row, effective_rule):
                    continue
                before = _membership(row, ordered_group_columns)
                changed_columns: list[str] = []
                if mode == "force_one":
                    target = selected_columns[:1]
                    for column in ordered_group_columns:
                        value = 1 if column in target else 0
                        if _set_flag(output, index, column, value):
                            changed_columns.append(column)
                elif mode == "force_all":
                    for column in selected_columns:
                        if _set_flag(output, index, column, 1):
                            changed_columns.append(column)
                elif mode == "deny":
                    for column in selected_columns:
                        if _set_flag(output, index, column, 0):
                            changed_columns.append(column)
                elif mode in {"allow_only", "preserve_if_inside"}:
                    allowed = set(selected_columns)
                    for column in ordered_group_columns:
                        if column in allowed:
                            continue
                        if _set_flag(output, index, column, 0):
                            changed_columns.append(column)
                elif mode == "prefer":
                    active_preferred = [
                        column
                        for column in selected_columns
                        if _truthy(output.at[index, column])
                    ]
                    if active_preferred:
                        keep = active_preferred[0]
                        for column in ordered_group_columns:
                            value = 1 if column == keep else 0
                            if _set_flag(output, index, column, value):
                                changed_columns.append(column)
                if changed_columns:
                    after = _membership(output.loc[index], ordered_group_columns)
                    audit_rows.append(
                        {
                            "frame_number": row.get(
                                "frame_number",
                                row.get("frame", ""),
                            ),
                            "instance_name": row.get("instance_name", ""),
                            "rule_name": effective_rule.get("name", ""),
                            "zone_group": effective_rule.get("zone_group", ""),
                            "mode": mode,
                            "raw_membership": "|".join(before),
                            "corrected_membership": "|".join(after),
                            "changed_columns": "|".join(changed_columns),
                        }
                    )

    return ZoneOccupancyPolicyResult(
        dataframe=output,
        audit=pd.DataFrame(audit_rows),
    )


def apply_zone_occupancy_policy_file(
    dataframe: pd.DataFrame,
    zone_specs: Sequence[ZoneShapeSpec],
    policy_path: str | Path | None,
) -> ZoneOccupancyPolicyResult:
    return apply_zone_occupancy_policy(
        dataframe,
        zone_specs,
        load_zone_occupancy_policy(policy_path),
    )
