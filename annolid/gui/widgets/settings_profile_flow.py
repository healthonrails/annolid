from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from typing import Callable, Optional

from qtpy import QtCore, QtWidgets

from annolid.features.dino_models import resolve_dino_model_id
from annolid.interfaces.memory.adapters.settings_model import SettingsProfile
from annolid.interfaces.memory.adapters.workspace import WorkspaceMemoryAdapter
from annolid.motion.optical_flow import OpticalFlowPreferences


def _show_status(window, text: str, timeout_ms: int = 3500) -> None:
    try:
        status_bar = window.statusBar()
        if status_bar is not None:
            status_bar.showMessage(text, timeout_ms)
    except Exception:
        pass


def _workspace_adapter(window) -> WorkspaceMemoryAdapter:
    workspace_id = str(getattr(window, "memory_workspace_id", "") or "default")
    return WorkspaceMemoryAdapter(workspace_id)


def _choose_profile_dialog(
    parent,
    profiles: list[SettingsProfile],
    panel_label: str,
) -> Optional[SettingsProfile]:
    if not profiles:
        QtWidgets.QMessageBox.information(
            parent,
            "No profile found",
            f"No saved settings profile found for {panel_label}.",
        )
        return None

    labels = [
        f"{idx + 1}. {profile.name} ({profile.workflow})"
        for idx, profile in enumerate(profiles)
    ]

    selected, ok = QtWidgets.QInputDialog.getItem(
        parent,
        f"Apply {panel_label} profile",
        "Choose a saved profile:",
        labels,
        0,
        False,
    )
    if not ok:
        return None
    try:
        selected_index = labels.index(str(selected))
    except ValueError:
        return None
    return profiles[selected_index]


def _persist_qsettings(window, key: str, value) -> None:
    settings = getattr(window, "settings", None)
    if isinstance(settings, QtCore.QSettings):
        settings.setValue(key, value)


def _tracker_config_payload(runtime) -> dict:
    if runtime is None:
        return {}
    if is_dataclass(runtime):
        payload = asdict(runtime)
    else:
        payload = {
            key: value
            for key, value in vars(runtime).items()
            if not str(key).startswith("_")
        }
    for transient_key in ("progress_hook", "error_hook", "analytics_hook"):
        payload.pop(transient_key, None)
    return payload


def _apply_advanced_parameters(window, profile: SettingsProfile) -> None:
    payload = dict(profile.settings or {})
    bool_keys = {
        "automatic_pause_enabled",
        "use_cpu_only",
        "save_video_with_color_mask",
        "auto_recovery_missing_instances",
        "follow_prediction_progress",
    }
    float_keys = {
        "epsilon_for_polygon",
        "t_max_value",
        "videomt_mask_threshold",
        "videomt_logit_threshold",
        "videomt_seed_iou_threshold",
    }
    int_keys = {"videomt_window", "videomt_input_height", "videomt_input_width"}

    if "optical_flow_enabled" in payload and getattr(
        window, "optical_flow_manager", None
    ):
        window.optical_flow_manager.set_compute_optical_flow(
            bool(payload["optical_flow_enabled"])
        )
    if "optical_flow_backend" in payload and getattr(
        window, "optical_flow_manager", None
    ):
        window.optical_flow_manager.set_backend(str(payload["optical_flow_backend"]))

    for key in bool_keys:
        if key in payload:
            value = bool(payload[key])
            if key == "follow_prediction_progress":
                setattr(window, "_follow_prediction_progress", value)
            else:
                setattr(window, key, value)

    for key in float_keys:
        if key in payload:
            value = float(payload[key])
            setattr(window, key, value)

    for key in int_keys:
        if key in payload:
            value = int(payload[key])
            setattr(window, key, value)

    tracker_cfg = payload.get("tracker_runtime_config")
    if tracker_cfg and hasattr(window, "tracker_runtime_config"):
        for key, value in dict(tracker_cfg).items():
            setattr(window.tracker_runtime_config, key, value)

    if "cutie_weights_override" in payload or (
        isinstance(tracker_cfg, dict) and "cutie_weights_path" in tracker_cfg
    ):
        raw_cutie_override = payload.get(
            "cutie_weights_override",
            tracker_cfg.get("cutie_weights_path")
            if isinstance(tracker_cfg, dict)
            else None,
        )
        cleaned_override = str(raw_cutie_override or "").strip() or None
        prev_override = getattr(window, "cutie_weights_override", None)
        setattr(window, "cutie_weights_override", cleaned_override)
        if hasattr(window, "tracker_runtime_config"):
            window.tracker_runtime_config.cutie_weights_path = cleaned_override
        if cleaned_override != prev_override:
            setattr(window, "_tracking_processor_cache_key_value", None)
        _persist_qsettings(window, "cutie/weights_override", cleaned_override or "")

    sam3_runtime = payload.get("sam3_runtime")
    if isinstance(sam3_runtime, dict):
        window._config.setdefault("sam3", {}).update(sam3_runtime)
        _persist_qsettings(window, "sam3d", sam3_runtime)
        try:
            _persist_qsettings(window, "sam3d_json", json.dumps(sam3_runtime))
        except Exception:
            pass


def _apply_optical_flow(window, profile: SettingsProfile) -> None:
    payload = dict(profile.settings or {})
    prefs = OpticalFlowPreferences.from_source(payload)
    manager = getattr(window, "optical_flow_manager", None)

    if manager is not None and callable(getattr(manager, "apply_preferences", None)):
        manager.apply_preferences(prefs, persist=True, refresh_overlay=True)
        return

    if manager is not None and callable(getattr(manager, "set_backend", None)):
        manager.set_backend(prefs.backend)
    prefs.apply_to_target(window)
    for key, value in prefs.qsettings_items():
        _persist_qsettings(window, key, value)


def _apply_depth(window, profile: SettingsProfile) -> None:
    payload = dict(profile.settings or {})
    window._config.setdefault("video_depth_anything", {}).update(payload)
    _persist_qsettings(window, "video_depth_anything", payload)


def _apply_sam3d(window, profile: SettingsProfile) -> None:
    payload = dict(profile.settings or {})
    window._config.setdefault("sam3d", {}).update(payload)
    _persist_qsettings(window, "sam3d", payload)
    try:
        _persist_qsettings(window, "sam3d_json", json.dumps(payload))
    except Exception:
        pass


def _apply_patch_similarity(window, profile: SettingsProfile) -> None:
    payload = dict(profile.settings or {})
    raw_model = payload.get("model", payload.get("patch_similarity/model", ""))
    model = resolve_dino_model_id(raw_model) if str(raw_model or "").strip() else ""
    alpha = float(payload.get("alpha", payload.get("patch_similarity/alpha", 0.55)))
    if model:
        window.patch_similarity_model = model
    window.patch_similarity_alpha = min(max(alpha, 0.05), 1.0)
    _persist_qsettings(window, "patch_similarity/model", window.patch_similarity_model)
    _persist_qsettings(window, "patch_similarity/alpha", window.patch_similarity_alpha)


def _apply_pca_map(window, profile: SettingsProfile) -> None:
    payload = dict(profile.settings or {})
    raw_model = payload.get("model", payload.get("pca_map/model", ""))
    model = resolve_dino_model_id(raw_model) if str(raw_model or "").strip() else ""
    alpha = float(payload.get("alpha", payload.get("pca_map/alpha", 0.65)))
    clusters = int(payload.get("clusters", payload.get("pca_map/clusters", 0)))
    if model:
        window.pca_map_model = model
    window.pca_map_alpha = min(max(alpha, 0.05), 1.0)
    window.pca_map_clusters = max(0, clusters)
    _persist_qsettings(window, "pca_map/model", window.pca_map_model)
    _persist_qsettings(window, "pca_map/alpha", window.pca_map_alpha)
    _persist_qsettings(window, "pca_map/clusters", window.pca_map_clusters)


_APPLIERS: dict[str, Callable[[object, SettingsProfile], None]] = {
    "advanced_parameters": _apply_advanced_parameters,
    "optical_flow": _apply_optical_flow,
    "video_depth_anything": _apply_depth,
    "sam3d": _apply_sam3d,
    "patch_similarity": _apply_patch_similarity,
    "pca_map": _apply_pca_map,
}


def apply_profile_for_workflow(
    window,
    workflow: str,
    panel_label: str,
) -> bool:
    adapter = _workspace_adapter(window)
    profiles = adapter.retrieve_settings_profiles(
        query="",
        top_k=100,
        workflow=workflow,
    )
    profile = _choose_profile_dialog(window, profiles, panel_label)
    if profile is None:
        return False
    applier = _APPLIERS.get(workflow)
    if applier is None:
        QtWidgets.QMessageBox.warning(
            window,
            "Unsupported workflow",
            f"Profile apply is not implemented for workflow '{workflow}'.",
        )
        return False
    applier(window, profile)
    _show_status(window, f"Applied profile '{profile.name}' to {panel_label}.")
    return True


def _collect_advanced_parameters(window) -> dict:
    tracker_cfg = _tracker_config_payload(
        getattr(window, "tracker_runtime_config", None)
    )
    return {
        "epsilon_for_polygon": float(getattr(window, "epsilon_for_polygon", 10.0)),
        "t_max_value": float(getattr(window, "t_max_value", 0.1)),
        "automatic_pause_enabled": bool(
            getattr(window, "automatic_pause_enabled", False)
        ),
        "use_cpu_only": bool(getattr(window, "use_cpu_only", False)),
        "save_video_with_color_mask": bool(
            getattr(window, "save_video_with_color_mask", False)
        ),
        "auto_recovery_missing_instances": bool(
            getattr(window, "auto_recovery_missing_instances", True)
        ),
        "follow_prediction_progress": bool(
            getattr(window, "_follow_prediction_progress", True)
        ),
        "videomt_mask_threshold": float(getattr(window, "videomt_mask_threshold", 0.5)),
        "videomt_logit_threshold": float(
            getattr(window, "videomt_logit_threshold", -2.0)
        ),
        "videomt_seed_iou_threshold": float(
            getattr(window, "videomt_seed_iou_threshold", 0.01)
        ),
        "videomt_window": int(getattr(window, "videomt_window", 8)),
        "videomt_input_height": int(getattr(window, "videomt_input_height", 0)),
        "videomt_input_width": int(getattr(window, "videomt_input_width", 0)),
        "optical_flow_enabled": bool(
            getattr(
                getattr(window, "optical_flow_manager", None),
                "compute_optical_flow",
                True,
            )
        ),
        "optical_flow_backend": str(
            getattr(
                getattr(window, "optical_flow_manager", None),
                "optical_flow_backend",
                "farneback",
            )
        ),
        "tracker_runtime_config": tracker_cfg,
        "cutie_weights_override": getattr(window, "cutie_weights_override", None),
        "sam3_runtime": dict((getattr(window, "_config", {}) or {}).get("sam3", {})),
    }


def _collect_optical_flow(window) -> dict:
    return OpticalFlowPreferences.from_source(window).to_profile_dict()


def _collect_depth(window) -> dict:
    return dict((getattr(window, "_config", {}) or {}).get("video_depth_anything", {}))


def _collect_sam3d(window) -> dict:
    return dict((getattr(window, "_config", {}) or {}).get("sam3d", {}))


def _collect_patch_similarity(window) -> dict:
    return {
        "model": str(getattr(window, "patch_similarity_model", "")),
        "alpha": float(getattr(window, "patch_similarity_alpha", 0.55)),
    }


def _collect_pca_map(window) -> dict:
    return {
        "model": str(getattr(window, "pca_map_model", "")),
        "alpha": float(getattr(window, "pca_map_alpha", 0.65)),
        "clusters": int(getattr(window, "pca_map_clusters", 0)),
    }


_COLLECTORS: dict[str, Callable[[object], dict]] = {
    "advanced_parameters": _collect_advanced_parameters,
    "optical_flow": _collect_optical_flow,
    "video_depth_anything": _collect_depth,
    "sam3d": _collect_sam3d,
    "patch_similarity": _collect_patch_similarity,
    "pca_map": _collect_pca_map,
}


def save_current_profile_for_workflow(window, workflow: str, panel_label: str) -> bool:
    collector = _COLLECTORS.get(workflow)
    if collector is None:
        QtWidgets.QMessageBox.warning(
            window,
            "Unsupported workflow",
            f"Profile save is not implemented for workflow '{workflow}'.",
        )
        return False
    profile_name, ok = QtWidgets.QInputDialog.getText(
        window,
        f"Save {panel_label} profile",
        "Profile name:",
    )
    if not ok:
        return False
    profile_name = str(profile_name).strip()
    if not profile_name:
        QtWidgets.QMessageBox.information(
            window,
            "Profile name required",
            "Please provide a profile name.",
        )
        return False
    tags_raw, ok = QtWidgets.QInputDialog.getText(
        window,
        f"Save {panel_label} profile",
        "Tags (comma-separated, optional):",
    )
    if not ok:
        return False
    tags = [x.strip() for x in str(tags_raw or "").split(",") if x.strip()]
    context, ok = QtWidgets.QInputDialog.getText(
        window,
        f"Save {panel_label} profile",
        "Context (optional):",
    )
    if not ok:
        return False
    payload = collector(window)
    profile = SettingsProfile(
        name=profile_name,
        workflow=workflow,
        settings=payload,
        tags=tags,
        context=str(context or "").strip() or None,
    )
    adapter = _workspace_adapter(window)
    memory_id = adapter.store_settings_profile(profile)
    if not memory_id:
        QtWidgets.QMessageBox.warning(
            window,
            "Save failed",
            f"Unable to save profile '{profile_name}'.",
        )
        return False
    _show_status(window, f"Saved profile '{profile_name}' for {panel_label}.")
    return True
