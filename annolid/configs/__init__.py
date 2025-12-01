import os.path as osp
import shutil

import yaml

from labelme.logger import logger


here = osp.dirname(osp.abspath(__file__))


def update_dict(target_dict, new_dict, validate_item=None):
    for key, value in new_dict.items():
        if validate_item:
            validate_item(key, value)
        if key not in target_dict:
            logger.warn("Skipping unexpected key in config: {}".format(key))
            continue
        if isinstance(target_dict[key], dict) and isinstance(value, dict):
            update_dict(target_dict[key], value, validate_item=validate_item)
        else:
            target_dict[key] = value


def _normalize_legacy_top_level_keys(config_dict):
    """Handle legacy / deprecated config keys without emitting warnings.

    - ``keep_prev_brightness_contrast`` used to control both brightness and
      contrast retention; now we have two dedicated flags.  When present, we
      map its boolean value onto ``keep_prev_brightness`` and
      ``keep_prev_contrast`` if they are not already set.
    - ``custom_models`` was historically used for custom AI model weights,
      which are now stored via QSettings.  We simply drop this key to avoid
      spurious warnings while keeping newer storage as the source of truth.
    """
    if not isinstance(config_dict, dict):
        return config_dict

    cfg = dict(config_dict)

    # Backwards-compat: map combined flag onto the newer, separate keys.
    if "keep_prev_brightness_contrast" in cfg:
        legacy_value = cfg.pop("keep_prev_brightness_contrast")
        legacy_bool = bool(legacy_value)
        cfg.setdefault("keep_prev_brightness", legacy_bool)
        cfg.setdefault("keep_prev_contrast", legacy_bool)

    # Legacy custom model list (now handled via QSettings in the GUI).
    cfg.pop("custom_models", None)

    return cfg


def _coerce_ai_default_if_unknown(config):
    """Ensure ``config['ai']['default']`` refers to a known AI model.

    Older configurations (or mismatched labelme versions) may reference a
    default model name that is not present in ``labelme.ai.MODELS``.  That
    causes noisy warnings like \"Default AI model is not found\" during startup.
    We silently fall back to a valid model name if available.
    """
    try:
        from labelme.ai import MODELS as LABELME_MODELS  # type: ignore
    except Exception:
        # If labelme.ai is unavailable, we cannot validate; keep config as-is.
        return config

    if not isinstance(config, dict):
        return config

    ai_section = config.get("ai") or {}
    default_name = ai_section.get("default")
    if not default_name:
        return config

    available_names = {
        getattr(m, "name", str(m)) for m in LABELME_MODELS
    }
    if default_name in available_names:
        return config

    # Prefer a stable, widely-available default; fall back to the first model.
    preferred_order = [
        "EfficientSam (speed)",
        "SegmentAnything (balanced)",
        "SegmentAnything (speed)",
    ]
    fallback_name = None
    for candidate in preferred_order:
        if candidate in available_names:
            fallback_name = candidate
            break

    if fallback_name is None and available_names:
        # Deterministic but arbitrary choice.
        fallback_name = sorted(available_names)[0]

    if fallback_name:
        ai_section["default"] = fallback_name
        config["ai"] = ai_section

    return config


# -----------------------------------------------------------------------------


def get_default_config():
    config_file = osp.join(here, "default_config.yaml")
    with open(config_file) as f:
        config = yaml.safe_load(f)

    # save default config to ~/.labelmerc
    user_config_file = osp.join(osp.expanduser("~"), ".labelmerc")
    if not osp.exists(user_config_file):
        try:
            shutil.copy(config_file, user_config_file)
        except Exception:
            logger.warn("Failed to save config: {}".format(user_config_file))

    return config


def validate_config_item(key, value):
    if key == "validate_label" and value not in [None, "exact"]:
        raise ValueError(
            "Unexpected value for config key 'validate_label': {}".format(
                value
            )
        )
    if key == "shape_color" and value not in [None, "auto", "manual"]:
        raise ValueError(
            "Unexpected value for config key 'shape_color': {}".format(value)
        )
    if key == "labels" and value is not None and len(value) != len(set(value)):
        raise ValueError(
            "Duplicates are detected for config key 'labels': {}".format(value)
        )


def get_config(config_file_or_yaml=None, config_from_args=None):
    # 1. default config
    config = get_default_config()

    # 2. specified as file or yaml
    if config_file_or_yaml is not None:
        config_from_yaml = yaml.safe_load(config_file_or_yaml)
        if not isinstance(config_from_yaml, dict):
            with open(config_from_yaml) as f:
                logger.info(
                    "Loading config file from: {}".format(config_from_yaml)
                )
                config_from_yaml = yaml.safe_load(f)
        config_from_yaml = _normalize_legacy_top_level_keys(config_from_yaml)
        update_dict(
            config, config_from_yaml, validate_item=validate_config_item
        )

    # 3. command line argument or specified config file
    if config_from_args is not None:
        # CLI shouldn't normally carry legacy keys, but normalize defensively.
        config_from_args = _normalize_legacy_top_level_keys(config_from_args)
        update_dict(
            config, config_from_args, validate_item=validate_config_item
        )

    config = _coerce_ai_default_if_unknown(config)
    return config
