from annolid.tracking.configuration import CutieDinoTrackerConfig


def test_kpseg_apply_mode_defaults_to_never():
    cfg = CutieDinoTrackerConfig()
    assert cfg.kpseg_apply_mode == "never"


def test_kpseg_apply_mode_normalizes_invalid_values():
    cfg = CutieDinoTrackerConfig(kpseg_apply_mode="banana")
    assert cfg.kpseg_apply_mode == "never"


def test_kpseg_apply_mode_accepts_auto_and_always():
    cfg = CutieDinoTrackerConfig(kpseg_apply_mode="AUTO")
    assert cfg.kpseg_apply_mode == "auto"
    cfg2 = CutieDinoTrackerConfig(kpseg_apply_mode="always")
    assert cfg2.kpseg_apply_mode == "always"


def test_kpseg_gate_params_normalize():
    cfg = CutieDinoTrackerConfig(
        kpseg_min_reliable_frames=0,
        kpseg_reliable_ratio=2.0,
        kpseg_disable_patience=0,
    )
    assert cfg.kpseg_min_reliable_frames >= 1
    assert 0.0 <= cfg.kpseg_reliable_ratio <= 1.0
    assert cfg.kpseg_disable_patience >= 1


def test_kpseg_temporal_fusion_params_normalize() -> None:
    cfg = CutieDinoTrackerConfig(
        kpseg_temporal_fusion_enable=1,
        kpseg_temporal_fusion_alpha=9.0,
        kpseg_temporal_fusion_low_conf_threshold=-4.0,
        kpseg_temporal_fusion_max_instances=0,
    )
    assert cfg.kpseg_temporal_fusion_enable is True
    assert 0.0 <= cfg.kpseg_temporal_fusion_alpha <= 0.95
    assert 0.0 <= cfg.kpseg_temporal_fusion_low_conf_threshold <= 1.0
    assert cfg.kpseg_temporal_fusion_max_instances >= 1
