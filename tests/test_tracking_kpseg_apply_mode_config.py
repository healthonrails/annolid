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
