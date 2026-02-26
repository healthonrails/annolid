from annolid.gui.mixins.prediction_execution_mixin import PredictionExecutionMixin


def test_cutie_start_prefers_existing_predictions() -> None:
    start = PredictionExecutionMixin._resolve_cutie_start_frame_from_seed_state(
        current_start_frame=1001,
        max_existing_frame=1200,
        manual_seed_max=1000,
    )
    assert start == 1201


def test_cutie_start_prefers_latest_manual_seed_when_no_existing_outputs() -> None:
    start = PredictionExecutionMixin._resolve_cutie_start_frame_from_seed_state(
        current_start_frame=1001,
        max_existing_frame=-1,
        manual_seed_max=1000,
    )
    assert start == 1001


def test_cutie_start_keeps_frame_zero_bootstrap_for_single_seed() -> None:
    start = PredictionExecutionMixin._resolve_cutie_start_frame_from_seed_state(
        current_start_frame=1,
        max_existing_frame=-1,
        manual_seed_max=0,
    )
    assert start == 0
