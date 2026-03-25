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


def test_cutie_brightness_contrast_kwargs_reads_video_settings() -> None:
    class _Host(PredictionExecutionMixin):
        video_file = "/tmp/demo.mp4"

        @staticmethod
        def get_video_brightness_contrast_values(_video_path):
            return (11, -9)

    host = _Host()
    assert host._cutie_brightness_contrast_kwargs() == {
        "brightness": 11,
        "contrast": -9,
    }


def test_cutie_brightness_contrast_kwargs_clamps_out_of_range_values() -> None:
    class _Host(PredictionExecutionMixin):
        video_file = "/tmp/demo.mp4"

        @staticmethod
        def get_video_brightness_contrast_values(_video_path):
            return (999, -999)

    host = _Host()
    assert host._cutie_brightness_contrast_kwargs() == {
        "brightness": 100,
        "contrast": -100,
    }
