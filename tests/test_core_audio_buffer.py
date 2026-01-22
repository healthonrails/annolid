import numpy as np

from annolid.core.media.audio import AudioBuffer


def test_audio_buffer_aligns_frames_to_samples():
    # 2 seconds of fake audio at 1000 Hz.
    data = np.arange(2000, dtype=np.float32)
    buf = AudioBuffer(audio_data=data, sample_rate=1000, fps=10.0)

    # Frame 0 should map to sample 0.
    assert buf.frame_to_sample_index(0) == 0
    # Frame 10 at 10 fps is 1 second => sample 1000.
    assert buf.frame_to_sample_index(10) == 1000

    frame_samples = buf.samples_for_frame(10)
    assert frame_samples.size > 0
    assert frame_samples[0] == 1000
