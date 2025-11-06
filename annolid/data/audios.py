import warnings
from typing import Optional
try:
    import sounddevice as sd
except ImportError:
    print("The 'sounddevice' module is required for audio playback.")
    print("Please install it by running: pip install sounddevice")
try:
    import librosa
except ImportError:
    print("The 'librosa' module is required for audio loading.")
    print("Please install it by running: pip install librosa")
# Suppress the Warnings
warnings.filterwarnings("ignore")


class AudioLoader:
    """
    A class for loading audio data and extracting audio samples for specific video frames.

    Parameters:
    - file_path (str): Path to the video or audio file.

    Usage:
    - Create an instance of the AudioLoader class by providing the file path.
    - Use the load_audio_for_frame() method to load audio samples for a specific video frame.
    """

    def __init__(self, file_path, fps=29.97):
        """
        Initialize the AudioLoader object.

        Parameters:
        - file_path (str): Path to the video or audio file.
        """
        # Load audio data and sample rate using librosa
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.fps = fps
        self._playhead_sample = 0

    def _frame_to_sample_index(self, frame_number: int) -> int:
        """
        Convert a video frame index to the closest audio sample index.

        Args:
            frame_number (int): Frame number to convert; negative values clamp to zero.

        Returns:
            int: Sample index within the audio buffer.
        """
        if frame_number is None:
            return 0
        frame_number = max(int(frame_number), 0)
        sample_index = int(round(frame_number / self.fps * self.sample_rate))
        return max(0, min(sample_index, len(self.audio_data)))

    def load_audio_for_frame(self, frame_number):
        """
        Load audio samples for a specific video frame.

        Parameters:
        - frame_number (int): Frame number for which to load the audio samples.
        - fps (float): Frames per second of the video.

        Returns:
        - audio_frame (ndarray): Numpy array containing the audio samples for the given frame.
        """
        # Calculate the corresponding audio sample index
        audio_sample_index = self._frame_to_sample_index(frame_number)

        # Calculate the audio duration corresponding to a single video frame
        audio_frame_duration = 1 / self.fps

        # Calculate the number of audio samples for a single video frame
        audio_frame_samples = int(audio_frame_duration * self.sample_rate)

        # Extract the audio samples for the given video frame
        end_index = audio_sample_index + audio_frame_samples
        audio_frame = self.audio_data[audio_sample_index:end_index]

        return audio_frame

    def set_playhead_frame(self, frame_number: int) -> None:
        """
        Set the internal playhead to align playback with a specific video frame.

        Args:
            frame_number (int): Frame number that should align with the playhead.
        """
        self._playhead_sample = self._frame_to_sample_index(frame_number)

    def play(self, start_frame: Optional[int] = None):
        """
        Play the audio from the current or specified frame position.

        Args:
            start_frame (int | None): Optional frame index from which to start playback.
        """
        if start_frame is not None:
            self.set_playhead_frame(start_frame)

        if self._playhead_sample >= len(self.audio_data):
            # Nothing to play from beyond the buffer length.
            return

        sd.play(
            self.audio_data[self._playhead_sample:],
            self.sample_rate,
            blocking=False,
        )

    def play_selected_part(self, x_start, x_end):
        """
        Play the selected part of the audio between the given x-axis values.

        Args:
            x_start (float): Start position on the x-axis.
            x_end (float): End position on the x-axis.
        """
        # Calculate the start and end indices based on x-axis values
        start_index = int(x_start * self.sample_rate)
        end_index = int(x_end * self.sample_rate)

        # Extract the selected part of the audio
        selected_audio = self.audio_data[start_index:end_index]

        # Play the selected audio using sounddevice
        sd.play(selected_audio, self.sample_rate, blocking=True)
        self._playhead_sample = start_index

    def stop(self):
        """
        Stop the currently playing audio.
        """
        sd.stop()
