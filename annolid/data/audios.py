import warnings

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
        import librosa
        # Load audio data and sample rate using librosa
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.fps = fps

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
        audio_sample_index = int(frame_number / self.fps * self.sample_rate)

        # Calculate the audio duration corresponding to a single video frame
        audio_frame_duration = 1 / self.fps

        # Calculate the number of audio samples for a single video frame
        audio_frame_samples = int(audio_frame_duration * self.sample_rate)

        # Extract the audio samples for the given video frame
        audio_frame = self.audio_data[audio_sample_index:
                                      audio_sample_index + audio_frame_samples]

        return audio_frame

    def play(self):
        import sounddevice as sd
        sd.play(self.audio_data)
        sd.wait()
