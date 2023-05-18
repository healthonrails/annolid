import warnings
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
        """
        Play the entire audio file.
        """
        sd.play(self.audio_data, self.sample_rate, blocking=False)

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
