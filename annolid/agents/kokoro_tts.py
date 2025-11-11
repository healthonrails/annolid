import os
import gdown
# --- Handle ImportError for kokoro_onnx ---
try:
    from kokoro_onnx import Kokoro
except ImportError:
    print(
        "\nError: kokoro_onnx library is not installed.\n"
        "Please install it using pip:\n"
        "   pip install kokoro-onnx\n"
    )

# --- Handle ImportError for sounddevice ---
try:
    import sounddevice as sd
except ImportError:
    print(
        "\nError: sounddevice library is not installed.\n"
        "Please install it using pip:\n"
        "   pip install sounddevice\n"
        "\nNote: sounddevice often requires system-level audio libraries (like PortAudio).\n"
        "Installation can be OS-specific. See:\n"
        "   https://python-sounddevice.readthedocs.io/en/latest/install.html\n"
    )

# --- Configuration ---
MODEL_FILENAME = "kokoro-v0_19.onnx"
VOICES_FILENAME = "voices.bin"
MODEL_URL = f"https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/{MODEL_FILENAME}"
VOICES_URL = f"https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/{VOICES_FILENAME}"
CACHE_DIR = ".kokoro_cache"  # Directory to store downloaded files

# --- Helper Functions ---


def download_file(url, filepath):
    """Downloads a file from a URL to a specified filepath using gdown."""
    os.makedirs(os.path.dirname(filepath),
                exist_ok=True)  # Ensure directory exists
    try:
        # quiet=False shows progress bar, fuzzy=True handles redirects
        gdown.download(url, filepath, quiet=False, fuzzy=True)
        print(f"Downloaded to '{filepath}'")
    except Exception as e:
        print(f"Error downloading from {url}: {e}")
        raise  # Re-raise the exception to be caught in text_to_speech


def ensure_files_exist():
    """Ensures model and voices files exist in the cache, downloads if not."""
    model_filepath = os.path.join(CACHE_DIR, MODEL_FILENAME)
    voices_filepath = os.path.join(CACHE_DIR, VOICES_FILENAME)

    if not os.path.exists(model_filepath):
        print(
            f"Model file '{MODEL_FILENAME}' not found in cache. Downloading...")
        download_file(MODEL_URL, model_filepath)
        print(f"Model file downloaded to '{model_filepath}'")
    else:
        print(f"Model file found in cache: '{model_filepath}'")

    if not os.path.exists(voices_filepath):
        print(
            f"Voices file '{VOICES_FILENAME}' not found in cache. Downloading...")
        download_file(VOICES_URL, voices_filepath)
        print(f"Voices file downloaded to '{voices_filepath}'")
    else:
        print(f"Voices file found in cache: '{voices_filepath}'")

    return model_filepath, voices_filepath


def text_to_speech(text, voice="af_sarah", speed=1.0, lang="en-us", output_path="audio.wav"):
    """
    Converts text to speech using Kokoro-ONNX.

    Args:
        text (str): The text to synthesize.
        voice (str): The voice to use (default: "af_sarah").
        speed (float): The speech speed (default: 1.0).
        lang (str): The language (default: "en-us").
        output_path (str, optional): Path to save the audio file. If None, audio is not saved. Defaults to "audio.wav".

    Returns:
        tuple: (samples, sample_rate) - NumPy array of audio samples and the sample rate.
               Returns None if there is an error.
    """
    try:
        model_path, voices_path = ensure_files_exist()
        kokoro = Kokoro(model_path, voices_path)
        samples, sample_rate = kokoro.create(
            text, voice=voice, speed=speed, lang=lang)

        if output_path:
            import soundfile as sf
            sf.write(output_path, samples, sample_rate)
            print(f"Audio saved to '{output_path}'")
        else:
            print("Audio generated but not saved.")

        return samples, sample_rate

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def play_audio(samples, sample_rate):
    """Plays audio data using sounddevice."""
    if samples is not None and sample_rate is not None and samples.size > 0:
        print("Playing audio...")
        sd.play(samples, sample_rate)
        sd.wait()  # Block until audio playback is finished
        print("Audio playback finished.")
    else:
        print("No audio data to play or audio data is empty.")


# --- Main execution (Example usage) ---
if __name__ == "__main__":
    input_text = "We can rewrite the integrand using polynomial long division, because the degree of the numerator is greater than or equal to the degree of the denominator."

    # Set output_path to None to not save to file
    audio_data = text_to_speech(
        input_text, voice="af_sarah", speed=1.1, output_path=None)

    if audio_data:
        samples, sample_rate = audio_data
        print("\nText-to-speech conversion successful!")
        play_audio(samples, sample_rate)  # Play the audio using sounddevice
    else:
        print("\nText-to-speech conversion failed.")
