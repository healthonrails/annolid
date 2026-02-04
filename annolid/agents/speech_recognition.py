import os
from datetime import datetime

try:
    import moonshine_onnx as moonshine
except ImportError:
    print(
        "\nError: The 'moonshine_onnx' library is not installed. \n"
        "Please install it to use this transcription function.\n\n"
        "You can install it using pip with the following command:\n\n"
        "   pip install useful-moonshine-onnx@git+https://git@github.com/usefulsensors/moonshine.git#subdirectory=moonshine-onnx\n\n"
        "For more information about moonshine and moonshine-onnx, please visit:\n"
        "   https://github.com/usefulsensors/moonshine\n\n"
        "After installation, please run your script again.\n"
    )


def transcribe_audio(
    audio_input, model_name="moonshine/base", output_dir="audio_files"
):
    """
    Transcribes audio from either a WAV file path or raw WAV data (bytes)
    using the moonshine_onnx library.

    For raw bytes input, it saves the audio to a WAV file in a structured directory
    before transcription for better organization and potential letter checking.

    Args:
        audio_input: Either:
                       - a string representing the path to a WAV file,
                       - raw WAV data as bytes (e.g., from audio_input.get_wav_data()).
        model_name (str, optional): The name of the moonshine model to use for transcription.
                                     Defaults to 'moonshine/base'.
        output_dir (str, optional): The directory where WAV files from raw bytes will be saved.
                                     Defaults to "audio_files".

    Returns:
        list[str]: A list containing the transcribed text as strings, as returned by moonshine_onnx.
                   Returns None if there is an error during processing.
    """
    wav_file_path = None

    if isinstance(audio_input, str):
        # Input is a WAV file path
        wav_file_path = audio_input
    elif isinstance(audio_input, bytes):
        # Input is raw WAV data (bytes)
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)

            # Create a timestamp for unique and ordered filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Best naming practice: informative filename with timestamp for raw bytes
            base_filename = f"audio_data_from_bytes_{timestamp}"
            wav_file_path = os.path.join(output_dir, f"{base_filename}.wav")

            # Save raw bytes to WAV file
            with open(wav_file_path, "wb") as f:
                f.write(audio_input)  # Directly write the bytes

            print(f"Raw WAV data saved to: {wav_file_path}")

        except Exception as e:
            print(f"Error saving raw WAV data to WAV: {e}")
            return None  # Indicate failure

    else:
        raise TypeError(
            "audio_input must be a WAV file path (str) or raw WAV data (bytes)"
        )

    if wav_file_path:
        try:
            transcribed_text = moonshine.transcribe(wav_file_path, model_name)
            print(f"Transcribed text from {wav_file_path}:")
            print(transcribed_text)
            return transcribed_text
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None  # Indicate transcription failure

    return None  # Should not reach here normally
