import os
import tempfile
from datetime import datetime
from typing import Optional

try:
    import moonshine_onnx as moonshine
except ImportError:
    moonshine = None


def transcribe_audio(
    audio_input, model_name="moonshine/base", output_dir: Optional[str] = None
):
    """
    Transcribes audio from either a WAV file path or raw WAV data (bytes)
    using the moonshine_onnx library.
    """
    if moonshine is None:
        raise ImportError(
            "The 'moonshine_onnx' library is not installed. "
            "Install it via: pip install useful-moonshine-onnx"
        )

    wav_file_path = None
    temp_file_path = None

    if isinstance(audio_input, str):
        wav_file_path = audio_input
    elif isinstance(audio_input, bytes):
        try:
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                wav_file_path = os.path.join(
                    output_dir, f"audio_data_from_bytes_{timestamp}.wav"
                )
                with open(wav_file_path, "wb") as f:
                    f.write(audio_input)
            else:
                fd, temp_file_path = tempfile.mkstemp(
                    prefix="annolid_audio_", suffix=".wav"
                )
                with os.fdopen(fd, "wb") as f:
                    f.write(audio_input)
                wav_file_path = temp_file_path
        except Exception as e:
            print(f"Error saving raw WAV data to WAV: {e}")
            return None
    else:
        raise TypeError(
            "audio_input must be a WAV file path (str) or raw WAV data (bytes)"
        )

    try:
        if wav_file_path:
            return moonshine.transcribe(wav_file_path, model_name)
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass

    return None
