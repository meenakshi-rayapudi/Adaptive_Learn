import os
import time
from utils.audio_utils import generate_audio
from langchain_core.tools import tool

def create_audio_narration(text: str, filename_prefix: str = "summary") -> str:
    """
    Synthesizes text into high-quality neural speech and saves it as an MP3 file.
    Returns the relative path to the generated audio file.
    """
    if not text or not text.strip():
        return "No text provided to generate audio."

    os.makedirs("data/audio", exist_ok=True)
    file_name = f"data/audio/{filename_prefix}_{int(time.time())}.mp3"
    try:
        audio_path = generate_audio(text, file_name)
        return audio_path
    except Exception as e:
        return f"Error generating audio narration: {str(e)}"
