from typing import Optional
from io import BytesIO
from urllib import response

from openai import OpenAI
from app.core.config import settings
from app.audio.types import TTSResponseFormat

if settings.OPENAI_API_KEY is None:
    raise RuntimeError("OPENAI_API_KEY is not set. Check your .env file.")

api_key = settings.OPENAI_API_KEY.get_secret_value()
client = OpenAI(api_key=api_key)

def synthesize_speech(
    text: str,
    voice: Optional[str] = None,
    format: TTSResponseFormat | None = None
) -> bytes:
    """
    Returns raw audio bytes for the given text.
    """
    if not settings.ENABLE_TTS:
        raise RuntimeError("TTS is disabled")

    voice = voice or settings.TTS_VOICE
    fmt: TTSResponseFormat = format or settings.TTS_FORMAT

    # Adjust to the exact API for your OpenAI version
    response = client.audio.speech.create(
        model=settings.TTS_MODEL_NAME,
        voice=voice,
        input=text,
        response_format=fmt,
    )

    if hasattr(response, "read") and callable(response.read):
        audio_bytes = response.read()
        if not isinstance(audio_bytes, (bytes, bytearray)):
            raise TypeError(f"response.read() did not return bytes, got {type(audio_bytes)}")
        return bytes(audio_bytes)

    raise TypeError(f"Unexpected TTS response type: {type(response)}")
