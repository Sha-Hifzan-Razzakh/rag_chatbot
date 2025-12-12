from typing import Optional
from openai import OpenAI
from app.core.config import settings

if settings.OPENAI_API_KEY is None:
    raise RuntimeError("OPENAI_API_KEY is not set. Check your .env file.")

api_key = settings.OPENAI_API_KEY.get_secret_value()
client = OpenAI(api_key=api_key)

def transcribe_audio(file_obj, language: Optional[str] = None) -> str:
    if not settings.ENABLE_STT:
        raise RuntimeError("STT is disabled")

    try:
        file_obj.seek(0)
    except Exception:
        pass

    if not getattr(file_obj, "name", None):
        file_obj.name = "audio.wav"

    kwargs = {"model": settings.STT_MODEL_NAME, "file": file_obj}
    if language:
        kwargs["language"] = language

    response = client.audio.transcriptions.create(**kwargs)
    return response.text if hasattr(response, "text") else response["text"]
