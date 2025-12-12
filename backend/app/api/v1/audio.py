from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import base64
from io import BytesIO

from app.audio.stt import transcribe_audio
from app.audio.tts import synthesize_speech
from app.models.schemas import STTResponse, TTSRequest
from app.core.config import settings

router = APIRouter(prefix="/audio", tags=["audio"])

@router.post("/stt", response_model=STTResponse)
async def stt_endpoint(
    file: UploadFile = File(...),
    language: str | None = None,
):
    """
    Speech-to-text: upload audio, get text.
    """
    if not settings.ENABLE_STT:
        raise HTTPException(status_code=503, detail="STT is disabled")

    try:
        audio_bytes = await file.read()
        bio = BytesIO(audio_bytes)
        bio.name = file.filename or "audio.wav"   # IMPORTANT
        text = transcribe_audio(bio, language=language)
        return STTResponse(text=text, language=language)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tts")
async def tts_endpoint(body: TTSRequest):
    """
    Text-to-speech: send text, stream back audio.
    """
    if not settings.ENABLE_TTS:
        raise HTTPException(status_code=503, detail="TTS is disabled")

    audio_bytes = synthesize_speech(
        text=body.text,
        voice=body.voice,
        format=body.format,
    )

    audio_format = body.format or settings.TTS_FORMAT
    media_type = (
        "audio/mpeg" if audio_format == "mp3" else f"audio/{audio_format}"
    )

    return StreamingResponse(
        BytesIO(audio_bytes),
        media_type=media_type,
        headers={"Content-Disposition": f'inline; filename="speech.{audio_format}"'},
    )
