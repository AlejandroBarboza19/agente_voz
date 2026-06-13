"""
Router de endpoints de voz:
  POST /api/v1/voice/chat       — Audio → Transcripción → LLM → Audio respuesta
  POST /api/v1/voice/transcribe — Audio → Transcripción (solo STT)
"""

import uuid
import structlog
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import Response

from app.config import get_settings, Settings
from app.models.chat import Message, Role, VoiceChatResponse, TranscribeResponse
from app.services.transcribe import TranscribeService
from app.services.llm import LLMService
from app.services.tts import TTSService
from app.services.dynamodb import DynamoDBService
from app.utils.audio import validate_audio, detect_format, convert_to_wav

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/voice", tags=["voice"])


def get_services(settings: Settings = Depends(get_settings)):
    return {
        "transcribe": TranscribeService(),
        "llm": LLMService(),
        "tts": TTSService(),
        "db": DynamoDBService(),
    }


@router.post("/chat", summary="Voice chat completo (audio → audio)")
async def voice_chat(
    audio: UploadFile = File(..., description="Archivo de audio del usuario"),
    session_id: str = Form(default=None, description="ID de sesión (se crea si no existe)"),
    language_code: str = Form(default=None, description="Código de idioma, ej: es-ES"),
    services: dict = Depends(get_services),
) -> Response:
    """
    Flujo completo de voz:
    1. Recibe audio del usuario
    2. Transcribe con Amazon Transcribe
    3. Genera respuesta con Ollama
    4. Sintetiza respuesta con ElevenLabs
    5. Retorna audio MP3 + headers con metadata

    El historial de conversación se guarda en DynamoDB por session_id.
    """
    session_id = session_id or str(uuid.uuid4())
    log = logger.bind(session_id=session_id)

    # --- 1. Leer y validar audio ---
    audio_bytes = await audio.read()
    try:
        validate_audio(audio_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    fmt = detect_format(audio_bytes)
    log.info("voice_chat_start", format=fmt, size_bytes=len(audio_bytes))

    # Convertir a WAV si es necesario para Transcribe
    if fmt != "wav":
        audio_bytes = convert_to_wav(audio_bytes, fmt)
        fmt = "wav"

    # --- 2. Transcripción ---
    try:
        transcribe_svc: TranscribeService = services["transcribe"]
        stt_result = await transcribe_svc.transcribe_audio(
            audio_bytes=audio_bytes,
            audio_format=fmt,
            language_code=language_code,
            session_id=session_id,
        )
    except Exception as e:
        log.error("transcription_failed", error=str(e))
        raise HTTPException(status_code=502, detail=f"Error en transcripción: {e}")

    transcript = stt_result["transcript"]
    if not transcript:
        raise HTTPException(
            status_code=422,
            detail="No se pudo transcribir el audio. Verifica la calidad del audio.",
        )
    log.info("transcription_done", transcript=transcript[:100])

    # --- 3. Historial + LLM ---
    db: DynamoDBService = services["db"]
    history = await db.get_session(session_id)

    llm: LLMService = services["llm"]
    try:
        response_text, tokens = await llm.chat(
            user_message=transcript, history=history
        )
    except Exception as e:
        log.error("llm_failed", error=str(e))
        raise HTTPException(status_code=502, detail=f"Error en LLM: {e}")

    log.info("llm_done", tokens=tokens, response_preview=response_text[:80])

    # Persistir en DynamoDB
    new_messages = [
        Message(role=Role.user, content=transcript),
        Message(role=Role.assistant, content=response_text),
    ]
    await db.append_messages(session_id, new_messages)

    # --- 4. TTS ---
    tts: TTSService = services["tts"]
    try:
        audio_response = await tts.synthesize(response_text)
    except Exception as e:
        log.error("tts_failed", error=str(e))
        raise HTTPException(status_code=502, detail=f"Error en síntesis de voz: {e}")

    log.info("tts_done", audio_bytes=len(audio_response))

    # --- 5. Responder con audio + metadata en headers ---
    return Response(
        content=audio_response,
        media_type="audio/mpeg",
        headers={
            "X-Session-ID": session_id,
            "X-Transcript": transcript[:200],
            "X-Response-Text": response_text[:200],
            "X-Tokens-Used": str(tokens),
            "Content-Disposition": f'attachment; filename="response_{session_id}.mp3"',
        },
    )


@router.post(
    "/transcribe",
    response_model=TranscribeResponse,
    summary="Solo transcripción STT",
)
async def transcribe_audio(
    audio: UploadFile = File(..., description="Archivo de audio a transcribir"),
    session_id: str = Form(default=None),
    language_code: str = Form(default=None),
    services: dict = Depends(get_services),
):
    """Transcribe audio a texto sin generar respuesta ni síntesis."""
    session_id = session_id or str(uuid.uuid4())

    audio_bytes = await audio.read()
    try:
        validate_audio(audio_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    fmt = detect_format(audio_bytes)
    if fmt != "wav":
        audio_bytes = convert_to_wav(audio_bytes, fmt)
        fmt = "wav"

    try:
        transcribe_svc: TranscribeService = services["transcribe"]
        result = await transcribe_svc.transcribe_audio(
            audio_bytes=audio_bytes,
            audio_format=fmt,
            language_code=language_code,
            session_id=session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    return TranscribeResponse(
        session_id=session_id,
        transcript=result["transcript"],
        confidence=result.get("confidence"),
        language=result.get("language", "unknown"),
    )
