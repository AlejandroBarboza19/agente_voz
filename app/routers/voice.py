"""
Router de endpoints de voz optimizado por Alejandro Barboza.
Híbrido: Baja latencia con Whisper Cloud + SQL Agent de LangChain + Respaldo S3 asíncrono.
"""

import uuid
import structlog
import base64
from io import BytesIO
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from fastapi.responses import Response
import requests
from concurrent.futures import ThreadPoolExecutor
import asyncio

from app.config import get_settings, Settings
from app.models.chat import Message, Role, TranscribeResponse
from app.services.llm import LLMService
from app.services.tts import TTSService
from app.services.dynamodb import DynamoDBService
from app.services.s3 import S3Service
from app.utils.audio import validate_audio, detect_format, convert_to_wav

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/voice", tags=["voice"])

# Thread pool para llamadas síncronas
executor = ThreadPoolExecutor(max_workers=4)

def get_services(settings: Settings = Depends(get_settings)):
    return {
        "llm": LLMService(),
        "tts": TTSService(),
        "db": DynamoDBService(),
        "s3": S3Service()
    }

# --- FUNCIÓN DE RESPALDO ASÍNCRONO (AWS S3) ---
async def ejecutar_respaldo_auditoria(s3_svc: S3Service, audio_bytes: bytes, filename: str, session_id: str):
    """Sube el audio a S3 en un hilo secundario sin hacer esperar al usuario."""
    try:
        logger.info("aws_s3_respaldo_inicio", session_id=session_id)
        await s3_svc.upload_audio(audio_bytes, filename, session_id)
        logger.info("aws_s3_respaldo_exitoso", session_id=session_id)
    except Exception as e:
        logger.error("aws_s3_respaldo_fallido", session_id=session_id, error=str(e))


def transcribe_sync(audio_bytes: bytes, language: str, api_key: str) -> str:
    """
    Transcribe audio usando OpenRouter.
    Formato correcto: input_audio como objeto con data y format.
    """
    try:
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }
        
        # Convertir audio a base64
        audio_b64 = base64.standard_b64encode(audio_bytes).decode('utf-8')
        
        payload = {
            'model': 'openai/whisper-1',
            'input_audio': {
                'data': audio_b64,
                'format': 'wav',
            },
            'language': language,
        }
        
        response = requests.post(
            'https://openrouter.ai/api/v1/audio/transcriptions',
            headers=headers,
            json=payload,
            timeout=60,
        )
        
        logger.info("openrouter_response", status_code=response.status_code)
        
        if response.status_code != 200:
            logger.error("openrouter_error", text=response.text, status=response.status_code)
            raise Exception(f"OpenRouter error: {response.text}")
        
        result = response.json()
        logger.info("openrouter_success", has_text='text' in result)
        return result.get('text', '')
    except Exception as e:
        logger.error("openrouter_transcribe_failed", error=str(e))
        raise


async def transcribe_with_openrouter(audio_bytes: bytes, filename: str, language: str, api_key: str) -> str:
    """
    Wrapper async que ejecuta la transcripción en un thread pool.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        transcribe_sync,
        audio_bytes,
        language,
        api_key
    )


@router.post("/chat", summary="Voice chat completo (audio → audio)")
async def voice_chat(
    audio: UploadFile = File(..., description="Archivo de audio del usuario"),
    session_id: str = Form(default=None, description="ID de sesión (se crea si no existe)"),
    language_code: str = Form(default=None, description="Código de idioma, ej: es"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    settings: Settings = Depends(get_settings),
    services: dict = Depends(get_services),
) -> Response:
    
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

    # --- 2. 🔥 RESPALDO ASÍNCRONO EN S3 🔥 ---
    s3_svc: S3Service = services["s3"]
    background_tasks.add_task(ejecutar_respaldo_auditoria, s3_svc, audio_bytes, audio.filename, session_id)

    # Convertir a WAV
    if fmt != "wav":
        try:
            audio_bytes = convert_to_wav(audio_bytes, fmt)
            fmt = "wav"
        except Exception as e:
            log.error("conversion_failed", error=str(e))
            raise HTTPException(status_code=400, detail=f"No se pudo convertir el audio: {e}")

    # --- 3. TRANSCRIPCIÓN CON WHISPER VÍA OPENROUTER ---
    try:
        log.info("enviando_a_whisper_openrouter")
        transcript = await transcribe_with_openrouter(
            audio_bytes,
            audio.filename or "audio.wav",
            language_code or "es",
            settings.openrouter_api_key
        )
        
    except Exception as e:
        log.error("whisper_failed", error=str(e))
        raise HTTPException(status_code=502, detail=f"Error en STT Whisper: {e}")

    if not transcript.strip():
        raise HTTPException(status_code=422, detail="No se escuchó voz clara.")
    log.info("transcription_done", transcript=transcript[:100])

    # --- 4. CEREBRO CON SQL AGENT (LangChain + DynamoDB) ---
    db: DynamoDBService = services["db"]
    history = await db.get_session(session_id)

    llm: LLMService = services["llm"]
    try:
        response_chunks = []
        async for chunk in llm.chat_stream(user_message=transcript, history=history):
            response_chunks.append(chunk)
            
        response_text = "".join(response_chunks)
        
    except Exception as e:
        log.error("sql_agent_failed", error=str(e))
        raise HTTPException(status_code=502, detail=f"Error en el Agente SQL: {e}")

    log.info("sql_agent_done", response_preview=response_text[:80])

    # Guardamos la conversación
    new_messages = [
        Message(role=Role.user, content=transcript),
        Message(role=Role.assistant, content=response_text),
    ]
    await db.append_messages(session_id, new_messages)

    # --- 5. VOZ CON ELEVENLABS ---
    tts: TTSService = services["tts"]
    audio_response = None
    tts_error = None
    
    try:
        audio_response = await tts.synthesize(response_text)
        log.info("tts_done", audio_bytes=len(audio_response))
    except Exception as e:
        log.warning("tts_failed", error=str(e))
        tts_error = str(e)

    # --- 6. RETORNAR AUDIO O TEXTO SEGÚN DISPONIBILIDAD ---
    def safe_header(text: str) -> str:
        return text.encode("ascii", errors="ignore").decode("ascii")[:200]

    if audio_response:
        return Response(
            content=audio_response,
            media_type="audio/mpeg",
            headers={
                "X-Session-ID": session_id,
                "X-Transcript": safe_header(transcript),
                "X-Response-Text": safe_header(response_text),
                "Content-Disposition": f'attachment; filename="response_{session_id}.mp3"',
            },
        )
    else:
        # Retornar JSON con texto si TTS falló
        import json
        response_data = {
            "session_id": session_id,
            "transcript": transcript,
            "response": response_text,
            "status": "success",
            "tts_available": False,
            "tts_error": tts_error
        }
        return Response(
            content=json.dumps(response_data, ensure_ascii=False),
            media_type="application/json",
            headers={
                "X-Session-ID": session_id,
            },
        )


@router.post("/transcribe", response_model=TranscribeResponse, summary="Solo transcripción STT")
async def transcribe_audio(
    audio: UploadFile = File(..., description="Archivo de audio a transcribir"),
    session_id: str = Form(default=None),
    language_code: str = Form(default=None),
    settings: Settings = Depends(get_settings),
):
    """Transcripción usando Whisper vía OpenRouter."""
    session_id = session_id or str(uuid.uuid4())
    audio_bytes = await audio.read()
    
    fmt = detect_format(audio_bytes)
    if fmt != "wav":
        try:
            audio_bytes = convert_to_wav(audio_bytes, fmt)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"No se pudo convertir el audio: {e}")

    try:
        transcript = await transcribe_with_openrouter(
            audio_bytes,
            audio.filename or "audio.wav",
            language_code or "es",
            settings.openrouter_api_key
        )
        
        return TranscribeResponse(
            session_id=session_id,
            transcript=transcript,
            confidence=1.0,
            language=language_code or "es",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
