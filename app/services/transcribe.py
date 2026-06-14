"""
Servicio STT usando faster-whisper (local, gratuito).
Reemplaza Amazon Transcribe — no requiere cuenta AWS.

Modelos disponibles (de menor a mayor calidad/tamaño):
  tiny, base, small, medium, large-v2, large-v3
Para un VPS de 4GB RAM se recomienda "small" o "base".
"""

import io
import structlog
from faster_whisper import WhisperModel
from functools import lru_cache
from typing import Optional

from app.config import get_settings

logger = structlog.get_logger()
settings = get_settings()

WHISPER_MODEL_SIZE = "tiny"    # ~150 MB, más ligero para VPS con poca RAM
WHISPER_DEVICE = "cpu"         # "cuda" si hay GPU
WHISPER_COMPUTE = "int8"       # int8 = más rápido en CPU


@lru_cache(maxsize=1)
def _load_model() -> WhisperModel:
    """Carga el modelo una sola vez y lo mantiene en memoria."""
    logger.info("loading_whisper_model", model=WHISPER_MODEL_SIZE, device=WHISPER_DEVICE)
    model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE,
    )
    logger.info("whisper_model_loaded")
    return model


class TranscribeService:
    def __init__(self):
        self.model = _load_model()
        self.default_language = settings.transcribe_language_code[:2]  # "es-ES" → "es"

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        audio_format: str = "wav",
        language_code: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> dict:
        """
        Transcribe audio a texto usando faster-whisper local.

        Args:
            audio_bytes: Bytes del audio
            audio_format: Formato (ignorado, whisper acepta cualquier formato via ffmpeg)
            language_code: Código de idioma ej "es", "es-ES", "en". None = autodetect.
            session_id: Para trazabilidad en logs

        Returns:
            dict con keys: transcript (str), confidence (float), language (str)
        """
        # Normalizar código de idioma: "es-ES" → "es"
        lang = None
        if language_code:
            lang = language_code.split("-")[0].lower()
        elif self.default_language:
            lang = self.default_language.split("-")[0].lower()

        logger.info(
            "transcribing_audio",
            session_id=session_id,
            language=lang,
            size_bytes=len(audio_bytes),
        )

        audio_io = io.BytesIO(audio_bytes)

        # faster-whisper acepta file-like objects directamente
        segments, info = self.model.transcribe(
            audio_io,
            language=lang,
            beam_size=5,
            vad_filter=True,          # filtra silencios automáticamente
            vad_parameters=dict(
                min_silence_duration_ms=500
            ),
        )

        # Materializar generator y concatenar texto
        text_parts = []
        confidences = []
        for segment in segments:
            text_parts.append(segment.text.strip())
            if hasattr(segment, "avg_logprob"):
                # avg_logprob es negativo; convertir a probabilidad aprox
                import math
                confidence = math.exp(segment.avg_logprob)
                confidences.append(min(confidence, 1.0))

        transcript = " ".join(text_parts).strip()
        avg_confidence = sum(confidences) / len(confidences) if confidences else None

        detected_lang = info.language if info else (lang or "unknown")

        logger.info(
            "transcription_done",
            session_id=session_id,
            transcript=transcript[:100],
            language=detected_lang,
            confidence=avg_confidence,
        )

        return {
            "transcript": transcript,
            "confidence": round(avg_confidence, 4) if avg_confidence else None,
            "language": detected_lang,
        }
