"""
Servicio Text-to-Speech usando ElevenLabs.
Convierte texto a audio en formato MP3.
"""

import structlog
from elevenlabs.client import AsyncElevenLabs
from elevenlabs import VoiceSettings

from app.config import get_settings

logger = structlog.get_logger()
settings = get_settings()


class TTSService:
    def __init__(self):
        self.client = AsyncElevenLabs(api_key=settings.elevenlabs_api_key)
        self.voice_id = settings.elevenlabs_voice_id
        self.model_id = settings.elevenlabs_model_id

    async def synthesize(
        self,
        text: str,
        voice_id: str | None = None,
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        style: float = 0.0,
        use_speaker_boost: bool = True,
    ) -> bytes:
        """
        Convierte texto a audio.

        Args:
            text: Texto a sintetizar
            voice_id: Override del voice_id configurado
            stability: Estabilidad de la voz (0.0 - 1.0)
            similarity_boost: Similitud con la voz original (0.0 - 1.0)
            style: Estilo expresivo (0.0 - 1.0, solo modelos v2+)
            use_speaker_boost: Mejora la claridad del hablante

        Returns:
            bytes del audio en formato MP3
        """
        if not text.strip():
            raise ValueError("El texto para síntesis no puede estar vacío")

        target_voice = voice_id or self.voice_id

        logger.info(
            "tts_request",
            voice_id=target_voice,
            model=self.model_id,
            text_length=len(text),
        )

        voice_settings = VoiceSettings(
            stability=stability,
            similarity_boost=similarity_boost,
            style=style,
            use_speaker_boost=use_speaker_boost,
        )

        # generate() retorna un generador de bytes; lo concatenamos
        audio_generator = self.client.text_to_speech.convert(
            voice_id=target_voice,
            text=text,
            model_id=self.model_id,
            voice_settings=voice_settings,
            output_format="mp3_44100_128",  # MP3, 44.1 kHz, 128 kbps
        )

        audio_bytes = b"".join([chunk async for chunk in await audio_generator])

        logger.info("tts_complete", bytes_generated=len(audio_bytes))
        return audio_bytes

    async def get_available_voices(self) -> list[dict]:
        """Lista todas las voces disponibles en la cuenta de ElevenLabs."""
        try:
            voices_response = await self.client.voices.get_all()
            return [
                {
                    "voice_id": v.voice_id,
                    "name": v.name,
                    "category": v.category,
                    "labels": v.labels,
                }
                for v in voices_response.voices
            ]
        except Exception as e:
            logger.error("elevenlabs_voices_error", error=str(e))
            return []

    async def check_health(self) -> bool:
        """Verifica que la API de ElevenLabs esté accesible."""
        try:
            await self.client.voices.get_all()
            return True
        except Exception as e:
            logger.error("elevenlabs_health_failed", error=str(e))
            return False
