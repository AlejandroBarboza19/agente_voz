"""
Utilidades de audio: conversión de formatos, validación, extracción de metadata.
Usa pydub internamente (requiere ffmpeg en PATH).
"""

import io
import structlog
from pydub import AudioSegment

logger = structlog.get_logger()

SUPPORTED_INPUT_FORMATS = {"wav", "mp3", "ogg", "flac", "webm", "m4a", "mp4"}
MAX_AUDIO_SIZE_MB = 25  # límite de Amazon Transcribe


def validate_audio(audio_bytes: bytes, max_mb: int = MAX_AUDIO_SIZE_MB) -> None:
    """Valida tamaño del audio antes de procesarlo."""
    size_mb = len(audio_bytes) / (1024 * 1024)
    if size_mb > max_mb:
        raise ValueError(
            f"Audio demasiado grande: {size_mb:.1f} MB. Máximo: {max_mb} MB"
        )
    if len(audio_bytes) == 0:
        raise ValueError("El archivo de audio está vacío")


def detect_format(audio_bytes: bytes) -> str:
    """
    Detecta el formato del audio por sus magic bytes.
    Retorna una extensión en minúsculas.
    """
    if audio_bytes[:4] == b"RIFF":
        return "wav"
    if audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
        return "mp3"
    if audio_bytes[:4] == b"fLaC":
        return "flac"
    if audio_bytes[:4] == b"OggS":
        return "ogg"
    if audio_bytes[:4] in (b"\x1a\x45\xdf\xa3",):
        return "webm"
    # Default: asumir wav
    logger.warning("unknown_audio_format_magic_bytes", fallback="wav")
    return "wav"


def convert_to_wav(audio_bytes: bytes, source_format: str) -> bytes:
    """
    Convierte audio a WAV mono 16kHz (óptimo para Amazon Transcribe).
    """
    try:
        segment = AudioSegment.from_file(
            io.BytesIO(audio_bytes), format=source_format
        )
        # Amazon Transcribe funciona mejor con mono 16kHz
        segment = segment.set_channels(1).set_frame_rate(16000)

        output = io.BytesIO()
        segment.export(output, format="wav")
        return output.getvalue()
    except Exception as e:
        logger.error("audio_conversion_error", format=source_format, error=str(e))
        raise ValueError(f"No se pudo convertir el audio: {e}") from e


def get_audio_metadata(audio_bytes: bytes, fmt: str) -> dict:
    """Extrae duración y propiedades del audio."""
    try:
        segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
        return {
            "duration_seconds": round(len(segment) / 1000, 2),
            "channels": segment.channels,
            "frame_rate": segment.frame_rate,
            "sample_width": segment.sample_width,
        }
    except Exception:
        return {}
