"""
Servicio Amazon Transcribe para convertir audio a texto (STT).

Flujo:
  1. Sube el archivo de audio a S3
  2. Inicia un job de transcripción en Amazon Transcribe
  3. Espera (con polling) a que el job termine
  4. Descarga y parsea el resultado
  5. Limpia el archivo temporal de S3
"""

import boto3
import uuid
import json
import time
import structlog
from io import BytesIO
from typing import Optional
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

logger = structlog.get_logger()
settings = get_settings()

# Formatos soportados por Amazon Transcribe
SUPPORTED_FORMATS = {"mp3", "mp4", "wav", "flac", "ogg", "amr", "webm"}


class TranscribeService:
    def __init__(self):
        session = boto3.Session(
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region,
        )
        self.transcribe = session.client("transcribe")
        self.s3 = session.client("s3")
        self.bucket = settings.transcribe_s3_bucket

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        audio_format: str = "wav",
        language_code: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> dict:
        """
        Transcribe un archivo de audio a texto.

        Args:
            audio_bytes: Bytes del audio
            audio_format: Formato del audio (wav, mp3, etc.)
            language_code: Código de idioma (ej: es-ES, en-US). None = autodetect.
            session_id: ID de sesión para trazabilidad

        Returns:
            dict con keys: transcript (str), confidence (float), language (str)
        """
        if audio_format.lower() not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Formato '{audio_format}' no soportado. "
                f"Usa uno de: {SUPPORTED_FORMATS}"
            )

        job_id = f"voice-agent-{session_id or uuid.uuid4().hex}-{int(time.time())}"
        s3_key = f"transcribe-jobs/{job_id}.{audio_format}"

        try:
            # 1. Subir audio a S3
            logger.info("uploading_audio_to_s3", job_id=job_id, s3_key=s3_key)
            self.s3.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=audio_bytes,
                ContentType=self._content_type(audio_format),
            )
            s3_uri = f"s3://{self.bucket}/{s3_key}"

            # 2. Iniciar job de transcripción
            transcribe_params = {
                "TranscriptionJobName": job_id,
                "Media": {"MediaFileUri": s3_uri},
                "MediaFormat": audio_format.lower(),
                "OutputBucketName": self.bucket,
                "OutputKey": f"transcribe-results/{job_id}.json",
            }

            if language_code:
                transcribe_params["LanguageCode"] = language_code
            else:
                # Identificación automática de idioma
                transcribe_params["IdentifyLanguage"] = True

            logger.info("starting_transcription_job", job_id=job_id)
            self.transcribe.start_transcription_job(**transcribe_params)

            # 3. Esperar resultado (polling)
            result = await self._wait_for_job(job_id)

            # 4. Descargar resultado de S3
            transcript_data = self._download_result(
                f"transcribe-results/{job_id}.json"
            )

            return self._parse_transcript(transcript_data, result)

        finally:
            # 5. Limpiar archivos temporales de S3
            self._cleanup_s3([s3_key, f"transcribe-results/{job_id}.json"])

    @retry(
        stop=stop_after_attempt(30),
        wait=wait_exponential(multiplier=2, min=2, max=30),
    )
    async def _wait_for_job(self, job_id: str) -> dict:
        """Polling hasta que el job de transcripción termine."""
        response = self.transcribe.get_transcription_job(
            TranscriptionJobName=job_id
        )
        job = response["TranscriptionJob"]
        status = job["TranscriptionJobStatus"]

        logger.debug("transcription_job_status", job_id=job_id, status=status)

        if status == "COMPLETED":
            return job
        elif status == "FAILED":
            reason = job.get("FailureReason", "Unknown error")
            raise RuntimeError(f"Transcription job failed: {reason}")
        else:
            # IN_PROGRESS o QUEUED → seguir esperando
            raise Exception(f"Job still running: {status}")

    def _download_result(self, s3_key: str) -> dict:
        """Descarga y parsea el JSON de resultados de S3."""
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
            return json.loads(response["Body"].read().decode("utf-8"))
        except ClientError as e:
            logger.error("s3_download_error", key=s3_key, error=str(e))
            raise

    def _parse_transcript(self, data: dict, job_metadata: dict) -> dict:
        """Extrae texto e información de confianza del resultado de Transcribe."""
        results = data.get("results", {})
        transcript = results.get("transcripts", [{}])[0].get("transcript", "")

        # Calcular confianza promedio de las palabras
        items = results.get("items", [])
        confidences = [
            float(item["alternatives"][0]["confidence"])
            for item in items
            if item.get("type") == "pronunciation"
            and item.get("alternatives")
            and item["alternatives"][0].get("confidence")
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else None

        # Idioma detectado
        language = job_metadata.get("LanguageCode") or job_metadata.get(
            "IdentifiedLanguageScore", {settings.transcribe_language_code: None}
        )
        if isinstance(language, dict):
            language = settings.transcribe_language_code

        return {
            "transcript": transcript.strip(),
            "confidence": round(avg_confidence, 4) if avg_confidence else None,
            "language": language,
        }

    def _cleanup_s3(self, keys: list[str]) -> None:
        """Elimina archivos temporales de S3 para evitar costos."""
        for key in keys:
            try:
                self.s3.delete_object(Bucket=self.bucket, Key=key)
                logger.debug("s3_cleanup", key=key)
            except ClientError:
                pass  # No crítico si falla la limpieza

    @staticmethod
    def _content_type(fmt: str) -> str:
        types = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "flac": "audio/flac",
            "ogg": "audio/ogg",
            "webm": "audio/webm",
            "mp4": "video/mp4",
            "amr": "audio/amr",
        }
        return types.get(fmt.lower(), "application/octet-stream")
