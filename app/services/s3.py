"""
Servicio de S3 para respaldo de audios.
"""

import boto3
import structlog
from botocore.exceptions import NoCredentialsError, ClientError
from app.config import Settings # Asegúrate de que esta ruta a tus settings sea la correcta en tu estructura

logger = structlog.get_logger()


class S3Service:
    """Servicio para subir audios a AWS S3."""
    
    def __init__(self):
        """Inicializa el cliente de S3 mapeando las credenciales del entorno."""
        try:
            # Forzamos a boto3 a usar tus credenciales y región exactas del archivo .env
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION
            )
            # Lee dinámicamente tu bucket real (alejandro-agente-voz-2026)
            self.bucket_name = settings.AWS_S3_BUCKET  
        except Exception as e:
            logger.error("s3_init_failed", error=str(e))
            self.s3_client = None
    
    async def upload_audio(self, audio_bytes: bytes, filename: str, session_id: str) -> bool:
        """
        Sube un archivo de audio a S3 de forma dinámica.
        
        Args:
            audio_bytes: Contenido del archivo en bytes
            filename: Nombre del archivo
            session_id: ID de sesión para organización
            
        Returns:
            True si se subió exitosamente, False si falló
        """
        if not self.s3_client:
            logger.warning("s3_upload_skipped", reason="No S3 client configured")
            return False
        
        try:
            key = f"audios/{session_id}/{filename}"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=audio_bytes,
                ContentType="audio/mpeg"
            )
            logger.info("s3_upload_success", key=key)
            return True
        except NoCredentialsError:
            logger.error("s3_upload_failed", reason="AWS credentials not found")
            return False
        except ClientError as e:
            logger.error("s3_upload_failed", error=str(e))
            return False
        except Exception as e:
            logger.error("s3_upload_failed", error=str(e))
            return False
    
    async def download_audio(self, session_id: str, filename: str) -> bytes:
        """
        Descarga un archivo de audio de S3.
        
        Args:
            session_id: ID de sesión
            filename: Nombre del archivo
            
        Returns:
            Contenido del archivo en bytes, o None si no se encontró
        """
        if not self.s3_client:
            return None
        
        try:
            key = f"audios/{session_id}/{filename}"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            return response['Body'].read()
        except ClientError as e:
            logger.error("s3_download_failed", error=str(e))
            return None