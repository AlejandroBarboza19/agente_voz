from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── FastAPI ──────────────────────────────────────────────
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    secret_key: str = "change-me-in-production"

    # ── LLM Cloud & STT (OpenRouter Whisper) ──────────────────
    base_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: str  # Obligatoria en el .env, sin valor por defecto
    llm_model: str = "google/gemini-2.5-flash"
    system_prompt: str = (
        "Eres un asistente de voz amigable y conciso de la Clínica SaludTotal. "
        "Responde siempre en el mismo idioma que el usuario. "
        "Mantén las respuestas cortas y naturales para conversación de voz."
    )

    # ── ElevenLabs ───────────────────────────────────────────
    elevenlabs_api_key: str  # Obligatoria en el .env
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    elevenlabs_model_id: str = "eleven_multilingual_v2"

    # ── AWS Infraestructura (Auditoría e Historial) ──────────
    aws_access_key_id: str  # Obligatoria en el .env
    aws_secret_access_key: str  # Obligatoria en el .env
    aws_region: str = "us-east-1"

    # ── DynamoDB & S3 ────────────────────────────────────────
    dynamodb_table_name: str = "voice_agent_sessions"
    dynamodb_session_ttl_hours: int = 24
    transcribe_s3_bucket: str = "alejandro-agente-voz-2026"  # Usado para el backup de audios

    # ── AWS Lambda ───────────────────────────────────────────
    lambda_cleanup_function: str = "voice-agent-session-cleanup"

    # ── PostgreSQL (Base de datos médica) ────────────────────
    postgres_dsn: str = "postgresql://postgres:[REDACTED]@db:5432/agente_voz"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
