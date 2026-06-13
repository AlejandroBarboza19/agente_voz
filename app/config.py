from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    secret_key: str = "change-me-in-production"

    # Ollama
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "llama3.2"
    ollama_system_prompt: str = (
        "Eres un asistente de voz amigable y conciso. "
        "Responde siempre en el mismo idioma que el usuario. "
        "Mantén las respuestas cortas y naturales para conversación de voz."
    )

    # ElevenLabs
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    elevenlabs_model_id: str = "eleven_multilingual_v2"

    # AWS
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"

    # Amazon Transcribe
    transcribe_language_code: str = "es-ES"
    transcribe_s3_bucket: str = ""

    # DynamoDB
    dynamodb_table_name: str = "voice_agent_sessions"
    dynamodb_session_ttl_hours: int = 24

    # PostgreSQL
    postgres_dsn: str = "postgresql://postgres:postgres@db:5432/agente_voz"

    # Lambda
    lambda_cleanup_function: str = "voice-agent-session-cleanup"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
