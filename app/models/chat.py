from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class Role(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class Message(BaseModel):
    role: Role
    content: str


class TextChatRequest(BaseModel):
    session_id: str = Field(..., description="ID único de la sesión de conversación")
    message: str = Field(..., description="Mensaje del usuario")
    stream: bool = Field(default=False, description="Streaming de respuesta")


class TextChatResponse(BaseModel):
    session_id: str
    message: str
    response: str
    tokens_used: Optional[int] = None


class VoiceChatResponse(BaseModel):
    session_id: str
    transcript: str
    response_text: str
    audio_url: Optional[str] = None  # URL temporal si se almacena en S3


class TranscribeResponse(BaseModel):
    session_id: str
    transcript: str
    confidence: Optional[float] = None
    language: str


class SessionHistory(BaseModel):
    session_id: str
    messages: list[Message]
    created_at: str
    updated_at: str
    ttl: Optional[int] = None
