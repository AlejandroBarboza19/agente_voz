"""
Router de endpoints de texto adaptado para Alejandro Barboza.
Conecta los chats de texto planos y de streaming al nuevo SQL Agent de LangChain.
"""

import uuid
import structlog
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse

from app.models.chat import (
    TextChatRequest,
    TextChatResponse,
    SessionHistory,
    Message,
    Role,
)
from app.services.llm import LLMService
from app.services.dynamodb import DynamoDBService

logger = structlog.get_logger()
router = APIRouter(tags=["text & sessions"])


def get_llm() -> LLMService:
    return LLMService()


def get_db() -> DynamoDBService:
    return DynamoDBService()


@router.post(
    "/api/v1/text/chat",
    response_model=TextChatResponse,
    summary="Chat de texto con historial",
)
async def text_chat(
    request: TextChatRequest,
    llm: LLMService = Depends(get_llm),
    db: DynamoDBService = Depends(get_db),
):
    """
    Chat de texto bloqueante adaptado al Agente SQL.
    Consume el generador asíncrono acumulando la respuesta antes de retornar.
    """
    log = logger.bind(session_id=request.session_id)
    log.info("text_chat_request", message=request.message[:80])

    # Historial previo desde DynamoDB
    history = await db.get_session(request.session_id)

    # 🔥 CORRECCIÓN CLAVE: Acumulamos el generador asíncrono porque 'llm.chat' ya no existe 🔥
    try:
        response_chunks = []
        async for chunk in llm.chat_stream(user_message=request.message, history=history):
            response_chunks.append(chunk)
        
        response_text = "".join(response_chunks)
    except Exception as e:
        log.error("llm_error", error=str(e))
        raise HTTPException(status_code=502, detail=f"Error al generar respuesta del agente: {e}")

    # Persistir mensajes en DynamoDB usando tu formato oficial
    await db.append_messages(
        request.session_id,
        [
            Message(role=Role.user, content=request.message),
            Message(role=Role.assistant, content=response_text),
        ],
    )

    return TextChatResponse(
        session_id=request.session_id,
        message=request.message,
        response=response_text,
        tokens_used=0,  # Ponemos 0 ya que el agente de LangChain oculta el conteo crudo de tokens de OpenRouter
    )


@router.post(
    "/api/v1/text/chat/stream",
    summary="Chat de texto con streaming",
)
async def text_chat_stream(
    request: TextChatRequest,
    llm: LLMService = Depends(get_llm),
    db: DynamoDBService = Depends(get_db),
):
    """
    Igual que /text/chat pero retorna la respuesta en streaming (SSE) directo desde Gemini.
    """
    history = await db.get_session(request.session_id)
    full_response = []

    async def event_generator():
        try:
            async for chunk in llm.chat_stream(request.message, history=history):
                full_response.append(chunk)
                yield f"data: {chunk}\n\n"
        except Exception as e:
            logger.error("stream_error", error=str(e))
            yield f"data: Error en el flujo: {str(e)}\n\n"

        # Al terminar el ciclo, guardamos la transcripción completa en tu DynamoDB
        complete_text = "".join(full_response)
        if complete_text:
            await db.append_messages(
                request.session_id,
                [
                    Message(role=Role.user, content=request.message),
                    Message(role=Role.assistant, content=complete_text),
                ],
            )
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get(
    "/api/v1/sessions/{session_id}",
    response_model=SessionHistory,
    summary="Obtener historial de sesión",
)
async def get_session(
    session_id: str,
    db: DynamoDBService = Depends(get_db),
):
    """Mantiene intacta tu lógica de consultas a DynamoDB"""
    messages = await db.get_session(session_id)
    if not messages:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")

    return SessionHistory(
        session_id=session_id,
        messages=messages,
        created_at="",
        updated_at="",
    )


@router.delete(
    "/api/v1/sessions/{session_id}",
    summary="Eliminar sesión",
)
async def delete_session(
    session_id: str,
    db: DynamoDBService = Depends(get_db),
):
    """Mantiene intacto el borrado de sesiones"""
    deleted = await db.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Sesión no encontrada o error al eliminar")
    return {"message": f"Sesión {session_id} eliminada"}