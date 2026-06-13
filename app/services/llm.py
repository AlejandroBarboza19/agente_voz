"""
Servicio LLM usando Ollama (local, gratuito).
Soporta chat con historial de conversación y streaming.
"""

import ollama
import structlog
from typing import AsyncIterator, Optional

from app.config import get_settings
from app.models.chat import Message, Role

logger = structlog.get_logger()
settings = get_settings()


class LLMService:
    def __init__(self):
        self.client = ollama.AsyncClient(host=settings.ollama_base_url)
        self.model = settings.ollama_model
        self.system_prompt = settings.ollama_system_prompt

    def _build_messages(
        self,
        history: list[Message],
        user_message: str,
    ) -> list[dict]:
        """
        Construye la lista de mensajes para la API de Ollama.
        Siempre antepone el system prompt.
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        for msg in history:
            messages.append({"role": msg.role.value, "content": msg.content})

        messages.append({"role": "user", "content": user_message})
        return messages

    async def chat(
        self,
        user_message: str,
        history: Optional[list[Message]] = None,
    ) -> tuple[str, int]:
        """
        Genera una respuesta de texto.

        Returns:
            (response_text, prompt_eval_count + eval_count)
        """
        history = history or []
        messages = self._build_messages(history, user_message)

        logger.info(
            "llm_request",
            model=self.model,
            history_length=len(history),
            message_preview=user_message[:80],
        )

        response = await self.client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": 0.7,
                "num_predict": 512,   # limitar tokens para respuestas de voz
            },
        )

        text = response["message"]["content"]
        tokens = response.get("prompt_eval_count", 0) + response.get("eval_count", 0)

        logger.info("llm_response", tokens=tokens, preview=text[:80])
        return text, tokens

    async def chat_stream(
        self,
        user_message: str,
        history: Optional[list[Message]] = None,
    ) -> AsyncIterator[str]:
        """
        Genera una respuesta en modo streaming.
        Yield de fragmentos de texto a medida que llegan.
        """
        history = history or []
        messages = self._build_messages(history, user_message)

        logger.info("llm_stream_start", model=self.model)

        async for chunk in await self.client.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options={"temperature": 0.7, "num_predict": 512},
        ):
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content

    async def check_health(self) -> bool:
        """Verifica que Ollama esté disponible y el modelo cargado."""
        try:
            models = await self.client.list()
            available = [m["name"] for m in models.get("models", [])]
            logger.info("ollama_models_available", models=available)

            if not any(self.model in m for m in available):
                logger.warning(
                    "model_not_found",
                    model=self.model,
                    available=available,
                )
                return False
            return True
        except Exception as e:
            logger.error("ollama_health_check_failed", error=str(e))
            return False

    async def pull_model_if_needed(self) -> None:
        """Descarga el modelo si no está disponible localmente."""
        try:
            healthy = await self.check_health()
            if not healthy:
                logger.info("pulling_model", model=self.model)
                await self.client.pull(self.model)
                logger.info("model_pulled", model=self.model)
        except Exception as e:
            logger.error("model_pull_failed", model=self.model, error=str(e))
            raise
