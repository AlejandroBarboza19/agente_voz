"""
Servicio LLM usando Ollama con function calling.
Soporta herramientas para agendar citas.
"""

import json
import ollama
import structlog
from typing import AsyncIterator, Optional
from datetime import date

from app.config import get_settings
from app.models.chat import Message, Role
from app.services.tools import TOOLS, execute_tool

logger = structlog.get_logger()
settings = get_settings()

SYSTEM_PROMPT = f"""Eres un asistente virtual de atención al cliente especializado en gestión de citas.
Hoy es {date.today().strftime('%A %d de %B de %Y')}.

Tu trabajo es:
- Saludar amablemente e identificar al cliente
- Agendar, consultar y cancelar citas
- Verificar disponibilidad de horarios antes de confirmar
- Confirmar siempre los detalles al final (nombre, servicio, fecha, hora)

Reglas importantes:
- Habla siempre en español, de forma natural y concisa (respuestas cortas, es una llamada de voz)
- Antes de agendar, consulta los horarios disponibles para la fecha que pide el cliente
- Siempre confirma nombre del cliente, servicio, fecha y hora antes de crear la cita
- Si el cliente dice "mañana", "el lunes", etc., convierte a fecha exacta YYYY-MM-DD
- Horario de atención: lunes a viernes de 9:00 a 18:00
- Si no hay disponibilidad, ofrece alternativas
"""


class LLMService:
    def __init__(self):
        self.client = ollama.AsyncClient(host=settings.ollama_base_url)
        self.model = settings.ollama_model

    def _build_messages(self, history: list[Message], user_message: str) -> list[dict]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
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
        Genera respuesta con soporte de function calling.
        El LLM puede invocar tools para gestionar citas.
        """
        history = history or []
        messages = self._build_messages(history, user_message)
        total_tokens = 0

        logger.info("llm_request", model=self.model, message=user_message[:80])

        # Loop de function calling: el LLM puede llamar múltiples tools
        for _ in range(5):  # máximo 5 rondas de tool calls
            response = await self.client.chat(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                options={"temperature": 0.4, "num_predict": 512},
            )

            total_tokens += response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
            msg = response["message"]

            # ¿El LLM quiere llamar a alguna función?
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                # Respuesta final de texto
                text = msg.get("content", "")
                logger.info("llm_response", tokens=total_tokens, preview=text[:80])
                return text, total_tokens

            # Ejecutar cada tool call
            messages.append({"role": "assistant", "content": msg.get("content", ""), "tool_calls": tool_calls})

            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = tc["function"].get("arguments", {})
                if isinstance(fn_args, str):
                    try:
                        fn_args = json.loads(fn_args)
                    except Exception:
                        fn_args = {}

                logger.info("tool_call", function=fn_name, args=fn_args)
                result = execute_tool(fn_name, fn_args)
                logger.info("tool_result", function=fn_name, result=result[:200])

                messages.append({
                    "role": "tool",
                    "content": result,
                    "name": fn_name,  # algunos modelos lo requieren
                })

        # Si llegamos aquí sin respuesta, forzar respuesta final
        response = await self.client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": 0.4, "num_predict": 256},
        )
        return response["message"].get("content", "Lo siento, ocurrió un error."), total_tokens

    async def chat_stream(
        self,
        user_message: str,
        history: Optional[list[Message]] = None,
    ) -> AsyncIterator[str]:
        """Streaming sin function calling (para el endpoint de texto)."""
        history = history or []
        messages = self._build_messages(history, user_message)

        async for chunk in await self.client.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options={"temperature": 0.4, "num_predict": 512},
        ):
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content

    async def check_health(self) -> bool:
        """Verifica que Ollama esté disponible y el modelo cargado."""
        try:
            response = await self.client.list()
            models_list = response.models if hasattr(response, "models") else response.get("models", [])
            available = []
            for m in models_list:
                name = getattr(m, "model", None) or getattr(m, "name", None) or (m.get("model") or m.get("name") if isinstance(m, dict) else None)
                if name:
                    available.append(name)

            logger.info("ollama_models_available", models=available)
            if not any(self.model in m for m in available):
                logger.warning("model_not_found", model=self.model, available=available)
                return False
            return True
        except Exception as e:
            logger.error("ollama_health_check_failed", error=str(e))
            return False

    async def pull_model_if_needed(self) -> None:
        """Descarga el modelo si no está disponible."""
        try:
            healthy = await self.check_health()
            if not healthy:
                logger.info("pulling_model", model=self.model)
                await self.client.pull(self.model)
                logger.info("model_pulled", model=self.model)
        except Exception as e:
            logger.error("model_pull_failed", model=self.model, error=str(e))
            raise
