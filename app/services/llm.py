"""
Servicio de LLM usando Gemini (vía OpenRouter) + LangChain SQL Agent.
Mantiene la compatibilidad con el historial de DynamoDB y el streaming del router de voz.
"""

import os
import structlog
from typing import AsyncGenerator
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
logger = structlog.get_logger()

# ── 1. CONEXIÓN A LA BASE DE DATOS (PostgreSQL en Docker) ──────────────────
# Usamos el DSN que viene de tu docker-compose
postgres_dsn = os.getenv("POSTGRES_DSN", "postgresql://postgres:postgres@db:5432/agente_voz")
db_engine = SQLDatabase.from_uri(postgres_dsn)


# ── 2. CONFIGURACIÓN DE GEMINI (VÍA OPENROUTER) ─────────────────────────────
llm = ChatOpenAI(
    model="google/gemini-2.5-flash",  # Rápido y preciso para SQL
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.0  # En 0.0 para que no invente citas ni queries falsos
)


# ── 3. PROMPT DEL AGENTE DE VOZ (ADAPTADO) ──────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", """Eres el asistente de voz inteligente de la clínica médica. Tu objetivo es ayudar a los pacientes a gestionar sus citas de forma amable, clara y muy concisa (hablas por teléfono).

## 🔐 REGLAS DE OPERACIÓN
1. Solo puedes consultar, agendar o cancelar citas usando la base de datos.
2. Si no encuentras disponibilidad o datos suficientes, indícaselo al paciente amablemente.
3. Habla con respuestas cortas y directas, adecuadas para ser leídas por un sintetizador de voz (ElevenLabs). Evita textos gigantes o listas interminables.
4. NUNCA menciones términos técnicos como "base de datos", "queries", "tablas" o "SQL".

## 👤 FLUJO PRINCIPAL
- Si es un paciente nuevo, solicita amablemente su nombre o identificación para verificarlo en el sistema.
- Puedes mirar los horarios disponibles, registrar una nueva cita, o cancelar una existente si te dan los datos necesarios.
"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])


# ── 4. CREACIÓN DEL AGENTE SQL ──────────────────────────────────────────────
# Tipo 'openai-functions' garantiza compatibilidad total con Gemini en OpenRouter
agente_sql = create_sql_agent(
    llm=llm,
    db=db_engine,
    prompt=prompt,
    agent_type="openai-functions",
    verbose=True  # Para que veas en los logs de Docker cómo escribe el SQL
)


class LLMService:
    def __init__(self):
        self.agent = agente_sql

    async def check_health(self) -> bool:
        """Verifica que la API de OpenRouter responda (usado por el main.py)."""
        try:
            # Una consulta básica para validar que el puente esté activo
            await llm.ainvoke("ping")
            return True
        except Exception as e:
            logger.error("llm_health_check_failed", error=str(e))
            return False

    async def chat_stream(self, user_message: str, history: list) -> AsyncGenerator[str, None]:
        """
        Procesa el mensaje del usuario usando el SQL Agent y el historial de DynamoDB.
        Retorna un generador de texto (Stream) compatible con tu router de voz.
        """
        try:
            # 1. Convertir el historial que viene de DynamoDB al formato de LangChain
            langchain_history = []
            for msg in history:
                if msg.role.value == "user":
                    langchain_history.append(HumanMessage(content=msg.content))
                elif msg.role.value == "assistant":
                    langchain_history.append(AIMessage(content=msg.content))

            logger.info("ejecutando_agente_sql_voz", message=user_message[:50])

            # 2. Ejecutar el agente de LangChain de forma asíncrona
            # Nota: Los agentes con tool-calling interno consolidan el pensamiento antes de responder,
            # por lo que hacemos un `ainvoke` y simulamos el stream en bloques para tu router actual.
            response = await self.agent.ainvoke({
                "input": user_message,
                "history": langchain_history
            })

            output_text = response.get("output", "Disculpa, tuve un problema procesando tu solicitud.")

            # Lo enviamos como yield para mantener la estructura asíncrona de bloques que espera tu voice.py
            yield output_text

        except Exception as e:
            logger.error("agent_execution_error", error=str(e))
            yield "Lo siento, experimenté un error interno al consultar el sistema de citas."