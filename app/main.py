"""
Entrypoint de la aplicación FastAPI.
Optimizado por Alejandro Barboza: Conexión en la nube + SQL Agent de LangChain.
"""

import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from app.config import get_settings
from app.routers import voice, text
from app.services.llm import LLMService
from app.services.dynamodb import DynamoDBService
# 🚨 Eliminamos el import de app.services.database ya que no usamos psycopg2 manual

logger = structlog.get_logger()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup y shutdown de la aplicación."""
    logger.info("app_startup", env=settings.app_env)

    # 1. Crear tabla DynamoDB si no existe (Historial de sesiones en AWS)
    try:
        DynamoDBService.create_table_if_not_exists(
            region=settings.aws_region,
            table_name=settings.dynamodb_table_name,
        )
    except Exception as e:
        logger.warning("dynamodb_setup_warning", error=str(e))

    # 🚨 Quitamos los bloques de init_db() y llm.pull_model_if_needed() 
    # ya que las tablas las manejas desde pgAdmin y el LLM corre por API en OpenRouter.

    logger.info("app_ready")
    yield

    logger.info("app_shutdown")


app = FastAPI(
    title="Voice AI Agent",
    description=(
        "Agente de IA de voz híbrido con FastAPI + LangChain SQL Agent + ElevenLabs + "
        "Whisper Cloud + DynamoDB + S3"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS (Mantenemos tu configuración abierta para desarrollo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(voice.router)
app.include_router(text.router)
# Nota: Si dejas appointments_router, asegúrate de que use SQLAlchemy o puedes quitarlo si solo el agente manejará las citas
# app.include_router(appointments_router)

# Static files (frontend)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/health", tags=["system"])
async def health_check():
    """Health check del servicio adaptado a APIs en la nube."""
    llm = LLMService()
    # Ahora verificamos la salud del servicio mapeado con la API Key externa
    api_ok = await llm.check_health()

    return JSONResponse(
        status_code=200 if api_ok else 503,
        content={
            "status": "healthy" if api_ok else "degraded",
            "services": {
                "openrouter_api": "ok" if api_ok else "unavailable"
            },
            "version": "1.0.0",
        },
    )


@app.get("/", tags=["system"])
async def root():
    static_index = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(static_index):
        return FileResponse(static_index)
    return {
        "message": "Voice AI Agent API con SQL Agent Listo",
        "docs": "/docs",
        "health": "/health",
    }