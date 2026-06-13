"""
Entrypoint de la aplicación FastAPI.
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
from app.services.database import init_db
from app.routers.appointments import router as appointments_router

logger = structlog.get_logger()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup y shutdown de la aplicación."""
    logger.info("app_startup", env=settings.app_env, model=settings.ollama_model)

    # Inicializar PostgreSQL
    try:
        init_db()
    except Exception as e:
        logger.warning("postgres_init_warning", error=str(e))

    # Crear tabla DynamoDB si no existe (útil en primera ejecución)
    try:
        DynamoDBService.create_table_if_not_exists(
            region=settings.aws_region,
            table_name=settings.dynamodb_table_name,
        )
    except Exception as e:
        logger.warning("dynamodb_setup_warning", error=str(e))

    # Descargar modelo Ollama si no está disponible
    try:
        llm = LLMService()
        await llm.pull_model_if_needed()
    except Exception as e:
        logger.warning("ollama_pull_warning", error=str(e))

    logger.info("app_ready")
    yield

    logger.info("app_shutdown")


app = FastAPI(
    title="Voice AI Agent",
    description=(
        "Agente de IA de voz con FastAPI + Ollama + ElevenLabs + "
        "Amazon Transcribe + DynamoDB"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ajustar en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(voice.router)
app.include_router(text.router)
app.include_router(appointments_router)


# Static files (frontend)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/health", tags=["system"])
async def health_check():
    """Health check del servicio y sus dependencias."""
    llm = LLMService()
    ollama_ok = await llm.check_health()

    return JSONResponse(
        status_code=200 if ollama_ok else 503,
        content={
            "status": "healthy" if ollama_ok else "degraded",
            "services": {
                "ollama": "ok" if ollama_ok else "unavailable",
                "model": settings.ollama_model,
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
        "message": "Voice AI Agent API",
        "docs": "/docs",
        "health": "/health",
    }
