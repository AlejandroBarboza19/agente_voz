# Voice AI Agent

Agente de IA de voz usando Python, FastAPI, Ollama, ElevenLabs, AWS (Lambda, DynamoDB, Transcribe) y Docker.

## Stack

| Componente | Tecnología |
|---|---|
| API | FastAPI |
| LLM | Ollama (local, gratuito) |
| STT | Amazon Transcribe |
| TTS | ElevenLabs |
| Base de datos | DynamoDB |
| Funciones serverless | AWS Lambda |
| Contenedores | Docker |

## Flujo de la conversación

```
Audio (usuario) → Amazon Transcribe → texto
texto → FastAPI → Ollama (LLM) → respuesta
respuesta → ElevenLabs → audio (respuesta)
Historial almacenado en DynamoDB
```

## Requisitos previos

- Docker y Docker Compose
- Cuenta AWS con credenciales configuradas
- API Key de ElevenLabs
- Ollama (corre automáticamente via Docker Compose)

## Inicio rápido

```bash
# 1. Copiar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales

# 2. Levantar servicios
docker-compose up --build

# 3. La API estará disponible en http://localhost:8000
# Documentación: http://localhost:8000/docs
```

## Endpoints principales

| Método | Ruta | Descripción |
|---|---|---|
| POST | `/api/v1/voice/chat` | Envía audio, recibe audio de respuesta |
| POST | `/api/v1/voice/transcribe` | Solo transcribe audio a texto |
| POST | `/api/v1/text/chat` | Chat en texto (sin voz) |
| GET | `/api/v1/sessions/{session_id}` | Historial de sesión |
| DELETE | `/api/v1/sessions/{session_id}` | Eliminar sesión |
| GET | `/health` | Estado del servicio |

## Estructura del proyecto

```
voice-ai-agent/
├── app/
│   ├── main.py                  # Entrypoint FastAPI
│   ├── config.py                # Configuración y variables de entorno
│   ├── models/                  # Modelos Pydantic
│   ├── routers/                 # Rutas de la API
│   ├── services/
│   │   ├── transcribe.py        # Amazon Transcribe (STT)
│   │   ├── llm.py               # Ollama (LLM)
│   │   ├── tts.py               # ElevenLabs (TTS)
│   │   └── dynamodb.py          # DynamoDB (historial)
│   └── utils/
│       └── audio.py             # Utilidades de audio
├── lambda/
│   └── session_cleanup/         # Lambda para limpiar sesiones antiguas
├── infrastructure/
│   └── terraform/               # IaC con Terraform (opcional)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```
