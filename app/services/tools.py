"""
Definición de tools (function calling) para Ollama.
El LLM decide cuándo llamar a cada función según la conversación.
"""

from app.services.database import (
    create_appointment,
    get_appointments,
    cancel_appointment,
    get_available_slots,
)
from datetime import date

# ── Definición de tools para Ollama ──────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_appointment",
            "description": "Agenda una nueva cita para un cliente. Usar cuando el cliente quiere reservar una cita.",
            "parameters": {
                "type": "object",
                "properties": {
                    "client_name": {"type": "string", "description": "Nombre completo del cliente"},
                    "service": {"type": "string", "description": "Tipo de servicio o motivo de la cita"},
                    "date": {"type": "string", "description": "Fecha en formato YYYY-MM-DD"},
                    "time": {"type": "string", "description": "Hora en formato HH:MM"},
                    "client_phone": {"type": "string", "description": "Teléfono del cliente (opcional)"},
                    "notes": {"type": "string", "description": "Notas adicionales (opcional)"},
                },
                "required": ["client_name", "service", "date", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_appointments",
            "description": "Consulta las citas próximas de un cliente.",
            "parameters": {
                "type": "object",
                "properties": {
                    "client_name": {"type": "string", "description": "Nombre del cliente"},
                },
                "required": ["client_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_appointment",
            "description": "Cancela una cita existente usando su ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "appointment_id": {"type": "integer", "description": "ID numérico de la cita a cancelar"},
                },
                "required": ["appointment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_available_slots",
            "description": "Consulta los horarios disponibles para una fecha específica.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Fecha en formato YYYY-MM-DD"},
                },
                "required": ["date"],
            },
        },
    },
]


def execute_tool(name: str, args: dict) -> str:
    """Ejecuta la función correspondiente y retorna el resultado como string."""
    import json

    handlers = {
        "create_appointment": create_appointment,
        "get_appointments": get_appointments,
        "cancel_appointment": cancel_appointment,
        "get_available_slots": get_available_slots,
    }

    fn = handlers.get(name)
    if not fn:
        return f"Función '{name}' no encontrada"

    try:
        result = fn(**args)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error ejecutando {name}: {e}"
