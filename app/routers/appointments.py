"""
Router de citas — endpoints REST para gestión directa.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.services.database import (
    create_appointment,
    get_appointments,
    cancel_appointment,
    get_available_slots,
)

router = APIRouter(prefix="/api/v1/appointments", tags=["appointments"])


class AppointmentCreate(BaseModel):
    client_name: str
    service: str
    date: str        # YYYY-MM-DD
    time: str        # HH:MM
    client_phone: Optional[str] = ""
    notes: Optional[str] = ""


@router.post("/", summary="Crear cita")
def create(body: AppointmentCreate):
    result = create_appointment(**body.model_dump())
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@router.get("/client/{client_name}", summary="Citas de un cliente")
def by_client(client_name: str):
    return get_appointments(client_name)


@router.delete("/{appointment_id}", summary="Cancelar cita")
def cancel(appointment_id: int):
    result = cancel_appointment(appointment_id)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["message"])
    return result


@router.get("/slots/{date}", summary="Horarios disponibles")
def slots(date: str):
    return get_available_slots(date)
