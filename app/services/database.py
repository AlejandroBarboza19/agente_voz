"""
Servicio PostgreSQL para citas y clientes.
Usa psycopg2 con connection pool.
"""

import psycopg2
import psycopg2.pool
import structlog
from datetime import datetime, timezone
from typing import Optional
from functools import lru_cache

from app.config import get_settings

logger = structlog.get_logger()
settings = get_settings()


@lru_cache(maxsize=1)
def get_pool():
    return psycopg2.pool.ThreadedConnectionPool(
        minconn=1,
        maxconn=10,
        dsn=settings.postgres_dsn,
    )


def get_conn():
    return get_pool().getconn()


def release_conn(conn):
    get_pool().putconn(conn)


def init_db():
    """Crea las tablas si no existen."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS clients (
                    id          SERIAL PRIMARY KEY,
                    name        VARCHAR(100) NOT NULL,
                    phone       VARCHAR(30),
                    email       VARCHAR(100),
                    created_at  TIMESTAMP DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS appointments (
                    id           SERIAL PRIMARY KEY,
                    client_name  VARCHAR(100) NOT NULL,
                    client_phone VARCHAR(30),
                    service      VARCHAR(100) NOT NULL,
                    date         DATE NOT NULL,
                    time         TIME NOT NULL,
                    status       VARCHAR(20) DEFAULT 'confirmed',
                    notes        TEXT,
                    created_at   TIMESTAMP DEFAULT NOW()
                );
            """)
            conn.commit()
            logger.info("database_initialized")
    finally:
        release_conn(conn)


# ── Funciones de citas ────────────────────────────────────────────────────────

def create_appointment(
    client_name: str,
    service: str,
    date: str,
    time: str,
    client_phone: str = "",
    notes: str = "",
) -> dict:
    """Crea una nueva cita. date: YYYY-MM-DD, time: HH:MM"""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO appointments (client_name, client_phone, service, date, time, notes)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (client_name, client_phone, service, date, time, notes))
            appt_id = cur.fetchone()[0]
            conn.commit()
            logger.info("appointment_created", id=appt_id, client=client_name)
            return {
                "success": True,
                "appointment_id": appt_id,
                "message": f"Cita confirmada para {client_name} el {date} a las {time} — {service}",
            }
    except Exception as e:
        conn.rollback()
        logger.error("appointment_create_error", error=str(e))
        return {"success": False, "message": f"Error al crear la cita: {e}"}
    finally:
        release_conn(conn)


def get_appointments(client_name: str) -> dict:
    """Consulta las citas de un cliente."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, service, date, time, status, notes
                FROM appointments
                WHERE LOWER(client_name) LIKE LOWER(%s)
                  AND date >= CURRENT_DATE
                ORDER BY date, time
                LIMIT 5
            """, (f"%{client_name}%",))
            rows = cur.fetchall()

        if not rows:
            return {
                "success": True,
                "appointments": [],
                "message": f"No hay citas próximas para {client_name}",
            }

        appts = []
        for row in rows:
            appts.append({
                "id": row[0],
                "service": row[1],
                "date": str(row[2]),
                "time": str(row[3])[:5],
                "status": row[4],
                "notes": row[5] or "",
            })

        return {"success": True, "appointments": appts}
    finally:
        release_conn(conn)


def cancel_appointment(appointment_id: int) -> dict:
    """Cancela una cita por ID."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE appointments SET status = 'cancelled'
                WHERE id = %s AND status = 'confirmed'
                RETURNING client_name, service, date, time
            """, (appointment_id,))
            row = cur.fetchone()
            conn.commit()

        if not row:
            return {"success": False, "message": "Cita no encontrada o ya cancelada"}

        return {
            "success": True,
            "message": f"Cita cancelada: {row[1]} del {row[0]} el {row[2]} a las {str(row[3])[:5]}",
        }
    except Exception as e:
        conn.rollback()
        return {"success": False, "message": str(e)}
    finally:
        release_conn(conn)


def get_available_slots(date: str) -> dict:
    """Retorna horarios disponibles para una fecha dada."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT time FROM appointments
                WHERE date = %s AND status = 'confirmed'
            """, (date,))
            occupied = {str(r[0])[:5] for r in cur.fetchall()}

        # Horarios de 9:00 a 18:00 cada 30 minutos
        all_slots = []
        for h in range(9, 18):
            for m in ("00", "30"):
                all_slots.append(f"{h:02d}:{m}")

        available = [s for s in all_slots if s not in occupied]
        return {
            "success": True,
            "date": date,
            "available_slots": available,
        }
    finally:
        release_conn(conn)
