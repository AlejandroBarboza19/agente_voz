"""
Servicio DynamoDB para almacenar el historial de conversaciones.

Esquema de la tabla:
  PK: session_id (String)
  Atributos: messages (List), created_at, updated_at, ttl (Number, para expiración automática)
"""

import boto3
import json
import structlog
from datetime import datetime, timezone, timedelta
from typing import Optional
from botocore.exceptions import ClientError

from app.config import get_settings
from app.models.chat import Message, Role

logger = structlog.get_logger()
settings = get_settings()


class DynamoDBService:
    def __init__(self):
        self.client = boto3.resource(
            "dynamodb",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        )
        self.table = self.client.Table(settings.dynamodb_table_name)

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _ttl_timestamp(self) -> int:
        """Retorna timestamp Unix para TTL (expiración automática en DynamoDB)."""
        expiry = datetime.now(timezone.utc) + timedelta(
            hours=settings.dynamodb_session_ttl_hours
        )
        return int(expiry.timestamp())

    async def get_session(self, session_id: str) -> list[Message]:
        """Recupera el historial de mensajes de una sesión."""
        try:
            response = self.table.get_item(Key={"session_id": session_id})
            item = response.get("Item")
            if not item:
                return []

            messages = [
                Message(role=Role(m["role"]), content=m["content"])
                for m in item.get("messages", [])
            ]
            logger.info("session_loaded", session_id=session_id, count=len(messages))
            return messages

        except ClientError as e:
            logger.error("dynamodb_get_error", session_id=session_id, error=str(e))
            return []

    async def save_session(self, session_id: str, messages: list[Message]) -> bool:
        """Guarda o actualiza el historial completo de una sesión."""
        try:
            serialized = [
                {"role": m.role.value, "content": m.content} for m in messages
            ]
            now = self._now_iso()

            self.table.put_item(
                Item={
                    "session_id": session_id,
                    "messages": serialized,
                    "updated_at": now,
                    "created_at": now,  # DynamoDB no hace upsert parcial con put_item
                    "ttl": self._ttl_timestamp(),
                }
            )
            logger.info("session_saved", session_id=session_id, count=len(messages))
            return True

        except ClientError as e:
            logger.error("dynamodb_save_error", session_id=session_id, error=str(e))
            return False

    async def append_messages(
        self, session_id: str, new_messages: list[Message]
    ) -> list[Message]:
        """Agrega mensajes nuevos al historial existente y persiste."""
        history = await self.get_session(session_id)
        history.extend(new_messages)
        await self.save_session(session_id, history)
        return history

    async def delete_session(self, session_id: str) -> bool:
        """Elimina una sesión por completo."""
        try:
            self.table.delete_item(Key={"session_id": session_id})
            logger.info("session_deleted", session_id=session_id)
            return True
        except ClientError as e:
            logger.error("dynamodb_delete_error", session_id=session_id, error=str(e))
            return False

    async def get_session_metadata(self, session_id: str) -> Optional[dict]:
        """Retorna metadata de la sesión sin cargar todos los mensajes."""
        try:
            response = self.table.get_item(
                Key={"session_id": session_id},
                ProjectionExpression="session_id, created_at, updated_at, #cnt",
                ExpressionAttributeNames={"#cnt": "message_count"},
            )
            return response.get("Item")
        except ClientError:
            return None

    @staticmethod
    def create_table_if_not_exists(region: str, table_name: str) -> None:
        """
        Crea la tabla DynamoDB si no existe.
        Útil para entornos de desarrollo / primeras ejecuciones.
        """
        client = boto3.client("dynamodb", region_name=region)
        existing = client.list_tables().get("TableNames", [])
        if table_name in existing:
            logger.info("table_already_exists", table=table_name)
            return

        client.create_table(
            TableName=table_name,
            KeySchema=[{"AttributeName": "session_id", "KeyType": "HASH"}],
            AttributeDefinitions=[
                {"AttributeName": "session_id", "AttributeType": "S"}
            ],
            BillingMode="PAY_PER_REQUEST",  # sin aprovisionamiento fijo
        )
        # Habilitar TTL para expiración automática de sesiones
        client.update_time_to_live(
            TableName=table_name,
            TimeToLiveSpecification={"Enabled": True, "AttributeName": "ttl"},
        )
        logger.info("table_created", table=table_name)
