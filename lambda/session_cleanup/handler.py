"""
AWS Lambda: Limpieza de sesiones antiguas en DynamoDB.

Trigger recomendado: EventBridge (CloudWatch Events) cada hora.
DynamoDB TTL ya maneja la expiración automática, pero esta función
permite limpiezas forzadas y generar reportes de uso.

Deployment:
  zip -r session_cleanup.zip .
  aws lambda update-function-code --function-name voice-agent-session-cleanup \
      --zip-file fileb://session_cleanup.zip
"""

import boto3
import os
import json
import logging
from datetime import datetime, timezone, timedelta
from boto3.dynamodb.conditions import Attr

logger = logging.getLogger()
logger.setLevel(logging.INFO)

TABLE_NAME = os.environ.get("DYNAMODB_TABLE_NAME", "voice_agent_sessions")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
MAX_SESSION_AGE_HOURS = int(os.environ.get("MAX_SESSION_AGE_HOURS", "48"))


def handler(event: dict, context) -> dict:
    """
    Handler principal del Lambda.

    Puede recibir:
    - Evento programado de EventBridge (limpieza automática)
    - Invocación directa con {"action": "cleanup", "session_id": "..."}
    """
    logger.info("Lambda invoked", event=json.dumps(event))

    action = event.get("action", "scheduled_cleanup")

    if action == "delete_session":
        return delete_single_session(event.get("session_id"))
    else:
        return run_scheduled_cleanup()


def run_scheduled_cleanup() -> dict:
    """Elimina sesiones cuyo TTL ya venció (limpieza adicional a la de DynamoDB)."""
    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    table = dynamodb.Table(TABLE_NAME)

    cutoff = datetime.now(timezone.utc) - timedelta(hours=MAX_SESSION_AGE_HOURS)
    cutoff_iso = cutoff.isoformat()

    logger.info(f"Scanning for sessions older than {cutoff_iso}")

    deleted_count = 0
    last_key = None

    while True:
        scan_kwargs = {
            "FilterExpression": Attr("updated_at").lt(cutoff_iso),
            "ProjectionExpression": "session_id",
        }
        if last_key:
            scan_kwargs["ExclusiveStartKey"] = last_key

        response = table.scan(**scan_kwargs)
        items = response.get("Items", [])

        for item in items:
            session_id = item["session_id"]
            table.delete_item(Key={"session_id": session_id})
            deleted_count += 1
            logger.info(f"Deleted session: {session_id}")

        last_key = response.get("LastEvaluatedKey")
        if not last_key:
            break

    result = {
        "statusCode": 200,
        "deleted_sessions": deleted_count,
        "cutoff_date": cutoff_iso,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    logger.info(f"Cleanup complete: {deleted_count} sessions deleted")
    return result


def delete_single_session(session_id: str) -> dict:
    """Elimina una sesión específica."""
    if not session_id:
        return {"statusCode": 400, "error": "session_id requerido"}

    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    table = dynamodb.Table(TABLE_NAME)

    table.delete_item(Key={"session_id": session_id})
    logger.info(f"Session deleted: {session_id}")

    return {
        "statusCode": 200,
        "message": f"Sesión {session_id} eliminada",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
