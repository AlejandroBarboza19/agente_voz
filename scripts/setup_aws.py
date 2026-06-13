"""
Script de setup inicial de AWS.
Crea el bucket S3 y la tabla DynamoDB necesarios.

Uso:
    python scripts/setup_aws.py
"""

import boto3
import os
import sys
from dotenv import load_dotenv

load_dotenv()

REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET = os.getenv("TRANSCRIBE_S3_BUCKET")
TABLE = os.getenv("DYNAMODB_TABLE_NAME", "voice_agent_sessions")
KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")

if not BUCKET:
    print("❌ ERROR: TRANSCRIBE_S3_BUCKET no definido en .env")
    sys.exit(1)


def create_s3_bucket():
    s3 = boto3.client(
        "s3",
        region_name=REGION,
        aws_access_key_id=KEY_ID,
        aws_secret_access_key=SECRET,
    )

    existing = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
    if BUCKET in existing:
        print(f"✅ S3 bucket ya existe: {BUCKET}")
        return

    if REGION == "us-east-1":
        s3.create_bucket(Bucket=BUCKET)
    else:
        s3.create_bucket(
            Bucket=BUCKET,
            CreateBucketConfiguration={"LocationConstraint": REGION},
        )

    # Bloquear acceso público
    s3.put_public_access_block(
        Bucket=BUCKET,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    print(f"✅ S3 bucket creado: {BUCKET}")


def create_dynamodb_table():
    client = boto3.client(
        "dynamodb",
        region_name=REGION,
        aws_access_key_id=KEY_ID,
        aws_secret_access_key=SECRET,
    )

    existing = client.list_tables().get("TableNames", [])
    if TABLE in existing:
        print(f"✅ DynamoDB tabla ya existe: {TABLE}")
        return

    client.create_table(
        TableName=TABLE,
        KeySchema=[{"AttributeName": "session_id", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "session_id", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )

    # Esperar a que la tabla esté activa
    waiter = client.get_waiter("table_exists")
    print(f"⏳ Esperando que la tabla {TABLE} esté lista...")
    waiter.wait(TableName=TABLE)

    # Activar TTL
    client.update_time_to_live(
        TableName=TABLE,
        TimeToLiveSpecification={"Enabled": True, "AttributeName": "ttl"},
    )
    print(f"✅ DynamoDB tabla creada: {TABLE} (TTL habilitado)")


if __name__ == "__main__":
    print(f"\n🚀 Setup AWS - Región: {REGION}\n")
    create_s3_bucket()
    create_dynamodb_table()
    print("\n✅ Setup completo. Ya puedes ejecutar: docker-compose up --build\n")
