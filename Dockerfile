# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --target /build/packages


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# ffmpeg requerido por pydub para conversión de audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copiar paquetes instalados a una ruta accesible para todos los usuarios
COPY --from=builder /build/packages /app/packages
ENV PYTHONPATH=/app/packages
ENV PATH=/app/packages/bin:$PATH

# Copiar código fuente
COPY app/ ./app/

# Crear directorio de cache con permisos correctos
RUN mkdir -p /app/.cache/huggingface && \
    useradd --no-create-home --no-log-init appuser && \
    chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
