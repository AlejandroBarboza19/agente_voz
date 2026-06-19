FROM python:3.12-slim

WORKDIR /app

# 1. Instalamos dependencias del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 2. Copiamos e instalamos los requerimientos de Python de forma global
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Copiamos el código fuente de la aplicación
COPY app/ ./app/

# 4. Configuración de usuario seguro (No root)
RUN useradd --no-create-home --no-log-init appuser && \
    chown -R appuser /app
USER appuser

EXPOSE 8000

# Healthcheck interno del contenedor
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# 🔥 Agregamos --reload para desarrollo activo 🔥
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--reload"]