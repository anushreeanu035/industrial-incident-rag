# Multi-stage lightweight container for Industrial Incident Investigator Backend
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# Install minimal build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pre-install lightweight CPU-only PyTorch to prevent downloading large CUDA wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and initial seed data
COPY pyproject.toml .
COPY src/ src/
COPY data/raw/ data/raw/
COPY data/benchmark/ data/benchmark/

# Install the package in editable/package mode
RUN pip install --no-cache-dir -e .

EXPOSE 8000

# Start Uvicorn bound to 0.0.0.0 and dynamically assigned $PORT
CMD ["sh", "-c", "uvicorn incident_rag.api:app --host 0.0.0.0 --port ${PORT:-8000} --app-dir src"]
