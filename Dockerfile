# KANAME [要] — AI-Powered Japanese Language & Productivity Intelligence System
# Multi-process container: FastAPI backend (internal :8000) + Streamlit dashboard (public :7860).
# Compatible with Hugging Face Spaces (Docker SDK), Render, Fly.io, etc.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# System dependencies: build toolchain + audio/FFmpeg runtime for speech-to-text.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ffmpeg \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching).
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the application source.
COPY . .

# Public port for the main (Streamlit) UI — matches Hugging Face Spaces default.
EXPOSE 7860

RUN chmod +x scripts/start.sh

CMD ["bash", "scripts/start.sh"]
