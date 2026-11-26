#!/usr/bin/env bash
# KANAME — container entrypoint (FastAPI backend + Streamlit dashboard).
# Designed for single-port hosts such as Hugging Face Spaces (Docker SDK).
set -e

# ---- Runtime configuration (overridable via host secrets/env) --------------
export BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:8000}"
export DATABASE_URL="${DATABASE_URL:-sqlite:///./kaname.db}"
export BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
export BACKEND_PORT="${BACKEND_PORT:-8000}"
# Public port the host exposes for the main UI.
export UI_PORT="${PORT:-7860}"

echo "[KANAME] DB=${DATABASE_URL}  backend=${BACKEND_HOST}:${BACKEND_PORT}  ui=${UI_PORT}"

# ---- 1. Initialize the SQLite database (idempotent) ------------------------
python backend/init_db.py || echo "[KANAME] init_db skipped/failed — continuing"

# ---- 2. Boot the FastAPI backend in the background -------------------------
uvicorn backend.main:app --host "${BACKEND_HOST}" --port "${BACKEND_PORT}" &
BACKEND_PID=$!

# ---- 3. Boot the Streamlit dashboard on the public port (foreground) -------
streamlit run frontend/streamlit_app/app.py \
    --server.address 0.0.0.0 \
    --server.port "${UI_PORT}" \
    --server.headless true \
    --browser.gatherUsageStats false &
STREAMLIT_PID=$!

# ---- Shut both down together -----------------------------------------------
trap 'kill "${BACKEND_PID}" "${STREAMLIT_PID}" 2>/dev/null || true' TERM INT
wait -n "${BACKEND_PID}" "${STREAMLIT_PID}"
