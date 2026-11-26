import os
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.api import nlp, voice, data, translation, documents, ml

load_dotenv()

app = FastAPI(
    title="KANAME Backend API",
    description="AI-Powered Japanese Language and Productivity Intelligence System — Backend Services (Phase 4)",
    version="0.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://localhost:7860",
        "http://127.0.0.1:8501",
        "http://127.0.0.1:7860",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
async def health_check(probe_db: bool = Query(False, description="If true, run a SELECT 1 connectivity probe against the database and include db_status field.")):
    response = {"status": "KANAME backend is running", "version": "0.4.0", "phase": "Phase 4 — Voice Intelligence, Custom Model, Persistence"}
    if probe_db:
        try:
            from backend.database.config import SessionLocal
            from sqlalchemy import text

            db = SessionLocal()
            try:
                db.execute(text("SELECT 1"))
                response["db_status"] = "connected"
                response["db_probe"] = "SELECT 1 OK"
            finally:
                db.close()
        except Exception as e:
            response["db_status"] = "disconnected"
            response["db_probe"] = f"failed: {e}"
    return response


@app.get("/", tags=["System"])
async def root():
    return {
        "project": "KANAME [要]",
        "full_name": "AI-Powered Japanese Language and Productivity Intelligence System",
        "version": "0.4.0",
        "phase": "Phase 4 — Voice Intelligence, Custom ML Model, Durable Persistence",
        "endpoints": {
            "health": "/health",
            "health_probe_db": "/health?probe_db=true",
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi.json": "/openapi.json",
            "api": {
                "nlp": "/api/nlp/",
                "voice": "/api/voice/",
                "data": "/api/data/",
                "translation": "/api/translation/",
                "documents": "/api/documents/",
                "ml": "/api/ml/",
            },
        },
    }


app.include_router(nlp.router, prefix="/api")
app.include_router(voice.router, prefix="/api")
app.include_router(data.router, prefix="/api")
app.include_router(translation.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(ml.router, prefix="/api")
