---
title: KANAME
emoji: ⚔️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# KANAME [要]

## AI-Powered Japanese Language and Productivity Intelligence System

MCA Minor Project — Semester-Long Individual Artificial Intelligence Project

---

## Project Description

KANAME is an AI-powered system focused on **Japanese language processing**, **voice intelligence**, **data analysis & visualization**, **Japanese ↔ English translation**, **bilingual document generation**, and a **custom-trained neural text classifier**.

The project uses **AI / NLP / Machine Learning** with custom-trained models for suitable tasks (not only API wrappers). Streamlit is the main dashboard UI, Gradio is used specifically for interactive voice/audio AI workflows, FastAPI serves as the backend, and SQLite provides durable persistence via SQLAlchemy.

---

## Technology Stack

| Layer                | Technologies                                     |
| -------------------- | ------------------------------------------------ |
| **Primary Language** | Python 3.10                                      |
| **Main Frontend**    | Streamlit                                        |
| **AI / Voice UI**    | Gradio (mic / audio specialized)                 |
| **Backend**          | FastAPI + Uvicorn                                |
| **Database**         | SQLite (file-based, zero external services)      |
| **ORM / Migrations** | SQLAlchemy 2.x + Alembic                         |
| **Japanese NLP**     | SudachiPy (+ SudachiDict-core)                   |
| **Machine Learning** | scikit-learn (LinearSVC / KMeans)                |
| **Deep Learning**    | PyTorch (custom MLP text classifier, CPU)        |
| **Speech-to-Text**   | faster-whisper (CTranslate2)                     |
| **Translation / LLM**| Google Gemini API (`google-generativeai`)        |
| **Data Processing**  | pandas, NumPy                                    |
| **Visualization**    | Plotly                                           |
| **Documents**        | python-docx (DOCX), ReportLab (PDF)              |

---

## Folder Structure

```
KANAME1/
├── frontend/
│   ├── streamlit_app/            # MAIN user interface (Streamlit)
│   │   ├── app.py                # Entry point — Dashboard + all module pages
│   │   ├── components/           # Reusable Streamlit components
│   │   └── styles/               # Custom CSS / themes
│   │
│   └── gradio_app/               # Specialized Voice / Audio AI UI
│       └── app.py                # STT + AI Response Gradio interface
│
├── backend/
│   ├── main.py                   # FastAPI application entry point
│   ├── api/                      # API routers (/api/nlp, /voice, /data, /translation, /documents, /ml)
│   ├── services/                 # Business logic layer (CRUD + persistence)
│   ├── models/                   # SQLAlchemy ORM models
│   ├── schemas/                  # Pydantic request / response schemas
│   ├── database/
│   │   └── config.py             # Engine, SessionLocal, Base, get_db()
│   └── init_db.py                # Create tables (SQLite by default)
│
├── ai_engine/                    # Core AI / NLP / ML logic
│   ├── japanese_nlp/             # SudachiPy tokenization, analysis, JLPT assessment
│   ├── machine_learning/         # scikit-learn models + custom PyTorch MLP (torch_classifier.py)
│   ├── voice_intelligence/       # STT (faster-whisper) / post-processing
│   ├── translation/              # JP ↔ EN translation + localization pipeline
│   └── data_intelligence/        # Analytics + Plotly visualization
│
├── document_engine/              # DOCX / PDF generation logic
├── alembic/                      # Migration environment + versions
├── datasets/                     # Training & analysis datasets
├── tests/                        # Unit & integration tests
│
├── requirements.txt              # Python dependencies (pinned)
├── .env.example                  # Environment variable template
├── .gitignore
└── README.md
```

---

## Environment Setup

### 1. Create a Python Virtual Environment

From the project root (`KANAME1/`):

**Windows (PowerShell / CMD):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Custom model (PyTorch):** the CPU build keeps installs lightweight.
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`

### 3. Configure Environment Variables

**Windows:**
```powershell
copy .env.example .env
```

**macOS / Linux:**
```bash
cp .env.example .env
```

Then edit `.env`. Persistence defaults to a local SQLite file — no external services required:

```
DATABASE_URL=sqlite:///./kaname.db

# LLM / translation (optional — translation falls back to offline mode when unset)
GOOGLE_API_KEY=your_gemini_api_key
GOOGLE_MODEL=gemini-3.5-flash
```

### 4. Initialize the Database

Tables are created automatically on first run. To create them explicitly (or reset):

```bash
python backend/init_db.py
# reset + recreate:
python backend/init_db.py --drop-first
```

---

## How to Run

> All commands are executed from the **project root** (`KANAME1/`) with the **virtual environment activated**.

### Run FastAPI Backend

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

- Interactive docs: Swagger UI http://localhost:8000/docs · ReDoc http://localhost:8000/redoc
- Health check: **GET http://localhost:8000/health** → `{"status":"KANAME backend is running","version":"0.4.0","phase":"Phase 4 …"}`

### Run Streamlit (Main Dashboard)

```bash
streamlit run frontend/streamlit_app/app.py
```

- Default URL: **http://localhost:8501**
- Dashboard + all module pages (NLP, Voice, Data, Translation, Documents, Machine Learning, Project Info).

### Run Gradio (Voice Intelligence Interface)

```bash
python frontend/gradio_app/app.py
```

- Default URL: **http://localhost:7860**
- Microphone recording, audio upload, and faster-whisper STT integration.

---

## API Overview (prefix `/api`)

| Module       | Highlights                                                                 |
| ------------ | -------------------------------------------------------------------------- |
| `nlp`        | Tokenization, morphological analysis, JLPT assessment, analysis history    |
| `voice`      | STT transcription, VAD, post-processing, keyword spotting, persisted jobs  |
| `data`       | Dataset upload/analysis, statistics, Plotly-ready chart payloads           |
| `translation`| JA ↔ EN translation, evaluation metrics                                    |
| `documents`  | DOCX / PDF generation with optional auto-translation, document ledger      |
| `ml`         | Classification, clustering, hybrid intent detection, **custom PyTorch MLP** (`/ml/torch/*`) |

---

## Current Phase

**Phase 4 — Voice Intelligence · Custom ML Model · Durable Persistence (complete)**

- ✅ **Persistence** — SQLite + SQLAlchemy 2.x + Alembic; tables for NLP analyses, voice jobs, datasets, and documents
- ✅ **Voice Intelligence** — faster-whisper STT end-to-end (Gradio UI + `/api/voice`), VAD, persisted jobs
- ✅ **Custom ML model** — PyTorch MLP text classifier trained end-to-end via gradient descent (`/api/ml/torch/*`)
- ✅ **Existing modules** — Japanese NLP (SudachiPy + JLPT), Data Intelligence (Plotly), JA↔EN Translation, bilingual Document generation
- ✅ **Deployment** — Docker configuration for free hosting (Hugging Face Spaces)

## Roadmap Workflows and Evaluation

The API and Streamlit dashboard also expose supervised BIO NER training from user-provided span annotations, topic discovery, confidence-gated auto-label suggestions, a demonstration Transformer classifier, request-scoped translation glossaries/examples, DOCX translation, batch document generation, PDF merging, outline drafting, and dataset quality profiling. Uploaded datasets are stored with their source content in the database and can be reused by `dataset_id`; keep database backups in mind because source files can be large.

The supervised NER model is a trainable token classifier using Sudachi morphology and neighboring-token features, not a pretrained language model. Submit aligned annotations to `POST /api/nlp/entities/train`, inspect `GET /api/nlp/entities/model`, and predict with `POST /api/nlp/entities/predict`. Its default dictionary/POS extractor remains available separately at `POST /api/nlp/entities`.

Voice-command parsing only suggests an action and never executes it. TTS and pretrained speaker diarization are optional host-dependent extensions:

```bash
pip install -r requirements-voice-optional.txt
```

TTS additionally needs a compatible system speech engine and installed voices. Diarization requires accepting the access terms for `pyannote/speaker-diarization-3.1` and setting `HF_TOKEN` in the backend environment. Without these prerequisites, those endpoints return a clear unavailable-feature error.

The Transformer, topic model, and built-in text classifiers are functional demonstrations, not evidence of production accuracy. Training accuracy on the bundled sample corpus is not a held-out metric. No project-wide or per-module 95% accuracy claim is valid without representative labeled test sets, leakage-safe splits, and separately reported metrics for each task. Add benchmark datasets and evaluate precision/recall/F1 (and task-appropriate metrics) before making accuracy guarantees.
