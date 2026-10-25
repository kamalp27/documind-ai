# DocuMind AI — AI-Powered Document Question Answering Chatbot

DocuMind AI is a production-grade Retrieval-Augmented Generation (RAG) system engineered to allow users to upload PDF documents, extract and chunk text with page-level metadata, index vectors in a vector database, and perform grounded question-answering with exact source citations.

---

## Current Status: Phase 1 — Workspace & Backend Foundation

Phase 1 establishes the core backend foundation:
- Modular FastAPI application with structured error handling.
- Environment-driven configuration management via Pydantic Settings.
- Standardized health check endpoint (`/api/v1/health`).
- Isolated Python 3.12 virtual environment and automated test suite via Pytest.

---

## Technology Stack

- **Backend:** Python 3.12, FastAPI, Uvicorn, Pydantic Settings
- **Testing:** Pytest, HTTPX
- **Vector DB & Retrieval (Upcoming):** ChromaDB, Sentence Transformers (`all-MiniLM-L6-v2`)
- **LLM Engine (Upcoming):** Google Gemini API
- **Document Ingestion (Upcoming):** PyMuPDF
- **Frontend (Upcoming):** React, TypeScript, Vite, Vitest

---

## Backend Getting Started (Phase 1)

### 1. Prerequisites
- Python 3.12+ installed.

### 2. Activate Virtual Environment
From the `backend/` directory:

**Windows (PowerShell):**
```powershell
cd backend
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
cd backend
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

### 5. Run the Backend Server
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
Open [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health) to view the health status.
API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 6. Run Test Suite
```bash
pytest
```
