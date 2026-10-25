# DocuMind AI — AI-Powered Document Question Answering Chatbot

DocuMind AI is a production-grade Retrieval-Augmented Generation (RAG) system engineered to allow users to upload PDF documents, extract and chunk text with page-level metadata, index vectors in a local vector database, and perform grounded question-answering with exact source citations.

---

## Project Status

- [x] **Phase 1: Workspace & Backend Foundation**
  - Modular FastAPI application with structured error handling.
  - Environment-driven configuration management via Pydantic Settings.
  - Standardized health check endpoint (`/api/v1/health`).
  - Isolated Python 3.12 virtual environment and automated test suite.
- [x] **Phase 2: PDF Ingestion & Vector-Storage Pipeline**
  - Robust PDF validation (extension, magic bytes signature, size limits, filename sanitization).
  - Page-by-page text extraction with 1-indexed page preservation via PyMuPDF (`pymupdf`).
  - Scanned PDF detection with explicit refusal messages (OCR limitation notice).
  - Deterministic text chunker with sliding overlap and word boundary preservation.
  - Lazy-loading embedding service using `sentence-transformers/all-MiniLM-L6-v2`.
  - Persistent ChromaDB vector store for chunk vectors and metadata.
  - Document metadata persistence across restarts.
  - Full CRUD operations: Upload, List, Get, and Delete documents.
- [ ] **Phase 3: RAG Retrieval Engine & Gemini Integration** (Upcoming)
- [ ] **Phase 4: Modern React + TypeScript Frontend** (Upcoming)
- [ ] **Phase 5: Verification, Edge Cases & Evaluation** (Upcoming)
- [ ] **Phase 6: Production Readiness & Portfolio Showcase** (Upcoming)

---

## Phase 2 Architecture

```
[ Upload PDF ] -> [ Validation & Signature Check ]
                         |
                         v
              [ PyMuPDF Text Extractor ]
                         |
           (Preserves 1-indexed page numbers)
                         |
                         v
             [ Recursive Text Chunker ]
       (Sliding window: 800 chars, 120 overlap)
                         |
                         v
          [ Sentence Transformers Embedder ]
              (all-MiniLM-L6-v2: 384 dims)
                         |
                         +-----------------------------+
                         |                             |
                         v                             v
           [ ChromaDB Vector Store ]       [ Metadata JSON Store ]
          (backend/data/chroma)      (backend/data/documents_metadata.json)
```

---

## Technology Stack

- **Backend Framework:** FastAPI, Uvicorn, Pydantic v2, Pydantic Settings
- **PDF Extraction:** PyMuPDF (`pymupdf 2.0`)
- **Embedding Generation:** Sentence Transformers (`sentence-transformers 6.2.0`, `torch 2.14.1`)
- **Vector Database:** ChromaDB (`chromadb 1.5.9`)
- **Testing:** Pytest, HTTPX

---

## Getting Started

### 1. Prerequisites
- Python 3.12+ installed on Windows, macOS, or Linux.

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

Key configuration variables:
* `MAX_UPLOAD_SIZE_BYTES`: Maximum upload file size (default: `10485760` = 10 MB).
* `CHUNK_SIZE`: Maximum character length of each chunk (default: `800`).
* `CHUNK_OVERLAP`: Overlap between consecutive chunks (default: `120`).
* `EMBEDDING_MODEL_NAME`: Sentence Transformers model (default: `sentence-transformers/all-MiniLM-L6-v2`).
* `CHROMA_PERSIST_DIR`: Local persistence path for ChromaDB (default: `data/chroma`).

### 5. Run Backend Server
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
* **API Documentation (Swagger UI):** [http://127.0.0.1:8000/api/v1/docs](http://127.0.0.1:8000/api/v1/docs)
* **ReDoc Documentation:** [http://127.0.0.1:8000/api/v1/redoc](http://127.0.0.1:8000/api/v1/redoc)
* **Health Check:** [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)

### 6. Run Automated Test Suite
```bash
pytest -v
```

---

## API Endpoints Reference

### Health Check
```bash
curl -X GET "http://127.0.0.1:8000/api/v1/health"
```

### Upload and Index PDF Document
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/documents/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.pdf;type=application/pdf"
```
**Response (201 Created):**
```json
{
  "document_id": "7f8b9c2a-1234-4567-89ab-cdef01234567",
  "filename": "sample.pdf",
  "status": "processed",
  "total_pages": 4,
  "total_chunks": 12,
  "created_at": "2026-10-25T18:00:00.000000+00:00"
}
```

### List Indexed Documents
```bash
curl -X GET "http://127.0.0.1:8000/api/v1/documents"
```

### Get Single Document Metadata
```bash
curl -X GET "http://127.0.0.1:8000/api/v1/documents/{document_id}"
```

### Delete Document and Embeddings
```bash
curl -X DELETE "http://127.0.0.1:8000/api/v1/documents/{document_id}"
```
**Response (200 OK):**
```json
{
  "success": true,
  "message": "Document and associated vector chunks deleted successfully.",
  "document_id": "7f8b9c2a-1234-4567-89ab-cdef01234567"
}
```

---

## Storage & Security Notes

1. **Storage Isolation:**
   - Raw vector embeddings and metadata are stored in `backend/data/chroma/`.
   - Document registry is persisted in `backend/data/documents_metadata.json`.
   - Both locations are strictly excluded from Git via `.gitignore`.
2. **File Validation:**
   - File extensions are validated (`.pdf`).
   - Magic bytes are checked (`%PDF-`), rejecting masqueraded files or scripts.
   - User-supplied filenames are sanitized to prevent directory traversal attacks (e.g. `../../etc/passwd`).
   - Maximum upload size is enforced (10 MB default) to prevent resource exhaustion.
3. **Data Privacy:**
   - Embeddings and vector searches run entirely locally on-premise without third-party external transmission.
   - No filesystem paths, internal stack traces, or sensitive environment keys are exposed in client responses.

---

## Known Limitations

- **Scanned PDFs & OCR:** This phase extracts digital text streams from PDFs. Scanned PDFs (pure images without embedded text) are detected and rejected with an informative error: `This document contains no extractable text. Scanned PDFs require OCR, which is not supported in this phase.`
