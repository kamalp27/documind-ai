# DocuMind AI — AI-Powered Document Question Answering Chatbot

DocuMind AI is a production-grade Retrieval-Augmented Generation (RAG) system engineered to allow users to upload PDF documents, extract and chunk text with page-level metadata, index vectors in a local vector database, and perform grounded question-answering with exact source citations backed by Google Gemini.

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
- [x] **Phase 3: RAG Retrieval Engine & Gemini Generation**
  - Semantic similarity search using ChromaDB cosine distance metric with distance threshold filtering.
  - Grounded prompt construction enforcing strict anti-hallucination guardrails and prompt injection defenses.
  - Structured page-level citations mapped directly from retrieved chunk metadata.
  - Lazy-loading Google Gemini API integration using the modern `google-genai` SDK.
  - Early-exit refusal when documents lack relevant information (conserves tokens and eliminates hallucinations).
  - Multi-document filtering via optional `document_ids` request parameter.
- [ ] **Phase 4: Modern React + TypeScript Frontend** (Upcoming)
- [ ] **Phase 5: Verification, Edge Cases & Evaluation** (Upcoming)
- [ ] **Phase 6: Production Readiness & Portfolio Showcase** (Upcoming)

---

## RAG Architecture & Data Flow

```
[ User Question ]
       |
       v
[ Embedding Engine ] -> Generates query vector (384-dim)
       |
       v
[ ChromaDB Semantic Search ] -> Filters chunks by Cosine Distance <= 0.75
       |
       +---> [ No Passages Meet Cutoff? ]
       |                 |
       |                YES -> Return "Insufficient Evidence" directly (0 token cost)
       |                 |
       v                NO
[ Grounded Prompt Builder ] -> Wraps passages in <document_context> XML boundaries
       |
       v
[ Google Gemini API ] (gemini-2.5-flash / T=0.1)
       |
       v
[ Synthesized Response with Verified Page Citations ]
```

---

## Technology Stack

- **Backend Framework:** FastAPI, Uvicorn, Pydantic v2, Pydantic Settings
- **PDF Extraction:** PyMuPDF (`pymupdf 2.0`)
- **Embedding Generation:** Sentence Transformers (`all-MiniLM-L6-v2`, `torch 2.14.1`)
- **Vector Database:** ChromaDB (`chromadb 1.5.9`)
- **Generative AI / LLM:** Google Gemini API (`google-genai 2.33.0`)
- **Testing:** Pytest, HTTPX

---

## Getting Started

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

#### How to Obtain and Configure your Google Gemini API Key:
1. Visit [Google AI Studio](https://aistudio.google.com/).
2. Sign in with your Google account and click **"Get API key"**.
3. Create a free API key and copy it.
4. Open your local `backend/.env` file and set:
   ```env
   GEMINI_API_KEY="your-gemini-api-key-here"
   ```
*(Note: Never commit your `.env` file to Git. It is already included in `.gitignore`)*.

Key configuration options in `.env`:
* `GEMINI_MODEL`: Model name (default: `gemini-2.5-flash`).
* `RAG_TOP_K`: Maximum number of passages retrieved (default: `4`).
* `RAG_RELEVANCE_THRESHOLD`: Maximum cosine distance for relevance (default: `0.75`, where smaller distance = higher similarity).
* `MAX_QUERY_LENGTH`: Maximum question character length (default: `1000`).
* `MAX_CONTEXT_CHARS`: Maximum context characters injected into prompt (default: `12000`).

### 5. Run Backend Server
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
* **Swagger UI:** [http://127.0.0.1:8000/api/v1/docs](http://127.0.0.1:8000/api/v1/docs)
* **ReDoc:** [http://127.0.0.1:8000/api/v1/redoc](http://127.0.0.1:8000/api/v1/redoc)
* **Health Check:** [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)

### 6. Run Automated Tests
```bash
pytest -v
```
*(Tests run completely offline using mock embeddings and mock Gemini clients without consuming API quota).*

---

## API Usage Guide

### 1. Upload a PDF Document
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/documents/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_report.pdf;type=application/pdf"
```

**Response (201 Created):**
```json
{
  "document_id": "7f8b9c2a-1234-4567-89ab-cdef01234567",
  "filename": "sample_report.pdf",
  "status": "processed",
  "total_pages": 3,
  "total_chunks": 8,
  "created_at": "2026-10-25T18:00:00.000000+00:00"
}
```

### 2. Query Documents with RAG
Ask a question across all uploaded documents:
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/chat/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What were the total operating expenses reported in Q3?"
  }'
```

Or restrict your search to a specific document:
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/chat/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the warranty policy?",
    "document_ids": ["7f8b9c2a-1234-4567-89ab-cdef01234567"]
  }'
```

**Response (200 OK):**
```json
{
  "answer": "Operating expenses for Q3 were $4.2 million, representing a 12% decrease compared to the prior quarter [1]. Capital expenditures were primarily allocated to cloud infrastructure [2].",
  "sources": [
    {
      "source_id": 1,
      "document_id": "7f8b9c2a-1234-4567-89ab-cdef01234567",
      "filename": "sample_report.pdf",
      "page_numbers": [2],
      "excerpt": "Total operating expenses in Q3 stood at $4.2M, driven down by 12% via operational efficiencies...",
      "similarity_score": 0.8842
    },
    {
      "source_id": 2,
      "document_id": "7f8b9c2a-1234-4567-89ab-cdef01234567",
      "filename": "sample_report.pdf",
      "page_numbers": [3],
      "excerpt": "Capital expenditures were directed towards cloud data center modernization...",
      "similarity_score": 0.8125
    }
  ],
  "insufficient_evidence": false,
  "model_used": "gemini-2.5-flash",
  "request_id": "3b1e32d6-11f8-4089-982d-bc4892c2db8f"
}
```

### 3. Out-of-Domain / Insufficient Evidence Query
When a question cannot be answered from the uploaded documents:
```json
{
  "answer": "The uploaded documents do not contain sufficient information to answer this question.",
  "sources": [],
  "insufficient_evidence": true,
  "model_used": null,
  "request_id": "c1f71490-3cb8-45a1-9457-320dae01d1b9"
}
```

---

## Source Citations & Security Guardrails

### How Citations Work
1. **Metadata Preservation:** During Phase 2 ingestion, every text chunk is stored in ChromaDB tagged with its 1-indexed source PDF page number (`page_number`) and sanitized filename.
2. **Context Numbering:** Chunks are formatted as `[Source 1]`, `[Source 2]` within the grounded prompt.
3. **Structured Mapping:** The API response returns structured `sources` objects linking each bracketed tag in the answer to the exact document ID, original filename, page number, and text excerpt.

### Security & Privacy Protections
1. **Prompt Injection Boundary:** All retrieved passages are isolated inside `<document_context>` XML tags. System instructions explicitly command the model to treat document content as inert, untrusted data and ignore override commands (e.g., *"Ignore previous instructions"*).
2. **Data Privacy:** Vector embeddings and semantic searches execute 100% locally on your machine. Only the top relevant text passages are dispatched to the Gemini API during generation.
3. **API Key Safety:** Credentials are read exclusively from environment variables via Pydantic Settings. Responses and error handlers never leak secrets or system paths.
4. **Early Rejection:** Queries without matching document passages bypass LLM inference entirely, preventing token waste and hallucinations.

---

## Known Limitations

- **Scanned PDFs & OCR:** Digitize PDFs containing text streams. Scanned PDFs (pure images) are detected and rejected with an explicit OCR limitation message.
- **Prompt Injection:** While structural XML framing and system instructions strongly mitigate prompt injection, application-level RAG should never process untrusted documents in environments requiring high-privilege tool execution.
