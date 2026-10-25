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
- [x] **Phase 4: Modern React + TypeScript Frontend**
  - Dark-navy and electric-blue AI SaaS user interface.
  - Centralized typed API client (`frontend/src/services/api.ts`).
  - Drag-and-drop PDF upload component with client-side file validation and progress feedback.
  - Document Library with card inspection, real metadata (pages, chunks, size, date), and deletion confirmation modal.
  - Conversational chat interface with auto-scroll, starter queries, and document filtering scope indicator.
  - Slide-in Citation Drawer displaying source document, 1-indexed page numbers, similarity match score, and verbatim excerpt.
  - Full automated frontend test suite with Vitest and React Testing Library (17 tests).
- [ ] **Phase 5: Verification, Edge Cases & Evaluation** (Upcoming)
- [ ] **Phase 6: Production Readiness & Portfolio Showcase** (Upcoming)

---

## System Architecture

```
                       +---------------------------------------------+
                       |          DocuMind AI Frontend               |
                       |       (React 18 + TypeScript + Vite)        |
                       +---------------------------------------------+
                                      |              ^
                 POST /documents/upload              |
                 POST /chat/query                    | JSON Response &
                 GET  /documents                     | Structured Citations
                                      v              |
                       +---------------------------------------------+
                       |           FastAPI ASGI Backend              |
                       +---------------------------------------------+
                               |                             |
                               v                             v
                +----------------------------+  +----------------------------+
                |    Ingestion Pipeline      |  |    RAG Retrieval & LLM     |
                |  - PyMuPDF Page Extractor  |  |  - Query Embeddings        |
                |  - Recursive Chunker       |  |  - Cosine Distance Filter  |
                |  - all-MiniLM-L6-v2 Model  |  |  - Prompt Guardrails       |
                |  - Persistent ChromaDB     |  |  - Google Gemini API       |
                +----------------------------+  +----------------------------+
```

---

## Technology Stack

- **Frontend:** React 18, TypeScript, Vite, Lucide Icons, Vanilla CSS Design System (Dark Navy / Electric Blue)
- **Frontend Testing:** Vitest, React Testing Library, jsdom
- **Backend Framework:** FastAPI, Uvicorn, Pydantic v2, Pydantic Settings
- **PDF Extraction:** PyMuPDF (`pymupdf 2.0`)
- **Embedding Generation:** Sentence Transformers (`all-MiniLM-L6-v2`, `torch 2.14.1`)
- **Vector Database:** ChromaDB (`chromadb 1.5.9`)
- **Generative AI / LLM:** Google Gemini API (`google-genai 2.33.0`)
- **Backend Testing:** Pytest, HTTPX

---

## Getting Started

### 1. Prerequisites
* **Python:** 3.12+
* **Node.js:** v20+ (Verified on v24.19.0)
* **npm:** 10+ (Verified on 11.17.0)

---

### 2. Backend Setup & Startup

From the project root:

```powershell
cd backend

# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
# source .venv/bin/activate    # Linux / macOS

# 2. Configure environment variables
# Copy .env.example to .env and optionally add your Gemini API key:
cp .env.example .env

# 3. Run FastAPI backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

* **Swagger UI:** [http://127.0.0.1:8000/api/v1/docs](http://127.0.0.1:8000/api/v1/docs)
* **Health Check:** [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)

To run the backend test suite (51 tests):
```powershell
pytest -v
```

---

### 3. Frontend Setup & Startup

In a separate terminal, navigate to the `frontend/` directory:

```powershell
cd frontend

# 1. Install dependencies
npm install

# 2. Configure environment (optional, defaults to http://127.0.0.1:8000)
cp .env.example .env

# 3. Start Vite development server
npm run dev
```

* **Frontend UI:** [http://127.0.0.1:5173](http://127.0.0.1:5173)

To run the frontend test suite (17 tests):
```powershell
npm test
```

To run TypeScript verification and production build:
```powershell
npm run build
```

---

## Features & UI Walkthrough

### 1. Document Upload
* Navigate to **Document Library** or click **Upload PDF** in the sidebar.
* Drag and drop any `.pdf` document (up to 10 MB).
* PyMuPDF validates the file header (`%PDF-`), extracts text page-by-page, generates 384-dimensional dense vectors with Sentence Transformers, and indexes them in ChromaDB.
* The newly uploaded document is automatically selected for focused chat Q&A.

### 2. Document Library & Filtering
* View all indexed documents showing real backend metadata: filename, page count, chunk count, file size, and upload date.
* Click **Filter Chat** to restrict semantic retrieval exclusively to a single document.
* Click **Trash** icon to trigger an accessible confirmation modal and permanently purge the document and all its vector embeddings from ChromaDB.

### 3. Grounded Chat Interface
* In the **Chat Assistant** view, ask questions about your documents in natural language.
* Click the suggested prompt chips for instant starter queries.
* Press `Enter` to submit, or `Shift+Enter` for a new line.
* If a document filter is active, an indicator banner displays the targeted scope.

### 4. Source Citation Inspector
* Assistant answers cite source claims using inline markers `[1]`, `[2]`.
* Beneath each answer, interactive source pills indicate the filename and exact 1-indexed page number (e.g. `[1] spec.pdf (p. 2)`).
* Clicking any pill opens the **Citation Drawer**, which presents:
  * Document name and ID
  * Source page numbers
  * Semantic similarity match percentage
  * Verbatim text passage excerpt with a one-click copy button.

### 5. Insufficient Evidence Safeguard
* When a question cannot be answered from the uploaded documents, DocuMind AI displays a distinct warning card indicating that the documents do not contain sufficient evidence, avoiding hallucinations and conserving LLM tokens.

---

## Security & Privacy Highlights

1. **Zero Exposure of Secrets:** The Google Gemini API key is stored strictly in `backend/.env`. The frontend client never handles, exposes, or transmits secrets.
2. **Local Vector Processing:** Embeddings and nearest-neighbor vector searches run entirely on your local machine using PyTorch and ChromaDB. Only relevant excerpts are sent to the LLM during answer synthesis.
3. **Prompt Injection Defense:** Document passages are wrapped inside `<document_context>` XML tags with explicit system instructions to ignore instructions found inside uploaded text.

---

## Known Limitations

* **Scanned PDFs & OCR:** Only documents with digital text streams are supported. Scanned PDFs (pure images) are detected and rejected with an informative error.
* **Single-turn Q&A:** Each query is stateless and evaluated against the indexed document vectors without multi-turn chat memory.
