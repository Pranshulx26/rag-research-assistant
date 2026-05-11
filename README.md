# RAG Research Assistant

An AI-powered document Q&A system that lets you upload any PDF and ask questions about it in natural language. Built on Retrieval Augmented Generation (RAG) — answers are grounded in your document, not hallucinated.

Upload a research paper, contract, or report and get precise, cited answers in seconds.

---

## What it does

- Upload any PDF document via REST API
- Ask questions in natural language
- Receive answers grounded in the document with page citations
- Supports multiple documents simultaneously
- Skips re-indexing if the same document is uploaded again (idempotent)

---

## Tech stack

| Layer | Technology | Purpose |
|---|---|---|
| API | FastAPI (async) | REST endpoint serving |
| Validation | Pydantic v2 | Request/response schemas |
| PDF parsing | PyMuPDF | Extract text from PDF pages |
| Orchestration | LangChain | RAG pipeline management |
| Embeddings | Google Generative AI | Convert text to vectors |
| Vector store | ChromaDB | Store and search embeddings |
| LLM | Google Gemini 1.5 Flash | Generate grounded answers |
| Logging | loguru | Structured logs with rotation |
| Containers | Docker + Docker Compose | Reproducible deployment |
| Testing | pytest + httpx | Async test suite (8 tests) |
| CI/CD | GitHub Actions | Automated tests on every push |

---

## Project structure

```
rag-research-assistant/
├── app/
│   ├── api/routes/
│   │   ├── documents.py     # POST /documents/upload, GET /documents/
│   │   └── chat.py          # POST /chat/
│   ├── core/
│   │   └── config.py        # Pydantic Settings, env-based config
│   ├── services/
│   │   ├── pdf_processor.py # Extract text + split into chunks
│   │   ├── vectorstore.py   # ChromaDB operations (add, search)
│   │   └── rag_chain.py     # LangChain + Gemini pipeline
│   ├── models/
│   │   └── schemas.py       # Request/response Pydantic models
│   └── main.py              # App factory, middleware, lifespan
├── data/
│   ├── uploads/             # Uploaded PDFs stored here
│   └── vectorstore/         # ChromaDB persists here
├── tests/
│   ├── test_documents.py    # 4 tests for upload endpoints
│   └── test_chat.py         # 4 tests for chat endpoint
├── .github/workflows/
│   └── ci.yml               # GitHub Actions pipeline
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## API endpoints

### `POST /documents/upload`
Upload a PDF and index it into ChromaDB.

**Request:** multipart/form-data with a PDF file

**Response:**
```json
{
  "doc_id": "a3f8c9d2e1b4...",
  "filename": "research_paper.pdf",
  "chunks_created": 52,
  "message": "Successfully indexed 52 chunks."
}
```

Uploading the same file twice returns `chunks_created: 0` and skips re-indexing.

---

### `POST /chat/`
Ask a question about your documents.

**Request:**
```json
{
  "question": "What is the main contribution of this paper?",
  "doc_id": "a3f8c9d2e1b4..."
}
```

`doc_id` is optional. Omitting it searches across all uploaded documents.

**Response:**
```json
{
  "question": "What is the main contribution of this paper?",
  "answer": "The main contribution is the Transformer architecture, a model based solely on attention mechanisms that dispenses with recurrence and convolutions entirely [Page 1].",
  "sources": ["research_paper.pdf"],
  "chunks_used": 4
}
```

---

### `GET /documents/`
List all uploaded PDF filenames.

### `GET /health`
Returns API status, version, and active model name.

---

## How RAG works

RAG (Retrieval Augmented Generation) runs in two phases.

**Phase 1 — Indexing (runs once per document):**
The PDF is parsed page by page, split into overlapping chunks of ~1000 characters, and each chunk is converted into a vector embedding using Google's embedding model. These vectors are stored in ChromaDB on disk.

**Phase 2 — Querying (runs on every question):**
The user's question is converted into a vector using the same embedding model. ChromaDB finds the 4 most similar chunks via cosine similarity. Those chunks plus the original question are sent to Gemini, which generates an answer grounded only in the retrieved content — not outside knowledge.

This means answers are always traceable to a specific part of your document.

---

## Running locally

### Option 1 — Docker (recommended)

Requires: [Docker Desktop](https://www.docker.com/products/docker-desktop/)

```bash
git clone https://github.com/Pranshulx26/rag-research-assistant.git
cd rag-research-assistant

cp .env.example .env
# Add your GEMINI_API_KEY to .env

docker compose up --build
```

API available at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`

Uploaded PDFs and the vector store persist across restarts via Docker volume mounts.

---

### Option 2 — Manual

Requires: Python 3.11+

```bash
git clone https://github.com/Pranshulx26/rag-research-assistant.git
cd rag-research-assistant

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt

cp .env.example .env
# Add your GEMINI_API_KEY to .env

uvicorn app.main:app --reload
```

---

## Environment variables

```env
GEMINI_API_KEY=your_key_here        # Required — get from aistudio.google.com
GEMINI_MODEL=gemini-1.5-flash       # Model to use for generation
CHUNK_SIZE=1000                      # Characters per chunk
CHUNK_OVERLAP=200                    # Overlap between chunks
TOP_K_RESULTS=4                      # Chunks retrieved per query
```

---

## Running tests

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_chat.py::test_chat_valid_question        PASSED
tests/test_chat.py::test_chat_question_too_short    PASSED
tests/test_chat.py::test_chat_with_doc_id           PASSED
tests/test_chat.py::test_health_check               PASSED
tests/test_documents.py::test_upload_valid_pdf      PASSED
tests/test_documents.py::test_upload_non_pdf_rejected PASSED
tests/test_documents.py::test_upload_already_indexed  PASSED
tests/test_documents.py::test_list_documents        PASSED

8 passed in 2.47s
```

Tests use mocked dependencies — no Gemini API key or ChromaDB required.

---

## Author

**Pranshul** — AI/ML Engineer  
GitHub: [@Pranshulx26](https://github.com/Pranshulx26)