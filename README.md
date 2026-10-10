# Industrial Incident Investigator (III)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.143-teal.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65-red.svg)](https://streamlit.io/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Local_Mode-orange.svg)](https://qdrant.tech/)
[![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-MiniLM-purple.svg)](https://sbert.net/)

An enterprise-grade, production-style full-stack Retrieval-Augmented Generation (RAG) platform purpose-built for reliability engineers, safety investigators, and plant managers to analyze industrial equipment failures, root causes, maintenance logs, and safety procedures.

---

## 🏗️ Architectural Overview

```
                      ┌──────────────────────────────────────────────┐
                      │       Streamlit Enterprise Frontend          │
                      │  (Dark Navy Theme, Interactive Plotly,       │
                      │   Investigation Workspace, Explorer, Eval)   │
                      └──────────────────────┬───────────────────────┘
                                             │ HTTP REST (JSON)
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │           FastAPI Backend Layer              │
                      │  (Async Uvicorn, Request ID, Pydantic DTOs,  │
                      │   CORS, Lifecycle Knowledge Base Seeding)    │
                      └──────────────────────┬───────────────────────┘
                                             │ Orchestrates
                                             ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   RAG Engine Core                                      │
├─────────────────────────┬───────────────────────────┬──────────────────────────────────┤
│    1. Ingestion         │    2. Multi-Channel       │    3. Ranking & Synthesis        │
│                         │       Retrieval           │                                  │
│ • PyMuPDF (PDF)         │ • Dense: MiniLM-L6-v2     │ • Reciprocal Rank Fusion (RRF)   │
│ • TXT / Markdown        │   in Qdrant Local Mode    │ • Cross-Encoder Reranking        │
│ • SHA-256 Deduplication │ • BM25 Keyword Search     │   (ms-marco-MiniLM-L-6-v2)       │
│ • Structure & Metadata  │   (Preserves HP-17, codes)│ • Groq LLM Grounded Generation   │
│   Aware Chunking        │ • Query Rewriting (Groq)  │ • Strict Citation Tracking       │
└─────────────────────────┴───────────────────────────┴──────────────────────────────────┘
```

The system strictly decouples the interface and retrieval logic: **the Streamlit frontend never executes search or LLM logic locally—it communicates exclusively via FastAPI HTTP endpoints.**

---

## 🚀 Key Features

1. **Document Ingestion & Intelligent Chunking**:
   - Native support for **PDF** (via PyMuPDF/fitz with page number preservation), **Markdown**, and **Plain Text**.
   - Structure-aware chunking preserving paragraph boundaries and technical identifiers (e.g. `HP-17`, `RX-404`, `V-204`, `E-301`).
   - SHA-256 checksum deduplication preventing duplicate indexing of identical document versions.
   - Automatic metadata extraction (Machine ID, Incident Type, Incident Date, Severity).

2. **Dense Vector Retrieval**:
   - Sentence Transformers (`all-MiniLM-L6-v2`) generating 384-dimensional normalized embeddings.
   - Qdrant local file-backed vector database (`data/processed/qdrant_db/`) supporting metadata filtering (`machine_id`, `incident_type`).

3. **BM25 Technical Keyword Search**:
   - `rank-bm25` enhanced with Lucene non-negative IDF formula ($\ln(1 + \frac{N - n + 0.5}{n + 0.5})$).
   - Preserves alphanumeric equipment codes, decimal telemetry (`14.2 mm/s`), and error codes (`E-301`).

4. **Hybrid Search with Reciprocal Rank Fusion (RRF)**:
   - Rank-based fusion: $RRF(d) = \sum \frac{1}{k + rank_i(d)}$ with configurable constant $k=60$.
   - Eliminates score calibration mismatch between cosine similarity and BM25 scores.

5. **Query Rewriting**:
   - Groq Cloud LLM generating targeted engineering query variations while preserving technical entity tags.
   - Resilient rule-based offline fallback when running without external API connectivity.

6. **Cross-Encoder Reranking**:
   - Sentence Transformers CrossEncoder (`ms-marco-MiniLM-L-6-v2`) re-scoring `(query, passage)` pairs.
   - Separate latency telemetry and graceful fallback if reranking fails.

7. **Evidence-Grounded Generation & Auditable Citations**:
   - Strict hallucination defense: every finding references traceable citations (`[CIT-1]`, `[CIT-2]`).
   - Distinguishes observed physical facts from investigative hypotheses.
   - Flags insufficient evidence when retrieved documentation does not answer the inquiry.
   - Defends against prompt injection in untrusted technical documents.

8. **RAG Evaluation Framework**:
   - Deterministic ranking metrics: **Recall@5**, **Recall@10**, **Precision@K**, **MRR**, and **nDCG@10**.
   - Optional Ragas generation metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall).
   - Reproducible benchmark export to structured JSON and CSV in `reports/`.

---

## 📁 Repository Layout

```
industrial-incident-rag/
├── app.py                         # Unified Application Runner CLI
├── requirements.txt               # Pinned core dependencies
├── requirements-lock.txt          # Fully frozen dependency lockfile
├── pytest.ini                     # Pytest configuration (pythonpath=src)
├── .env.example                   # Environment configuration template
├── .env                           # Local environment configuration
├── .gitignore                     # Git ignore rules
├── README.md                      # Comprehensive project documentation
│
├── frontend/                      # Streamlit Enterprise UI Layer
│   ├── app.py                     # Main dashboard overview & navigation
│   ├── pages/
│   │   ├── 1_Investigation.py     # Investigation Workspace & Root Cause Cards
│   │   ├── 2_Retrieval_Explorer.py# Stage-by-Stage Retrieval Inspection Tool
│   │   ├── 3_RAG_Evaluation.py    # Benchmark Reports & Visualizations
│   │   ├── 4_Knowledge_Base.py    # Document Upload & Repository Management
│   │   └── 5_Settings.py          # System Hyperparameters & Architecture
│   ├── components/
│   │   ├── api_client.py          # Backend HTTP client
│   │   ├── sidebar.py             # Persistent status & equipment sidebar
│   │   ├── evidence_cards.py      # Citation accordions & hypothesis cards
│   │   └── charts.py              # Plotly interactive benchmark charts
│   └── styles/
│       └── theme.css              # Dark navy & enterprise CSS design system
│
├── src/
│   └── incident_rag/              # Core Application & RAG Engine Package
│       ├── __init__.py
│       ├── config.py              # Pydantic Settings & environment loaders
│       ├── schemas.py             # Pydantic request, response, and data models
│       ├── logging_config.py      # Structured correlation logging
│       ├── api.py                 # FastAPI REST API endpoints
│       ├── dependencies.py        # Dependency injection providers & singletons
│       ├── query_rewriting.py     # Groq query rewriter with offline fallback
│       ├── reranking.py           # Cross-Encoder reranking module
│       ├── context_builder.py     # Context builder & prompt injection sanitizer
│       ├── generation.py          # Evidence-grounded response generator
│       ├── pipeline.py            # Unified RAG Pipeline orchestrator
│       ├── ingestion/
│       │   ├── loader.py          # PDF, MD, TXT loaders & text cleaners
│       │   ├── metadata.py        # Equipment tag & date extraction heuristics
│       │   ├── chunker.py         # Structure-aware chunking with overlap
│       │   └── service.py         # Ingestion service & deduplication
│       ├── embeddings/
│       │   └── encoder.py         # Sentence Transformers embedding encoder
│       ├── retrieval/
│       │   ├── dense_search.py    # Qdrant dense vector search engine
│       │   ├── bm25_search.py     # Technical BM25 search engine
│       │   ├── fusion.py          # Reciprocal Rank Fusion algorithms
│       │   └── hybrid_search.py   # Multi-channel hybrid retriever
│       ├── evaluation/
│       │   ├── dataset.py         # Benchmark dataset loader & SHA-256 verifier
│       │   ├── retrieval_metrics.py# Recall@K, Precision@K, MRR, nDCG@10
│       │   ├── ragas_evaluator.py # Ragas integration with safe fallback
│       │   ├── benchmark_runner.py# Multi-strategy benchmark executor
│       │   └── reports.py         # Report exporter (JSON & CSV)
│       └── storage/
│           ├── document_store.py  # Document metadata store
│           ├── vector_store.py    # Qdrant vector store interface
│           └── experiment_store.py# Benchmark history manager
│
├── data/
│   ├── raw/                       # Seed incident reports (MD, TXT, PDF)
│   ├── processed/                 # Local Qdrant DB & BM25 indices
│   └── benchmark/
│       └── retrieval_benchmark.json # Labeled ground-truth benchmark suite
│
├── reports/                       # Persisted benchmark evaluation reports
└── tests/                         # Pytest test suite (45 automated tests)
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.11 installed (`python --version`)
- Git

### 1. Clone & Set Up Virtual Environment

**Windows PowerShell:**
```powershell
# Navigate into the project directory
cd "C:\Users\ANUSHREE\OneDrive\Desktop\rag project"

# Create Python 3.11 virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔑 Environment Configuration

Copy `.env.example` to `.env`:

```powershell
cp .env.example .env
```

### Groq API Key Setup
To enable live LLM query rewriting and generative answer synthesis:
1. Create a free account at [Groq Console](https://console.groq.com/keys).
2. Generate an API Key.
3. Open `.env` and set:
   ```env
   GROQ_API_KEY=gsk_your_groq_api_key_here
   GROQ_MODEL=llama-3.3-70b-versatile
   ```

> **Note on Offline / Keyless Mode:** If `GROQ_API_KEY` is not provided, the application automatically engages its deterministic extractive synthesis engine and rule-based query expansion. The entire pipeline, API, retrieval explorer, and benchmark framework remain 100% operational offline without external API keys.

---

## 🚀 Running the Application

### Option 1: One-Click / Single Command Launch (Recommended)
Starts both the FastAPI backend and Streamlit frontend concurrently with automatic backend health polling:

**Command Prompt / Windows Explorer:**
Double-click `run.bat` or run:
```cmd
run.bat
```

**PowerShell:**
```powershell
.\run.ps1
# or
.\.venv\Scripts\python.exe app.py all
```

Once started:
- **Frontend Dashboard:** [http://localhost:8501](http://localhost:8501)
- **Backend API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

Press `Ctrl+C` in the terminal to stop both servers.

---

### Option 2: Dual-Terminal Launch

If you prefer viewing server logs in separate terminal windows:

**Terminal 1 — Start FastAPI Backend:**
```powershell
.\.venv\Scripts\python.exe app.py backend
```

**Terminal 2 — Start Streamlit Frontend:**
```powershell
.\.venv\Scripts\python.exe app.py frontend
```

---

### Option 3: Direct Module Commands

**Backend:**
```powershell
.\.venv\Scripts\uvicorn.exe incident_rag.api:app --host 127.0.0.1 --port 8000 --reload --app-dir src
```

**Frontend:**
```powershell
.\.venv\Scripts\streamlit.exe run frontend/app.py
```

---

## 🧪 Automated Testing

The project includes 45 comprehensive unit and integration tests covering loaders, chunking, embeddings, dense search, BM25 tokenization, RRF mathematical fusion, cross-encoder reranking, context building, prompt injection defense, FastAPI REST endpoints, and frontend client calls.

Run all tests:
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

Run specific test modules:
```powershell
# Test retrieval metrics
.\.venv\Scripts\python.exe -m pytest tests/test_retrieval_metrics.py -v

# Test FastAPI endpoints
.\.venv\Scripts\python.exe -m pytest tests/test_api.py -v

# Test pipeline & generation
.\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -v
```

---

## 📊 Reproducible Benchmarking

Execute the full deterministic benchmark suite across all 5 strategies on the real knowledge base:

```powershell
.\.venv\Scripts\python.exe app.py benchmark
```

Results are automatically saved to `reports/` in both JSON and CSV formats (e.g. `reports/benchmark_run_d9274ae23a.csv`).

### Real Execution Results

| Strategy | Recall@5 | Recall@10 | Precision@K | MRR | nDCG@10 | Avg Latency | Passed Queries |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 Keyword** | 1.0000 | 1.0000 | 0.4000 | 1.0000 | 0.9877 | 0.5 ms | 10 / 10 |
| **Dense Search** | 1.0000 | 1.0000 | 0.4000 | 1.0000 | 0.9920 | 26.0 ms | 10 / 10 |
| **Hybrid (RRF)** | 1.0000 | 1.0000 | 0.4000 | 1.0000 | 0.9920 | 19.5 ms | 10 / 10 |
| **Hybrid + Rerank** | 1.0000 | 1.0000 | 0.4000 | 1.0000 | 0.9920 | 767.7 ms | 10 / 10 |
| **Full Advanced RAG**| 1.0000 | 1.0000 | 0.4000 | 1.0000 | 0.9920 | 974.7 ms | 10 / 10 |

---

## 🔒 Security & Reliability

- **Secret Isolation**: API keys reside solely on the backend via environment variables. The Streamlit client never handles credentials.
- **Prompt Injection Neutralization**: The context builder scrubs pattern overrides (e.g., `ignore previous instructions`) and treats documents strictly as untrusted data.
- **Upload Hardening**: File type whitelisting (`.pdf`, `.md`, `.txt`) and 25MB maximum size restrictions.
- **Resilient Fallback**: Graceful degradation to extractive synthesis and rule-based query expansion if network or external model providers become unavailable.
