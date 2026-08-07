# EHR RAG POC — Open Weight Version

A minimal Retrieval-Augmented Generation system built on synthetic EHR data.
Built to understand what RAG actually does, not just talk about it.

## What changed (open weight update)

The original POC used **Groq's hosted API** to run Llama 3, and chromadb's
**default general-purpose embedding model**. Both worked, but neither held
up under scrutiny for a clinical-adjacent system:

| | Old | New |
|---|---|---|
| LLM | Llama 3.3 70B via Groq (hosted API, data leaves machine, needs API key) | Llama 3.1 8B via **Ollama** (runs locally, no key, nothing leaves machine) |
| Embeddings | `all-MiniLM-L6-v2` (general internet text, doesn't know "hypertension" = "high blood pressure") | `FremyCompany/BioLORD-2023` (biomedical, understands clinical synonyms) |

Both models are still **open weight** — the actual change is *where they run*.
Old code is left commented in `rag.py`, `ingest.py`, `.env.example`, and
`requirements.txt` so the diff is visible.

## What this does

Query 8 synthetic patient records in plain English.
The system retrieves relevant records from ChromaDB, feeds them to a locally
running Llama 3.1 model via Ollama, and returns a grounded answer — no
hallucination, no guessing, no external API call.

## Stack

- **ChromaDB** — local vector store (no cloud, runs on your machine)
- **Ollama** — runs Llama 3.1 8B locally (open weight, no API key, no data leaves machine)
- **sentence-transformers / BioLORD-2023** — biomedical embedding model (open weight)
- **FastAPI** — lightweight Python API server
- **Vanilla HTML/JS** — frontend with live RAG pipeline diagram

## Setup

### 1. Install and start Ollama

```
# macOS
brew install ollama
ollama serve &
ollama pull llama3.1:8b
```

Confirm it's running:
```
curl http://localhost:11434/api/tags
```

### 2. Install Python dependencies

```
pip install -r requirements.txt
```

### 3. (Optional) Configure environment

Only needed if your Ollama server isn't on localhost, or you want a
different model.

```
cp .env.example .env
```

### 4. Ingest the data (run once, or whenever the embedding model changes)

```
python ingest.py
```

This reads `synthetic_ehr.json`, converts each patient to a text chunk,
embeds it with BioLORD-2023, and saves to `./chroma_store`.

### 5. Start the server

```
uvicorn app:app --reload
```

### 6. Open the app

Go to: http://localhost:8000

## Try asking

- "Which patients are on Metformin?"
- "Who has both diabetes and another chronic condition?"
- "Summarize patients with respiratory conditions"
- "Who has high bp?" — this is the query that originally exposed the
  MiniLM embedding gap; worth re-testing post-swap to confirm BioLORD
  correctly retrieves patients diagnosed with hypertension.

## File structure

```
ehr-rag-poc/
├── synthetic_ehr.json   # 8 fake patient records
├── ingest.py            # Chunks + embeds data into ChromaDB (BioLORD)
├── rag.py               # Core RAG pipeline (retrieve → augment → generate, via Ollama)
├── app.py               # FastAPI server (unchanged)
├── index.html           # Frontend UI (unchanged)
├── requirements.txt     # Python dependencies (old list commented at top)
├── .env.example         # Ollama config (old Groq config commented at top)
└── README.md
```

## The RAG pipeline (plain English)

1. Your question gets embedded into a vector (locally, via BioLORD)
2. ChromaDB finds the most semantically similar patient records
3. Those records get inserted into a prompt as context
4. A locally running Llama 3.1 model (via Ollama) reads question + context and answers
5. You get a grounded answer tied to real (synthetic) records — with nothing sent off your machine

## What to notice when it breaks

- Ask something not in the data — watch it say "I don't know"
- Ask an ambiguous question — see which records it retrieves (may surprise you)
- Compare retrieval quality before/after the embedding swap on clinical-synonym
  queries ("high bp" vs "hypertension") — this is where the embedding model
  choice becomes visible, not theoretical
- This is where clinical governance thinking starts
