# EHR RAG POC

A minimal Retrieval-Augmented Generation system built on synthetic EHR data.
Built to understand what RAG actually does, not just talk about it.

## What this does
Query 8 synthetic patient records in plain English.
The system retrieves relevant records from ChromaDB, feeds them to Llama 3 via Groq,
and returns a grounded answer — no hallucination, no guessing.

## Stack
- **ChromaDB** — local vector store (no cloud, runs on your machine)
- **Groq** — free LLM API (Llama 3)
- **FastAPI** — lightweight Python API server
- **Vanilla HTML/JS** — frontend with live RAG pipeline diagram

## Setup

### 1. Install dependencies
```bash
pip install fastapi uvicorn chromadb openai groq langchain langchain-openai python-dotenv
```

### 2. Add your Groq API key
```bash
cp .env.example .env
# Open .env and replace 'your_groq_api_key_here' with your actual key
# Get a free key at: https://console.groq.com
```

### 3. Ingest the data (run once)
```bash
python ingest.py
```
This reads synthetic_ehr.json, converts each patient to a text chunk,
embeds it into vectors, and saves to ./chroma_store

### 4. Start the server
```bash
uvicorn app:app --reload
```

### 5. Open the app
Go to: http://localhost:8000

## Try asking
- "Which patients are on Metformin?"
- "Who has both diabetes and another chronic condition?"
- "Summarize patients with respiratory conditions"
- "Which patients had improving lab results?"

## File structure
```
ehr-rag-poc/
├── synthetic_ehr.json   # 8 fake patient records
├── ingest.py            # Chunks + embeds data into ChromaDB
├── rag.py               # Core RAG pipeline (retrieve → augment → generate)
├── app.py               # FastAPI server
├── index.html           # Frontend UI
├── .env.example         # Copy to .env and add your key
└── README.md
```

## The RAG pipeline (plain English)
1. Your question gets embedded into a vector
2. ChromaDB finds the most semantically similar patient records
3. Those records get inserted into a prompt as context
4. Groq's LLM reads question + context and answers
5. You get a grounded answer tied to real (synthetic) records

## What to notice when it breaks
- Ask something not in the data — watch it say "I don't know"
- Ask an ambiguous question — see which records it retrieves (may surprise you)
- This is where clinical governance thinking starts
