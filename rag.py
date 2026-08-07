"""
STEP 2: RAG CORE
This is the heart of the system.

ask_question() does exactly 3 things:
    1. RETRIEVE  — search ChromaDB for the most relevant patient chunks
    2. AUGMENT   — build a prompt that includes those chunks as context
    3. GENERATE  — send that prompt to a local LLM and return the answer

This is what RAG means: Retrieval-Augmented Generation.

── OPEN WEIGHT UPDATE ─────────────────────────────────────────────────────
Original version called Groq's hosted API. Llama 3's weights are open,
but routing through Groq still sends every question + patient record
off your machine to a third-party server.

Swapped to Ollama, which runs the same open-weight Llama model locally.
Nothing leaves your machine. Old code kept below (commented) so you can
see exactly what changed and revert if needed.
────────────────────────────────────────────────────────────────────────────
"""

import os
import requests
import chromadb
from chromadb.utils import embedding_functions
# from groq import Groq                          # OLD: hosted API client
from dotenv import load_dotenv

load_dotenv()

# ── Clients ──────────────────────────────────────────────────────────────────
# OLD — Groq hosted API (requires GROQ_API_KEY, sends data off-machine)
# groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# NEW — Local Ollama server, no API key, no external calls
OLLAMA_URL   = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

chroma_client = chromadb.PersistentClient(path="./chroma_store")

# OLD — chromadb's DefaultEmbeddingFunction (all-MiniLM-L6-v2, general-purpose,
# does not know "hypertension" == "high blood pressure" — see README notes)
# embedding_fn = embedding_functions.DefaultEmbeddingFunction()

# NEW — Biomedical embedding model, still open weight, understands clinical
# synonyms and drug-disease relationships. Must match the model used in ingest.py.
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="FremyCompany/BioLORD-2023"
)

collection = chroma_client.get_collection(
    name="ehr_patients",
    embedding_function=embedding_fn
)


def ask_question(question: str, n_results: int = 3) -> dict:
    """
    Full RAG pipeline in one function.

    Args:
        question:  Plain English question from the user
        n_results: How many patient records to retrieve as context

    Returns:
        dict with 'answer' and 'sources' (which patients were retrieved)
    """

    # ── 1. RETRIEVE ──────────────────────────────────────────────────────────
    # ChromaDB embeds the question and finds the closest matching records.
    # "Closest" = most semantically similar, not just keyword match.
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )

    retrieved_chunks = results["documents"][0]   # list of patient text chunks
    retrieved_ids    = results["ids"][0]          # e.g. ["P003", "P001"]

    # ── 2. AUGMENT ───────────────────────────────────────────────────────────
    # We build a prompt that gives the LLM the retrieved context.
    # Without this, the LLM would guess from its training data.
    # With this, it answers from YOUR data.
    context = "\n\n---\n\n".join(retrieved_chunks)

    prompt = f"""You are a clinical data assistant. Answer the question below using ONLY the patient records provided.
If the answer is not in the records, say so clearly. Do not guess or use outside knowledge.

PATIENT RECORDS:
{context}

QUESTION:
{question}

ANSWER:"""

    # ── 3. GENERATE ──────────────────────────────────────────────────────────
    # OLD — Groq hosted API call
    # response = groq_client.chat.completions.create(
    #     model="llama-3.3-70b-versatile",   # Fast, free, good enough for POC
    #     messages=[
    #         {"role": "user", "content": prompt}
    #     ],
    #     temperature=0.2           # Low temp = more factual, less creative
    # )
    # answer = response.choices[0].message.content.strip()

    # NEW — Local Ollama call, same open-weight Llama model, runs on your machine
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2}
        },
        timeout=120
    )
    response.raise_for_status()
    answer = response.json()["response"].strip()

    return {
        "answer": answer,
        "sources": retrieved_ids
    }
