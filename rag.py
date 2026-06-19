"""
STEP 2: RAG CORE
This is the heart of the system.

ask_question() does exactly 3 things:
    1. RETRIEVE  — search ChromaDB for the most relevant patient chunks
    2. AUGMENT   — build a prompt that includes those chunks as context
    3. GENERATE  — send that prompt to Groq's LLM and return the answer

This is what RAG means: Retrieval-Augmented Generation.
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Clients ──────────────────────────────────────────────────────────────────
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_store")
embedding_fn  = embedding_functions.DefaultEmbeddingFunction()
collection    = chroma_client.get_collection(
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
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",   # Fast, free, good enough for POC
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2           # Low temp = more factual, less creative
    )

    answer = response.choices[0].message.content.strip()

    return {
        "answer": answer,
        "sources": retrieved_ids
    }
