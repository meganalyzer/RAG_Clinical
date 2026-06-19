"""
STEP 1: INGEST
This script reads your EHR JSON, converts each patient record into
a text chunk, embeds it, and stores it in a local ChromaDB vector store.

Run this ONCE before starting the app:
    python ingest.py
"""

import json
import chromadb
from chromadb.utils import embedding_functions

# ── Load synthetic EHR data ──────────────────────────────────────────────────
with open("synthetic_ehr.json", "r") as f:
    patients = json.load(f)

# ── Convert each patient record into a readable text chunk ───────────────────
# This is the "chunking" step in RAG.
# We turn structured JSON into natural language so the LLM can reason over it.
def patient_to_text(p):
    return f"""
Patient ID: {p['patient_id']}
Name: {p['name']}
Age: {p['age']} | Gender: {p['gender']}
Diagnosis: {p['diagnosis']}
Medications: {', '.join(p['medications'])}
Last Visit: {p['last_visit']}
Clinical Notes: {p['notes']}
""".strip()

documents = [patient_to_text(p) for p in patients]
ids       = [p["patient_id"] for p in patients]

# ── Set up ChromaDB (runs locally, no cloud needed) ──────────────────────────
# Uses a default sentence-transformer model to embed text into vectors.
# These vectors are what enable semantic search ("meaning-based" search).
client = chromadb.PersistentClient(path="./chroma_store")

embedding_fn = embedding_functions.DefaultEmbeddingFunction()

collection = client.get_or_create_collection(
    name="ehr_patients",
    embedding_function=embedding_fn
)

# Clear existing data so re-runs don't duplicate
existing = collection.get()
if existing["ids"]:
    collection.delete(ids=existing["ids"])

# ── Embed and store ──────────────────────────────────────────────────────────
collection.add(
    documents=documents,
    ids=ids
)

print(f"✅ Ingested {len(documents)} patient records into ChromaDB.")
print("   Vector store saved to ./chroma_store")
print("   You can now run: python app.py")
