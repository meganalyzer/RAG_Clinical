"""
STEP 1: INGEST
This script reads your EHR JSON, converts each patient record into
a text chunk, embeds it, and stores it in a local ChromaDB vector store.

Run this ONCE before starting the app:
    python ingest.py

── OPEN WEIGHT UPDATE ─────────────────────────────────────────────────────
Swapped the embedding model from chromadb's default (all-MiniLM-L6-v2,
general-purpose) to BioLORD-2023 (biomedical, open weight). MiniLM does not
reliably place "hypertension" and "high blood pressure" close together in
vector space — it was trained on general internet text, not clinical
literature. BioLORD is trained on medical text and understands clinical
synonyms, ICD codes, and drug-disease relationships.

If you re-run ingest with a new embedding model, you MUST re-run it fully —
old vectors from MiniLM and new vectors from BioLORD are not compatible in
the same collection. The script below already clears existing data before
re-embedding, so this is handled automatically.
────────────────────────────────────────────────────────────────────────────
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
client = chromadb.PersistentClient(path="./chroma_store")

# OLD — general-purpose default embedding model (all-MiniLM-L6-v2)
# Doesn't know "hypertension" == "high blood pressure". See README notes.
# embedding_fn = embedding_functions.DefaultEmbeddingFunction()

# NEW — biomedical embedding model, open weight, understands clinical synonyms
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="FremyCompany/BioLORD-2023"
)

collection = client.get_or_create_collection(
    name="ehr_patients",
    embedding_function=embedding_fn
)

# Clear existing data so re-runs don't duplicate (and so old MiniLM vectors
# never mix with new BioLORD vectors in the same collection)
existing = collection.get()
if existing["ids"]:
    collection.delete(ids=existing["ids"])

# ── Embed and store ──────────────────────────────────────────────────────────
collection.add(
    documents=documents,
    ids=ids
)

print(f"✅ Ingested {len(documents)} patient records into ChromaDB.")
print("   Embedding model: FremyCompany/BioLORD-2023 (biomedical, open weight)")
print("   Vector store saved to ./chroma_store")
print("   You can now run: uvicorn app:app --reload")
