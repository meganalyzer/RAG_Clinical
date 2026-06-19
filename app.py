"""
STEP 3: API SERVER
FastAPI server that exposes the RAG pipeline as an endpoint.
The frontend calls this when you submit a question.

Run with:
    uvicorn app:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from rag import ask_question

app = FastAPI(title="EHR RAG POC")

# Allow browser to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend
app.mount("/static", StaticFiles(directory="."), name="static")

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]

@app.get("/")
def serve_ui():
    return FileResponse("index.html")

@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    """
    Main RAG endpoint.
    Receives a plain English question, returns an answer + which patient
    records were retrieved as sources.
    """
    result = ask_question(request.question)
    return AnswerResponse(
        answer=result["answer"],
        sources=result["sources"]
    )

@app.get("/health")
def health():
    return {"status": "ok"}
