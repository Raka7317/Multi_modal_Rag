import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.ingest import ingest_file
from app.graph import ask
from app.evaluation.ragas_eval import run_ragas_eval

app = FastAPI(title="Multimodal RAG API")

UPLOAD_DIR = "/tmp/rag_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    local_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(local_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    n_chunks = ingest_file(local_path, s3_key=f"raw/{uuid.uuid4()}_{file.filename}")
    return {"filename": file.filename, "chunks_indexed": n_chunks}


class QueryRequest(BaseModel):
    question: str
    user_id: str
    thread_id: str


@app.post("/query")
def query(req: QueryRequest):
    result = ask(question=req.question, user_id=req.user_id, thread_id=req.thread_id)
    return {
        "answer": result["answer"],
        "sources": [
            {"modality": c["modality"], "source_path": c["source_path"], "score": c.get("rerank_score")}
            for c in result["reranked"]
        ],
    }


class EvalRequest(BaseModel):
    questions: list[str]
    answers: list[str]
    contexts: list[list[str]]
    ground_truths: list[str] | None = None


@app.post("/evaluate")
def evaluate_endpoint(req: EvalRequest):
    df = run_ragas_eval(req.questions, req.answers, req.contexts, req.ground_truths)
    return df.to_dict(orient="records")
