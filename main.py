import os
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import rag

load_dotenv()
app = FastAPI(title="Verdict AI")

MAX_MB = 15
workspaces: dict[str, rag.Workspace] = {}


def ws_for(wid: str) -> rag.Workspace:
    if not wid or len(wid) > 64:
        raise HTTPException(400, "Bad workspace id")
    return workspaces.setdefault(wid, rag.Workspace())


@app.post("/api/upload")
def upload(workspace_id: str = Form(...), files: list[UploadFile] = File(...)):
    ws = ws_for(workspace_id)
    results = []
    for f in files:
        data = f.file.read()
        if len(data) > MAX_MB * 1024 * 1024:
            results.append({"name": f.filename, "error": f"Larger than {MAX_MB} MB"})
            continue
        try:
            pages = rag.read_file(f.filename, data)
            doc_id = uuid.uuid4().hex[:8]
            chunks = rag.chunk_pages(pages, doc_id, f.filename)
            if not chunks:
                raise ValueError("No readable text (scanned PDFs need OCR).")
            ws.add(doc_id, f.filename, len(pages), chunks)
            results.append(ws.docs[doc_id])
        except Exception as e:
            results.append({"name": f.filename, "error": str(e)})
    return {"results": results, "docs": list(ws.docs.values())}


@app.get("/api/docs")
def list_docs(workspace_id: str):
    return {"docs": list(ws_for(workspace_id).docs.values())}


@app.delete("/api/docs/{doc_id}")
def delete_doc(doc_id: str, workspace_id: str):
    ws_for(workspace_id).remove(doc_id)
    return {"docs": list(ws_for(workspace_id).docs.values())}


class Ask(BaseModel):
    workspace_id: str
    question: str


@app.post("/api/ask")
def ask(body: Ask):
    ws = ws_for(body.workspace_id)
    if not ws.docs:
        raise HTTPException(400, "Upload a document first.")
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise HTTPException(500, "Server is missing GROQ_API_KEY.")
    try:
        return rag.answer_question(ws, body.question.strip(), key)
    except Exception as e:
        raise HTTPException(502, f"LLM error: {e}")


@app.get("/")
def home():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
