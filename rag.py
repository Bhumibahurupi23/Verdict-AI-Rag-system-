from __future__ import annotations

import io
import json
import os
import re
from dataclasses import dataclass

import numpy as np
from docx import Document
from dotenv import load_dotenv
from fastembed import TextEmbedding
from groq import Groq
from pypdf import PdfReader

load_dotenv()

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-safeguard-20b")
MIN_SCORE = 0.35  # below this, we refuse without calling the LLM. Tune it on your own docs.

SYSTEM_PROMPT = """You are Verdict AI, a strict document question-answering engine.
Rules:
1. Use ONLY the numbered passages provided. Never use outside knowledge.
2. If the passages do not clearly contain the answer, set "answerable" to false.
3. Write the answer in simple, plain language.
4. For every claim, give a citation: the passage id and a SHORT EXACT quote
   copied word-for-word from that passage.
5. Never guess, infer beyond the text, or invent numbers, names, or dates.
Respond with JSON only, in this exact shape:
{"answerable": true, "answer": "text with [1] style markers",
 "citations": [{"id": 1, "quote": "exact words from passage 1"}]}"""


@dataclass
class Chunk:
    doc_id: str
    source: str
    page: int
    text: str


_embedder: TextEmbedding | None = None


def embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBED_MODEL)
    return _embedder


# ---------- reading + chunking ----------
def read_file(name: str, data: bytes) -> list[tuple[int, str]]:
    ext = name.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        reader = PdfReader(io.BytesIO(data))
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    if ext == "docx":
        doc = Document(io.BytesIO(data))
        return [(1, "\n".join(p.text for p in doc.paragraphs))]
    if ext in ("txt", "md"):
        return [(1, data.decode("utf-8", errors="ignore"))]
    raise ValueError("Unsupported file type. Use PDF, DOCX, TXT or MD.")


def chunk_pages(pages, doc_id: str, source: str, size=900, overlap=150) -> list[Chunk]:
    chunks, step = [], size - overlap
    for page, text in pages:
        text = " ".join(text.split())
        for start in range(0, len(text), step):
            piece = text[start : start + size].strip()
            if len(piece) > 40:
                chunks.append(Chunk(doc_id, source, page, piece))
    return chunks


# ---------- workspace (one per browser) ----------
class Workspace:
    def __init__(self):
        self.docs: dict[str, dict] = {}
        self.chunks: list[Chunk] = []
        self.vecs: np.ndarray | None = None

    def add(self, doc_id: str, name: str, pages: int, chunks: list[Chunk]):
        v = np.array(list(embedder().embed([c.text for c in chunks])), dtype="float32")
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        self.vecs = v if self.vecs is None else np.vstack([self.vecs, v])
        self.chunks += chunks
        self.docs[doc_id] = {"id": doc_id, "name": name, "pages": pages, "chunks": len(chunks)}

    def remove(self, doc_id: str):
        keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
        self.chunks = [self.chunks[i] for i in keep]
        self.vecs = self.vecs[keep] if keep else None
        self.docs.pop(doc_id, None)

    def search(self, question: str, k=5):
        if self.vecs is None:
            return []
        q = np.array(list(embedder().query_embed(question)), dtype="float32")[0]
        q /= np.linalg.norm(q)
        scores = self.vecs @ q
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


# ---------- answering with verification ----------
def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


NOT_FOUND = {
    "answerable": False,
    "answer": "I couldn't find this in your documents.",
    "sources": [],
}


def answer_question(ws: Workspace, question: str, api_key: str) -> dict:
    hits = ws.search(question, k=3)
    if not hits or hits[0][1] < MIN_SCORE:
        return NOT_FOUND

    context = "\n\n".join(
        f"[{i + 1}] ({c.source}, page {c.page})\n{c.text}" for i, (c, _) in enumerate(hits)
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {question}"},
    ]

    client = Groq(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0,
            max_tokens=700,
            response_format={"type": "json_object"},
            messages=messages,
        )
    except Exception as e:
        if "rate_limit" in str(e) or "429" in str(e) or "413" in str(e):
            raise
        # some models don't support JSON mode, so retry without it
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0,
            max_tokens=700,
            messages=messages,
        )

    try:
        raw = resp.choices[0].message.content or ""
        out = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
    except (AttributeError, json.JSONDecodeError):
        return {
            "answerable": False,
            "answer": "Something went wrong. Please try again.",
            "sources": [],
        }

    if not out.get("answerable"):
        return NOT_FOUND

    # Verify every quote really exists in the passage it cites
    verified = []
    for cit in out.get("citations", []):
        try:
            idx = int(cit["id"]) - 1
            chunk, score = hits[idx]
        except (KeyError, ValueError, IndexError, TypeError):
            continue
        quote = _norm(str(cit.get("quote", "")))
        if len(quote) > 10 and quote in _norm(chunk.text):
            verified.append(
                {
                    "id": idx + 1,
                    "source": chunk.source,
                    "page": chunk.page,
                    "quote": cit["quote"],
                    "text": chunk.text,
                    "score": round(score, 2),
                }
            )

    if not verified:
        return {
            "answerable": False,
            "answer": "I found related text but couldn't verify an exact answer, so I won't guess.",
            "sources": [],
        }
    return {"answerable": True, "answer": out["answer"], "sources": verified}
