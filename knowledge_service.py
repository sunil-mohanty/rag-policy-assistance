"""
knowledge_service.py
Backend service owning the Chroma vector store + Ollama LLM connections.
This is the piece that gets restarted after every ingestion — NOT the
Streamlit frontend, which stays running the whole time.

WHY restarting THIS instead of fighting caching:
We spent several rounds trying to make a single long-running Streamlit
process reliably pick up a freshly-ingested vector store — hitting, in
order: (1) chromadb caching its internal client by directory path,
(2) Streamlit's own resource cache needing explicit invalidation across
processes, (3) chromadb's SharedSystemClient caching internal state
even across genuinely different paths within one process. Each fix
closed one gap but another opened.

The actual fix is architectural, not another cache-busting trick: stop
trying to keep one long-lived process's internal state in sync with
disk changes at all. A freshly started process has NO stale state of
any kind to fight, by construction. So: this service loads the vector
store + LLM ONCE at startup, and admin_app.py simply restarts this
whole process after every successful ingest. The Streamlit frontend
(policy_app.py) never restarts — it just makes small, fast HTTP calls
to whatever instance of this service happens to be running.

Run with:
    uvicorn knowledge_service:app --host 127.0.0.1 --port 8600

(admin_app.py starts/restarts this for you automatically — you
generally won't run this command by hand except for initial testing.)
"""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import rag_core

PID_FILE = Path("./knowledge_service.pid")
PID_FILE.write_text(str(os.getpid()))

SYSTEM_PROMPT = """You are an HR Policy Assistant for the company.

Answer the employee's question using the policy context provided below.
The context has already been filtered to only include content judged
relevant to the question, so treat it as trustworthy — you do not need
an exact word-for-word match; a reasonable, well-supported inference from
the context counts as answerable.

Answering rules:
1. Give a concise, synthesized answer in your own words — 2 to 4
   sentences for most questions. NEVER paste or quote large blocks of the
   source text verbatim. Summarize; don't reproduce the document.
2. Always end your answer with the source document name(s), like:
   (Source: Leave_Policy.pdf)

Context:
{context}
"""

# --- Loaded ONCE at process startup. This module-level state IS the
# cache — there's no explicit caching decorator needed, because the
# process's own lifetime is exactly the cache's lifetime. Restarting
# the process (which admin_app.py does after every ingest) is what
# guarantees a fresh view of the vector store. ---------------------------
_version = rag_core.current_version()
if _version is None:
    raise RuntimeError(
        "No documents ingested yet. Use the Admin app to ingest at "
        "least one document, THEN start knowledge_service (admin_app.py "
        "does this restart for you automatically after ingestion)."
    )

_embeddings = OllamaEmbeddings(model=rag_core.EMBED_MODEL)
_vectordb = Chroma(
    collection_name=rag_core.COLLECTION_NAME,
    embedding_function=_embeddings,
    persist_directory=str(rag_core.version_dir(_version)),
)
_llm = ChatOllama(
    model=rag_core.LLM_MODEL,
    temperature=0.1,
    num_ctx=2048,
    num_predict=300,
    keep_alive="30m",
)
_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])
_chain = _prompt | _llm | StrOutputParser()

app = FastAPI(title="HR Policy Knowledge Service")


@app.get("/health")
def health():
    return {"status": "ok", "version": _version}


class SearchRequest(BaseModel):
    question: str
    k: int = 4


@app.post("/search")
def search(req: SearchRequest):
    """Embedding + similarity search only — no LLM call. Returns raw
    scored chunks; the frontend decides what to do with the scores
    (answer / off-topic / not-covered thresholds live in policy_app.py,
    not here — this service stays a dumb, fast, restartable data layer)."""
    try:
        results = _vectordb.similarity_search_with_relevance_scores(
            req.question, k=req.k
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return [
        {
            "text": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "score": score,
        }
        for doc, score in results
    ]


class GenerateRequest(BaseModel):
    context: str
    question: str


@app.post("/generate")
def generate(req: GenerateRequest):
    """Streams the LLM's answer back token-by-token as plain text, so
    the Streamlit frontend can pipe it straight into st.write_stream
    for the same live-typing feel as before."""
    def token_stream():
        for token in _chain.stream({"context": req.context, "question": req.question}):
            yield token

    return StreamingResponse(token_stream(), media_type="text/plain")