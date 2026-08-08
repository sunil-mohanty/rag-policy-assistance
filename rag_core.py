"""
rag_core.py
Shared ingestion/versioning logic. Used by admin_app.py (to ingest) and
knowledge_service.py (to know which version to load at startup).

No Streamlit dependency anymore — this is now plain Python, usable from
any process. The retrieval/generation logic that used to live here
(get_vectordb, get_llm) has moved to knowledge_service.py, which owns
those connections for its own process lifetime instead of trying to
cache-and-invalidate them within a long-running Streamlit process. See
knowledge_service.py's docstring for why that's the actual fix.
"""

import time
import uuid
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DATA_DIR = Path("./data")
PERSIST_BASE_DIR = Path("./chroma_db")
POINTER_FILE = PERSIST_BASE_DIR / "CURRENT_VERSION"
COLLECTION_NAME = "hr_policies"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}
SUPPORTED_EXTENSIONS = list(LOADER_MAP.keys())


def list_documents():
    """Every currently-ingestable file sitting in data/."""
    if not DATA_DIR.exists():
        return []
    return sorted(
        f for f in DATA_DIR.iterdir()
        if f.suffix.lower() in LOADER_MAP
    )


def load_documents():
    docs = []
    files = list_documents()
    for f in files:
        loader_cls = LOADER_MAP[f.suffix.lower()]
        loader = loader_cls(str(f))
        loaded = loader.load()
        for d in loaded:
            d.metadata["source"] = f.name
        docs.extend(loaded)
    return docs, files


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def current_version():
    """Reads the pointer file. Returns None if no ingestion has ever
    completed successfully yet."""
    if POINTER_FILE.exists():
        v = POINTER_FILE.read_text().strip()
        return v if v else None
    return None


def version_dir(version):
    return PERSIST_BASE_DIR / version


def run_full_ingestion():
    """Rebuild the vector store from scratch using everything currently
    in data/, into a NEW uniquely-named directory. Returns
    (num_files, num_chunks). Does NOT restart knowledge_service.py —
    that's admin_app.py's job, right after this returns successfully.
    """
    DATA_DIR.mkdir(exist_ok=True)
    PERSIST_BASE_DIR.mkdir(exist_ok=True)

    docs, files = load_documents()
    if not files:
        raise ValueError(f"No supported documents found in {DATA_DIR}. "
                          f"Supported: {SUPPORTED_EXTENSIONS}")

    chunks = chunk_documents(docs)

    new_version = f"v_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    new_dir = version_dir(new_version)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(new_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )

    POINTER_FILE.write_text(new_version)
    return len(files), len(chunks)


def cleanup_old_versions():
    """Deletes every version directory except the current one. Safe to
    call anytime now — knowledge_service.py is restarted immediately
    after every ingest, so there's no long-lived process left holding a
    stale reference to an old version the way there used to be."""
    import shutil

    current = current_version()
    if not current:
        raise RuntimeError(
            "Could not determine the current version — refusing to "
            "clean up to avoid deleting the active version by mistake."
        )
    if not version_dir(current).exists():
        raise RuntimeError(
            f"Current version '{current}' has no matching folder on "
            f"disk — refusing to clean up. Re-ingest first."
        )

    removed = 0
    if not PERSIST_BASE_DIR.exists():
        return removed
    for entry in PERSIST_BASE_DIR.iterdir():
        if entry.is_dir() and entry.name != current:
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
    return removed