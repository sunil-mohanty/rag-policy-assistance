# HR Policy RAG Chatbot — Self-Hosted (Ollama + Chroma + Streamlit)

Zero SaaS LLM calls. Everything runs locally on your Mac.

## 3-Hour Build Plan

### Hour 1 — Environment setup (20–30 min actual work)
```bash
# 1. Install Ollama (if not already)
brew install ollama

# 2. Start Ollama service
ollama serve &

# 3. Pull the models (this downloads ~5-8GB, do this FIRST while you set up code)
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# 4. Set up your Python env (reuse your manim-env pattern or make a new one)
cd hr-rag-chatbot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Hour 1 (parallel) — Gather policy docs
- Drop your HR policy files (PDF, DOCX, TXT, MD) into `./data/`
- Start with 3-5 real policies: Leave Policy, Code of Conduct, WFH Policy,
  Benefits, Grievance Procedure — whatever you have handy.

### Hour 2 — Ingest + first test
```bash
python ingest.py
```
This chunks the docs, embeds them locally via `nomic-embed-text`, and
persists to `./chroma_db`. Takes seconds to a couple minutes depending on
doc volume.

```bash
streamlit run app.py
```
Opens at `http://localhost:8501`. Ask a test question immediately —
"How many casual leave days do I get?" — and check the retrieved excerpts
in the expander to confirm it's grounding on the right document.

### Hour 3 — Polish + demo prep
- Tune `TOP_K` in `app.py` (default 4) if answers miss context — bump to 6
- Tune `chunk_size`/`chunk_overlap` in `ingest.py` if answers feel fragmented
- Swap `LLM_MODEL` to `qwen2.5:7b` or `mistral:7b` if llama3.1 is too slow
  on your machine — test 2-3 questions on each and pick the best
- Add your bank/company logo + rename title in `app.py` for the showcase
- Optional: add a "last updated" timestamp per doc in the sidebar

## Why this stack for a bank HR use case
- **No data leaves the machine** — Ollama runs the LLM and embeddings
  entirely locally, no API keys, no external calls. Good story for a
  banking compliance/security review.
- **Swappable models** — if leadership later wants a bigger model, just
  `ollama pull` a different one and change one line.
- **Cheap to run** — no per-token cost, unlike OpenAI/Anthropic APIs, which
  matters if this scales to org-wide usage.

## Known limitations to mention in your demo
- Retrieval quality depends heavily on doc chunking — long policy PDFs with
  tables (e.g. leave-day matrices) sometimes need custom parsing.
- No user authentication / access control yet — fine for a demo, not for
  production with sensitive HR data.
- No conversation memory across sessions (in-memory only, resets on reload).
- For production: add citation click-through, feedback buttons (👍/👎),
  and ideally swap Chroma for a more scalable vector DB (pgvector, Qdrant)
  if the policy corpus grows large.

## Troubleshooting
- **"Could not load vector store"** → run `python ingest.py` first
- **Slow responses** → try a smaller/quantized model, e.g. `ollama pull llama3.1:8b-instruct-q4_0`
- **Ollama connection refused** → make sure `ollama serve` is running
