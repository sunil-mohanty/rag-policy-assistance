#!/bin/bash
# runpod/start-runpod.sh
#
# Runs the WHOLE stack in one RunPod pod: Ollama, knowledge_service,
# admin_app, and policy_app. Unlike the Railway deployment, everything
# stays on one machine here — no cross-container networking needed,
# policy_app talks to knowledge_service over plain 127.0.0.1 exactly
# like local dev.
#
# First-time setup:
#   1. Upload hr-rag-chatbot/ to /workspace/hr-rag-chatbot (use
#      /workspace specifically — the persistent Network Volume mount
#      point, so models/data survive pod restarts)
#   2. scp this script + runpod/requirements.txt into /workspace/
#   3. chmod +x start-runpod.sh && ./start-runpod.sh
#
# On every later pod start, just re-run this same script — every step
# is idempotent (skips whatever's already done).

set -e
cd /workspace

echo "=== 1. Install Ollama (skipped if already installed) ==="
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo "=== 2. Start Ollama server in the background ==="
ollama serve > /workspace/ollama.log 2>&1 &
sleep 5

echo "=== 3. Pull models (skipped if already cached on the volume) ==="
ollama pull llama3.2:3b
ollama pull nomic-embed-text

echo "=== 4. Python environment (skipped if already built) ==="
cd /workspace/hr-rag-chatbot
if [ ! -d "venv" ]; then
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

echo "=== 5. Resume existing knowledge base, if any ==="
# If a previous ingestion already exists on the volume (e.g. this is a
# pod restart, not a first boot), start knowledge_service right away so
# policy_app doesn't error out waiting for someone to hit "Ingest Now"
# again. If this is the first-ever boot, admin_app's own
# restart_knowledge_service() starts it after the first ingest instead.
if [ -f "chroma_db/CURRENT_VERSION" ]; then
    echo "    Existing knowledge base found — starting knowledge_service..."
    uvicorn knowledge_service:app --host 0.0.0.0 --port 8600 \
        > /workspace/knowledge_service.log 2>&1 &
fi

echo "=== 6. Starting Admin app (background) ==="
streamlit run admin_app.py \
    --server.port 8502 --server.address 0.0.0.0 --server.headless true \
    --server.enableCORS false --server.enableXsrfProtection false \
    > /workspace/admin_app.log 2>&1 &

echo ""
echo "=== Setup complete. Starting policy_app (foreground — this keeps the pod alive) ==="
echo "In RunPod's dashboard, use the HTTP Service URLs for ports 8501"
echo "(chat — your demo link) and 8502 (admin) to get the actual URLs."
echo ""

# Foreground on purpose — keeps the pod's main process alive. If you
# need to detach safely instead of leaving this terminal open, wrap the
# whole script in tmux: `tmux new -s demo`, run this script, Ctrl+B D.
exec streamlit run policy_app.py \
    --server.port 8501 --server.address 0.0.0.0 --server.headless true \
    --server.enableCORS false --server.enableXsrfProtection false

 #run the mobile app
exec streamlit run policy_mobile_app.py \
    --server.port 8503 --server.address 0.0.0.0 --server.headless true \
    --server.enableCORS false --server.enableXsrfProtection false   