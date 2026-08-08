"""
admin_app.py
Standalone admin app for uploading and ingesting HR policy documents —
runs as a SEPARATE process/port from policy_app.py.

After every successful ingest, this automatically restarts
knowledge_service.py — see that file's docstring for why restarting a
small backend service, rather than trying to invalidate caches within
a long-running process, is what actually fixes the "still shows old
data after ingest" problem for good.

Run with:
    streamlit run admin_app.py --server.port 8502
"""

import os
import signal
import subprocess
import time
from pathlib import Path

import streamlit as st
import requests
import rag_core

# --- CHANGE THIS before real use — placeholder gate, not real auth. ---
ADMIN_PASSWORD = "changeme123"

KNOWLEDGE_SERVICE_HOST = "127.0.0.1"
KNOWLEDGE_SERVICE_PORT = 8600
KNOWLEDGE_SERVICE_URL = f"http://{KNOWLEDGE_SERVICE_HOST}:{KNOWLEDGE_SERVICE_PORT}"
PID_FILE = Path("./knowledge_service.pid")


def restart_knowledge_service():
    """Stops whatever knowledge_service instance is running (if any),
    then starts a fresh one. The fresh process loads the CURRENT
    version from the pointer file at its own startup — no stale state
    possible, since nothing carries over from the old process."""
    if PID_FILE.exists():
        try:
            old_pid = int(PID_FILE.read_text().strip())
            os.kill(old_pid, signal.SIGTERM)
        except (ValueError, ProcessLookupError, PermissionError):
            pass  # already gone, or PID file was stale — fine either way

    time.sleep(1.0)  # give the OS a moment to free the port

    subprocess.Popen(
        [
            "uvicorn", "knowledge_service:app",
            "--host", KNOWLEDGE_SERVICE_HOST,
            "--port", str(KNOWLEDGE_SERVICE_PORT),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for _ in range(20):  # up to ~10s
        try:
            resp = requests.get(f"{KNOWLEDGE_SERVICE_URL}/health", timeout=1)
            if resp.status_code == 200:
                return resp.json()
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.5)

    raise RuntimeError(
        "knowledge_service didn't come back up within 10 seconds after "
        "restart — check the terminal running admin_app.py for errors, "
        "or start it manually to see the actual error: "
        "uvicorn knowledge_service:app --host 127.0.0.1 --port 8600"
    )


st.set_page_config(page_title="Admin — HR Policy Assistant", page_icon="🔐")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; }
.admin-header {
    background: linear-gradient(135deg, #041153 0%, #0C1F6B 100%);
    border-radius: 16px;
    padding: 20px 28px;
    margin-bottom: 24px;
    color: #FFFFFF;
}
.admin-header h1 { font-size: 1.4rem; margin: 0; color: #FFFFFF; }
.stButton > button {
    border-radius: 9999px !important;
    font-weight: 600 !important;
    border: none !important;
    background-color: #B02435 !important;
    color: #FFFFFF !important;
}
</style>
<div class="admin-header"><h1>🔐 Admin — Manage HR Policy Documents</h1></div>
""", unsafe_allow_html=True)

# --- Simple password gate ------------------------------------------------
if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

if not st.session_state.admin_authenticated:
    pw = st.text_input("Admin password", type="password")
    if st.button("Log in"):
        if pw == ADMIN_PASSWORD:
            st.session_state.admin_authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

# --- Knowledge service status --------------------------------------------
st.subheader("⚙️ Knowledge service status")
try:
    health = requests.get(f"{KNOWLEDGE_SERVICE_URL}/health", timeout=2).json()
    st.success(f"Running — serving version `{health['version']}`")
except requests.exceptions.RequestException:
    st.warning(
        "Not currently running. It will start automatically after your "
        "first ingest below, or start it manually: "
        "`uvicorn knowledge_service:app --host 127.0.0.1 --port 8600`"
    )

st.divider()

# --- Current documents ----------------------------------------------------
st.subheader("📄 Current policy documents")
existing = rag_core.list_documents()

if not existing:
    st.info("No documents ingested yet — upload some below to get started.")
else:
    for f in existing:
        col1, col2 = st.columns([5, 1])
        with col1:
            size_kb = f.stat().st_size / 1024
            st.markdown(f"**{f.name}** · {size_kb:.0f} KB")
        with col2:
            if st.button("🗑️ Remove", key=f"remove_{f.name}"):
                f.unlink()
                st.rerun()

st.divider()

# --- Upload new documents --------------------------------------------------
st.subheader("⬆️ Upload new documents")
uploaded_files = st.file_uploader(
    "PDF, DOCX, TXT, or MD files",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
)

if uploaded_files:
    rag_core.DATA_DIR.mkdir(exist_ok=True)
    for uploaded in uploaded_files:
        dest = rag_core.DATA_DIR / uploaded.name
        with open(dest, "wb") as f:
            f.write(uploaded.getbuffer())
    st.success(f"Saved {len(uploaded_files)} file(s) to data/. "
               "Click 'Ingest Now' below to make them searchable.")

st.divider()

# --- Ingest ------------------------------------------------------------
st.subheader("🔄 Ingest")
st.caption(
    "Rebuilds the search index from everything currently in data/, then "
    "automatically restarts the knowledge service so the chat app picks "
    "up the change immediately — no restart needed on the chat side."
)

if st.button("Ingest Now", type="primary"):
    if not rag_core.list_documents():
        st.error("No documents to ingest — upload some first.")
    else:
        try:
            with st.spinner("Ingesting documents..."):
                num_files, num_chunks = rag_core.run_full_ingestion()

            with st.spinner("Restarting knowledge service with the new data..."):
                health = restart_knowledge_service()

            st.success(
                f"✅ Ingested {num_files} document(s) into {num_chunks} "
                f"chunks. Knowledge service restarted and is now serving "
                f"version `{health['version']}`. The chat app is already "
                f"up to date — no restart needed there."
            )
        except Exception as e:
            st.error(f"Failed: {e}")

st.divider()

# --- Manual cleanup -----------------------------------------------------
st.subheader("🧹 Clean up old versions")
st.caption(
    "Safe to run anytime now — knowledge_service restarts after every "
    "ingest, so there's no long-lived process left holding a reference "
    "to an old version the way there used to be."
)
if st.button("Clean up old versions"):
    try:
        removed = rag_core.cleanup_old_versions()
        st.success(f"Removed {removed} old version folder(s).")
    except RuntimeError as e:
        st.error(str(e))

st.divider()
if st.button("Log out"):
    st.session_state.admin_authenticated = False
    st.rerun()