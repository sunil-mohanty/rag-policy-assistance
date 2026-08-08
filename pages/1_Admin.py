"""
pages/1_Admin.py
Upload + ingest HR policy documents from the browser — no terminal, no
manual `python ingest.py`, and no restarting the chat app afterward.

Streamlit auto-discovers any file in pages/ and adds it to the sidebar
navigation automatically alongside app.py.
"""

import streamlit as st
import rag_core

# --- CHANGE THIS before real use — this is a placeholder gate, not real
# auth. Fine for a demo; swap for proper SSO before production per the
# earlier requirements-doc discussion. -------------------------------
ADMIN_PASSWORD = "changeme123"

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
    "Rebuilds the search index from everything currently in data/ "
    "(full rebuild each time — simple and avoids stale duplicate "
    "content from removed or edited files)."
)

if st.button("Ingest Now", type="primary"):
    if not rag_core.list_documents():
        st.error("No documents to ingest — upload some first.")
    else:
        with st.spinner("Ingesting documents... this can take a little while depending on document size."):
            try:
                num_files, num_chunks = rag_core.run_full_ingestion()

                # THIS is the step that means no restart is needed —
                # it invalidates the cached vector store connection that
                # app.py shares with this page via rag_core, so the chat
                # page picks up the new content on its very next use.
                rag_core.get_vectordb.clear()

                st.success(
                    f"✅ Ingested {num_files} document(s) into {num_chunks} "
                    f"chunks. The chat assistant is already up to date — "
                    f"no restart needed."
                )
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

st.divider()
if st.button("Log out"):
    st.session_state.admin_authenticated = False
    st.rerun()