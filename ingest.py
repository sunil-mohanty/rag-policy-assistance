"""
ingest.py
Optional command-line ingestion — the Admin page (pages/1_Admin.py) now
handles this from the browser instead, so you generally won't need this
script anymore. Kept for scripting/CI use or as a fallback if you prefer
the terminal. Both call the exact same logic in rag_core.py, so results
are identical either way.

Run with:
    python ingest.py
"""

import rag_core

if __name__ == "__main__":
    try:
        num_files, num_chunks = rag_core.run_full_ingestion()
        print(f"✅ Ingested {num_files} document(s) into {num_chunks} chunks "
              f"in '{rag_core.PERSIST_BASE_DIR}'.")
        print("If the Streamlit app is already running, use the Admin "
              "page's ingest button instead next time to avoid needing "
              "a restart — this CLI script doesn't clear the app's cache.")
    except ValueError as e:
        print(f"❌ {e}")