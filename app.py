import streamlit as st
import tempfile
import os
from pathlib import Path
import rag

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="DocIntel", page_icon="📜", layout="wide")

# ==============================
# SESSION STATE
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "index_ready" not in st.session_state:
    st.session_state.index_ready = False

if "boot_checked" not in st.session_state:
    st.session_state.boot_checked = True
    st.session_state.index_ready  = rag.load_existing_index()

# ==============================
# SIDEBAR — Upload & Document Library
# ==============================
st.sidebar.title("🤖 DocIntel")
st.sidebar.caption("Document Intelligence System 💡")
st.sidebar.divider()

# ── Upload ────────────────────────────────────────────────────────────────────
st.sidebar.subheader("📂 Upload Documents")
st.sidebar.caption("PDF, TXT, DOCX, CSV, XLSX, XLS, PPTX, MD, HTML — up to 100 files")

uploaded_files = st.sidebar.file_uploader(
    "Choose files",
    type=["pdf", "txt", "docx", "doc", "csv", "xlsx", "xls", "pptx", "md", "html", "htm"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if uploaded_files:
    st.sidebar.caption(f"**{len(uploaded_files)}** file(s) selected")

if st.sidebar.button("⌛ Process files", use_container_width=True):
    if not uploaded_files:
        st.sidebar.warning("Select at least one file first.")
    else:
        tmp_dir   = tempfile.mkdtemp()
        tmp_paths = []
        for uf in uploaded_files:
            p = os.path.join(tmp_dir, uf.name)
            with open(p, "wb") as f:
                f.write(uf.read())
            tmp_paths.append(p)

        progress_bar = st.sidebar.progress(0, text="Starting…")

        def on_progress(current, total, fname):
            pct = int((current / total) * 100) if total else 0
            progress_bar.progress(pct, text=f"{fname}")

        try:
            result = rag.process_documents(tmp_paths, on_progress)
            progress_bar.progress(100, text="Done!")
            st.session_state.index_ready = True
            st.sidebar.success(
                f"Indexed **{result['total_files']}** file(s) · "
                f"**{result['total_chunks']}** chunks"
            )
            if result["skipped"]:
                st.sidebar.warning(
                    f"Skipped {len(result['skipped'])}: {', '.join(result['skipped'])}"
                )
            st.rerun()
        except ValueError as e:
            st.sidebar.error(str(e))

st.sidebar.divider()

# ── Document Library ──────────────────────────────────────────────────────────
st.sidebar.subheader("🗂️ Indexed Documents")

stats = rag.get_index_stats()

if stats["status"] == "ready" and stats["indexed_files"]:
    st.sidebar.caption(f"{stats['total_vectors']} vectors · {len(stats['indexed_files'])} files")
    for fname in stats["indexed_files"]:
        ext = Path(fname).suffix.lower().lstrip(".")
        ext_icons = {
            "pdf": "📕", "txt": "📄", "docx": "📘", "doc": "📘",
            "csv": "📊", "xlsx": "📊", "xls": "📊",
            "pptx": "📑", "md": "📝", "html": "🌐", "htm": "🌐",
        }
        icon = ext_icons.get(ext, "📄")
        st.sidebar.caption(f"{icon} `{fname}`")
else:
    st.sidebar.info("No documents indexed yet.\nUpload files above to begin.")

st.sidebar.divider()

# ── Clear chat ────────────────────────────────────────────────────────────────
if st.sidebar.button("🧹 Clear Chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# ==============================
# MAIN — Chat Interface
# ==============================
st.title("🤖 DocIntel — Document Intelligence System")

if not st.session_state.index_ready:
    st.info("👈 Upload and process documents from the sidebar to start chatting.")

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if "sources" in msg and msg["sources"]:
            st.markdown("**Sources:**")
            for s in msg["sources"].split(","):
                st.markdown(f"- {s.strip()}")

        if "chunks" in msg and msg["chunks"]:
            with st.expander("🔍 View Supporting Context"):
                for cm in msg["chunks"]:
                    page_s = str(cm["page"] + 1) if cm["page"] != "" else "—"
                    st.caption(
                        f"Chunk {cm['chunk']} · `{cm['source']}` · "
                        f"page {page_s} · score {cm['score']} · {cm['tokens']} tokens"
                    )

# ── Chat input ────────────────────────────────────────────────────────────────
prompt = st.chat_input("💬 Ask something about your documents…")

if prompt:
    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({
        "role":    "user",
        "content": prompt,
    })

    # Generate and show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            if not rag.is_ready():
                response_text = "No documents indexed yet. Please upload and process files from the sidebar first."
                sources       = ""
                chunk_meta    = []
                st.warning(response_text)
            else:
                try:
                    answer, sources, chunk_meta = rag.generate_answer(prompt)
                    response_text = answer
                    st.markdown(response_text)

                    if sources:
                        st.markdown("**Sources:**")
                        for s in sources.split(","):
                            st.markdown(f"- {s.strip()}")

                    if chunk_meta:
                        with st.expander("🔍 View Supporting Context"):
                            for cm in chunk_meta:
                                page_s = str(cm["page"] + 1) if cm["page"] != "" else "—"
                                st.caption(
                                    f"Chunk {cm['chunk']} · `{cm['source']}` · "
                                    f"page {page_s} · score {cm['score']} · {cm['tokens']} tokens"
                                )

                except RuntimeError as e:
                    response_text = str(e)
                    sources       = ""
                    chunk_meta    = []
                    st.error(response_text)

    st.session_state.messages.append({
        "role":    "assistant",
        "content": response_text,
        "sources": sources,
        "chunks":  chunk_meta,
    })
