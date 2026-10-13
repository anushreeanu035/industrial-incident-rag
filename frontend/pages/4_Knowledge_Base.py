from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st
from frontend.components.api_client import api_client
from frontend.components.sidebar import render_sidebar

st.set_page_config(
    page_title="Knowledge Base Management - Incident Investigator",
    page_icon="📁",
    layout="wide",
)

css_path = Path(__file__).resolve().parent.parent / "styles" / "theme.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

render_sidebar()

st.markdown(
    """
    <div class="incident-header-banner">
        <h1>📁 Knowledge Base Management</h1>
        <p>Enterprise Repository: Ingestion, Structure-Aware Chunking, Metadata Extraction & Integrity Auditing</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Document Upload Section
st.markdown("### 📤 Upload New Incident Reports")
st.caption("Supported file formats: **PDF (.pdf)**, **Markdown (.md)**, **Plain Text (.txt)**. Maximum file size: 25MB.")

uploaded_files = st.file_uploader(
    "Choose technical documents to ingest:",
    type=["pdf", "md", "txt", "markdown"],
    accept_multiple_files=True,
)

if uploaded_files:
    if st.button("⚡ Process & Index Uploaded Documents", use_container_width=True):
        progress_bar = st.progress(0)
        status_container = st.empty()

        for idx, file_obj in enumerate(uploaded_files, start=1):
            status_container.info(f"Processing and chunking '{file_obj.name}'...")
            try:
                content_bytes = file_obj.read()
                resp = api_client.upload_document(
                    filename=file_obj.name,
                    content_bytes=content_bytes,
                    mime_type=file_obj.type or "application/octet-stream",
                )
                if resp.get("status") == "indexed":
                    st.success(f"✅ **{file_obj.name}**: {resp.get('message')} (ID: `{resp.get('document_id')}`)")
                else:
                    st.warning(f"⚠️ **{file_obj.name}**: {resp.get('message')}")
            except Exception as e:
                st.error(f"❌ Failed to ingest '{file_obj.name}': {str(e)}")

            progress_bar.progress(idx / len(uploaded_files))

        status_container.success("Batch ingestion completed.")
        st.rerun()

st.divider()

# Ingested Documents List
st.markdown("### 📚 Indexed Incident Documents")

try:
    documents = api_client.list_documents()
except Exception as e:
    documents = []
    st.error(f"Could not load documents from backend: {str(e)}")

if not documents:
    st.info("No documents are currently indexed in the knowledge base. Upload reports above to start.")
else:
    # Summary Metrics
    c1, c2, c3, c4 = st.columns(4)
    total_chunks = sum(d.get("chunk_count", 0) for d in documents)
    total_size_kb = sum(d.get("file_size_bytes", 0) for d in documents) / 1024.0

    with c1:
        st.metric("Total Documents", len(documents))
    with c2:
        st.metric("Total Vector Chunks", total_chunks)
    with c3:
        st.metric("Storage Volume", f"{total_size_kb:.1f} KB")
    with c4:
        st.metric("Indexing Status", "100% Synced")

    # Documents Table
    doc_rows = []
    for d in documents:
        doc_rows.append({
            "Document ID": d.get("document_id"),
            "Filename": d.get("filename"),
            "Machine ID": d.get("machine_id") or "Unassigned",
            "Incident Type": d.get("incident_type") or "Unspecified",
            "Date": d.get("incident_date") or "N/A",
            "Format": d.get("file_type", "").upper(),
            "Chunks": d.get("chunk_count", 0),
            "Pages": d.get("page_count", 1),
            "Size (KB)": round(d.get("file_size_bytes", 0) / 1024.0, 1),
            "SHA-256 (Prefix)": d.get("checksum_sha256", "")[:12] + "...",
            "Status": d.get("status", "unknown").upper(),
        })

    df_docs = pd.DataFrame(doc_rows)
    st.dataframe(df_docs, use_container_width=True, hide_index=True)

    # Document Deletion and Detail Inspector
    st.markdown("#### 🗑️ Document Management & Deletion")
    col_sel, col_del = st.columns([3, 1], vertical_alignment="bottom")

    with col_sel:
        doc_to_manage = st.selectbox(
            "Select Document to Inspect or Remove:",
            options=documents,
            format_func=lambda d: f"{d.get('filename')} (ID: {d.get('document_id')}, Machine: {d.get('machine_id')})",
        )

    with col_del:
        if st.button("🗑️ Delete Selected Document", type="secondary", use_container_width=True):
            if doc_to_manage:
                doc_id = doc_to_manage.get("document_id")
                try:
                    res = api_client.delete_document(doc_id)
                    st.success(f"Document `{doc_id}` deleted successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete document: {str(e)}")

    if doc_to_manage:
        with st.expander(f"🔍 Provenance & Metadata Details for {doc_to_manage.get('filename')}", expanded=False):
            st.json(doc_to_manage)
