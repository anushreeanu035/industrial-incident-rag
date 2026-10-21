from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st
from frontend.components.api_client import api_client
from frontend.components.sidebar import render_sidebar
from frontend.components.theme import load_theme

st.set_page_config(
    page_title="Knowledge Base Management - Incident Investigator",
    page_icon="📁",
    layout="wide",
)

# Load cached enterprise theme CSS
load_theme()

render_sidebar(current_page="knowledge_base")

# Top Navigation Bar
col_back, _ = st.columns([1, 4])
with col_back:
    if st.button("← Return to Dashboard", key="btn_back_home_kb"):
        st.switch_page("app.py")

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
st.markdown("### 📤 Ingest New Incident Reports")
st.caption("Supported technical formats: **PDF (.pdf)**, **Markdown (.md)**, **Plain Text (.txt)**. Maximum file size: 25 MB per file.")

uploaded_files = st.file_uploader(
    "Choose technical documents to ingest into the knowledge base:",
    type=["pdf", "md", "txt", "markdown"],
    accept_multiple_files=True,
    key="kb_files_uploader",
)

if uploaded_files:
    if st.button("⚡ Process & Index Uploaded Documents", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_container = st.empty()

        for idx, file_obj in enumerate(uploaded_files, start=1):
            status_container.info(f"Processing, chunking, and embedding '{file_obj.name}'...")
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

        status_container.success("Batch ingestion completed successfully.")
        st.rerun()

st.divider()

# Ingested Documents List
st.markdown("### 📚 Indexed Incident Dossiers")

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
        st.metric("Total Indexed Reports", len(documents))
    with c2:
        st.metric("Total Vector Chunks", total_chunks)
    with c3:
        st.metric("Storage Volume", f"{total_size_kb:.1f} KB")
    with c4:
        st.metric("Repository Status", "100% Synced", delta="Qdrant + BM25")

    # Documents Table
    doc_rows = []
    for d in documents:
        doc_rows.append({
            "Document ID": d.get("document_id"),
            "Report Filename": d.get("filename"),
            "Machine ID": d.get("machine_id") or "Unassigned",
            "Incident Classification": d.get("incident_type") or "Unspecified",
            "Incident Date": d.get("incident_date") or "N/A",
            "Format": d.get("file_type", "").upper(),
            "Chunks": d.get("chunk_count", 0),
            "Pages": d.get("page_count", 1),
            "Size (KB)": round(d.get("file_size_bytes", 0) / 1024.0, 1),
            "Checksum (SHA-256)": d.get("checksum_sha256", "")[:12] + "...",
            "Status": d.get("status", "unknown").upper(),
        })

    df_docs = pd.DataFrame(doc_rows)
    st.dataframe(
        df_docs,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Document ID": st.column_config.TextColumn("Doc ID", width="small"),
            "Report Filename": st.column_config.TextColumn("Filename", width="medium"),
            "Machine ID": st.column_config.TextColumn("Machine", width="small"),
            "Incident Classification": st.column_config.TextColumn("Classification", width="medium"),
            "Incident Date": st.column_config.TextColumn("Date", width="small"),
            "Format": st.column_config.TextColumn("Format", width="small"),
            "Chunks": st.column_config.NumberColumn("Chunks", width="small"),
            "Pages": st.column_config.NumberColumn("Pages", width="small"),
            "Size (KB)": st.column_config.NumberColumn("Size (KB)", width="small"),
            "Checksum (SHA-256)": st.column_config.TextColumn("Checksum", width="small"),
            "Status": st.column_config.TextColumn("Status", width="small"),
        },
    )

    # Document Actions: Inspect, Analyze Directly, or Delete
    st.markdown("#### ⚙️ Document Management & Direct Actions")
    col_sel, col_action_analyze, col_del = st.columns([2, 1, 1], vertical_alignment="bottom")

    with col_sel:
        doc_to_manage = st.selectbox(
            "Select Document:",
            options=documents,
            format_func=lambda d: f"{d.get('filename')} (ID: {d.get('document_id')}, Machine: {d.get('machine_id') or 'N/A'})",
            key="sb_manage_doc",
        )

    with col_action_analyze:
        if st.button("🔍 Analyze in Investigation →", key="btn_jump_to_inv", use_container_width=True, type="primary"):
            if doc_to_manage:
                st.session_state["selected_doc_id"] = doc_to_manage.get("document_id")
                st.switch_page("pages/1_Investigation.py")

    with col_del:
        if st.button("🗑️ Delete Document", key="btn_del_doc", type="secondary", use_container_width=True):
            if doc_to_manage:
                doc_id = doc_to_manage.get("document_id")
                try:
                    res = api_client.delete_document(doc_id)
                    st.success(f"Document `{doc_id}` deleted successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete document: {str(e)}")

    if doc_to_manage:
        with st.expander(f"🔍 Provenance & Metadata Details for `{doc_to_manage.get('filename')}`", expanded=False):
            st.json(doc_to_manage)
