from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from app import _ensure_backend_running
    _ensure_backend_running()
except Exception:
    pass


import math
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
        <p>Enterprise Repository: Ingestion, Structure-Aware Chunking, Metadata Extraction & Forensic Dossier Inspection</p>
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
    help="Files will be parsed, cleaned, split into structure-aware chunks, and indexed across Qdrant & BM25.",
)

if uploaded_files:
    if st.button("⚡ Process & Index Uploaded Documents", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_container = st.empty()

        for idx, file_obj in enumerate(uploaded_files, start=1):
            status_container.info(f"Processing, chunking, and embedding '{file_obj.name}'...")
            try:
                content_bytes = file_obj.getvalue()
                if not content_bytes:
                    st.error(f"❌ Upload Error: '{file_obj.name}' is empty.")
                    continue
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
        api_client.clear_cache("documents")
        api_client.clear_cache("health")
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
    st.info("No documents are currently indexed in the knowledge base. Upload incident reports above to get started.")
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
        st.metric("Repository Status", "Synchronized", delta="Qdrant + BM25")

    st.markdown("#### 🔍 Filter & Search Dossiers")
    col_search, col_mach_filter, col_sort = st.columns([2, 1, 1])

    with col_search:
        search_kw = st.text_input(
            "Search dossiers:",
            placeholder="Search by filename, machine ID, or classification...",
            key="kb_search_kw",
        ).strip().lower()

    with col_mach_filter:
        existing_machines = sorted(list({d.get("machine_id") or "Unassigned" for d in documents}))
        selected_filter_mach = st.selectbox(
            "Filter by Machine:",
            options=["All Machines"] + existing_machines,
            key="kb_filter_mach",
        )

    with col_sort:
        sort_choice = st.selectbox(
            "Sort By:",
            options=["Date (Newest First)", "Filename (A-Z)", "Chunks (Highest First)", "Size (Largest First)"],
            key="kb_sort_choice",
        )

    # Apply search and filtering
    filtered_docs = documents
    if search_kw:
        filtered_docs = [
            d for d in filtered_docs
            if search_kw in d.get("filename", "").lower()
            or search_kw in str(d.get("machine_id", "")).lower()
            or search_kw in str(d.get("incident_type", "")).lower()
            or search_kw in str(d.get("document_id", "")).lower()
        ]

    if selected_filter_mach != "All Machines":
        filtered_docs = [
            d for d in filtered_docs
            if (d.get("machine_id") or "Unassigned") == selected_filter_mach
        ]

    # Apply Sorting
    if sort_choice == "Date (Newest First)":
        filtered_docs.sort(key=lambda d: d.get("upload_timestamp", "") or "", reverse=True)
    elif sort_choice == "Filename (A-Z)":
        filtered_docs.sort(key=lambda d: d.get("filename", "").lower())
    elif sort_choice == "Chunks (Highest First)":
        filtered_docs.sort(key=lambda d: d.get("chunk_count", 0), reverse=True)
    elif sort_choice == "Size (Largest First)":
        filtered_docs.sort(key=lambda d: d.get("file_size_bytes", 0), reverse=True)

    # Pagination
    PAGE_SIZE = 8
    total_pages = max(1, math.ceil(len(filtered_docs) / PAGE_SIZE))
    curr_page = st.session_state.get("kb_curr_page", 1)
    if curr_page > total_pages:
        curr_page = 1
        st.session_state["kb_curr_page"] = 1

    start_idx = (curr_page - 1) * PAGE_SIZE
    end_idx = start_idx + PAGE_SIZE
    page_docs = filtered_docs[start_idx:end_idx]

    # Documents Table
    doc_rows = []
    for d in page_docs:
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
            "Checksum (SHA-256)": (d.get("checksum_sha256") or "")[:12] + "...",
            "Status": (d.get("status") or "INDEXED").upper(),
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

    # Pagination controls
    if total_pages > 1:
        col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
        with col_p1:
            if st.button("◀ Previous Page", disabled=(curr_page <= 1), key="btn_kb_prev"):
                st.session_state["kb_curr_page"] = curr_page - 1
                st.rerun()
        with col_p2:
            st.markdown(
                f"<div style='text-align: center; color: var(--text-secondary); font-size: 13px; line-height: 2.2;'>"
                f"Page <b>{curr_page}</b> of <b>{total_pages}</b> ({len(filtered_docs)} total matches)"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_p3:
            if st.button("Next Page ▶", disabled=(curr_page >= total_pages), key="btn_kb_next"):
                st.session_state["kb_curr_page"] = curr_page + 1
                st.rerun()

    # Document Actions: Inspect, Analyze Directly, or Delete
    st.markdown("#### ⚙️ Document Management & Direct Actions")
    col_sel, col_action_analyze = st.columns([3, 1], vertical_alignment="bottom")

    with col_sel:
        doc_to_manage = st.selectbox(
            "Select Document for Management & Inspection:",
            options=documents,
            format_func=lambda d: f"{d.get('filename')} (ID: {d.get('document_id')}, Machine: {d.get('machine_id') or 'N/A'}, Chunks: {d.get('chunk_count', 0)})",
            key="sb_manage_doc",
        )

    with col_action_analyze:
        if st.button("🔍 Analyze in Investigation →", key="btn_jump_to_inv", use_container_width=True, type="primary"):
            if doc_to_manage:
                st.session_state["selected_doc_id"] = doc_to_manage.get("document_id")
                st.switch_page("pages/1_Investigation.py")

    if doc_to_manage:
        doc_id = doc_to_manage.get("document_id")

        # Document Details & Preview Tab
        t_meta, t_preview, t_del = st.tabs(["📋 Metadata & Provenance", "🔍 Extracted Text & Chunks Preview", "🗑️ Delete Document"])

        with t_meta:
            st.json(doc_to_manage)

        with t_preview:
            st.caption(f"Showing indexed text chunks for `{doc_to_manage.get('filename')}` directly from Qdrant vector database.")
            try:
                chunks_data = api_client.get_document_chunks(doc_id, limit=50)
                if not chunks_data:
                    st.info("No chunk passages found for this document.")
                else:
                    st.markdown(f"**Total Chunks Available:** {len(chunks_data)}")
                    for c in chunks_data:
                        meta = c.get("metadata") or {}
                        chunk_idx = meta.get("chunk_index", 0)
                        page_num = meta.get("page_number")
                        page_str = f"Page {page_num}" if page_num else "Page N/A"
                        with st.expander(f"Chunk #{chunk_idx} [{page_str}] (Chars {meta.get('start_char', 0)}–{meta.get('end_char', 0)})", expanded=(chunk_idx == 0)):
                            st.markdown(f"```text\n{c.get('text', '')}\n```")
            except Exception as e:
                st.warning(f"Could not load chunk preview: {str(e)}")

        with t_del:
            st.warning("⚠️ **Caution**: Deleting this document removes all metadata, BM25 keywords, and vector embeddings permanently.")
            confirm_del = st.checkbox(f"Confirm permanent deletion of '{doc_to_manage.get('filename')}' (ID: `{doc_id}`)", key=f"chk_del_{doc_id}")
            if st.button("🗑️ Permanently Delete Dossier", key="btn_del_doc_confirmed", type="secondary", disabled=not confirm_del):
                try:
                    res = api_client.delete_document(doc_id)
                    st.success(f"Document `{doc_id}` deleted successfully.")
                    api_client.clear_cache("documents")
                    api_client.clear_cache("health")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete document: {str(e)}")
