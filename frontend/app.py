from pathlib import Path
import sys

# Ensure project root is in sys.path regardless of how Streamlit is launched
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st
from frontend.components.api_client import api_client
from frontend.components.sidebar import render_sidebar
from frontend.components.theme import load_theme

st.set_page_config(
    page_title="Industrial Incident Investigator",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load cached enterprise theme CSS
load_theme()

# Render Global Enterprise Sidebar with active page
render_sidebar(current_page="dashboard")

# Main Dashboard Banner with Industrial Badge
st.markdown(
    """
    <div class="incident-header-banner">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div>
                <h1>⚙️ Industrial Incident Investigator</h1>
                <p>Enterprise AI Platform for Root Cause Analysis, Multi-Stage Forensic Retrieval & Diagnostic Auditing</p>
            </div>
            <div>
                <span class="capability-tag" style="background: rgba(37, 99, 235, 0.25); color: #93c5fd; border-color: rgba(147, 197, 253, 0.4); font-size: 11.5px; padding: 4px 10px;">
                    HYBRID RAG • RRF • CROSS-ENCODER
                </span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Fetch System Status & Health Metrics Bar (Strictly using real backend data)
try:
    health = api_client.check_health()
    online = True
except Exception:
    health = {}
    online = False

# Count completed investigations in session state
session_investigations = st.session_state.get("investigation_history", [])
completed_inv_count = len(session_investigations)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label="System Status",
        value="Operational" if online else "Offline",
        delta="API Online" if online else "Backend Offline",
        delta_color="normal" if online else "inverse",
    )
with col2:
    st.metric(
        label="Indexed Documents",
        value=health.get("indexed_documents", 0),
        help="Number of active incident reports in the knowledge base repository",
    )
with col3:
    st.metric(
        label="Total Chunks",
        value=health.get("indexed_chunks", 0),
        help="Embeddings indexed in local persistent vector database",
    )
with col4:
    st.metric(
        label="Completed Investigations",
        value=completed_inv_count,
        delta=f"{completed_inv_count} Active Session Run(s)" if completed_inv_count > 0 else "Ready for Inquiries",
        help="Investigations executed during active session",
    )

st.markdown("---")

# Quick Actions Section
st.markdown("### ⚡ Quick Actions")
q1, q2, q3, q4 = st.columns(4)

with q1:
    with st.container(border=True):
        st.markdown(
            """
            <div style="min-height: 90px;">
                <div style="font-weight: 700; color: #0F1E36; font-size: 14.5px; margin-bottom: 4px;">📤 Upload Documents</div>
                <div style="font-size: 12.5px; color: #64748B; line-height: 1.45;">
                    Ingest technical incident reports (.pdf, .md, .txt) with automated structure-aware chunking.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Upload Reports →", key="btn_quick_upload", use_container_width=True, type="primary"):
            st.switch_page("pages/4_Knowledge_Base.py")

with q2:
    with st.container(border=True):
        st.markdown(
            """
            <div style="min-height: 90px;">
                <div style="font-weight: 700; color: #0F1E36; font-size: 14.5px; margin-bottom: 4px;">🛠️ Start Investigation</div>
                <div style="font-size: 12.5px; color: #64748B; line-height: 1.45;">
                    Directly analyze selected reports or submit forensic questions with equipment mismatch validation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Start Investigation →", key="btn_quick_investigate", use_container_width=True, type="primary"):
            st.switch_page("pages/1_Investigation.py")

with q3:
    with st.container(border=True):
        st.markdown(
            """
            <div style="min-height: 90px;">
                <div style="font-weight: 700; color: #0F1E36; font-size: 14.5px; margin-bottom: 4px;">🔬 Retrieval Explorer</div>
                <div style="font-size: 12.5px; color: #64748B; line-height: 1.45;">
                    Inspect candidate scoring stage-by-stage across BM25, Dense vectors, RRF, and Cross-Encoder.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Explore Retrieval →", key="btn_quick_retrieval", use_container_width=True):
            st.switch_page("pages/2_Retrieval_Explorer.py")

with q4:
    with st.container(border=True):
        st.markdown(
            """
            <div style="min-height: 90px;">
                <div style="font-weight: 700; color: #0F1E36; font-size: 14.5px; margin-bottom: 4px;">📊 Evaluate RAG</div>
                <div style="font-size: 12.5px; color: #64748B; line-height: 1.45;">
                    Benchmark retrieval strategies measuring Recall@K, Precision@K, MRR, nDCG@10, and latencies.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Evaluate Pipeline →", key="btn_quick_eval", use_container_width=True):
            st.switch_page("pages/3_RAG_Evaluation.py")

st.markdown("---")

# Recent Investigations (Session data if available)
if session_investigations:
    st.markdown("### 🕒 Recent Investigations (Active Session)")
    inv_rows = []
    for inv in session_investigations[-5:]:
        inv_rows.append({
            "Target / Query": inv.get("query_or_doc", "Investigation"),
            "Mode": inv.get("mode", "Analysis"),
            "Machine ID": inv.get("machine_id", "N/A"),
            "Timestamp": inv.get("timestamp", "Just now"),
            "Evidence Grounded": "✅ Yes" if inv.get("is_sufficient", True) else "⚠️ Mismatch / Insufficient",
        })
    df_inv = pd.DataFrame(inv_rows)
    st.dataframe(df_inv, use_container_width=True, hide_index=True)
    st.markdown("---")

# Ingested Incident Reports Table (Real data from knowledge base)
st.markdown("### 📚 Knowledge Base Incident Dossiers")
try:
    documents = api_client.list_documents()
except Exception:
    documents = []

if documents:
    doc_table_rows = []
    for d in documents:
        doc_table_rows.append({
            "Report Filename": d.get("filename"),
            "Machine ID": d.get("machine_id") or "Unassigned",
            "Incident Classification": d.get("incident_type") or "Report",
            "Incident Date": d.get("incident_date") or "N/A",
            "Chunks": d.get("chunk_count", 0),
            "Format": d.get("file_type", "").upper(),
            "Doc ID": d.get("document_id"),
        })
    df_preview = pd.DataFrame(doc_table_rows)
    st.dataframe(
        df_preview,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Report Filename": st.column_config.TextColumn("Report Filename", width="medium"),
            "Machine ID": st.column_config.TextColumn("Machine ID", width="small"),
            "Incident Classification": st.column_config.TextColumn("Incident Classification", width="medium"),
            "Incident Date": st.column_config.TextColumn("Date", width="small"),
            "Chunks": st.column_config.NumberColumn("Chunks", width="small"),
            "Format": st.column_config.TextColumn("Format", width="small"),
            "Doc ID": st.column_config.TextColumn("Document ID", width="small"),
        },
    )
else:
    st.info("No incident reports found in the knowledge base. Ingest sample documents from the Knowledge Base page.")

st.info(
    "💡 **Quick Navigation:** Open **Investigation Workspace** to directly analyze incident documents "
    "like `INC-2023-044_conveyor_motor_overheating.md` or submit evidence-grounded queries."
)
