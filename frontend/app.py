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

# Fetch System Connectivity & Health Status (Real backend data)
conn_info = api_client.get_connection_info()
online = conn_info["online"]
health = conn_info["data"]
latency_ms = conn_info["latency_ms"]
base_url = conn_info["base_url"]

if not online:
    st.markdown(
        f"""
        <div class="mismatch-banner" style="margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <strong style="font-size: 14px;">⚠️ FastAPI Backend is Offline or Sleeping (Cold Start)</strong>
                    <div style="font-size: 12.5px; margin-top: 4px;">
                        Target Host: <code>{base_url}</code><br/>
                        On Render's Free Tier, services spin down after 15 minutes of inactivity and take 30–50s to complete cold start.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_wake, _ = st.columns([1, 3])
    with col_wake:
        if st.button("🔄 Wake Up / Reconnect Backend", key="btn_wake_dash", type="primary", use_container_width=True):
            api_client.clear_cache()
            with st.spinner("Pinging FastAPI backend at " + base_url + "..."):
                try:
                    api_client.check_health(ttl=0.0)
                    st.toast("Backend responded successfully!", icon="✅")
                except Exception as ex:
                    st.toast(f"Connection attempt: {str(ex)[:80]}", icon="⏳")
            st.rerun()
else:
    st.markdown(
        f"""
        <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 6px; padding: 7px 14px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; font-size: 12.5px; color: #166534;">
            <span>🟢 <b>FastAPI Backend Connected:</b> <code>{base_url}</code> &bull; v{health.get('version', '1.0.0')} &bull; Env: <b>{health.get('environment', 'production').upper()}</b></span>
            <span>⚡ Health Latency: <b>{latency_ms:.1f} ms</b> &bull; RSS Memory: <b>{health.get('process_memory_mb', 0):.0f} MB</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Count completed investigations in session state
session_investigations = st.session_state.get("investigation_history", [])
completed_inv_count = len(session_investigations)

# Top Executive KPI Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label="System Status",
        value="Operational" if online else "Sleeping / Offline",
        delta=f"{latency_ms:.1f}ms latency" if online else "Connection Pending",
        delta_color="normal" if online else "inverse",
    )
with col2:
    if online:
        doc_count = health.get("indexed_documents", 0)
        st.metric(
            label="Indexed Documents",
            value=doc_count,
            delta="Active in Repository" if doc_count > 0 else "0 Documents Ingested",
            help="Number of active incident reports in the knowledge base repository",
        )
    else:
        st.metric(
            label="Indexed Documents",
            value="—",
            delta="Backend Offline",
            delta_color="inverse",
            help="Backend unreachable; data cannot be retrieved",
        )
with col3:
    if online:
        chunk_count = health.get("indexed_chunks", 0)
        st.metric(
            label="Total Chunks",
            value=chunk_count,
            delta="Vector Embeddings Indexed" if chunk_count > 0 else "0 Chunks in Qdrant",
            help="Embeddings indexed in vector database",
        )
    else:
        st.metric(
            label="Total Chunks",
            value="—",
            delta="Backend Offline",
            delta_color="inverse",
            help="Backend unreachable; data cannot be retrieved",
        )
with col4:
    st.metric(
        label="Active Session Investigations",
        value=completed_inv_count,
        delta=f"{completed_inv_count} Run(s) Recorded" if completed_inv_count > 0 else "Ready for Inquiries",
        help="Investigations executed during the active user session",
    )

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

# Sub-system / Pipeline Infrastructure Status
st.markdown("##### 🏗️ Pipeline Infrastructure & Models")
p1, p2, p3, p4 = st.columns(4)
with p1:
    vdb_status = "Online (Persistent)" if online else "Unavailable"
    vdb_color = "#16A34A" if online else "#DC2626"
    st.markdown(
        f"""
        <div class="stat-card" style="padding: 10px 14px;">
            <span class="label">Vector Database</span>
            <span class="value" style="font-size: 15px; color: {vdb_color}; font-weight: 700;">Qdrant</span>
            <span class="delta">{vdb_status} &bull; 384-Dim Cosine</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with p2:
    groq_ok = health.get("groq_configured", False)
    llm_color = "#16A34A" if groq_ok else "#D97706"
    llm_label = "Groq LLaMA 3.3" if groq_ok else "Key Required"
    st.markdown(
        f"""
        <div class="stat-card" style="padding: 10px 14px;">
            <span class="label">LLM Engine</span>
            <span class="value" style="font-size: 15px; color: {llm_color}; font-weight: 700;">{llm_label}</span>
            <span class="delta">{"70B Versatile Cloud" if groq_ok else "Fallback Synthesis"}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with p3:
    enc_loaded = health.get("embedding_model_loaded", False)
    enc_color = "#2563EB" if enc_loaded else "#64748B"
    st.markdown(
        f"""
        <div class="stat-card" style="padding: 10px 14px;">
            <span class="label">Dense Encoder</span>
            <span class="value" style="font-size: 14px; color: {enc_color}; font-weight: 700;">all-MiniLM-L6-v2</span>
            <span class="delta">{"Loaded in Memory" if enc_loaded else "Lazy-Loaded (Standby)"}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with p4:
    rerank_on = health.get("reranker_enabled", True)
    rerank_loaded = health.get("reranker_model_loaded", False)
    rerank_color = "#7C3AED" if rerank_on else "#94A3B8"
    st.markdown(
        f"""
        <div class="stat-card" style="padding: 10px 14px;">
            <span class="label">Cross-Encoder Reranker</span>
            <span class="value" style="font-size: 14px; color: {rerank_color}; font-weight: 700;">ms-marco-MiniLM</span>
            <span class="delta">{"Active & Loaded" if rerank_loaded else ("Enabled (Lazy-Loaded)" if rerank_on else "Disabled (Free Tier)")}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Quick Actions Section
st.markdown("### ⚡ Operational Workflows")
q1, q2, q3, q4 = st.columns(4)

with q1:
    with st.container(border=True):
        st.markdown(
            """
            <div style="min-height: 80px;">
                <div style="font-weight: 700; color: #0F1E36; font-size: 14px; margin-bottom: 4px;">📤 Knowledge Base</div>
                <div style="font-size: 12px; color: #64748B; line-height: 1.45;">
                    Upload technical reports (.pdf, .md, .txt), filter documents, preview indexed chunks, and manage dossiers.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Knowledge Base →", key="btn_quick_upload", use_container_width=True, type="primary"):
            st.switch_page("pages/4_Knowledge_Base.py")

with q2:
    with st.container(border=True):
        st.markdown(
            """
            <div style="min-height: 80px;">
                <div style="font-weight: 700; color: #0F1E36; font-size: 14px; margin-bottom: 4px;">🛠️ Investigation Workspace</div>
                <div style="font-size: 12px; color: #64748B; line-height: 1.45;">
                    Dual forensic workflows: single-dossier root cause analysis or cross-corpus evidence Q&A with mismatch validation.
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
            <div style="min-height: 80px;">
                <div style="font-weight: 700; color: #0F1E36; font-size: 14px; margin-bottom: 4px;">🔬 Retrieval Explorer</div>
                <div style="font-size: 12px; color: #64748B; line-height: 1.45;">
                    Inspect candidate scoring stage-by-stage across BM25, Dense vectors, RRF, and Cross-Encoder layers.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Explore Pipeline →", key="btn_quick_retrieval", use_container_width=True):
            st.switch_page("pages/2_Retrieval_Explorer.py")

with q4:
    with st.container(border=True):
        st.markdown(
            """
            <div style="min-height: 80px;">
                <div style="font-weight: 700; color: #0F1E36; font-size: 14px; margin-bottom: 4px;">📊 Evaluate RAG</div>
                <div style="font-size: 12px; color: #64748B; line-height: 1.45;">
                    Run empirical benchmarks comparing Recall@K, Precision@K, MRR, nDCG@10, and latencies.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Run Benchmarks →", key="btn_quick_eval", use_container_width=True):
            st.switch_page("pages/3_RAG_Evaluation.py")

st.markdown("---")

# Recent Investigations (Session data if available)
st.markdown("### 🕒 Recent Investigations (Active Session)")
if session_investigations:
    inv_rows = []
    for inv in session_investigations[-8:]:
        inv_rows.append({
            "Target / Query": inv.get("query_or_doc", "Investigation"),
            "Mode": inv.get("mode", "Analysis"),
            "Machine ID": inv.get("machine_id", "N/A"),
            "Timestamp": inv.get("timestamp", "Just now"),
            "Evidence Grounded": "✅ Yes" if inv.get("is_sufficient", True) else "⚠️ Mismatch / Insufficient",
        })
    df_inv = pd.DataFrame(inv_rows)
    st.dataframe(df_inv, use_container_width=True, hide_index=True)
    st.caption("💡 To reopen or export full diagnostic reports from past investigations, visit the **Investigation Workspace**.")
else:
    st.markdown(
        """
        <div style="background: #F8FAFC; border: 1px dashed #CBD5E1; border-radius: 6px; padding: 18px; text-align: center; color: #64748B; font-size: 13px;">
            No investigations executed in this session yet. Click <b>Start Investigation</b> above to analyze an incident report or ask a technical question.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Ingested Incident Reports Table (Real data from knowledge base)
st.markdown("### 📚 Knowledge Base Incident Dossiers")
try:
    documents = api_client.list_documents() if online else []
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
elif online:
    st.info("No incident reports found in the knowledge base. Ingest sample documents from the **Knowledge Base** page.")
else:
    st.warning("Connect to the backend to view indexed incident dossiers.")

st.info(
    "💡 **Quick Navigation:** Open **Investigation Workspace** to directly analyze incident documents "
    "like `INC-2023-044_conveyor_motor_overheating.md` or submit evidence-grounded queries."
)
