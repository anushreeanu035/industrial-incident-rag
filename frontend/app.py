from pathlib import Path
import sys

# Ensure project root is in sys.path regardless of how Streamlit is launched
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
from frontend.components.api_client import api_client
from frontend.components.sidebar import render_sidebar

st.set_page_config(
    page_title="Industrial Incident Investigator",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load theme CSS
css_path = Path(__file__).resolve().parent / "styles" / "theme.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Render Global Sidebar
render_sidebar()

# Main Dashboard Banner
st.markdown(
    """
    <div class="incident-header-banner">
        <h1>⚙️ Industrial Incident Investigator</h1>
        <p>Advanced Full-Stack RAG Platform for Equipment Failures, Root Cause Analysis & Forensic Auditing</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Quick System Status & Health Metrics
try:
    health = api_client.check_health()
    online = True
except Exception as e:
    health = {}
    online = False

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label="System Status",
        value="Operational" if online else "Offline",
        delta="API Online" if online else "Backend Offline",
    )
with col2:
    st.metric(
        label="Indexed Documents",
        value=health.get("indexed_documents", 0),
        help="Number of active incident reports in the knowledge base",
    )
with col3:
    st.metric(
        label="Vector Chunks",
        value=health.get("indexed_chunks", 0),
        help="Chunks indexed in local Qdrant vector database",
    )
with col4:
    groq_active = health.get("groq_configured", False)
    st.metric(
        label="AI Synthesis Engine",
        value="Groq Cloud" if groq_active else "Extractive",
        help="LLM provider for query rewriting and evidence-grounded synthesis",
    )

st.markdown("---")

st.markdown("### 🔍 Platform Capabilities")

# Row 1: Investigation Workspace and Retrieval Explorer
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    with st.container(border=True):
        st.markdown(
            """
            <div class="capability-header">
                <div class="capability-title">🛠️ 1. Investigation Workspace</div>
                <span class="capability-tag">Forensic RCA</span>
            </div>
            <p class="capability-desc">
                Perform in-depth forensic queries with machine ID and incident filters. Receive evidence-grounded executive summaries, root cause hypotheses with confidence ratings, and fully auditable passage citations.
            </p>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Investigation Workspace →", key="btn_nav_investigation", use_container_width=True, type="primary"):
            st.switch_page("pages/1_Investigation.py")
        st.page_link("pages/1_Investigation.py", label="Direct Link: Investigation Workspace", icon="🛠️", use_container_width=True)

with row1_col2:
    with st.container(border=True):
        st.markdown(
            """
            <div class="capability-header">
                <div class="capability-title">🔬 2. Retrieval Explorer</div>
                <span class="capability-tag">Inspection</span>
            </div>
            <p class="capability-desc">
                Inspect pipeline behavior stage-by-stage. View original and rewritten queries, candidate rankings across BM25 keyword search, dense vector retrieval, reciprocal rank fusion (RRF), and cross-encoder reranking.
            </p>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Launch Retrieval Explorer →", key="btn_nav_retrieval", use_container_width=True, type="primary"):
            st.switch_page("pages/2_Retrieval_Explorer.py")
        st.page_link("pages/2_Retrieval_Explorer.py", label="Direct Link: Retrieval Explorer", icon="🔬", use_container_width=True)

# Row 2: RAG Evaluation and Knowledge Base
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    with st.container(border=True):
        st.markdown(
            """
            <div class="capability-header">
                <div class="capability-title">📊 3. RAG Evaluation Dashboard</div>
                <span class="capability-tag">Benchmarking</span>
            </div>
            <p class="capability-desc">
                Benchmark retrieval strategies against deterministic ground-truth incident datasets. Compare Recall@5, Recall@10, Precision@K, MRR, nDCG@10, and latencies across Dense, BM25, Hybrid RRF, and Cross-Encoder reranking.
            </p>
            """,
            unsafe_allow_html=True,
        )
        if st.button("View Evaluation Dashboard →", key="btn_nav_evaluation", use_container_width=True, type="primary"):
            st.switch_page("pages/3_RAG_Evaluation.py")
        st.page_link("pages/3_RAG_Evaluation.py", label="Direct Link: RAG Evaluation Dashboard", icon="📊", use_container_width=True)

with row2_col2:
    with st.container(border=True):
        st.markdown(
            """
            <div class="capability-header">
                <div class="capability-title">📁 4. Knowledge Base Management</div>
                <span class="capability-tag">Repository</span>
            </div>
            <p class="capability-desc">
                Upload, manage, and inspect incident reports (PDF, Markdown, TXT). View chunk counts, metadata tags, SHA-256 deduplication checksums, and trigger re-indexing or deletion.
            </p>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Manage Knowledge Base →", key="btn_nav_knowledge_base", use_container_width=True, type="primary"):
            st.switch_page("pages/4_Knowledge_Base.py")
        st.page_link("pages/4_Knowledge_Base.py", label="Direct Link: Knowledge Base Management", icon="📁", use_container_width=True)

st.info(
    "💡 **Quick Start:** Navigate to **Investigation Workspace** to ask questions about high-pressure steam turbine HP-17, reactor RX-404, or compressor K-102."
)
