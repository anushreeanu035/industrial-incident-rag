from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
from frontend.components.api_client import api_client
from frontend.components.sidebar import render_sidebar
from frontend.components.theme import load_theme

st.set_page_config(
    page_title="Settings & System Diagnostics - Incident Investigator",
    page_icon="⚙️",
    layout="wide",
)

# Load cached theme CSS
load_theme()

render_sidebar(current_page="settings")

st.markdown(
    """
    <div class="incident-header-banner">
        <h1>⚙️ System & Pipeline Architecture Settings</h1>
        <p>Active Model Configurations, Fusion Hyperparameters, Storage Verifications, and Diagnostics</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Check backend health
try:
    health = api_client.health_check()
    is_healthy = health.get("status") == "healthy"
except Exception:
    health = {}
    is_healthy = False

try:
    settings_data = api_client.get_settings()
except Exception as e:
    settings_data = {}
    if not is_healthy:
        st.warning(
            f"⚠️ **FastAPI Backend is unreachable at `{api_client.base_url}`.**\n\n"
            "Please ensure the local FastAPI backend is running via:\n`uvicorn incident_rag.api:app --host 127.0.0.1 --port 8000 --reload`"
        )
    else:
        st.error(f"Failed to fetch system settings from backend: {str(e)}")

# System Status Overview
st.markdown("### 🖥️ Backend Service Status")
c1, c2, c3, c4 = st.columns(4)
with c1:
    status_color = "#16A34A" if is_healthy else "#DC2626"
    status_label = "ONLINE / READY" if is_healthy else "OFFLINE"
    display_url = api_client.base_url
    if len(display_url) > 28:
        display_url = display_url[:25] + "..."
    st.markdown(
        f"""
        <div class="stat-card">
            <span class="label">FastAPI Backend</span>
            <span class="value" style="color: {status_color}; font-size: 19px;">{status_label}</span>
            <span class="delta" title="{api_client.base_url}">{display_url}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
        <div class="stat-card">
            <span class="label">Dense Vector Index</span>
            <span class="value" style="color: #2563EB;">Qdrant</span>
            <span class="delta">384-Dim &bull; Cosine Distance</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""
        <div class="stat-card">
            <span class="label">Sparse Keyword Search</span>
            <span class="value" style="color: #0D9488;">BM25 Okapi</span>
            <span class="delta">Rank-BM25 Tokenized</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f"""
        <div class="stat-card">
            <span class="label">Fusion Algorithm</span>
            <span class="value" style="color: #7C3AED;">RRF + Cross-Enc</span>
            <span class="delta">k = {settings_data.get('rrf_k', 60)} &bull; ms-marco</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.expander("🔗 Backend Service Connection Settings", expanded=not is_healthy):
    st.markdown(
        "<p style='font-size: 13px; color: #64748B; margin-bottom: 8px;'>"
        "Configure or test your local FastAPI backend host. Default is <code>http://127.0.0.1:8000</code>."
        "</p>",
        unsafe_allow_html=True,
    )
    col_u1, col_u2 = st.columns([3, 1])
    with col_u1:
        new_url_input = st.text_input(
            "Target FastAPI Backend Host:",
            value=api_client.base_url,
            help="Local FastAPI backend host (default: http://127.0.0.1:8000)",
            key="input_backend_url_settings",
        )
    with col_u2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Connect & Test →", key="btn_apply_backend_url", type="primary", use_container_width=True):
            cleaned_url = new_url_input.strip().rstrip("/")
            st.session_state["backend_url_override"] = cleaned_url
            api_client.base_url = cleaned_url
            api_client.clear_cache()
            st.rerun()

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# Architecture Overview Cards
st.markdown("### 🤖 Neural & Language Model Architecture")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        f"""
        <div class="industrial-card" style="border-top: 3px solid #2563EB; min-height: 175px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <h4 style="margin: 0; color: #0F1E36; font-size: 15px; font-weight: 700;">Dense Embeddings</h4>
                <span class="capability-tag" style="background: #EFF6FF; color: #2563EB; border-color: #BFDBFE;">384-DIM</span>
            </div>
            <p style="font-size: 14.5px; font-weight: 700; color: #2563EB; margin: 4px 0 0 0; font-family: monospace;">
                {settings_data.get("embedding_model", "all-MiniLM-L6-v2")}
            </p>
            <p style="font-size: 12.5px; color: #64748B; margin-top: 8px; line-height: 1.45; margin-bottom: 0;">
                Normalized dense vectors via Sentence Transformers. Runs local CPU/GPU similarity search across industrial document chunks.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""
        <div class="industrial-card" style="border-top: 3px solid #0D9488; min-height: 175px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <h4 style="margin: 0; color: #0F1E36; font-size: 15px; font-weight: 700;">Cross-Encoder Reranker</h4>
                <span class="capability-tag" style="background: #F0FDF4; color: #0D9488; border-color: #99F6E4;">RERANKER</span>
            </div>
            <p style="font-size: 14.5px; font-weight: 700; color: #0D9488; margin: 4px 0 0 0; font-family: monospace;">
                {settings_data.get("reranker_model", "ms-marco-MiniLM-L-6-v2")}
            </p>
            <p style="font-size: 12.5px; color: #64748B; margin-top: 8px; line-height: 1.45; margin-bottom: 0;">
                Joint cross-attentive relevance scoring over query-passage candidate pairs to filter false-positive keyword or semantic matches.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"""
        <div class="industrial-card" style="border-top: 3px solid #7C3AED; min-height: 175px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <h4 style="margin: 0; color: #0F1E36; font-size: 15px; font-weight: 700;">Inference & Rewriting LLM</h4>
                <span class="capability-tag" style="background: #FAF5FF; color: #7C3AED; border-color: #E9D5FF;">CLOUD LLM</span>
            </div>
            <p style="font-size: 14.5px; font-weight: 700; color: #7C3AED; margin: 4px 0 0 0; font-family: monospace;">
                {settings_data.get("groq_model", "llama-3.3-70b-versatile")}
            </p>
            <p style="font-size: 12.5px; color: #64748B; margin-top: 8px; line-height: 1.45; margin-bottom: 0;">
                Groq Cloud ultra-fast inference for multi-query rewriting, forensic report generation, and automated fallback offline synthesis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# Hyperparameter Tunables
st.markdown("### 🎛️ Pipeline Hyperparameters")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown(
        """
        <div class="industrial-card" style="padding: 16px 20px; margin-bottom: 16px;">
            <h4 style="margin-top: 0; color: #0F1E36; font-size: 15px; font-weight: 700; border-bottom: 1px solid #E2E8F0; padding-bottom: 8px;">
                🔍 Retrieval & Fusion Parameters
            </h4>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.number_input(
        "Dense Top-K Candidates:",
        value=int(settings_data.get("dense_top_k", 10)),
        disabled=True,
        help="Configured in .env via DENSE_TOP_K",
    )
    st.number_input(
        "BM25 Top-K Candidates:",
        value=int(settings_data.get("bm25_top_k", 10)),
        disabled=True,
        help="Configured in .env via BM25_TOP_K",
    )
    st.number_input(
        "Reciprocal Rank Fusion k Constant:",
        value=int(settings_data.get("rrf_k", 60)),
        disabled=True,
        help="Formula: RRF(d) = Σ 1 / (k + rank_i(d)). Configured via RRF_K.",
    )
    st.number_input(
        "Cross-Encoder Candidate Pool Size:",
        value=int(settings_data.get("rerank_top_k", 5)),
        disabled=True,
        help="Configured in .env via RERANK_TOP_K",
    )

with col_b:
    st.markdown(
        """
        <div class="industrial-card" style="padding: 16px 20px; margin-bottom: 16px;">
            <h4 style="margin-top: 0; color: #0F1E36; font-size: 15px; font-weight: 700; border-bottom: 1px solid #E2E8F0; padding-bottom: 8px;">
                📑 Chunking & Query Expansion
            </h4>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.number_input(
        "Chunk Word Size:",
        value=int(settings_data.get("chunk_size", 400)),
        disabled=True,
        help="Target length in words/tokens for text chunking.",
    )
    st.number_input(
        "Chunk Word Overlap:",
        value=int(settings_data.get("chunk_overlap", 80)),
        disabled=True,
        help="Number of words overlapping between adjacent chunks to maintain context continuity.",
    )
    st.checkbox(
        "Query Rewriting Active by Default",
        value=bool(settings_data.get("query_rewriting_enabled", True)),
        disabled=True,
    )
    st.number_input(
        "Query Rewriting Variant Count:",
        value=int(settings_data.get("num_query_rewrites", 3)),
        disabled=True,
    )

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# Environment & Storage Audit
st.markdown("### 🔒 Storage & Environment Verification")

is_remote = not ("127.0.0.1" in api_client.base_url or "localhost" in api_client.base_url)

if is_remote and is_healthy:
    chunks_count = health.get("indexed_chunks", 0)
    docs_count = health.get("indexed_documents", 0)
    qdrant_exists = is_healthy
    bm25_exists = is_healthy
    docs_exists = is_healthy
    qdrant_loc = f"Backend Persistent DB ({chunks_count} chunks)"
    bm25_loc = "Backend BM25 Corpus Index (Active)"
    docs_loc = f"Backend Registry ({docs_count} dossiers)"
else:
    qdrant_path = _PROJECT_ROOT / "data" / "processed" / "qdrant_db"
    bm25_path = _PROJECT_ROOT / "data" / "processed" / "bm25_index.pkl"
    docs_path = _PROJECT_ROOT / "data" / "processed" / "documents.json"
    qdrant_exists = qdrant_path.exists()
    bm25_exists = bm25_path.exists()
    docs_exists = docs_path.exists()
    qdrant_loc = "data/processed/qdrant_db/"
    bm25_loc = "data/processed/bm25_index.pkl"
    docs_loc = "data/processed/documents.json"

st.markdown(
    f"""
    <div class="industrial-card" style="padding: 20px;">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px;">
            <div style="border-left: 3px solid {'#16A34A' if qdrant_exists else '#DC2626'}; padding-left: 12px;">
                <div style="font-weight: 700; color: #0F1E36; font-size: 14px;">Qdrant Vector Database</div>
                <div style="font-size: 12px; color: #64748B; font-family: monospace; margin: 2px 0;">{qdrant_loc}</div>
                <div style="font-size: 12px; font-weight: 600; color: {'#16A34A' if qdrant_exists else '#DC2626'};">
                    {'✓ Initialized & Verified' if qdrant_exists else '✗ Not Initialized'}
                </div>
            </div>
            <div style="border-left: 3px solid {'#16A34A' if bm25_exists else '#DC2626'}; padding-left: 12px;">
                <div style="font-weight: 700; color: #0F1E36; font-size: 14px;">BM25 Keyword Index</div>
                <div style="font-size: 12px; color: #64748B; font-family: monospace; margin: 2px 0;">{bm25_loc}</div>
                <div style="font-size: 12px; font-weight: 600; color: {'#16A34A' if bm25_exists else '#DC2626'};">
                    {'✓ Persisted & Loaded' if bm25_exists else '✗ Not Found'}
                </div>
            </div>
            <div style="border-left: 3px solid {'#16A34A' if docs_exists else '#DC2626'}; padding-left: 12px;">
                <div style="font-weight: 700; color: #0F1E36; font-size: 14px;">Document Metadata Registry</div>
                <div style="font-size: 12px; color: #64748B; font-family: monospace; margin: 2px 0;">{docs_loc}</div>
                <div style="font-size: 12px; font-weight: 600; color: {'#16A34A' if docs_exists else '#DC2626'};">
                    {'✓ Metadata Registry Active' if docs_exists else '✗ Missing'}
                </div>
            </div>
        </div>
        <div style="margin-top: 16px; padding-top: 12px; border-top: 1px solid #E2E8F0; font-size: 12px; color: #64748B; display: flex; justify-content: space-between; align-items: center;">
            <span><strong>Security Isolation:</strong> GROQ_API_KEY and environment credentials are restricted to the backend and never passed to the browser client.</span>
            <span style="font-size: 11px; background: #F1F5F9; padding: 3px 8px; border-radius: 4px; font-weight: 600;">ENTERPRISE COMPLIANT</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
