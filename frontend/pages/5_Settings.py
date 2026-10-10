from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
from frontend.components.api_client import api_client
from frontend.components.sidebar import render_sidebar

st.set_page_config(
    page_title="Settings - Incident Investigator",
    page_icon="⚙️",
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
        <h1>⚙️ System & Pipeline Settings</h1>
        <p>Active Model Architectures, Fusion Hyperparameters, and Ingestion Tunables</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    settings_data = api_client.get_settings()
except Exception as e:
    settings_data = {}
    st.error(f"Failed to fetch system settings from backend: {str(e)}")

# Architecture Overview Cards
st.markdown("### 🤖 Active Model Configuration")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        f"""
        <div class="industrial-card">
            <h4 style="margin-top: 0; color: #0F1E36;">Dense Embeddings</h4>
            <p style="font-size: 0.95rem; font-weight: 600; color: #1E88E5; margin: 0;">
                {settings_data.get("embedding_model", "all-MiniLM-L6-v2")}
            </p>
            <p style="font-size: 0.8rem; color: #64748B; margin-top: 0.5rem;">
                384-dimensional normalized vectors via Sentence Transformers. Local CPU/GPU inference.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
        <div class="industrial-card">
            <h4 style="margin-top: 0; color: #0F1E36;">Cross-Encoder Reranker</h4>
            <p style="font-size: 0.95rem; font-weight: 600; color: #00897B; margin: 0;">
                {settings_data.get("reranker_model", "ms-marco-MiniLM-L-6-v2")}
            </p>
            <p style="font-size: 0.8rem; color: #64748B; margin-top: 0.5rem;">
                Cross-attentive relevance re-scoring over (query, passage) pairs.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""
        <div class="industrial-card">
            <h4 style="margin-top: 0; color: #0F1E36;">Inference & Rewriting LLM</h4>
            <p style="font-size: 0.95rem; font-weight: 600; color: #7C3AED; margin: 0;">
                {settings_data.get("groq_model", "llama-3.3-70b-versatile")}
            </p>
            <p style="font-size: 0.8rem; color: #64748B; margin-top: 0.5rem;">
                Groq Cloud ultra-fast inference with fallback extractive offline synthesis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# Hyperparameter Tunables
st.markdown("### 🎛️ Pipeline Hyperparameters")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("#### Retrieval & Fusion Parameters")
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
    st.markdown("#### Chunking & Query Rewriting")
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
        help="Number of words overlapping between adjacent chunks.",
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

st.markdown("---")
st.markdown("### 🔒 Environment & Security Verification")
st.markdown(
    """
    - **API Keys**: Stored securely in `.env` and system environment. Never exposed to frontend clients.
    - **Local Storage**: Vectors stored in local file-backed Qdrant DB (`data/processed/qdrant_db/`).
    - **Keyword Index**: BM25 index persisted in `data/processed/bm25_index.pkl`.
    - **Metadata Store**: Document registry persisted in `data/processed/documents.json`.
    """
)
