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
    page_title="Retrieval Explorer - Incident Investigator",
    page_icon="🔬",
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
        <h1>🔬 Retrieval Explorer</h1>
        <p>Stage-by-Stage Forensic Inspection: Query Rewriting, BM25, Dense Embeddings, RRF, and Cross-Encoder</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p style="color: #475569; font-size: 14px; line-height: 1.55; margin-bottom: 0.75rem;">
        Examine how candidate evidence passages evolve through each stage of the RAG pipeline.
        Observe exact code matches (e.g. <code>V-204</code>) under <b>BM25</b>, semantic similarity under <b>Dense Search</b>,
        and re-ranked candidate ordering under <b>Cross-Encoder Reranking</b>.
    </p>
    """,
    unsafe_allow_html=True,
)

# Search Controls in Compact, Well-Aligned Layout
with st.form("retrieval_explorer_form"):
    col_q, col_strat = st.columns([5, 3])
    with col_q:
        search_query = st.text_input(
            "Search Query:",
            value="orifice valve V-204 particulate blockage on steam turbine HP-17",
            placeholder="Enter technical query or failure code to inspect retrieval pipeline...",
        )
    with col_strat:
        strategy_label = st.selectbox(
            "Target Strategy:",
            options=[
                "Full Advanced RAG (Rewriting + RRF + Rerank)",
                "Hybrid + Cross-Encoder Rerank",
                "Hybrid (BM25 + Dense RRF)",
                "Dense Vector Search",
                "BM25 Keyword Search",
            ],
            index=0,
        )

    col_k, col_rw, col_btn = st.columns([3, 2, 3], vertical_alignment="bottom")
    with col_k:
        top_k = st.slider("Top-K Candidates per Channel:", min_value=1, max_value=20, value=8)
    with col_rw:
        enable_rewriting = st.checkbox("Enable Query Rewriting", value=True)
    with col_btn:
        explore_button = st.form_submit_button("🚀 Run Pipeline Inspection", use_container_width=True)

STRATEGY_MAP = {
    "Full Advanced RAG (Rewriting + RRF + Rerank)": "full_advanced",
    "Hybrid + Cross-Encoder Rerank": "hybrid_rerank",
    "Hybrid (BM25 + Dense RRF)": "hybrid_rrf",
    "Dense Vector Search": "dense",
    "BM25 Keyword Search": "bm25",
}

if explore_button:
    if not search_query.strip():
        st.warning("Please enter a search query.")
    else:
        with st.spinner("Executing multi-stage retrieval pipeline..."):
            try:
                resp = api_client.retrieve(
                    query=search_query,
                    strategy=STRATEGY_MAP[strategy_label],
                    top_k=top_k,
                    enable_rewriting=enable_rewriting,
                )
                st.session_state["explorer_data"] = resp
            except Exception as e:
                st.error(f"Retrieval inspection failed: {str(e)}")

# Display Explorer Results
if "explorer_data" in st.session_state:
    data = st.session_state["explorer_data"]
    stages = data.get("stage_results", {})

    st.markdown("---")

    # Query Rewriting Inspection
    rewritten = data.get("rewritten_queries", [])
    st.markdown("### 🔄 Query Rewriting Stage")
    col_q1, col_q2 = st.columns(2)
    with col_q1:
        st.markdown(
            f"""
            <div class="industrial-card" style="border-left: 4px solid #2563EB; padding: 14px 18px; margin-bottom: 0;">
                <div style="font-size: 12px; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.03em;">Original Query</div>
                <div style="font-size: 14px; font-weight: 600; color: #0F1E36; margin-top: 6px;"><code>{data.get('query')}</code></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_q2:
        if rewritten:
            variants_html = "".join(f"<li style='margin-bottom: 4px;'><code>{q}</code></li>" for q in rewritten)
            st.markdown(
                f"""
                <div class="industrial-card" style="border-left: 4px solid #0D9488; padding: 14px 18px; margin-bottom: 0;">
                    <div style="font-size: 12px; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.03em;">Rewritten Variants ({len(rewritten)})</div>
                    <ul style="margin: 6px 0 0 16px; padding: 0; font-size: 13px; color: #1E293B; line-height: 1.5;">{variants_html}</ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Query rewriting was disabled or produced no additional variants.")

    st.markdown("---")

    # Multi-Stage Tab Layout
    tab_final, tab_rerank, tab_fused, tab_dense, tab_bm25 = st.tabs([
        "🏆 Final Pipeline Context",
        "⚖️ Cross-Encoder Reranked",
        "🔀 RRF Fused Candidates",
        "🧠 Dense Vector Search",
        "🔤 BM25 Keyword Search",
    ])

    def render_items_table(items: list, score_col_name: str, score_key: str, stage_tag: str = "stage"):
        if not items:
            st.info("No candidates retrieved for this stage.")
            return

        rows = []
        for it in items:
            meta = it.get("metadata", {})
            rows.append({
                "Rank": it.get("final_rank") or it.get(f"{score_key}_rank", "-"),
                score_col_name: round(it.get(score_key, 0.0), 4) if it.get(score_key) is not None else "N/A",
                "Source Document": meta.get("source_filename", ""),
                "Page": meta.get("page_number", "N/A"),
                "Machine ID": meta.get("machine_id", "N/A"),
                "Chunk ID": it.get("chunk_id", ""),
                "Text Preview": it.get("text", "")[:120] + "...",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        with st.expander("🔍 Inspect Full Passages in this Stage", expanded=False):
            for idx, it in enumerate(items):
                cid = it.get("chunk_id", f"chk_{idx}")
                score_val = it.get(score_key)
                st.markdown(f"**Chunk `{cid}`** | {score_col_name}: `{score_val}`")
                st.markdown(
                    f"""
                    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 12px 14px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 13px; color: #1E293B; white-space: pre-wrap; margin-bottom: 8px; line-height: 1.55;">
{it.get("text", "")}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with tab_final:
        st.markdown(f"#### Final Context Passages Passed to LLM ({len(data.get('results', []))} items)")
        st.caption("These top-ranked chunks form the evidence context for answer synthesis.")
        render_items_table(data.get("results", []), "Rerank / Final Score", "rerank_score", "final")

    with tab_rerank:
        st.markdown("#### Cross-Encoder Reranked Candidates")
        st.caption("Scored using Sentence Transformers `ms-marco-MiniLM-L-6-v2` cross-encoder over `(query, passage)` pairs.")
        render_items_table(stages.get("reranked_results", []), "Cross-Encoder Score", "rerank_score", "rerank")

    with tab_fused:
        st.markdown("#### Reciprocal Rank Fusion (RRF) Candidates")
        st.caption("Scores computed as RRF_score(d) = Σ 1 / (k + rank_i(d)) with k = 60.")
        render_items_table(stages.get("fused_results", []), "RRF Fusion Score", "rrf_score", "fused")

    with tab_dense:
        st.markdown("#### Dense Vector Candidates (Sentence Transformers `all-MiniLM-L6-v2`)")
        st.caption("Scored by Cosine Similarity between query embedding vector and document chunk vector (scale: -1.0 to 1.0).")
        render_items_table(stages.get("dense_results", []), "Cosine Similarity", "dense_score", "dense")

    with tab_bm25:
        st.markdown("#### BM25 Keyword Search Candidates (rank-bm25)")
        st.caption("Scored using BM25Okapi / Lucene non-negative IDF over technical token vocabulary.")
        render_items_table(stages.get("bm25_results", []), "BM25 Score", "bm25_score", "bm25")
