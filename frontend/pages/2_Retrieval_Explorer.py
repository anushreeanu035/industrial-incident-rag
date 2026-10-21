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
    page_title="Retrieval Explorer - Incident Investigator",
    page_icon="🔬",
    layout="wide",
)

# Load cached theme CSS
load_theme()

render_sidebar(current_page="retrieval")

# Top Navigation Bar
col_back, _ = st.columns([1, 4])
with col_back:
    if st.button("← Return to Dashboard", key="btn_back_home_ret"):
        st.switch_page("app.py")

st.markdown(
    """
    <div class="incident-header-banner">
        <h1>🔬 Retrieval Explorer</h1>
        <p>Stage-by-Stage Forensic Inspection: Query Rewriting, BM25 Keyword Search, Dense Embeddings, RRF, and Cross-Encoder</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Pipeline Flow Visual Indicator
st.markdown(
    """
    <div class="pipeline-flow-container">
        <span class="pipeline-step-badge active">1. Original Query</span>
        <span class="pipeline-step-arrow">→</span>
        <span class="pipeline-step-badge">2. Query Rewriting</span>
        <span class="pipeline-step-arrow">→</span>
        <span class="pipeline-step-badge">3. Dense Vector + BM25 Search</span>
        <span class="pipeline-step-arrow">→</span>
        <span class="pipeline-step-badge">4. Reciprocal Rank Fusion (RRF)</span>
        <span class="pipeline-step-arrow">→</span>
        <span class="pipeline-step-badge">5. Cross-Encoder Reranker</span>
        <span class="pipeline-step-arrow">→</span>
        <span class="pipeline-step-badge active">6. Final Grounded Context</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p style="color: var(--text-secondary); font-size: 13.5px; line-height: 1.55; margin-bottom: 0.75rem;">
        Audit how candidate evidence passages are scored and reranked through each pipeline layer.
        Observe exact machine code matches (e.g. <code>V-204</code>, <code>M-301</code>) under <b>BM25</b>, semantic similarity under <b>Dense Search</b>,
        balanced candidate blending under <b>RRF Fusion</b>, and deep cross-attentive relevance under <b>Cross-Encoder Reranking</b>.
    </p>
    """,
    unsafe_allow_html=True,
)

# Search Controls
with st.form("retrieval_explorer_form"):
    col_q, col_strat = st.columns([5, 3])
    with col_q:
        search_query = st.text_input(
            "Search Query / Technical Diagnostic Inquiry:",
            value="orifice valve V-204 particulate blockage on steam turbine HP-17",
            placeholder="Enter technical query, failure code, or machine identifier to inspect pipeline...",
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
        explore_button = st.form_submit_button("🚀 Run Pipeline Inspection", type="primary", use_container_width=True)

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

    # Execution Metadata Bar
    c_meta1, c_meta2, c_meta3 = st.columns(3)
    with c_meta1:
        st.metric("Strategy Executed", data.get("strategy", "").upper())
    with c_meta2:
        st.metric("Total Passages Retrieved", len(data.get("results", [])))
    with c_meta3:
        st.metric("Pipeline Latency", f"{data.get('execution_time_ms', 0.0):.1f} ms")

    # Query Rewriting Inspection
    rewritten = data.get("rewritten_queries", [])
    st.markdown("### 🔄 Query Rewriting Stage")
    col_q1, col_q2 = st.columns(2)
    with col_q1:
        st.markdown(
            f"""
            <div class="industrial-card" style="border-left: 4px solid var(--accent-blue); padding: 14px 18px; margin-bottom: 0;">
                <div style="font-size: 11px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.04em;">Original User Query</div>
                <div style="font-size: 13.5px; font-weight: 600; color: var(--primary-navy); margin-top: 6px;"><code>{data.get('query')}</code></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_q2:
        if rewritten:
            variants_html = "".join(f"<li style='margin-bottom: 4px;'><code>{q}</code></li>" for q in rewritten)
            st.markdown(
                f"""
                <div class="industrial-card" style="border-left: 4px solid var(--accent-teal); padding: 14px 18px; margin-bottom: 0;">
                    <div style="font-size: 11px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.04em;">Rewritten Query Variants ({len(rewritten)})</div>
                    <ul style="margin: 6px 0 0 16px; padding: 0; font-size: 12.5px; color: var(--text-primary); line-height: 1.5;">{variants_html}</ul>
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

    def render_items_table(items: list, score_col_name: str, score_key: str):
        if not items:
            st.info("No candidates retrieved for this stage.")
            return

        rows = []
        for it in items:
            meta = it.get("metadata", {})
            score_val = it.get(score_key)
            formatted_score = f"{score_val:.4f}" if isinstance(score_val, (int, float)) else "N/A"
            rows.append({
                "Rank": it.get("final_rank") or it.get(f"{score_key}_rank", "-"),
                score_col_name: formatted_score,
                "Source Document": meta.get("source_filename", ""),
                "Page": meta.get("page_number", "N/A"),
                "Machine ID": meta.get("machine_id", "N/A"),
                "Chunk ID": it.get("chunk_id", ""),
                "Passage Preview": it.get("text", "")[:130].replace("\n", " ") + "...",
            })
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.TextColumn("Rank", width="small"),
                score_col_name: st.column_config.TextColumn(score_col_name, width="small"),
                "Source Document": st.column_config.TextColumn("Source Doc", width="medium"),
                "Page": st.column_config.TextColumn("Page", width="small"),
                "Machine ID": st.column_config.TextColumn("Machine", width="small"),
                "Chunk ID": st.column_config.TextColumn("Chunk ID", width="small"),
                "Passage Preview": st.column_config.TextColumn("Passage Preview", width="large"),
            },
        )

        with st.expander(f"🔍 Inspect Full Passages ({len(items)} items)", expanded=False):
            for idx, it in enumerate(items):
                cid = it.get("chunk_id", f"chk_{idx}")
                score_val = it.get(score_key)
                st.markdown(f"**Chunk `{cid}`** | {score_col_name}: `{score_val}`")
                st.markdown(
                    f"""
                    <div style="background-color: #F8FAFC; border: 1px solid var(--border-color); border-radius: 6px; padding: 12px 14px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 13px; color: var(--text-primary); white-space: pre-wrap; margin-bottom: 8px; line-height: 1.55;">
{it.get("text", "")}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with tab_final:
        st.markdown(f"#### Final Context Passages Passed to LLM ({len(data.get('results', []))} items)")
        st.caption("These top-ranked passages form the final grounded evidence context for root cause synthesis.")
        render_items_table(data.get("results", []), "Rerank Score", "rerank_score")

    with tab_rerank:
        st.markdown("#### Cross-Encoder Reranked Candidates")
        st.caption("Scored using Sentence Transformers `ms-marco-MiniLM-L-6-v2` cross-encoder over `(query, passage)` pairs.")
        render_items_table(stages.get("reranked_results", []), "Cross-Encoder Score", "rerank_score")

    with tab_fused:
        st.markdown("#### Reciprocal Rank Fusion (RRF) Candidates")
        st.caption("Fused ranks computed via RRF(d) = Σ 1 / (k + rank_i(d)) with k = 60.")
        render_items_table(stages.get("fused_results", []), "RRF Fusion Score", "rrf_score")

    with tab_dense:
        st.markdown("#### Dense Vector Search Candidates")
        st.caption("Cosine similarity over 384-dimensional dense vectors (`all-MiniLM-L6-v2`).")
        render_items_table(stages.get("dense_results", []), "Cosine Sim", "dense_score")

    with tab_bm25:
        st.markdown("#### BM25 Keyword Search Candidates")
        st.caption("Lexical keyword scoring with Lucene non-negative IDF over preserved technical vocabulary.")
        render_items_table(stages.get("bm25_results", []), "BM25 Score", "bm25_score")
