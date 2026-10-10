from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st
from frontend.components.api_client import api_client
from frontend.components.charts import render_benchmark_metrics_chart, render_latency_chart
from frontend.components.sidebar import render_sidebar

st.set_page_config(
    page_title="RAG Evaluation Dashboard - Incident Investigator",
    page_icon="📊",
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
        <h1>📊 RAG Evaluation & Benchmarking Dashboard</h1>
        <p>Scientific Evaluation of Retrieval Quality, Ranking Metrics, Latencies, and Multi-Strategy Ablation</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Benchmark Controls
st.markdown("### 🧪 Run New Benchmark Evaluation")
with st.expander("Configure and Execute Live Benchmark Run", expanded=True):
    col_strats, col_opt = st.columns([2, 1])
    with col_strats:
        st.markdown("**Strategies to Evaluate:**")
        selected_strategies = st.multiselect(
            "Select Pipeline Configurations:",
            options=["dense", "bm25", "hybrid_rrf", "hybrid_rerank", "full_advanced"],
            default=["dense", "bm25", "hybrid_rrf", "hybrid_rerank", "full_advanced"],
            format_func=lambda s: {
                "dense": "Dense Only (all-MiniLM-L6-v2)",
                "bm25": "BM25 Keyword Only",
                "hybrid_rrf": "Hybrid Search (Dense + BM25 + RRF)",
                "hybrid_rerank": "Hybrid + Cross-Encoder Rerank",
                "full_advanced": "Full Advanced RAG (Rewriting + RRF + Rerank)",
            }.get(s, s),
        )
    with col_opt:
        top_k_bench = st.slider("Top-K Evaluation Depth:", min_value=3, max_value=15, value=5)
        run_ragas_toggle = st.checkbox("Run Ragas Generation Metrics (requires LLM judge)", value=False)

    if st.button("🚀 Execute Live Benchmark Run", use_container_width=True):
        if not selected_strategies:
            st.error("Please select at least one strategy to benchmark.")
        else:
            with st.spinner("Executing benchmark across test cases. Computing Recall@K, Precision@K, MRR, nDCG@10, and latencies..."):
                try:
                    report = api_client.run_evaluation(
                        strategies=selected_strategies,
                        top_k=top_k_bench,
                        run_ragas=run_ragas_toggle,
                    )
                    st.success(f"Benchmark completed successfully! Run ID: `{report.get('run_id')}`")
                    st.session_state["active_report"] = report
                except Exception as e:
                    st.error(f"Benchmark run failed: {str(e)}")

st.divider()

# Historical Benchmark Selection
st.markdown("### 📁 Benchmark Reports History")
history_runs = []
try:
    history_runs = api_client.list_evaluations()
except Exception as e:
    st.warning("Could not fetch historical reports.")

active_report = st.session_state.get("active_report")

if history_runs:
    run_options = {r.get("run_id"): f"{r.get('run_id')} ({r.get('timestamp')})" for r in history_runs}
    selected_run_id = st.selectbox(
        "Select Historical Evaluation Report to Inspect:",
        options=list(run_options.keys()),
        format_func=lambda x: run_options[x],
        index=0,
    )
    if not active_report or active_report.get("run_id") != selected_run_id:
        active_report = api_client.get_evaluation(selected_run_id)
else:
    if not active_report:
        st.info("No benchmark evaluations have been run yet. Click 'Execute Live Benchmark Run' above to generate real metrics.")

# Display Report Metrics and Visualizations
if active_report:
    st.markdown("---")
    st.markdown(f"### 📈 Benchmark Results: `{active_report.get('run_id')}`")

    # Metadata Banner
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f"**Timestamp:** `{active_report.get('timestamp')}`")
        st.markdown(f"**Dataset Version:** `{active_report.get('dataset_version')}`")
    with m_col2:
        st.markdown(f"**Dataset Hash:** `{active_report.get('dataset_hash', '')[:12]}...`")
        st.markdown(f"**Embedding Model:** `{active_report.get('embedding_model')}`")
    with m_col3:
        st.markdown(f"**Reranker Model:** `{active_report.get('reranker_model')}`")
        st.markdown(f"**LLM Model:** `{active_report.get('llm_model')}`")
    with m_col4:
        config = active_report.get("configuration", {})
        st.markdown(f"**Top-K:** `{config.get('top_k')}` | **RRF k:** `{config.get('rrf_k')}`")
        st.markdown(f"**Chunk Size:** `{config.get('chunk_size')}` | **Overlap:** `{config.get('chunk_overlap')}`")

    # Strategy Results Table
    results = active_report.get("results", [])
    table_rows = []
    for r in results:
        rm = r.get("retrieval_metrics", {})
        ragas = r.get("ragas_metrics") or {}
        table_rows.append({
            "Strategy": r.get("strategy"),
            "Recall@5": f"{rm.get('recall_at_5', 0.0):.4f}",
            "Recall@10": f"{rm.get('recall_at_10', 0.0):.4f}",
            "Precision@K": f"{rm.get('precision_at_k', 0.0):.4f}",
            "MRR": f"{rm.get('mrr', 0.0):.4f}",
            "nDCG@10": f"{rm.get('ndcg_at_10', 0.0):.4f}",
            "Faithfulness": f"{ragas.get('faithfulness'):.4f}" if ragas.get("faithfulness") is not None else "Not evaluated",
            "Answer Relevancy": f"{ragas.get('answer_relevancy'):.4f}" if ragas.get("answer_relevancy") is not None else "Not evaluated",
            "Context Precision": f"{ragas.get('context_precision'):.4f}" if ragas.get("context_precision") is not None else "Not evaluated",
            "Context Recall": f"{ragas.get('context_recall'):.4f}" if ragas.get("context_recall") is not None else "Not evaluated",
            "Avg Latency (ms)": f"{r.get('avg_latency_ms', 0.0):.1f}",
            "P95 Latency (ms)": f"{r.get('p95_latency_ms', 0.0):.1f}",
            "Passed Queries": f"{r.get('successful_queries', 0)}/{r.get('total_queries', 0)}",
        })

    df_results = pd.DataFrame(table_rows)
    st.markdown("#### Comparative Strategy Metrics")
    st.dataframe(df_results, use_container_width=True, hide_index=True)

    # Interactive Charts
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig_metrics = render_benchmark_metrics_chart(results)
        st.plotly_chart(fig_metrics, use_container_width=True)

    with chart_col2:
        fig_latency = render_latency_chart(results)
        st.plotly_chart(fig_latency, use_container_width=True)

    # Download CSV Report
    csv_data = df_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Benchmark Report (CSV)",
        data=csv_data,
        file_name=f"benchmark_{active_report.get('run_id')}.csv",
        mime="text/csv",
        use_container_width=True,
    )
