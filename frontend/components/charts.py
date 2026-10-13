"""Interactive Plotly visualizations for benchmark evaluation metrics."""

from typing import Any, Dict, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_benchmark_metrics_chart(results: List[Dict[str, Any]]):
    """Generates comparative grouped bar chart for Recall@5, Recall@10, Precision@K, MRR, and nDCG@10."""
    records = []
    for r in results:
        strat = r.get("strategy", "Unknown")
        m = r.get("retrieval_metrics", {})
        records.append({
            "Strategy": strat,
            "Recall@5": m.get("recall_at_5", 0.0),
            "Recall@10": m.get("recall_at_10", 0.0),
            "Precision@K": m.get("precision_at_k", 0.0),
            "MRR": m.get("mrr", 0.0),
            "nDCG@10": m.get("ndcg_at_10", 0.0),
        })

    df = pd.DataFrame(records)
    df_melted = df.melt(id_vars=["Strategy"], var_name="Metric", value_name="Score")

    fig = px.bar(
        df_melted,
        x="Strategy",
        y="Score",
        color="Metric",
        barmode="group",
        color_discrete_sequence=["#0F1E36", "#1E88E5", "#00897B", "#D97706", "#7C3AED"],
        title="Retrieval Metrics Comparison Across Strategies",
    )
    fig.update_layout(
        font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", size=12.5, color="#1E293B"),
        title=dict(font=dict(size=15, color="#0F1E36")),
        yaxis=dict(range=[0, 1.05], title="Score (0.0 to 1.0)", tickfont=dict(size=11)),
        xaxis=dict(title=None, tickfont=dict(size=11)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11.5),
        ),
        margin=dict(l=45, r=25, t=45, b=40),
        height=380,
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def render_latency_chart(results: List[Dict[str, Any]]):
    """Generates comparative latency bar chart comparing average vs P95 execution time."""
    records = []
    for r in results:
        records.append({
            "Strategy": r.get("strategy", "Unknown"),
            "Avg Latency (ms)": r.get("avg_latency_ms", 0.0),
            "P95 Latency (ms)": r.get("p95_latency_ms", 0.0),
        })
    df = pd.DataFrame(records)
    df_melted = df.melt(id_vars=["Strategy"], var_name="Latency Metric", value_name="Milliseconds")

    fig = px.bar(
        df_melted,
        x="Strategy",
        y="Milliseconds",
        color="Latency Metric",
        barmode="group",
        color_discrete_sequence=["#2563EB", "#DC2626"],
        title="Query Execution Latency (Average vs. P95)",
    )
    fig.update_layout(
        font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", size=12.5, color="#1E293B"),
        title=dict(font=dict(size=15, color="#0F1E36")),
        yaxis=dict(title="Milliseconds", tickfont=dict(size=11)),
        xaxis=dict(title=None, tickfont=dict(size=11)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11.5),
        ),
        margin=dict(l=45, r=25, t=45, b=40),
        height=380,
        template="plotly_white",
    )
    return fig
