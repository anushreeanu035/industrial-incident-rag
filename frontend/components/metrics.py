"""Reusable UI metric cards and badges for the Streamlit dashboard."""

from typing import Any, Dict, Optional
import streamlit as st


def render_metric_card(
    title: str,
    value: Any,
    subtitle: Optional[str] = None,
    delta: Optional[str] = None,
    delta_color: str = "normal",
    help_text: Optional[str] = None,
):
    """Renders a standardized industrial metric card."""
    st.metric(
        label=title,
        value=value,
        delta=delta,
        delta_color=delta_color,
        help=help_text,
    )
    if subtitle:
        st.caption(subtitle)


def render_confidence_badge(confidence: str) -> str:
    """Returns HTML for styled confidence badge."""
    conf_lower = confidence.lower()
    if "high" in conf_lower or "fact" in conf_lower:
        badge_class = "badge-fact"
    elif "low" in conf_lower or "insufficient" in conf_lower:
        badge_class = "badge-warning"
    else:
        badge_class = "badge-hypothesis"
    return f'<span class="{badge_class}">{confidence}</span>'


def render_retrieval_stats(stats: Dict[str, Any]):
    """Renders a 4-column summary of retrieval latency and counts."""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Strategy", stats.get("strategy", "Hybrid"))
    with c2:
        st.metric("Candidates Retrieved", stats.get("total_retrieved", 0))
    with c3:
        st.metric("Final Passages", stats.get("final_passages", 0))
    with c4:
        st.metric("Latency", f"{stats.get('latency_ms', 0):.1f} ms")
