"""Reusable UI cards for rendering evidence citations, hypotheses, and retrieved passages."""

from typing import Any, Dict, List
import streamlit as st


def render_hypothesis_card(hypothesis: Dict[str, Any]):
    """Renders a potential root cause hypothesis card with confidence indicator."""
    hyp_text = hypothesis.get("hypothesis", "")
    confidence = hypothesis.get("confidence_level", "Medium (Strong Evidence)")
    evidence_points = hypothesis.get("supporting_evidence", [])
    citation_ids = hypothesis.get("citation_ids", [])

    badge_class = "badge-hypothesis"
    if "high" in confidence.lower() or "fact" in confidence.lower():
        badge_class = "badge-fact"
    elif "low" in confidence.lower() or "insufficient" in confidence.lower():
        badge_class = "badge-warning"

    st.markdown(
        f"""
        <div class="industrial-card" style="border-left: 4px solid #1E88E5;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
                <h4 style="margin: 0; color: #0F1E36; font-size: 1.05rem;">{hyp_text}</h4>
                <span class="{badge_class}">{confidence}</span>
            </div>
        """,
        unsafe_allow_html=True,
    )

    if evidence_points:
        st.markdown("**Supporting Evidence:**")
        for pt in evidence_points:
            st.markdown(f"- {pt}")

    if citation_ids:
        chips_html = "".join(f'<span class="citation-chip">{cid}</span>' for cid in citation_ids)
        st.markdown(f"**Referenced Sources:** {chips_html}", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_citation_accordion(citations: List[Dict[str, Any]]):
    """Renders expandable citation cards allowing investigators to audit original passages."""
    if not citations:
        st.info("No citations available for this response.")
        return

    st.markdown("### 📑 Traceable Evidence Passages")
    for cit in citations:
        cid = cit.get("citation_id", "[CIT]")
        source = cit.get("source_name", "Unknown File")
        page = cit.get("page_number")
        page_str = f"Page {page}" if page else "Page N/A"
        chunk_id = cit.get("chunk_id", "")
        excerpt = cit.get("excerpt", "")
        reason = cit.get("relevance_reason", "")

        label = f"{cid} — {source} ({page_str})"
        with st.expander(label, expanded=False):
            st.markdown(f"**Chunk Identifier:** `{chunk_id}`")
            st.markdown(f"**Source File:** `{source}` | **Location:** {page_str}")
            if reason:
                st.markdown(f"**Relevance Metric:** {reason}")
            st.markdown("**Original Passage Excerpt:**")
            st.markdown(
                f"""
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.8rem; font-family: monospace; font-size: 0.85rem; color: #1E293B; white-space: pre-wrap; line-height: 1.5;">
{excerpt}
                </div>
                """,
                unsafe_allow_html=True,
            )
