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


def generate_report_markdown(resp: Dict[str, Any], query: str) -> str:
    """Formats investigation response into an auditable 6-part Markdown incident report."""
    retrieval_meta = resp.get("retrieval_metadata", {})
    citations = resp.get("citations", [])

    # Infer equipment context from citations/metadata
    machine_id = retrieval_meta.get("machine_id_filter")
    if not machine_id and citations:
        sources = [c.get("source_name", "") for c in citations]
        source_str = ", ".join(set(s for s in sources if s))
    else:
        source_str = "Knowledge Base Repository"

    lines = [
        "# Industrial Incident Investigation Report",
        f"**Inquiry / Query:** {query}",
        "",
        "## 1. Executive Summary: Incident Summary",
        resp.get("executive_summary", "No summary provided."),
        "",
        "## 2. Documented Observations & Telemetry (Equipment & Operating Context)",
        f"- **Operating Context & Equipment Focus:** {query}",
        f"- **Referenced Documentation:** {source_str}",
        f"- **Evidence Grounding Status:** {'Confirmed Grounded Evidence' if resp.get('is_sufficient_evidence', True) else 'Insufficient / Mismatched Evidence'}",
    ]

    obs_list = resp.get("documented_observations", [])
    if obs_list:
        lines.append("- **Sensor Telemetry & Physical Inspection:**")
        for obs in obs_list:
            clean_obs = str(obs).strip()
            if clean_obs:
                lines.append(f"  - {clean_obs}")
    else:
        lines.append("*No sensor telemetry or physical observations documented in retrieved passages.*")

    lines.append("")
    lines.append("## 3. Potential Root Causes & Hypotheses (Root Cause Analysis)")
    hypotheses = resp.get("potential_root_causes", [])
    if hypotheses:
        for hyp in hypotheses:
            hyp_title = hyp.get("hypothesis", "Causal Hypothesis")
            conf = hyp.get("confidence_level", "Medium")
            lines.append(f"### {hyp_title}")
            lines.append(f"- **Confidence Level:** {conf}")
            for ev in hyp.get("supporting_evidence", []):
                clean_ev = str(ev).strip()
                if clean_ev:
                    lines.append(f"  - Supporting Evidence: {clean_ev}")
            cids = ", ".join(hyp.get("citation_ids", []))
            if cids:
                lines.append(f"  - Referenced Citations: {cids}")
    else:
        lines.append("*No definitive root causes established in the reviewed passages.*")

    lines.append("")
    lines.append("## 4. Mandated Corrective & Preventive Actions")
    actions = resp.get("corrective_actions", [])
    if actions:
        for act in actions:
            clean_act = str(act).strip()
            if clean_act:
                lines.append(f"- {clean_act}")
    else:
        lines.append("*No specific corrective actions recorded in the reviewed passages.*")

    lines.append("")
    lines.append("## 5. Missing Evidence & Forensic Uncertainties (Evidence Limitations and Gaps)")
    uncertainties = resp.get("missing_evidence_and_uncertainties", [])
    if uncertainties:
        for u in uncertainties:
            clean_u = str(u).strip()
            if clean_u:
                lines.append(f"- {clean_u}")
    elif not resp.get("is_sufficient_evidence", True):
        lines.append("- Critical gap: Available evidence does not establish the incident cause for the requested equipment.")
    else:
        lines.append("- No unresolved forensic evidence gaps identified in reviewed passages.")

    if citations:
        lines.append("")
        lines.append("## 6. Traceable Evidence Citations")
        for cit in citations:
            page_info = f" (Page {cit.get('page_number')})" if cit.get("page_number") else ""
            lines.append(f"### {cit.get('citation_id', '[CIT]')} — {cit.get('source_name', 'Unknown')}{page_info}")
            lines.append(f"> {cit.get('excerpt', '')}")
            if cit.get("relevance_reason"):
                lines.append(f"*Relevance:* {cit.get('relevance_reason')}")
            lines.append("")

    return "\n".join(lines)
