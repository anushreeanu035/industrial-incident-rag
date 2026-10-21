"""Reusable UI cards for rendering evidence citations, hypotheses, and retrieved passages."""

import html
from typing import Any, Dict, List
import streamlit as st


def render_hypothesis_card(hypothesis: Dict[str, Any]):
    """Renders a potential root cause hypothesis card with confidence indicator."""
    hyp_text = html.escape(str(hypothesis.get("hypothesis", "")))
    confidence = html.escape(str(hypothesis.get("confidence_level", "Medium (Strong Evidence)")))
    evidence_points = hypothesis.get("supporting_evidence", [])
    citation_ids = hypothesis.get("citation_ids", [])

    badge_class = "badge-hypothesis"
    if "high" in confidence.lower() or "fact" in confidence.lower():
        badge_class = "badge-fact"
    elif "low" in confidence.lower() or "insufficient" in confidence.lower():
        badge_class = "badge-warning"

    ev_html = ""
    if evidence_points:
        ev_items = "".join(f"<li>{html.escape(str(pt))}</li>" for pt in evidence_points)
        ev_html = f"<div style='margin-top: 6px;'><strong>Supporting Evidence:</strong><ul style='margin: 4px 0 0 16px; padding: 0; font-size: 13px; color: var(--text-primary); line-height: 1.5;'>{ev_items}</ul></div>"

    chips_html = ""
    if citation_ids:
        chips = "".join(f'<span class="citation-chip">{html.escape(str(cid))}</span>' for cid in citation_ids)
        chips_html = f"<div style='margin-top: 6px;'><strong>Referenced Sources:</strong> {chips}</div>"

    card_html = (
        f'<div class="industrial-card" style="border-left: 4px solid var(--accent-blue); margin-bottom: 10px;">'
        f'<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.4rem;">'
        f'<h4 style="margin: 0; color: var(--primary-navy); font-size: 15px; font-weight: 600;">{hyp_text}</h4>'
        f'<span class="{badge_class}">{confidence}</span>'
        f'</div>'
        f'{ev_html}'
        f'{chips_html}'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


def render_citation_accordion(citations: List[Dict[str, Any]]):
    """Renders expandable citation cards allowing investigators to audit original passages."""
    if not citations:
        st.info("No citations available for this response.")
        return

    st.markdown("##### 📑 Traceable Source Citations & Evidence Passages")
    for cit in citations:
        cid = cit.get("citation_id", "[CIT]")
        source = cit.get("source_name", "Unknown File")
        page = cit.get("page_number")
        page_str = f"Page {page}" if page else "Page N/A"
        chunk_id = cit.get("chunk_id", "")
        excerpt = (cit.get("excerpt") or "").strip()
        reason = cit.get("relevance_reason", "")

        label = f"{cid} — {source} ({page_str})"
        with st.expander(label, expanded=False):
            if not excerpt:
                st.warning("⚠️ Citation cannot be resolved to a valid source passage.")
                continue

            c_info1, c_info2 = st.columns([2, 1])
            with c_info1:
                st.markdown(f"**Source Document:** `{source}` | **Location:** {page_str}")
                st.markdown(f"**Chunk Identifier:** `{chunk_id}`")
            with c_info2:
                if reason and not reason.startswith("-"):
                    st.markdown(f"**Passage Context:** {reason}")

            st.markdown("**Original Passage Excerpt:**")
            st.markdown(
                f"""
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 12px 14px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 13px; color: #1E293B; white-space: pre-wrap; line-height: 1.55;">
{excerpt}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Advanced Retrieval Details Panel
            with st.expander("🔬 Advanced Retrieval Details", expanded=False):
                st.json({
                    "citation_id": cid,
                    "source_name": source,
                    "chunk_id": chunk_id,
                    "page_number": page,
                    "relevance_metadata": reason,
                })


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


def generate_document_analysis_markdown(analysis: Dict[str, Any]) -> str:
    """Formats structured document analysis into a 9-section incident report in exact order."""
    overview = analysis.get("incident_overview") or analysis.get("overview") or {}
    filename = (
        analysis.get("source_filename")
        or analysis.get("filename")
        or analysis.get("source_document")
        or "Incident Document"
    )

    def _val(k: str, default: str = "Not specified in the source") -> str:
        v = overview.get(k)
        if not v or str(v).strip().lower() in ["not specified", "none", "", "null"]:
            return default
        return str(v).strip()

    inc_id = _val("incident_id")
    mach_id = _val("machine_id")
    mach_name = _val("machine_name")
    inc_type = _val("incident_type")
    inc_date = _val("incident_date")
    facility = _val("facility")
    severity = _val("severity_level")
    op_impact = _val("operational_impact")
    status = _val("status")
    lead_inv = _val("lead_investigator")

    lines = [
        f"# Forensic Document Analysis Report: {inc_id}",
        f"**Source Document:** {filename}",
        f"**Machine / Equipment:** {mach_id} ({mach_name})",
        "",
        "## 1. Incident Overview",
        f"- **Incident ID:** {inc_id}",
        f"- **Equipment ID:** {mach_id}",
        f"- **Equipment Description:** {mach_name}",
        f"- **Incident Classification:** {inc_type}",
        f"- **Incident Date & Time:** {inc_date}",
        f"- **Facility / Location:** {facility}",
        f"- **Severity Level:** {severity}",
        f"- **Operational Impact:** {op_impact}",
        f"- **Operational Status:** {status}",
        f"- **Lead Investigator:** {lead_inv}",
        "",
        "## 2. Executive Summary",
        analysis.get("executive_summary") or "Not specified in the source",
        "",
        "## 3. Incident Timeline",
    ]

    timeline = analysis.get("timeline", [])
    if timeline:
        for t in timeline:
            ts = t.get("timestamp", "Timestamp")
            ev = t.get("event", "")
            cat = t.get("category", "Operational")
            lines.append(f"- **{ts}** [{cat}]: {ev}")
    else:
        lines.append("- Not specified in the source: No chronological sequence of events was documented in this report.")
    lines.append("")

    lines.append("## 4. Physical Observations & Telemetry")
    telemetry = analysis.get("telemetry") or analysis.get("telemetry_and_observations") or []
    if telemetry:
        lines.append("### Sensor Telemetry & Measured Values")
        lines.append("| Parameter | Observed Value | Reference / Threshold | Nature / Grounding | Interpretation | Evidence Citation |")
        lines.append("|---|---|---|---|---|---|")
        for te in telemetry:
            param = te.get("parameter") or "Not specified in the source"
            val = te.get("measured_value") or "Not specified in the source"
            ref = te.get("nominal_threshold") or te.get("nominal_range") or "Not specified in the source"
            stat = te.get("status") or "Observed"
            nature = "Source-Stated Measurement" if any(w in stat.lower() for w in ["critical", "observed", "alarm", "trip"]) else "Diagnostic Inspection"
            interp = te.get("interpretation") or "Not specified in the source"
            cit = te.get("citation_id") or "[CIT-1]"
            lines.append(f"| {param} | **{val}** | {ref} | {nature} | {interp} | {cit} |")
        lines.append("")

    obs = analysis.get("physical_observations", [])
    if obs:
        lines.append("### Physical Inspection Observations")
        for o in obs:
            lines.append(f"- {o}")
        lines.append("")
    elif not telemetry:
        lines.append("- Not specified in the source: No sensor telemetry or physical observations documented in report.")
        lines.append("")

    lines.append("## 5. Root Cause Analysis")
    root_causes = analysis.get("root_causes", [])
    if root_causes:
        for rc in root_causes:
            lines.append(f"### {rc.get('hypothesis', 'Root Cause Hypothesis')}")
            conf = rc.get("confidence_level") or rc.get("likelihood") or "High (Documented Fact)"
            lines.append(f"- **Confidence Level:** {conf}")
            for ev in rc.get("supporting_evidence", []):
                lines.append(f"  - Supporting Evidence: {ev}")
            cids = ", ".join(rc.get("citation_ids", []))
            if cids:
                lines.append(f"  - Supporting Citations: {cids}")
    else:
        lines.append("- Not specified in the source: No definitive root cause identified in source text.")
    lines.append("")

    lines.append("## 6. Contributing Factors")
    factors = analysis.get("contributing_factors", [])
    if factors:
        for f in factors:
            lines.append(f"- {f}")
    else:
        lines.append("- Not specified in the source: No secondary contributing factors specified.")
    lines.append("")

    lines.append("## 7. Corrective & Preventive Actions")
    actions = analysis.get("corrective_actions", [])
    if actions:
        for a in actions:
            clean_a = str(a).strip()
            lines.append(f"- {clean_a}")
    else:
        lines.append("- Not specified in the source: No corrective actions specified.")
    lines.append("")

    lines.append("## 8. Missing Evidence & Uncertainties")
    missing = (
        analysis.get("missing_evidence_and_limitations")
        or analysis.get("missing_evidence_and_uncertainties")
        or []
    )
    if missing:
        for m in missing:
            lines.append(f"- {m}")
    else:
        lines.append("- No unresolved forensic evidence limitations identified in this report.")
    lines.append("")

    lines.append("## 9. Evidence & Citations")
    citations = analysis.get("citations", [])
    if citations:
        for cit in citations:
            p_str = f" (Page {cit.get('page_number')})" if cit.get("page_number") else ""
            lines.append(f"### {cit.get('citation_id', '[CIT]')} — {cit.get('source_name', 'Document')}{p_str}")
            lines.append(f"> {cit.get('excerpt', '')}")
            if cit.get("relevance_reason") and not cit.get("relevance_reason").startswith("-"):
                lines.append(f"*Note:* {cit.get('relevance_reason')}")
            lines.append("")
    else:
        lines.append("- No traceable source citations available.")
        lines.append("")

    return "\n".join(lines)


def render_document_analysis_report(analysis: Dict[str, Any]):
    """Renders the comprehensive structured Document Analysis report with 9 standard sections in exact order."""
    import re
    import pandas as pd
    from frontend.components.charts import (
        render_dynamic_telemetry_chart,
        render_incident_timeline_figure,
        render_root_cause_matrix_figure,
    )

    overview = analysis.get("incident_overview") or analysis.get("overview") or {}
    filename = (
        analysis.get("source_filename")
        or analysis.get("filename")
        or analysis.get("source_document")
        or "Incident Document"
    )

    def _val(k: str, default: str = "Not specified in the source") -> str:
        v = overview.get(k)
        if not v or str(v).strip().lower() in ["not specified", "none", "", "null"]:
            return default
        return str(v).strip()

    inc_id = _val("incident_id")
    inc_type = _val("incident_type")
    mach_id = _val("machine_id")
    mach_name = _val("machine_name")
    facility = _val("facility")
    inc_date = _val("incident_date")
    severity = _val("severity_level")
    op_impact = _val("operational_impact")
    status = _val("status")
    lead_inv = _val("lead_investigator")

    # Header and Download Bar
    col_hdr, col_dl = st.columns([3, 1], vertical_alignment="bottom")
    with col_hdr:
        st.markdown(f"### 📋 Forensic Analysis Report: `{filename}`")
    with col_dl:
        md_content = generate_document_analysis_markdown(analysis)
        st.download_button(
            "📥 Download Analysis (.md)",
            data=md_content,
            file_name=f"investigation_{mach_id}_{inc_id}.md",
            mime="text/markdown",
            use_container_width=True,
            type="primary",
        )

    # -----------------------------------------------------------------------
    # Section 1 — Incident Overview
    # -----------------------------------------------------------------------
    st.markdown("#### Section 1 — Incident Overview")
    st.caption("Core equipment tags, operational event classification, and site metadata extracted from source report.")

    cards_data = [
        {
            "label": "Incident ID",
            "value": inc_id,
            "caption": f"Status: {status}",
            "border_color": "var(--accent-blue)",
        },
        {
            "label": "Incident Type",
            "value": inc_type,
            "caption": f"Lead Investigator: {lead_inv}",
            "border_color": "var(--navy-light)",
        },
        {
            "label": "Machine / Equipment ID",
            "value": mach_id,
            "caption": mach_name,
            "border_color": "var(--accent-teal)",
        },
        {
            "label": "Incident Date & Time",
            "value": inc_date,
            "caption": f"Facility: {facility}",
            "border_color": "var(--accent-blue)",
        },
        {
            "label": "Severity Level",
            "value": severity,
            "caption": f"Facility: {facility}",
            "border_color": "var(--danger-red)",
        },
        {
            "label": "Operational Impact",
            "value": op_impact,
            "caption": f"Equipment: {mach_id}",
            "border_color": "var(--warning-amber)",
        },
    ]

    cards_html = []
    for c in cards_data:
        lbl = html.escape(c["label"])
        val = html.escape(c["value"])
        cap = html.escape(c["caption"])
        border = c["border_color"]
        cards_html.append(
            f'<div class="stat-card" style="border-top-color: {border};">'
            f'<div class="stat-label">{lbl}</div>'
            f'<div class="stat-value">{val}</div>'
            f'<div class="stat-caption">{cap}</div>'
            f'</div>'
        )

    overview_grid_html = f'<div class="overview-grid">{"".join(cards_html)}</div>'
    st.markdown(overview_grid_html, unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Section 2 — Executive Summary
    # -----------------------------------------------------------------------
    st.markdown("#### Section 2 — Executive Summary")
    raw_exec = analysis.get("executive_summary")
    exec_summary_text = (
        str(raw_exec).strip()
        if raw_exec and str(raw_exec).strip().lower() not in ["none", "", "not specified"]
        else "Not specified in the source"
    )
    exec_escaped = html.escape(exec_summary_text)
    exec_html = (
        f'<div class="report-section-card" style="border-left: 4px solid var(--accent-blue); background-color: var(--bg-card);">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">'
        f'<span style="font-size: 11px; font-weight: 700; color: var(--accent-blue); text-transform: uppercase; letter-spacing: 0.04em;">'
        f'Forensic Executive Briefing'
        f'</span>'
        f'<span class="badge-fact">Source-Documented Summary</span>'
        f'</div>'
        f'<p style="margin: 0; font-size: 14.5px; line-height: 1.65; color: var(--text-primary); font-weight: 500;">'
        f'{exec_escaped}'
        f'</p>'
        f'</div>'
    )
    st.markdown(exec_html, unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Section 3 — Incident Timeline
    # -----------------------------------------------------------------------
    st.markdown("#### Section 3 — Incident Timeline")
    st.caption("Chronological sequence of operational events, telemetry warnings, and protection trips.")
    timeline = analysis.get("timeline", [])
    if timeline:
        cleaned_timeline = []
        seen_ts = set()
        for item in timeline:
            ts = str(item.get("timestamp", "Timestamp")).strip()
            ev = str(item.get("event", "")).strip()
            cat = str(item.get("category", "Operational")).strip()
            if ts in seen_ts or not ev:
                continue
            seen_ts.add(ts)
            cleaned_timeline.append({"timestamp": ts, "event": ev, "category": cat})

        cleaned_timeline.sort(key=lambda x: x["timestamp"])

        timeline_fig = render_incident_timeline_figure(cleaned_timeline)
        if timeline_fig:
            st.plotly_chart(timeline_fig, use_container_width=True)

        items_html = []
        for item in cleaned_timeline:
            ts_esc = html.escape(item["timestamp"])
            ev_esc = html.escape(item["event"])
            cat_esc = html.escape(item["category"])
            items_html.append(
                f'<div class="timeline-item">'
                f'<div class="timeline-badge">{ts_esc}</div>'
                f'<div class="timeline-content">'
                f'<span class="timeline-category">{cat_esc}</span>'
                f'<span class="timeline-text">{ev_esc}</span>'
                f'</div>'
                f'</div>'
            )
        timeline_html = f'<div class="timeline-container">{"".join(items_html)}</div>'
        st.markdown(timeline_html, unsafe_allow_html=True)
    else:
        st.info("Not specified in the source: No chronological sequence of events was documented in this report.")

    # -----------------------------------------------------------------------
    # Section 4 — Physical Observations & Telemetry
    # -----------------------------------------------------------------------
    st.markdown("#### Section 4 — Physical Observations & Telemetry")
    st.caption("Measured parameters, engineering thresholds, physical inspection findings, and interpretations.")

    telemetry = analysis.get("telemetry") or analysis.get("telemetry_and_observations") or []
    obs = analysis.get("physical_observations", [])

    if telemetry:
        t_rows = []
        for te in telemetry:
            param = te.get("parameter") or "Not specified in the source"
            val = te.get("measured_value") or "Not specified in the source"
            ref = te.get("nominal_threshold") or te.get("nominal_range") or "Not specified in the source"
            interp = te.get("interpretation") or "Not specified in the source"
            cit = te.get("citation_id") or "[CIT-1]"
            stat = te.get("status") or "Observed"
            nature = "Source-Stated Measurement" if any(w in stat.lower() for w in ["critical", "observed", "alarm", "trip", "elevated", "choked", "degraded"]) else "Diagnostic Inspection"
            t_rows.append({
                "Parameter": param,
                "Observed Value": val,
                "Reference / Threshold": ref,
                "Nature / Grounding": nature,
                "Engineering Interpretation": interp,
                "Evidence Citation": cit,
            })
        df_te = pd.DataFrame(t_rows)
        st.dataframe(
            df_te,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Parameter": st.column_config.TextColumn("Parameter", width="medium"),
                "Observed Value": st.column_config.TextColumn("Observed Value", width="small"),
                "Reference / Threshold": st.column_config.TextColumn("Reference / Threshold", width="medium"),
                "Nature / Grounding": st.column_config.TextColumn("Grounding Status", width="small"),
                "Engineering Interpretation": st.column_config.TextColumn("Engineering Interpretation", width="large"),
                "Evidence Citation": st.column_config.TextColumn("Citation", width="small"),
            },
        )

        # Dynamic comparative chart for telemetry with numeric thresholds
        chart = render_dynamic_telemetry_chart(telemetry)
        if chart:
            st.markdown("##### 📈 Telemetry vs Documented Design Limits")
            st.plotly_chart(chart, use_container_width=True)

    if obs:
        st.markdown("##### 🔍 Physical Inspection Observations")
        for o in obs:
            st.markdown(f"- {o}")

    if not telemetry and not obs:
        st.info("Not specified in the source: No sensor telemetry or physical diagnostic readings were documented in this report.")

    # -----------------------------------------------------------------------
    # Section 5 — Root Cause Analysis
    # -----------------------------------------------------------------------
    st.markdown("#### Section 5 — Root Cause Analysis")
    st.caption("Documented root causes, causal mechanisms, and diagnostic evidence validation.")

    root_causes = analysis.get("root_causes", [])
    contributing_factors = analysis.get("contributing_factors", [])
    missing = (
        analysis.get("missing_evidence_and_limitations")
        or analysis.get("missing_evidence_and_uncertainties")
        or []
    )

    if root_causes:
        primary_rc = next((rc for rc in root_causes if "primary" in rc.get("hypothesis", "").lower()), None)
        if not primary_rc and root_causes:
            primary_rc = root_causes[0]

        secondary_rc = next((rc for rc in root_causes if "secondary" in rc.get("hypothesis", "").lower()), None)
        if not secondary_rc and len(root_causes) > 1 and root_causes[1] != primary_rc:
            secondary_rc = root_causes[1]

        # Primary Cause Card
        if primary_rc:
            p_hyp = html.escape(str(primary_rc.get('hypothesis', 'Primary Root Cause')))
            p_conf = html.escape(str(primary_rc.get('confidence_level', 'High (Documented Fact)')))
            p_ev_items = "".join(f"<li>{html.escape(str(ev))}</li>" for ev in primary_rc.get('supporting_evidence', []))
            p_html = (
                f'<div class="report-section-card" style="border-left: 4px solid var(--danger-red); margin-bottom: 12px;">'
                f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">'
                f'<span style="font-size: 11px; font-weight: 700; color: var(--danger-red); text-transform: uppercase; letter-spacing: 0.04em;">'
                f'🔴 Primary Documented Root Cause'
                f'</span>'
                f'<span class="badge-fact">{p_conf}</span>'
                f'</div>'
                f'<div style="font-size: 14.5px; font-weight: 600; color: var(--primary-navy); margin-bottom: 6px;">'
                f'{p_hyp}'
                f'</div>'
                f'<div style="font-size: 13px; color: var(--text-secondary); line-height: 1.5;">'
                f'<strong>Supporting Evidence:</strong>'
                f'</div>'
                f'<ul style="margin: 4px 0 0 16px; padding: 0; font-size: 13px; color: var(--text-primary); line-height: 1.5;">'
                f'{p_ev_items}'
                f'</ul>'
                f'</div>'
            )
            st.markdown(p_html, unsafe_allow_html=True)

        # Secondary Cause Card
        if secondary_rc:
            s_hyp = html.escape(str(secondary_rc.get('hypothesis', 'Secondary Root Cause')))
            s_conf = html.escape(str(secondary_rc.get('confidence_level', 'High (Documented Fact)')))
            s_ev_items = "".join(f"<li>{html.escape(str(ev))}</li>" for ev in secondary_rc.get('supporting_evidence', []))
            s_html = (
                f'<div class="report-section-card" style="border-left: 4px solid var(--warning-amber); margin-bottom: 12px;">'
                f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">'
                f'<span style="font-size: 11px; font-weight: 700; color: var(--warning-amber); text-transform: uppercase; letter-spacing: 0.04em;">'
                f'🟡 Secondary Documented Root Cause'
                f'</span>'
                f'<span class="badge-fact">{s_conf}</span>'
                f'</div>'
                f'<div style="font-size: 14.5px; font-weight: 600; color: var(--primary-navy); margin-bottom: 6px;">'
                f'{s_hyp}'
                f'</div>'
                f'<div style="font-size: 13px; color: var(--text-secondary); line-height: 1.5;">'
                f'<strong>Supporting Evidence:</strong>'
                f'</div>'
                f'<ul style="margin: 4px 0 0 16px; padding: 0; font-size: 13px; color: var(--text-primary); line-height: 1.5;">'
                f'{s_ev_items}'
                f'</ul>'
                f'</div>'
            )
            st.markdown(s_html, unsafe_allow_html=True)

        other_rcs = [rc for rc in root_causes if rc not in (primary_rc, secondary_rc)]
        for rc in other_rcs:
            render_hypothesis_card(rc)

        # Dynamic Root Cause Evidence Validation Matrix Table
        st.markdown("##### 🧩 Root Cause & Evidence Validation Matrix")
        st.plotly_chart(
            render_root_cause_matrix_figure(root_causes, contributing_factors, missing),
            use_container_width=True,
        )
    else:
        st.info("Not specified in the source: No root causes identified in source document.")

    # -----------------------------------------------------------------------
    # Section 6 — Contributing Factors
    # -----------------------------------------------------------------------
    st.markdown("#### Section 6 — Contributing Factors")
    st.caption("Environmental, mechanical, or operational conditions that exacerbated the incident.")

    if contributing_factors:
        items_html = []
        for f in contributing_factors:
            f_esc = html.escape(str(f))
            items_html.append(
                f'<div class="contributing-factor-item">'
                f'<span style="color: var(--accent-blue); font-weight: bold; margin-right: 8px;">▪</span>'
                f'<span>{f_esc}</span>'
                f'</div>'
            )
        st.markdown("".join(items_html), unsafe_allow_html=True)
    else:
        st.info("Not specified in the source: No secondary contributing factors specified.")

    # -----------------------------------------------------------------------
    # Section 7 — Corrective & Preventive Actions
    # -----------------------------------------------------------------------
    st.markdown("#### Section 7 — Corrective & Preventive Actions")
    st.caption("Engineering remediation, maintenance schedule revisions, and monitoring upgrades.")

    actions = analysis.get("corrective_actions", [])
    if actions:
        act_rows = []
        for a in actions:
            clean_a = str(a).replace("[CIT-1]", "").replace("[CIT-2]", "").strip()
            a_lower = clean_a.lower()

            # Dynamic status extraction reflecting source text
            status = "Not specified in the source"
            if any(w in a_lower for w in ["completed", "implemented", "installed", "replaced", "cleaned", "flushed", "revised"]):
                status = "Implemented / Completed"
            elif any(w in a_lower for w in ["scheduled", "pending"]):
                status = "Scheduled / Pending"
            elif any(w in a_lower for w in ["recommend", "proposed"]):
                status = "Recommended"
            elif "mandate" in a_lower or "require" in a_lower:
                status = "Mandated Revision"
            else:
                status = "Documented in Report"

            # Dynamic component attribution
            comp = mach_id
            for c_kw in ["bearing", "fan", "cooling ribs", "fins", "cowl", "housing", "sensor", "valve", "impeller", "transformer", "bushing"]:
                if c_kw in a_lower:
                    comp = f"{mach_id} ({c_kw.title()})"
                    break

            # Purpose extraction
            purpose = "Engineering restoration of design tolerances"
            purp_m = re.search(r"(?:to|prevent|restore|eliminate|maintain)\s+([^,.;]+)", clean_a, re.IGNORECASE)
            if purp_m:
                purpose = purp_m.group(0).strip().capitalize()

            cit_m = re.search(r"\[(CIT-\d+)\]", str(a))
            cit_id = cit_m.group(0) if cit_m else "[CIT-1]"

            act_rows.append({
                "Action Description": clean_a,
                "Purpose / Scope": purpose,
                "Equipment / Component": comp,
                "Status": status,
                "Source Citation": cit_id,
            })

        df_act = pd.DataFrame(act_rows)
        st.dataframe(
            df_act,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Action Description": st.column_config.TextColumn("Action Description", width="large"),
                "Purpose / Scope": st.column_config.TextColumn("Purpose / Scope", width="large"),
                "Equipment / Component": st.column_config.TextColumn("Equipment / Component", width="medium"),
                "Status": st.column_config.TextColumn("Status", width="small"),
                "Source Citation": st.column_config.TextColumn("Citation", width="small"),
            },
        )
    else:
        st.info("Not specified in the source: No corrective actions specified.")

    # -----------------------------------------------------------------------
    # Section 8 — Missing Evidence & Uncertainties
    # -----------------------------------------------------------------------
    st.markdown("#### Section 8 — Missing Evidence & Uncertainties")
    st.caption("Forensic boundaries, unrecorded telemetry intervals, and confirmed non-occurrences.")

    if missing:
        li_items = "".join(f"<li style='margin-bottom: 5px;'>{html.escape(str(m))}</li>" for m in missing)
        miss_html = (
            f'<div class="industrial-card" style="border-left: 4px solid var(--warning-amber); background-color: #FFFDF5; margin-bottom: 12px;">'
            f'<div style="font-weight: 700; font-size: 13.5px; color: #92400E; margin-bottom: 6px;">'
            f'⚠️ Forensic Boundaries & Document Limitations'
            f'</div>'
            f'<ul style="margin: 0; padding-left: 20px; font-size: 13px; color: #78350F; line-height: 1.6;">'
            f'{li_items}'
            f'</ul>'
            f'</div>'
        )
        st.markdown(miss_html, unsafe_allow_html=True)
    else:
        st.info("No unresolved forensic evidence limitations identified.")

    # -----------------------------------------------------------------------
    # Section 9 — Evidence & Citations
    # -----------------------------------------------------------------------
    st.markdown("#### Section 9 — Evidence & Citations")
    st.caption("Traceable passage audit trail linking each finding directly to original indexed text chunks.")
    citations = analysis.get("citations", [])
    render_citation_accordion(citations)

    # Optional Advanced Diagnostics
    diagnostics = analysis.get("diagnostics", {})
    if diagnostics:
        with st.expander("⚙️ Advanced Retrieval & Pipeline Diagnostics", expanded=False):
            st.json(diagnostics)

