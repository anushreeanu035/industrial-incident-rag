from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datetime import datetime
import hashlib
import html
import json
from typing import Any, Dict, List, Optional
import uuid
import streamlit as st

from frontend.components.api_client import api_client
from frontend.components.evidence_cards import (
    generate_document_analysis_markdown,
    generate_report_markdown,
    render_citation_accordion,
    render_document_analysis_report,
    render_hypothesis_card,
)
from frontend.components.sidebar import render_sidebar
from frontend.components.theme import load_theme

st.set_page_config(
    page_title="Investigation Workspace - Incident Investigator",
    page_icon="🛠️",
    layout="wide",
)

# Load cached enterprise theme CSS
load_theme()

render_sidebar(current_page="investigation")

# Top Navigation Bar
col_back, _ = st.columns([1, 4])
with col_back:
    if st.button("← Return to Dashboard", key="btn_back_home_inv"):
        st.switch_page("app.py")

st.markdown(
    """
    <div class="incident-header-banner">
        <h1>🛠️ Investigation Workspace</h1>
        <p>Industrial Forensic Investigation Workbench: Document-Driven Report Auditing & Evidence-Grounded Q&A</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Primary Two-Workflow Tabs
tab_doc, tab_qa = st.tabs([
    "📄 Analyze Incident Document (Document-Driven)",
    "❓ Ask a Question (Evidence-Grounded RAG)",
])


# ===========================================================================
# WORKFLOW A: Analyze Incident Document (Document-Driven)
# ===========================================================================
with tab_doc:
    st.markdown("### 📄 Analyze Incident Report")
    st.caption(
        "Directly evaluate an industrial incident report. Extracts the incident overview, machine telemetry, "
        "chronological sequence of events, root cause determinations, contributing factors, and corrective actions "
        "grounded solely in the selected document."
    )

    # Document Selection Source: Existing KB or Upload New
    source_mode = st.radio(
        "Document Source:",
        options=["Select from Knowledge Base", "Upload New Technical Document"],
        horizontal=True,
        key="doc_source_mode_radio",
    )

    active_doc_id: Optional[str] = None
    active_doc_filename: Optional[str] = None
    active_doc_meta: Dict[str, Any] = {}

    if source_mode == "Select from Knowledge Base":
        try:
            available_docs = api_client.list_documents()
        except Exception as e:
            available_docs = []
            st.error(f"Failed to fetch documents from knowledge base: {str(e)}")

        if not available_docs:
            st.warning("No incident documents are currently indexed in the knowledge base. Please upload a report below or seed sample data.")
        else:
            # Sort with M-301 conveyor motor report near the top if present
            sorted_docs = sorted(
                available_docs,
                key=lambda d: 0 if "INC-2023-044" in d.get("filename", "") else 1,
            )

            doc_options = {
                d["document_id"]: f"{d['filename']}  [Machine: {d.get('machine_id') or 'N/A'} | {d.get('incident_type') or 'Report'}]"
                for d in sorted_docs
            }

            doc_keys = list(doc_options.keys())
            preselected_id = st.session_state.get("selected_doc_id")
            default_idx = doc_keys.index(preselected_id) if preselected_id in doc_keys else 0

            selected_doc_id = st.selectbox(
                "Choose Incident Report to Analyze:",
                options=doc_keys,
                index=default_idx,
                format_func=lambda x: doc_options[x],
                key="sb_select_existing_doc",
            )

            if selected_doc_id:
                active_doc_id = selected_doc_id
                doc_dict = next((d for d in sorted_docs if d["document_id"] == selected_doc_id), {})
                active_doc_filename = doc_dict.get("filename", "Report")
                active_doc_meta = doc_dict

    else:  # Upload New Technical Document
        uploaded_file = st.file_uploader(
            "Upload technical incident report (.pdf, .txt, .md):",
            type=["pdf", "txt", "md", "markdown"],
            key="inv_tab_file_uploader",
            help="Maximum permitted file size: 25 MB. The document is indexed with isolated document ID.",
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            if len(file_bytes) > 25 * 1024 * 1024:
                st.error("❌ Validation Error: File size exceeds the maximum permitted limit of 25 MB.")
            elif len(file_bytes) == 0:
                st.error(f"❌ Validation Error: Uploaded file '{uploaded_file.name}' is empty.")
            else:
                current_hash = hashlib.sha256(file_bytes).hexdigest()
                cached_upload = st.session_state.get("tab_doc_uploaded_info")

                if not cached_upload or cached_upload.get("hash") != current_hash:
                    with st.spinner(f"Ingesting '{uploaded_file.name}' into knowledge base..."):
                        try:
                            resp = api_client.upload_document(
                                filename=uploaded_file.name,
                                content_bytes=file_bytes,
                                mime_type=uploaded_file.type or "application/octet-stream",
                            )
                            if resp.get("status") == "indexed":
                                st.session_state["tab_doc_uploaded_info"] = {
                                    "hash": current_hash,
                                    "filename": uploaded_file.name,
                                    "document_id": resp.get("document_id"),
                                    "chunks_created": resp.get("chunks_created", 0),
                                }
                            else:
                                st.session_state["tab_doc_uploaded_info"] = None
                                st.error(f"❌ Ingestion failed: {resp.get('message')}")
                        except Exception as e:
                            st.session_state["tab_doc_uploaded_info"] = None
                            st.error(f"❌ Ingestion error: {str(e)}")

                cached_upload = st.session_state.get("tab_doc_uploaded_info")
                if cached_upload and cached_upload.get("hash") == current_hash:
                    active_doc_id = cached_upload["document_id"]
                    active_doc_filename = cached_upload["filename"]
                    active_doc_meta = {
                        "filename": active_doc_filename,
                        "document_id": active_doc_id,
                        "chunk_count": cached_upload.get("chunks_created", 0),
                    }
                    st.success(f"✅ Document `{active_doc_filename}` ready for forensic analysis.")

    # Invalidate cached analysis results if the active document has changed
    prev_doc_id = st.session_state.get("analyzed_doc_id")
    if active_doc_id and prev_doc_id and prev_doc_id != active_doc_id:
        st.session_state.pop("doc_analysis_result", None)
        st.session_state["analyzed_doc_id"] = None

    # Selected Document Details Card
    if active_doc_id:
        doc_card_html = (
            f'<div class="industrial-card" style="border-left: 4px solid var(--accent-blue); padding: 10px 16px; margin: 10px 0 16px 0;">'
            f'<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">'
            f'<div>'
            f'<strong style="color: var(--primary-navy); font-size: 14px;">Target Report:</strong> '
            f'<code style="font-size: 13.5px; font-weight: 600; color: var(--accent-blue);">{html.escape(str(active_doc_filename))}</code>'
            f'</div>'
            f'<div style="font-size: 12.5px; color: var(--text-secondary);">'
            f'<strong>Doc ID:</strong> <code>{html.escape(str(active_doc_id))}</code> &nbsp;|&nbsp; '
            f'<strong>Machine:</strong> <code>{html.escape(str(active_doc_meta.get("machine_id", "Unassigned")))}</code>'
            f'</div>'
            f'</div>'
            f'</div>'
        )
        st.markdown(doc_card_html, unsafe_allow_html=True)

        col_btn_doc, _ = st.columns([1, 2])
        with col_btn_doc:
            btn_analyze = st.button(
                f"🔍 Analyze Document",
                key="btn_execute_doc_analysis",
                type="primary",
                use_container_width=True,
            )

        if btn_analyze:
            with st.spinner(f"Analyzing '{active_doc_filename}': extracting timeline, telemetry, root causes, and citations..."):
                try:
                    analysis_result = api_client.analyze_document(active_doc_id)
                    st.session_state["doc_analysis_result"] = analysis_result
                    st.session_state["analyzed_doc_id"] = active_doc_id

                    # Record in session investigation history for Dashboard
                    if "investigation_history" not in st.session_state:
                        st.session_state["investigation_history"] = []
                    st.session_state["investigation_history"].append({
                        "id": str(uuid.uuid4())[:8],
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "mode": "Document Analysis",
                        "query_or_doc": f"Doc: {active_doc_filename}",
                        "machine_id": active_doc_meta.get("machine_id", "M-301"),
                        "status": "Completed",
                        "is_sufficient": True,
                        "findings": (analysis_result.get("executive_summary") or "Structured forensic report generated.")[:130] + "...",
                        "data": analysis_result,
                    })
                except Exception as e:
                    st.error(f"❌ Document analysis failed: {str(e)}")

        # Display Structured Analysis Results if available for current document
        if "doc_analysis_result" in st.session_state and st.session_state.get("analyzed_doc_id") == active_doc_id:
            st.divider()
            render_document_analysis_report(st.session_state["doc_analysis_result"])
    else:
        st.info("Select or upload an incident document above to begin document analysis.")


# ===========================================================================
# WORKFLOW B: Ask a Question (Evidence-Grounded RAG)
# ===========================================================================
with tab_qa:
    st.markdown("### ❓ Ask a Forensic Question")
    st.caption(
        "Submit technical questions regarding equipment failures, root causes, or telemetry across the knowledge base. "
        "Answers are synthesized strictly from relevant retrieved evidence passages with cross-encoder validation."
    )

    # Sample Query Presets
    st.markdown("##### ⚡ Quick Investigation Presets:")
    st.caption("Motor M-301 Overheating Inquiry Benchmarks:")
    p_row1 = st.columns(4)
    with p_row1[0]:
        if st.button("🔥 Overheating Root Cause", key="preset_m301_cause", use_container_width=True):
            st.session_state["qa_query_text"] = "What caused motor M-301 to overheat?"
            st.session_state["qa_machine_choice"] = "M-301"
    with p_row1[1]:
        if st.button("🔎 Supporting Evidence", key="preset_m301_evidence", use_container_width=True):
            st.session_state["qa_query_text"] = "What evidence supports the reported root cause?"
            st.session_state["qa_machine_choice"] = "M-301"
    with p_row1[2]:
        if st.button("🛠️ Corrective Actions", key="preset_m301_actions", use_container_width=True):
            st.session_state["qa_query_text"] = "Which corrective actions were recommended?"
            st.session_state["qa_machine_choice"] = "M-301"
    with p_row1[3]:
        if st.button("⚡ Insulation Failure?", key="preset_m301_insulation", use_container_width=True):
            st.session_state["qa_query_text"] = "Is there evidence of insulation failure?"
            st.session_state["qa_machine_choice"] = "M-301"

    st.caption("Cross-Equipment Plant Incidents:")
    p_row2 = st.columns(3)
    with p_row2[0]:
        if st.button("Turbine HP-17 Vibration", key="preset_hp17", use_container_width=True):
            st.session_state["qa_query_text"] = "What caused the high radial vibration and babbitt wiping on steam turbine HP-17?"
            st.session_state["qa_machine_choice"] = "HP-17"
    with p_row2[1]:
        if st.button("Reactor RX-404 Excursion", key="preset_rx404", use_container_width=True):
            st.session_state["qa_query_text"] = "Why did reactor RX-404 overheat and why did emergency quench pump QP-02 fail to start?"
            st.session_state["qa_machine_choice"] = "RX-404"
    with p_row2[2]:
        if st.button("Transformer TR-200 Flash", key="preset_tr200", use_container_width=True):
            st.session_state["qa_query_text"] = "What was the failure mechanism on Bushing B-Phase that caused transformer TR-200 explosion?"
            st.session_state["qa_machine_choice"] = "TR-200"

    # Investigation QA Form
    with st.form("qa_investigation_form"):
        user_query = st.text_area(
            "Investigation Question / Technical Equipment Inquiry:",
            value=st.session_state.get("qa_query_text", "What caused motor M-301 to overheat?"),
            height=85,
            placeholder="Enter technical query regarding equipment alarms, failure mechanisms, trip events, or repair history...",
        )

        c1, c2, c3, c4 = st.columns([2, 3, 2, 2], vertical_alignment="bottom")
        with c1:
            all_machine_options = ["All Equipment", "HP-17", "RX-404", "M-301", "K-102", "BFP-02", "TR-200"]
            default_machine = st.session_state.get("qa_machine_choice", "All Equipment")
            machine_idx = all_machine_options.index(default_machine) if default_machine in all_machine_options else 0
            selected_machine = st.selectbox(
                "Machine ID Filter (Optional):",
                options=all_machine_options,
                index=machine_idx,
            )
        with c2:
            strategy_label = st.selectbox(
                "Retrieval Strategy:",
                options=[
                    "Full Advanced RAG (Rewriting + RRF + Rerank)",
                    "Hybrid + Cross-Encoder Rerank",
                    "Hybrid (BM25 + Dense RRF)",
                    "Dense Vector Search",
                    "BM25 Keyword Search",
                ],
                index=0,
            )
        with c3:
            top_k_val = st.slider("Context Passages (Top-K):", min_value=1, max_value=15, value=5)
        with c4:
            enable_rewriting = st.checkbox("Enable Query Rewriting", value=True)

        qa_submit_button = st.form_submit_button("🔍 Run Incident Investigation", type="primary", use_container_width=True)

    STRATEGY_MAP = {
        "Full Advanced RAG (Rewriting + RRF + Rerank)": "full_advanced",
        "Hybrid + Cross-Encoder Rerank": "hybrid_rerank",
        "Hybrid (BM25 + Dense RRF)": "hybrid_rrf",
        "Dense Vector Search": "dense",
        "BM25 Keyword Search": "bm25",
    }

    if qa_submit_button:
        if not user_query.strip():
            st.warning("Please enter an investigation question.")
        else:
            m_filter = None if selected_machine == "All Equipment" else selected_machine
            strat_key = STRATEGY_MAP[strategy_label]

            with st.spinner("Retrieving evidence passages, evaluating equipment alignment, and synthesizing grounded findings..."):
                try:
                    qa_response = api_client.investigate(
                        query=user_query,
                        strategy=strat_key,
                        top_k=top_k_val,
                        machine_id_filter=m_filter,
                        enable_rewriting=enable_rewriting,
                    )
                    st.session_state["qa_latest_investigation"] = qa_response
                    st.session_state["qa_last_query"] = user_query

                    # Record in session investigation history for Dashboard
                    if "investigation_history" not in st.session_state:
                        st.session_state["investigation_history"] = []
                    st.session_state["investigation_history"].append({
                        "id": str(uuid.uuid4())[:8],
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "mode": "Evidence Q&A",
                        "query_or_doc": user_query[:80] + ("..." if len(user_query) > 80 else ""),
                        "machine_id": selected_machine if selected_machine != "All Equipment" else "All",
                        "status": "Completed" if qa_response.get("is_sufficient_evidence", True) else "Mismatch / Limited",
                        "is_sufficient": qa_response.get("is_sufficient_evidence", True),
                        "findings": (qa_response.get("executive_summary") or "Grounded Q&A forensic findings")[:130] + "...",
                        "data": qa_response,
                    })
                except Exception as e:
                    st.error(f"Investigation execution failed: {str(e)}")

    # Display Question-Answering Results
    if "qa_latest_investigation" in st.session_state:
        qa_resp = st.session_state["qa_latest_investigation"]
        investigated_query = st.session_state.get("qa_last_query", user_query)

        # Mismatch or Insufficient Evidence Banner
        if not qa_resp.get("is_sufficient_evidence", True):
            st.markdown(
                """
                <div class="industrial-card" style="border-left: 4px solid var(--warning-amber); background-color: #FFFDF5; margin-bottom: 14px;">
                    <div style="font-weight: 700; font-size: 14px; color: #92400E; margin-bottom: 4px;">
                        ⚠️ Equipment Mismatch / Insufficient Grounded Evidence
                    </div>
                    <p style="margin: 0; font-size: 13.5px; color: #78350F; line-height: 1.55;">
                        The retrieved documentation does not match the requested machine or provide verified evidence to confirm the cause of the specified incident.
                        See the forensic explanation and missing evidence breakdown below.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        col_qa_hdr, col_qa_dl1, col_qa_dl2 = st.columns([3, 1, 1], vertical_alignment="bottom")
        with col_qa_hdr:
            st.markdown("### 📋 Executive Investigation Summary")
        with col_qa_dl1:
            qa_md = generate_report_markdown(qa_resp, investigated_query)
            st.download_button(
                "📥 Download Report (.md)",
                data=qa_md,
                file_name="incident_investigation_qa_report.md",
                mime="text/markdown",
                use_container_width=True,
                type="primary",
            )
        with col_qa_dl2:
            st.download_button(
                "📥 Export JSON",
                data=json.dumps(qa_resp, indent=2),
                file_name="incident_investigation_qa_report.json",
                mime="application/json",
                use_container_width=True,
            )

        # Executive Summary Box
        border_color = "var(--warning-amber)" if not qa_resp.get("is_sufficient_evidence", True) else "var(--accent-blue)"
        st.markdown(
            f"""
            <div class="report-section-card" style="border-left: 4px solid {border_color}; background-color: var(--bg-card);">
                <p style="font-size: 14.5px; color: var(--text-primary); line-height: 1.65; margin: 0; font-weight: 500;">
                    {qa_resp.get("executive_summary", "No summary provided.")}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Documented Observations & Telemetry
        obs_list = [str(o).strip() for o in qa_resp.get("documented_observations", []) if str(o).strip()]
        if obs_list:
            st.markdown("### 🔍 Documented Physical Observations & Telemetry")
            for obs in obs_list:
                st.markdown(f"- {obs}")

        # Root Cause Hypotheses
        hypotheses = qa_resp.get("potential_root_causes", [])
        if hypotheses:
            st.markdown("### 🎯 Potential Root Causes & Hypotheses")
            for hyp in hypotheses:
                render_hypothesis_card(hyp)

        # Corrective Actions
        actions = [str(a).strip() for a in qa_resp.get("corrective_actions", []) if str(a).strip()]
        if actions:
            st.markdown("### 🛠️ Mandated Corrective Actions")
            for act in actions:
                st.markdown(f"- {act}")

        # Missing Evidence / Forensic Uncertainties
        uncertainties = [str(u).strip() for u in qa_resp.get("missing_evidence_and_uncertainties", []) if str(u).strip()]
        if uncertainties:
            st.markdown("### ❓ Missing Evidence & Forensic Uncertainties")
            for u in uncertainties:
                st.markdown(f"- {u}")

        st.divider()

        # Expandable Citations and Audit Trail
        render_citation_accordion(qa_resp.get("citations", []))

# ===========================================================================
# Investigation Session History Drawer (Reopen & Audit)
# ===========================================================================
inv_history = st.session_state.get("investigation_history", [])
if inv_history:
    st.divider()
    with st.expander(f"🕒 Investigation Session History ({len(inv_history)} Investigations)", expanded=False):
        for idx, h_item in enumerate(reversed(inv_history), start=1):
            col_h1, col_h2, col_h3 = st.columns([4, 2, 2])
            with col_h1:
                st.markdown(f"**#{idx} [{h_item.get('mode')}]** {h_item.get('query_or_doc')}")
                st.caption(f"Machine: `{h_item.get('machine_id')}` | Status: `{h_item.get('status')}` | Time: `{h_item.get('timestamp')}`")
            with col_h2:
                st.markdown(f"<div style='font-size: 12px; color: var(--text-secondary);'>{h_item.get('findings')}</div>", unsafe_allow_html=True)
            with col_h3:
                if st.button("🔄 Reopen Report", key=f"btn_reopen_hist_{idx}", use_container_width=True):
                    if h_item.get("mode") == "Document Analysis":
                        st.session_state["doc_analysis_result"] = h_item.get("data")
                        st.session_state["doc_source_mode_radio"] = "Select from Knowledge Base"
                    else:
                        st.session_state["qa_latest_investigation"] = h_item.get("data")
                    st.rerun()
