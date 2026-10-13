from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import hashlib
from typing import Any, Dict
import streamlit as st
from frontend.components.api_client import api_client
from frontend.components.evidence_cards import (
    generate_report_markdown,
    render_citation_accordion,
    render_hypothesis_card,
)
from frontend.components.sidebar import render_sidebar

st.set_page_config(
    page_title="Investigation Workspace - Incident Investigator",
    page_icon="🛠️",
    layout="wide",
)

# Load theme CSS
css_path = Path(__file__).resolve().parent.parent / "styles" / "theme.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

render_sidebar()

col_back, _ = st.columns([1, 3])
with col_back:
    if st.button("← Return to Home Dashboard", key="btn_back_home_inv"):
        st.switch_page("app.py")

st.markdown(
    """
    <div class="incident-header-banner">
        <h1>🛠️ Investigation Workspace</h1>
        <p>Forensic Root Cause Analysis, Telemetry Correlation & Evidence-Grounded Synthesis</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sample Query Shortcuts
st.markdown("##### ⚡ Quick Investigation Presets:")
preset_cols = st.columns(3)
with preset_cols[0]:
    if st.button("🔥 HP-17 Bearing Vibration Spikes", use_container_width=True):
        st.session_state["query_input"] = "What caused the high radial vibration and babbitt wiping on steam turbine HP-17?"
        st.session_state["machine_filter"] = "HP-17"
with preset_cols[1]:
    if st.button("🚨 RX-404 Thermal Runaway & Burst Disk", use_container_width=True):
        st.session_state["query_input"] = "Why did reactor RX-404 overheat and why did emergency quench pump QP-02 fail to start?"
        st.session_state["machine_filter"] = "RX-404"
with preset_cols[2]:
    if st.button("⚡ TR-200 Transformer Arc Flash", use_container_width=True):
        st.session_state["query_input"] = "What was the failure mechanism on Bushing B-Phase that caused transformer TR-200 explosion?"
        st.session_state["machine_filter"] = "TR-200"

# User Document Upload Option
st.markdown("##### 📄 Analyze User-Uploaded Document:")
uploaded_file = st.file_uploader(
    "Choose a technical document (.pdf, .txt, .md) to analyze directly:",
    type=["pdf", "txt", "md", "markdown"],
    key="inv_file_uploader",
    help="Upload limit: 25 MB. The document is chunked, embedded, and indexed with strict document-level isolation.",
)

doc_filter = None
active_doc_name = None

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    if len(file_bytes) > 25 * 1024 * 1024:
        st.error("❌ Validation Error: File size exceeds the maximum permitted limit of 25 MB.")
    elif len(file_bytes) == 0:
        st.error(f"❌ Validation Error: Uploaded file '{uploaded_file.name}' is empty.")
    else:
        current_hash = hashlib.sha256(file_bytes).hexdigest()
        cached_info = st.session_state.get("uploaded_doc_info")

        if not cached_info or cached_info.get("hash") != current_hash:
            with st.spinner(f"Ingesting '{uploaded_file.name}': extracting text, chunking, and indexing into vector & BM25 stores..."):
                try:
                    resp = api_client.upload_document(
                        filename=uploaded_file.name,
                        content_bytes=file_bytes,
                        mime_type=uploaded_file.type or "application/octet-stream",
                    )
                    if resp.get("status") == "indexed":
                        st.session_state["uploaded_doc_info"] = {
                            "hash": current_hash,
                            "filename": uploaded_file.name,
                            "document_id": resp.get("document_id"),
                            "chunks_created": resp.get("chunks_created", 0),
                            "status": "Ready for Analysis",
                        }
                    else:
                        st.session_state["uploaded_doc_info"] = None
                        st.error(f"❌ Processing failed: {resp.get('message')}")
                except Exception as e:
                    st.session_state["uploaded_doc_info"] = None
                    st.error(f"❌ Ingestion error: {str(e)}")

        cached_info = st.session_state.get("uploaded_doc_info")
        if cached_info and cached_info.get("hash") == current_hash:
            col_u1, col_u2, col_u3 = st.columns([2, 1, 1])
            with col_u1:
                st.success(f"✅ **{cached_info['filename']}** uploaded and indexed ({cached_info['chunks_created']} chunks)")
            with col_u2:
                st.info(f"**Doc ID:** `{cached_info['document_id']}`")
            with col_u3:
                st.caption(f"**Status:** {cached_info['status']}")

            scope_opts = [
                f"Uploaded Document: {cached_info['filename']} (Isolated)",
                "Entire Knowledge Base",
            ]
            scope_choice = st.radio(
                "Investigation Target Scope:",
                options=scope_opts,
                index=0,
                horizontal=True,
                key="scope_choice_radio",
            )
            if scope_choice == scope_opts[0]:
                doc_filter = cached_info["document_id"]
                active_doc_name = cached_info["filename"]
else:
    if "uploaded_doc_info" in st.session_state and st.session_state["uploaded_doc_info"] is not None:
        st.session_state["uploaded_doc_info"] = None

# Investigation Form Controls
with st.form("investigation_form"):
    user_query = st.text_area(
        "Investigation Query / Equipment Inquiry:",
        value=st.session_state.get("query_input", "What caused the high vibration and babbitt wiping on steam turbine HP-17?"),
        height=85,
        placeholder="Enter technical question regarding machine trips, root causes, sensor alarms, or maintenance records...",
    )

    c1, c2, c3, c4 = st.columns([2, 3, 2, 2], vertical_alignment="bottom")
    with c1:
        machine_id_choice = st.selectbox(
            "Machine ID Filter (Optional):",
            options=["All Equipment", "HP-17", "RX-404", "M-301", "K-102", "BFP-02", "TR-200"],
            index=0 if "machine_filter" not in st.session_state else (
                ["All Equipment", "HP-17", "RX-404", "M-301", "K-102", "BFP-02", "TR-200"].index(st.session_state["machine_filter"])
                if st.session_state["machine_filter"] in ["HP-17", "RX-404", "M-301", "K-102", "BFP-02", "TR-200"] else 0
            ),
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

    submit_label = (
        f"🔍 Analyze Uploaded Document ({active_doc_name})"
        if doc_filter
        else "🔍 Run Incident Investigation"
    )
    submit_button = st.form_submit_button(submit_label, use_container_width=True)

# Map human-readable strategy to API enum
STRATEGY_MAP = {
    "Full Advanced RAG (Rewriting + RRF + Rerank)": "full_advanced",
    "Hybrid + Cross-Encoder Rerank": "hybrid_rerank",
    "Hybrid (BM25 + Dense RRF)": "hybrid_rrf",
    "Dense Vector Search": "dense",
    "BM25 Keyword Search": "bm25",
}

if submit_button:
    if not user_query.strip():
        st.warning("Please enter an investigation question.")
    else:
        m_filter = None if machine_id_choice == "All Equipment" else machine_id_choice
        strat_key = STRATEGY_MAP[strategy_label]

        with st.spinner("Analyzing incident telemetry, retrieving evidence passages, and generating grounded report..."):
            try:
                response = api_client.investigate(
                    query=user_query,
                    strategy=strat_key,
                    top_k=top_k_val,
                    machine_id_filter=m_filter,
                    enable_rewriting=enable_rewriting,
                    document_id_filter=doc_filter,
                )

                st.session_state["latest_investigation"] = response
                st.session_state["last_investigated_query"] = user_query
            except Exception as e:
                st.error(f"Investigation execution failed: {str(e)}")

# Display Investigation Results
if "latest_investigation" in st.session_state:
    resp = st.session_state["latest_investigation"]

    # Warning if evidence is insufficient
    if not resp.get("is_sufficient_evidence", True):
        st.warning(
            "⚠️ **Warning: Insufficient Evidence in Knowledge Base.** "
            "The retrieved documentation does not provide definitive evidence to answer this inquiry with certainty. "
            "Review missing uncertainties below."
        )

    col_res_header, col_download = st.columns([3, 1], vertical_alignment="bottom")
    with col_res_header:
        st.markdown("### 📋 Executive Investigation Summary")
    with col_download:
        report_md = generate_report_markdown(
            resp, st.session_state.get("last_investigated_query", user_query)
        )
        st.download_button(
            "📥 Download Report",
            data=report_md,
            file_name="incident_investigation_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    st.markdown(
        f"""
        <div class="industrial-card" style="border-left: 4px solid #2563EB; background-color: #F8FAFC;">
            <p style="font-size: 14px; color: #1E293B; line-height: 1.6; margin: 0;">
                {resp.get("executive_summary", "No summary provided.")}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Documented Observations & Telemetry
    obs_list = [str(o).strip() for o in resp.get("documented_observations", []) if str(o).strip()]
    if obs_list:
        st.markdown("### 🔍 Documented Physical Observations & Telemetry")
        for obs in obs_list:
            st.markdown(f"- {obs}")

    # Root Cause Hypotheses
    hypotheses = resp.get("potential_root_causes", [])
    if hypotheses:
        st.markdown("### 🎯 Potential Root Causes & Hypotheses")
        for hyp in hypotheses:
            render_hypothesis_card(hyp)

    # Corrective and Preventive Actions
    actions = [str(a).strip() for a in resp.get("corrective_actions", []) if str(a).strip()]
    if actions:
        st.markdown("### 🛠️ Mandated Corrective Actions")
        for act in actions:
            st.markdown(f"- {act}")

    # Missing Evidence / Uncertainties
    uncertainties = [str(u).strip() for u in resp.get("missing_evidence_and_uncertainties", []) if str(u).strip()]
    if uncertainties:
        st.markdown("### ❓ Missing Evidence & Forensic Uncertainties")
        for u in uncertainties:
            st.markdown(f"- {u}")

    st.divider()

    # Expandable Citations and Audit Trail
    render_citation_accordion(resp.get("citations", []))
