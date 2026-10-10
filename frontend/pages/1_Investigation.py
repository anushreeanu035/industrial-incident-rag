from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
from frontend.components.api_client import api_client
from frontend.components.evidence_cards import render_citation_accordion, render_hypothesis_card
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

# Investigation Form Controls
with st.form("investigation_form"):
    user_query = st.text_area(
        "Investigation Query / Equipment Inquiry:",
        value=st.session_state.get("query_input", "What caused the high vibration and babbitt wiping on steam turbine HP-17?"),
        height=90,
        placeholder="Enter technical question regarding machine trips, root causes, sensor alarms, or maintenance records...",
    )

    c1, c2, c3, c4 = st.columns(4)
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

    submit_button = st.form_submit_button("🔍 Run Incident Investigation", use_container_width=True)

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
                )

                st.session_state["latest_investigation"] = response
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

    st.markdown("### 📋 Executive Investigation Summary")
    st.markdown(
        f"""
        <div class="industrial-card" style="border-left: 4px solid #0F1E36; background-color: #F8FAFC;">
            <p style="font-size: 1.05rem; color: #1E293B; line-height: 1.6; margin: 0;">
                {resp.get("executive_summary", "No summary provided.")}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Documented Observations & Telemetry
    obs_list = resp.get("documented_observations", [])
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
    actions = resp.get("corrective_actions", [])
    if actions:
        st.markdown("### 🛠️ Mandated Corrective Actions")
        for act in actions:
            st.markdown(f"- {act}")

    # Missing Evidence / Uncertainties
    uncertainties = resp.get("missing_evidence_and_uncertainties", [])
    if uncertainties:
        st.markdown("### ❓ Missing Evidence & Forensic Uncertainties")
        for u in uncertainties:
            st.markdown(f"- {u}")

    st.divider()

    # Expandable Citations and Audit Trail
    render_citation_accordion(resp.get("citations", []))
