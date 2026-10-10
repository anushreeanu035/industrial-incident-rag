"""Sidebar component displaying system health, active models, and equipment quick-filters."""

from typing import Dict, Optional
import streamlit as st
from frontend.components.api_client import api_client


def render_sidebar():
    """Renders persistent enterprise sidebar."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.5rem 0 1rem 0;">
                <h2 style="margin: 0; color: #0F1E36; font-size: 1.3rem; font-weight: 700;">
                    ⚙️ Industrial Incident
                </h2>
                <div style="font-size: 0.85rem; color: #64748B; font-weight: 500;">
                    Investigator Platform
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # Check Backend Health
        try:
            health = api_client.check_health()
            status_color = "#16A34A"
            status_text = "Backend Online"
            docs_count = health.get("indexed_documents", 0)
            chunks_count = health.get("indexed_chunks", 0)
            groq_status = "Connected" if health.get("groq_configured") else "Offline (Extractive Fallback)"
        except Exception:
            status_color = "#DC2626"
            status_text = "Backend Disconnected"
            docs_count = 0
            chunks_count = 0
            groq_status = "Unavailable"

        st.markdown(
            f"""
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.8rem; margin-bottom: 1rem;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.4rem;">
                    <span style="font-size: 0.8rem; font-weight: 600; color: #475569;">System Status</span>
                    <span style="display: inline-flex; align-items: center; font-size: 0.75rem; font-weight: 600; color: {status_color};">
                        <span style="height: 8px; width: 8px; background-color: {status_color}; border-radius: 50%; display: inline-block; margin-right: 5px;"></span>
                        {status_text}
                    </span>
                </div>
                <div style="font-size: 0.75rem; color: #64748B;">
                    <strong>Indexed Documents:</strong> {docs_count}<br>
                    <strong>Vector Chunks:</strong> {chunks_count}<br>
                    <strong>Groq LLM:</strong> {groq_status}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 🧭 Quick Navigation")
        if st.button("🏠 Home Dashboard", key="sb_btn_home", use_container_width=True):
            st.switch_page("app.py")
        if st.button("🛠️ Investigation Workspace", key="sb_btn_inv", use_container_width=True):
            st.switch_page("pages/1_Investigation.py")
        if st.button("🔬 Retrieval Explorer", key="sb_btn_ret", use_container_width=True):
            st.switch_page("pages/2_Retrieval_Explorer.py")
        if st.button("📊 RAG Evaluation", key="sb_btn_eval", use_container_width=True):
            st.switch_page("pages/3_RAG_Evaluation.py")
        if st.button("📁 Knowledge Base", key="sb_btn_kb", use_container_width=True):
            st.switch_page("pages/4_Knowledge_Base.py")
        if st.button("⚙️ Settings", key="sb_btn_set", use_container_width=True):
            st.switch_page("pages/5_Settings.py")

        st.divider()

        st.markdown("### 🏷️ Active Machinery")
        st.markdown(
            """
            - **HP-17**: High-Pressure Steam Turbine
            - **RX-404**: Chemical Batch Reactor
            - **M-301**: 400 kW Stacker Motor
            - **K-102**: Centrifugal Feed Gas Compressor
            - **BFP-02**: Boiler Feed Water Pump
            - **TR-200**: 50 MVA Power Transformer
            """
        )

        st.divider()

        st.markdown(
            """
            <div style="font-size: 0.75rem; color: #94A3B8; text-align: center;">
                Industrial Incident Investigator v1.0<br>
                Dense + BM25 + RRF + CrossEncoder
            </div>
            """,
            unsafe_allow_html=True,
        )
