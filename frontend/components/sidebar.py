"""Sidebar component displaying system health, active models, and equipment quick-filters."""

from typing import Dict, Optional
import streamlit as st
from frontend.components.api_client import api_client


def render_sidebar():
    """Renders persistent enterprise sidebar."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.4rem 0 0.8rem 0;">
                <h2 style="margin: 0; color: #0F1E36; font-size: 16px; font-weight: 700; line-height: 1.3;">
                    ⚙️ Industrial Incident
                </h2>
                <div style="font-size: 12px; color: #64748B; font-weight: 500; margin-top: 2px;">
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
            status_text = "Online"
            docs_count = health.get("indexed_documents", 0)
            chunks_count = health.get("indexed_chunks", 0)
            groq_status = "Connected" if health.get("groq_configured") else "Extractive"
        except Exception:
            status_color = "#DC2626"
            status_text = "Disconnected"
            docs_count = 0
            chunks_count = 0
            groq_status = "Unavailable"

        st.markdown(
            f"""
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px 14px; margin-bottom: 0.9rem; box-shadow: 0 1px 3px rgba(15, 30, 54, 0.03);">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.4rem;">
                    <span style="font-size: 12px; font-weight: 600; color: #475569; text-transform: uppercase; letter-spacing: 0.04em;">System Status</span>
                    <span style="display: inline-flex; align-items: center; font-size: 12px; font-weight: 600; color: {status_color};">
                        <span style="height: 8px; width: 8px; background-color: {status_color}; border-radius: 50%; display: inline-block; margin-right: 6px; box-shadow: 0 0 6px {status_color}80;"></span>
                        {status_text}
                    </span>
                </div>
                <div style="font-size: 12.5px; color: #64748B; line-height: 1.5;">
                    <strong>Documents:</strong> {docs_count} &nbsp;|&nbsp; <strong>Chunks:</strong> {chunks_count}<br>
                    <strong>LLM Engine:</strong> {groq_status}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='font-size: 12.5px; font-weight: 600; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;'>🧭 Quick Navigation</div>",
            unsafe_allow_html=True,
        )
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

        st.markdown(
            "<div style='font-size: 12.5px; font-weight: 600; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;'>🏷️ Active Machinery</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <ul style="font-size: 12.5px; color: #334155; padding-left: 16px; margin: 0; line-height: 1.55;">
                <li><b>HP-17</b>: High-Pressure Steam Turbine</li>
                <li><b>RX-404</b>: Chemical Batch Reactor</li>
                <li><b>M-301</b>: 400 kW Stacker Motor</li>
                <li><b>K-102</b>: Gas Compressor</li>
                <li><b>BFP-02</b>: Feed Water Pump</li>
                <li><b>TR-200</b>: 50 MVA Power Transformer</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        st.markdown(
            """
            <div style="font-size: 11px; color: #94A3B8; text-align: center; line-height: 1.4;">
                Industrial Incident Investigator v1.0<br>
                Dense + BM25 + RRF + CrossEncoder
            </div>
            """,
            unsafe_allow_html=True,
        )
