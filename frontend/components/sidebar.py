"""Sidebar component displaying system health, active models, navigation, and equipment quick-filters."""

import html
from typing import Dict, Optional
import streamlit as st
from frontend.components.api_client import api_client


def render_sidebar(current_page: Optional[str] = None):
    """Renders persistent enterprise sidebar with active page highlighting and live health polling."""
    if current_page is None:
        current_page = st.session_state.get("active_nav_page", "dashboard")
    else:
        st.session_state["active_nav_page"] = current_page

    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.5rem 0 0.8rem 0;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 22px;">⚙️</span>
                    <div>
                        <h2 style="margin: 0; color: #0F1E36; font-size: 15px; font-weight: 700; line-height: 1.25;">
                            Industrial Incident
                        </h2>
                        <div style="font-size: 11.5px; color: #2563EB; font-weight: 600; letter-spacing: 0.03em;">
                            Investigator Platform
                        </div>
                    </div>
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
            groq_status = "Connected" if health.get("groq_configured") else "Offline Fallback"
            is_connected = True
        except Exception:
            status_color = "#DC2626"
            status_text = "Disconnected"
            docs_count = 0
            chunks_count = 0
            groq_status = "Unavailable"
            is_connected = False

        backend_url_display = api_client.base_url
        if len(backend_url_display) > 30:
            backend_url_display = backend_url_display[:28] + "..."

        st.markdown(
            f"""
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 12px; margin-bottom: 0.8rem; box-shadow: 0 1px 2px rgba(15, 30, 54, 0.03);">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.35rem;">
                    <span style="font-size: 11px; font-weight: 600; color: #475569; text-transform: uppercase; letter-spacing: 0.04em;">Backend Status</span>
                    <span style="display: inline-flex; align-items: center; font-size: 11.5px; font-weight: 600; color: {status_color};">
                        <span style="height: 7px; width: 7px; background-color: {status_color}; border-radius: 50%; display: inline-block; margin-right: 5px; box-shadow: 0 0 5px {status_color}80;"></span>
                        {status_text}
                    </span>
                </div>
                <div style="font-size: 11px; color: #64748B; line-height: 1.45; word-break: break-all;">
                    <strong>Host:</strong> <code>{html.escape(backend_url_display)}</code><br>
                    <strong>Dossiers:</strong> {docs_count} &nbsp;|&nbsp; <strong>Chunks:</strong> {chunks_count}<br>
                    <strong>LLM Engine:</strong> {groq_status}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not is_connected:
            col_reconn, _ = st.columns([1, 0.1])
            with col_reconn:
                if st.button("🔄 Reconnect Backend", key="btn_sb_reconnect", use_container_width=True):
                    api_client.clear_cache("health")
                    st.rerun()

        st.markdown(
            "<div style='font-size: 11px; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;'>🧭 Navigation</div>",
            unsafe_allow_html=True,
        )

        nav_links = [
            ("dashboard", "Dashboard", "📊", "app.py"),
            ("investigation", "Investigation", "🛠️", "pages/1_Investigation.py"),
            ("retrieval", "Retrieval Explorer", "🔬", "pages/2_Retrieval_Explorer.py"),
            ("evaluation", "RAG Evaluation", "📈", "pages/3_RAG_Evaluation.py"),
            ("knowledge_base", "Knowledge Base", "📁", "pages/4_Knowledge_Base.py"),
            ("settings", "Settings", "⚙️", "pages/5_Settings.py"),
        ]

        for page_id, name, icon, path in nav_links:
            is_active = (current_page == page_id)
            btn_label = f"{icon}  {name}  ◀" if is_active else f"{icon}  {name}"
            btn_type = "primary" if is_active else "secondary"
            if st.button(btn_label, key=f"sb_nav_{page_id}", type=btn_type, use_container_width=True):
                if not is_active:
                    st.session_state["active_nav_page"] = page_id
                    st.switch_page(path)

        st.divider()

        st.markdown(
            "<div style='font-size: 12.5px; font-weight: 600; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;'>🏷️ Active Machinery</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <ul style="font-size: 12px; color: #334155; padding-left: 16px; margin: 0; line-height: 1.55;">
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
