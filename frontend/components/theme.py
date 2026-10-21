"""Shared cached theme loader for the Streamlit frontend."""

from pathlib import Path
import streamlit as st

_THEME_CSS_PATH = Path(__file__).resolve().parent.parent / "styles" / "theme.css"


@st.cache_data(show_spinner=False)
def get_theme_css() -> str:
    """Reads and caches theme CSS from disk to avoid repeated I/O on every page rerun."""
    if _THEME_CSS_PATH.exists():
        with open(_THEME_CSS_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def load_theme():
    """Injects cached theme CSS into the active Streamlit page."""
    css = get_theme_css()
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
