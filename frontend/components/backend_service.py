"""Backend service manager for lifecycle, health polling, and Streamlit Cloud secrets bridging."""

import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional

import streamlit as st

logger = logging.getLogger("incident_rag.backend_service")


def sync_streamlit_secrets(env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Bridges Streamlit Community Cloud Secrets to the subprocess environment.

    Handles flat key-values as well as nested sections (e.g., [general], [secrets]).
    """
    if env is None:
        target_env = os.environ
    else:
        target_env = env

    try:
        if hasattr(st, "secrets"):
            for key, val in st.secrets.items():
                if isinstance(val, (str, int, float, bool)):
                    str_val = str(val).strip()
                    target_env[str(key)] = str_val
                    os.environ[str(key)] = str_val
                elif isinstance(val, dict):
                    for sub_k, sub_v in val.items():
                        if isinstance(sub_v, (str, int, float, bool)):
                            str_val = str(sub_v).strip()
                            target_env[str(sub_k)] = str_val
                            os.environ[str(sub_k)] = str_val
    except Exception as ex:
        logger.debug(f"Streamlit secrets sync skipped or failed: {ex}")

    return target_env


def is_backend_healthy(host: str = "127.0.0.1", port: int = 8000, timeout: float = 1.5) -> bool:
    """Synchronous health probe to check if the FastAPI backend is operational."""
    url = f"http://{host}:{port}/health"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "BackendHealthProbe"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_python_executable() -> str:
    """Finds the active virtual environment Python executable, falling back to sys.executable."""
    project_root = Path(__file__).resolve().parent.parent.parent
    # Check Windows venv
    venv_win = project_root / ".venv" / "Scripts" / "python.exe"
    if venv_win.is_file():
        return str(venv_win)
    # Check Linux/macOS venv
    venv_nix = project_root / ".venv" / "bin" / "python"
    if venv_nix.is_file():
        return str(venv_nix)
    return sys.executable


def _build_subprocess_env() -> Dict[str, str]:
    """Prepares sanitized environment variables for the FastAPI uvicorn subprocess."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    sync_streamlit_secrets(env)

    project_root = str(Path(__file__).resolve().parent.parent.parent)
    src_dir = str(Path(project_root) / "src")

    existing_pythonpath = env.get("PYTHONPATH", "")
    paths = [project_root, src_dir]
    if existing_pythonpath:
        paths.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


@st.cache_resource(show_spinner=False)
def ensure_backend_running(host: str = "127.0.0.1", port: int = 8000) -> bool:
    """Starts FastAPI in background if not already active.

    Cached via st.cache_resource to guarantee idempotent, single-process execution
    across all page navigations, browser refreshes, and multi-user sessions.
    """
    # 1. Check if a remote backend URL is configured (e.g. Render, Railway, Cloud Run)
    backend_url = os.environ.get("BACKEND_URL", "").strip()
    try:
        if hasattr(st, "secrets"):
            for k in ("BACKEND_URL", "backend_url", "API_URL", "api_url"):
                if k in st.secrets and str(st.secrets[k]).strip():
                    backend_url = str(st.secrets[k]).strip()
                    break
    except Exception:
        pass

    # If backend points to a remote cloud host, do not spawn local uvicorn
    if backend_url and not any(local in backend_url for local in ("127.0.0.1", "localhost", "0.0.0.0")):
        logger.info(f"Using remote backend at {backend_url}. Skipping local process spawn.")
        return True

    # 2. Check if local backend is already healthy
    if is_backend_healthy(host=host, port=port, timeout=1.0):
        logger.info(f"FastAPI backend already healthy on http://{host}:{port}")
        return True

    # 3. Spawn local uvicorn subprocess
    logger.info(f"Starting FastAPI backend on http://{host}:{port} ...")
    cmd = [
        get_python_executable(),
        "-m",
        "uvicorn",
        "incident_rag.api:app",
        "--host",
        host,
        "--port",
        str(port),
        "--app-dir",
        "src",
    ]

    env = _build_subprocess_env()

    try:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except Exception as e:
        logger.error(f"Failed to spawn uvicorn backend subprocess: {e}")
        return False

    # 4. Poll /health until ready
    health_url = f"http://{host}:{port}/health"
    max_retries = 30
    delay_sec = 0.8
    for attempt in range(1, max_retries + 1):
        if is_backend_healthy(host=host, port=port, timeout=1.0):
            logger.info(f"FastAPI backend is online and healthy on attempt {attempt}")
            return True
        if proc.poll() is not None:
            # Process terminated prematurely
            stderr_out = ""
            if proc.stderr:
                try:
                    stderr_out = proc.stderr.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
            logger.error(f"FastAPI backend exited with code {proc.returncode}. Stderr: {stderr_out}")
            return False
        time.sleep(delay_sec)

    logger.warning("FastAPI backend health check timed out after 24 seconds.")
    return False
