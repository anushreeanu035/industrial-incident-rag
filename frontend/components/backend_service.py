"""Backend service manager for lifecycle, health polling, and Streamlit Cloud secrets bridging."""

import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

import streamlit as st

logger = logging.getLogger("incident_rag.backend_service")


def extract_groq_credentials() -> Tuple[Optional[str], str]:
    """Extracts Groq API key and model from st.secrets or os.environ across all common formats."""
    api_key: Optional[str] = None
    model: str = "openai/gpt-oss-120b"

    # 1. Check os.environ
    for k in ("GROQ_API_KEY", "groq_api_key", "GROQ_KEY", "groq_key"):
        val = os.environ.get(k, "").strip().strip("\"'").strip()
        if val:
            api_key = val
            break

    for k in ("GROQ_MODEL", "groq_model"):
        val = os.environ.get(k, "").strip().strip("\"'").strip()
        if val:
            model = val
            break

    # 2. Check st.secrets (flat keys)
    try:
        if hasattr(st, "secrets"):
            for k in ("GROQ_API_KEY", "groq_api_key", "GROQ_KEY", "groq_key"):
                if k in st.secrets and str(st.secrets[k]).strip():
                    api_key = str(st.secrets[k]).strip().strip("\"'").strip()
                    break

            for k in ("GROQ_MODEL", "groq_model"):
                if k in st.secrets and str(st.secrets[k]).strip():
                    model = str(st.secrets[k]).strip().strip("\"'").strip()
                    break

            # 3. Check nested sections in st.secrets (e.g. [general], [secrets], [groq], [default])
            for section in ("general", "secrets", "groq", "default"):
                if section in st.secrets:
                    sec_obj = st.secrets[section]
                    for k in ("GROQ_API_KEY", "groq_api_key", "api_key", "GROQ_KEY", "groq_key"):
                        try:
                            if k in sec_obj and str(sec_obj[k]).strip():
                                api_key = str(sec_obj[k]).strip().strip("\"'").strip()
                                break
                        except Exception:
                            pass
                    for k in ("GROQ_MODEL", "groq_model", "model"):
                        try:
                            if k in sec_obj and str(sec_obj[k]).strip():
                                model = str(sec_obj[k]).strip().strip("\"'").strip()
                                break
                        except Exception:
                            pass
    except Exception as ex:
        logger.debug(f"Error reading st.secrets: {ex}")

    return api_key, model


def sync_streamlit_secrets(env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Bridges Streamlit Community Cloud Secrets to the subprocess environment."""
    if env is None:
        target_env = os.environ
    else:
        target_env = env

    api_key, model = extract_groq_credentials()
    if api_key:
        target_env["GROQ_API_KEY"] = api_key
        target_env["groq_api_key"] = api_key
        os.environ["GROQ_API_KEY"] = api_key
        os.environ["groq_api_key"] = api_key

    if model:
        target_env["GROQ_MODEL"] = model
        target_env["groq_model"] = model
        os.environ["GROQ_MODEL"] = model
        os.environ["groq_model"] = model

    try:
        if hasattr(st, "secrets"):
            for key in st.secrets:
                val = st.secrets[key]
                if isinstance(val, (str, int, float, bool)):
                    str_val = str(val).strip()
                    target_env[str(key)] = str_val
                    os.environ[str(key)] = str_val
                elif hasattr(val, "items"):
                    for sub_k, sub_v in val.items():
                        if isinstance(sub_v, (str, int, float, bool)):
                            str_val = str(sub_v).strip()
                            target_env[str(sub_k)] = str_val
                            os.environ[str(sub_k)] = str_val
    except Exception as ex:
        logger.debug(f"Streamlit secrets sync skipped or failed: {ex}")

    return target_env


def sync_backend_config(host: str = "127.0.0.1", port: int = 8000) -> bool:
    """Pushes extracted Groq credentials to the running FastAPI backend over HTTP."""
    api_key, model = extract_groq_credentials()
    if not api_key:
        return False

    url = f"http://{host}:{port}/api/v1/system/configure"
    payload = json.dumps({"groq_api_key": api_key, "groq_model": model}).encode("utf-8")
    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "FrontendConfigSync"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            return resp.status == 200
    except Exception as ex:
        logger.debug(f"Could not push config to backend at {url}: {ex}")
        return False


def is_backend_healthy(host: str = "127.0.0.1", port: int = 8000, timeout: float = 1.5) -> bool:
    """Synchronous health probe to check if the FastAPI backend is operational."""
    url = f"http://{host}:{port}/health"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "BackendHealthProbe"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def is_backend_current(host: str = "127.0.0.1", port: int = 8000) -> bool:
    """Verifies that the running backend supports runtime configuration (/api/v1/system/configure)."""
    url = f"http://{host}:{port}/api/v1/system/configure"
    req = urllib.request.Request(
        url,
        data=b"{}",
        headers={"Content-Type": "application/json", "User-Agent": "BackendVersionProbe"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            return resp.status in (200, 422)
    except urllib.error.HTTPError as e:
        return e.code in (200, 422)
    except Exception:
        return False


def terminate_stale_backend(port: int = 8000) -> None:
    """Kills any stale process listening on the backend port."""
    logger.warning(f"Terminating stale backend process on port {port}...")
    try:
        if sys.platform == "win32":
            subprocess.run(
                f"for /f \"tokens=5\" %a in ('netstat -aon ^| findstr :{port}') do taskkill /F /PID %a",
                shell=True,
                capture_output=True,
                timeout=5,
            )
        else:
            # Linux container environment (Streamlit Cloud)
            subprocess.run(["pkill", "-9", "-f", "uvicorn.*incident_rag"], capture_output=True, timeout=5)
            subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True, timeout=5)
        time.sleep(1.0)
    except Exception as ex:
        logger.warning(f"Error terminating stale backend: {ex}")


def get_python_executable() -> str:
    """Finds the active virtual environment Python executable, falling back to sys.executable."""
    project_root = Path(__file__).resolve().parent.parent.parent
    venv_win = project_root / ".venv" / "Scripts" / "python.exe"
    if venv_win.is_file():
        return str(venv_win)
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

    # 2. Check if local backend is already healthy and up to date
    if is_backend_healthy(host=host, port=port, timeout=1.0):
        if is_backend_current(host=host, port=port):
            logger.info(f"FastAPI backend already healthy and current on http://{host}:{port}")
            sync_backend_config(host=host, port=port)
            return True
        else:
            logger.warning(
                f"Outdated FastAPI backend detected on http://{host}:{port} (missing /api/v1/system/configure). "
                f"Terminating stale process to launch updated version..."
            )
            terminate_stale_backend(port=port)

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
    max_retries = 30
    delay_sec = 0.8
    for attempt in range(1, max_retries + 1):
        if is_backend_healthy(host=host, port=port, timeout=1.0):
            logger.info(f"FastAPI backend is online and healthy on attempt {attempt}")
            sync_backend_config(host=host, port=port)
            return True
        if proc.poll() is not None:
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
