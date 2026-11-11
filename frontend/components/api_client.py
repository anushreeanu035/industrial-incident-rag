"""HTTP API client connecting the Streamlit frontend to the FastAPI backend with connection pooling, retries, and caching."""

import os
import time
from typing import Any, Dict, List, Optional
import httpx


def get_configured_backend_url() -> str:
    """Dynamically resolves backend API URL from session state, environment variables, or local default."""
    # 0. User runtime override in session state (e.g., from Settings page)
    try:
        import streamlit as st
        if hasattr(st, "session_state") and st.session_state.get("backend_url_override"):
            val = str(st.session_state["backend_url_override"]).strip().rstrip("/")
            if val:
                return val
    except Exception:
        pass

    # 1. Streamlit Secrets (for local .streamlit/secrets.toml if present)
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for key in ("BACKEND_URL", "backend_url", "API_URL", "api_url"):
                if key in st.secrets and st.secrets[key]:
                    val = str(st.secrets[key]).strip().rstrip("/")
                    if val:
                        return val
    except Exception:
        pass

    # 2. Environment Variables (.env or system)
    for key in ("BACKEND_URL", "backend_url", "API_URL", "api_url"):
        val = os.environ.get(key)
        if val and val.strip():
            return val.strip().rstrip("/")

    # 3. Default fallback for local development
    return "http://127.0.0.1:8000"
DEFAULT_BACKEND_URL = get_configured_backend_url()


class APIClient:
    """High-performance client with connection pooling, controlled retries, and intelligent caching."""

    def __init__(self, base_url: Optional[str] = None):
        self._explicit_base_url = base_url.rstrip("/") if base_url else None
        self._client: Optional[httpx.Client] = None
        self._active_url: Optional[str] = None
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._is_online: bool = False
        self._last_error: Optional[str] = None
        self._last_latency_ms: float = 0.0

    @property
    def base_url(self) -> str:
        """Returns active backend base URL, resolving dynamically from secrets/env if not manually specified."""
        if self._explicit_base_url:
            return self._explicit_base_url
        return get_configured_backend_url()

    @base_url.setter
    def base_url(self, url: Optional[str]) -> None:
        """Allows dynamically overriding backend URL at runtime."""
        self._explicit_base_url = url.rstrip("/") if url else None
        if self._client and not self._client.is_closed:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
            self._active_url = None

    @property
    def is_online(self) -> bool:
        """Returns the most recent confirmed online status of the backend."""
        return self._is_online

    @property
    def last_error(self) -> Optional[str]:
        """Returns the last encountered communication error message."""
        return self._last_error

    @property
    def last_latency_ms(self) -> float:
        """Returns the latency of the last successful health check or request in ms."""
        return self._last_latency_ms

    def _get_client(self) -> httpx.Client:
        """Returns or lazily creates a persistent connection-pooled HTTP client."""
        target_url = self.base_url
        if self._client is None or self._client.is_closed or self._active_url != target_url:
            if self._client and not self._client.is_closed:
                try:
                    self._client.close()
                except Exception:
                    pass
            self._client = httpx.Client(
                base_url=target_url,
                timeout=httpx.Timeout(connect=5.0, read=60.0, write=60.0, pool=5.0),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=30, keepalive_expiry=60.0),
            )
            self._active_url = target_url
        return self._client

    def _format_http_error(self, status_code: int, response_text: str) -> str:
        """Translates raw HTTP status errors into actionable messages."""
        # Sanitize HTML proxy pages (e.g., Cloudflare/Render 502 HTML pages)
        clean_text = response_text
        if "<!doctype html" in response_text.lower() or "<html" in response_text.lower():
            clean_text = "(Reverse proxy returned HTML gateway error)"

        if status_code in (502, 503, 504):
            return (
                f"Render Gateway Error ({status_code}): The FastAPI backend at {self.base_url} is currently starting up or restarting. "
                "On Render's Free Tier, services spin down after 15 minutes of inactivity and take 30–50 seconds to complete cold start. "
                "Please wait a moment and refresh."
            )
        elif status_code == 404:
            return f"Not Found (404): The requested resource does not exist on {self.base_url}."
        elif status_code == 422:
            return f"Validation Error (422): Input parameters failed backend schema verification: {clean_text}"
        elif status_code == 429:
            return f"Rate Limited (429): Too many concurrent requests on the backend. Please wait a few seconds and try again."
        return f"Backend error ({status_code}): {clean_text}"

    def _execute_with_retry(
        self,
        method: str,
        endpoint: str,
        max_retries: int = 3,
        backoff_sec: float = 1.5,
        **kwargs,
    ) -> httpx.Response:
        """Executes HTTP request with controlled exponential backoff for transient server errors (502/503/504/timeout)."""
        client = self._get_client()
        last_exception: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                if method.upper() == "GET":
                    res = client.get(endpoint, **kwargs)
                elif method.upper() == "POST":
                    res = client.post(endpoint, **kwargs)
                elif method.upper() == "DELETE":
                    res = client.delete(endpoint, **kwargs)
                else:
                    res = client.request(method, endpoint, **kwargs)

                # If transient 502/503/504, retry if attempts remain
                if res.status_code in (502, 503, 504) and attempt < max_retries:
                    time.sleep(backoff_sec * attempt)
                    continue

                res.raise_for_status()
                self._is_online = True
                self._last_error = None
                return res

            except (httpx.ConnectTimeout, httpx.ConnectError, httpx.RemoteProtocolError) as e:
                self._client = None
                last_exception = e
                if attempt < max_retries:
                    time.sleep(backoff_sec * attempt)
                    client = self._get_client()
                    continue
                self._is_online = False
                self._last_error = (
                    f"Unable to connect to local backend at {self.base_url} after {max_retries} attempts. "
                    "Please ensure the local FastAPI server is running via: uvicorn incident_rag.api:app --reload"
                )
                raise ConnectionError(self._last_error) from e

            except httpx.HTTPStatusError as e:
                if e.response.status_code in (502, 503, 504) and attempt < max_retries:
                    time.sleep(backoff_sec * attempt)
                    continue
                self._last_error = self._format_http_error(e.response.status_code, e.response.text)
                if e.response.status_code in (502, 503, 504):
                    self._is_online = False
                raise ValueError(self._last_error) from e

            except Exception as e:
                self._client = None
                self._last_error = str(e)
                raise

        # Should only reach here if loop exhausted on a status error
        raise ValueError(f"Request to {endpoint} failed after {max_retries} retries: {self._last_error}")

    def _get(self, endpoint: str, timeout: float = 15.0, max_retries: int = 3) -> Dict[str, Any]:
        """Executes GET request using pooled client with retries."""
        res = self._execute_with_retry("GET", endpoint, max_retries=max_retries, timeout=timeout)
        return res.json()

    def _post(self, endpoint: str, json_data: Dict[str, Any], timeout: float = 60.0, max_retries: int = 3) -> Dict[str, Any]:
        """Executes POST request using pooled client with retries."""
        res = self._execute_with_retry("POST", endpoint, max_retries=max_retries, json=json_data, timeout=timeout)
        return res.json()

    def clear_cache(self, key_prefix: Optional[str] = None) -> None:
        """Invalidates all or specific cached API responses."""
        if key_prefix:
            keys_to_del = [k for k in self._cache if k.startswith(key_prefix)]
            for k in keys_to_del:
                self._cache.pop(k, None)
                self._cache_timestamps.pop(k, None)
        else:
            self._cache.clear()
            self._cache_timestamps.clear()

    def check_health(self, ttl: float = 10.0) -> Dict[str, Any]:
        """Returns system health, serving from memory cache if called within ttl seconds."""
        now = time.time()
        cache_key = "health"
        if cache_key in self._cache and (now - self._cache_timestamps.get(cache_key, 0.0)) < ttl:
            return self._cache[cache_key]

        start_t = time.perf_counter()
        try:
            data = self._get("/health", timeout=5.0, max_retries=2)
            self._last_latency_ms = (time.perf_counter() - start_t) * 1000.0

            # If groq not configured on backend, attempt auto-sync from st.secrets
            if not data.get("groq_configured"):
                try:
                    from frontend.components.backend_service import extract_groq_credentials
                    sec_key, sec_model = extract_groq_credentials()
                    if sec_key:
                        self.configure_groq(sec_key, sec_model)
                        data = self._get("/health", timeout=5.0, max_retries=2)
                except Exception:
                    pass

            self._cache[cache_key] = data
            self._cache_timestamps[cache_key] = now
            self._is_online = True
            self._last_error = None
            return data
        except Exception as e:
            self._is_online = False
            self._last_error = str(e)
            raise

    def configure_groq(self, api_key: str, model: str = "openai/gpt-oss-120b") -> Dict[str, Any]:
        """Dynamically configures Groq LLM credentials and triggers live connection testing."""
        url = "/api/v1/system/configure"
        payload = {"groq_api_key": api_key.strip(), "groq_model": model.strip()}
        res = self._post(url, json_data=payload, timeout=10.0, max_retries=1)
        self.clear_cache("health")
        return res

    def health_check(self, ttl: float = 10.0) -> Dict[str, Any]:
        """Alias for check_health to support callers using health_check()."""
        return self.check_health(ttl=ttl)

    def get_connection_info(self) -> Dict[str, Any]:
        """Returns a snapshot of backend connectivity, URL, and status."""
        try:
            health = self.check_health(ttl=5.0)
            return {
                "online": True,
                "base_url": self.base_url,
                "latency_ms": self._last_latency_ms,
                "error": None,
                "data": health,
            }
        except Exception as e:
            return {
                "online": False,
                "base_url": self.base_url,
                "latency_ms": 0.0,
                "error": str(e),
                "data": {},
            }

    def get_settings(self, ttl: float = 60.0) -> Dict[str, Any]:
        """Returns backend settings with 60s memory caching."""
        now = time.time()
        cache_key = "settings"
        if cache_key in self._cache and (now - self._cache_timestamps.get(cache_key, 0.0)) < ttl:
            return self._cache[cache_key]

        data = self._get("/api/v1/settings")
        self._cache[cache_key] = data
        self._cache_timestamps[cache_key] = now
        return data

    def list_documents(self, ttl: float = 30.0) -> List[Dict[str, Any]]:
        """Returns indexed documents with 30s caching (invalidated on upload/delete)."""
        now = time.time()
        cache_key = "documents"
        if cache_key in self._cache and (now - self._cache_timestamps.get(cache_key, 0.0)) < ttl:
            return self._cache[cache_key]

        data = self._get("/api/v1/documents")
        self._cache[cache_key] = data
        self._cache_timestamps[cache_key] = now
        return data

    def get_document(self, document_id: str) -> Dict[str, Any]:
        """Retrieves metadata for a specific document."""
        return self._get(f"/api/v1/documents/{document_id}")

    def get_document_chunks(self, document_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieves indexed chunks and preview text for a specific document."""
        return self._get(f"/api/v1/documents/{document_id}/chunks?limit={limit}")

    def upload_document(
        self,
        filename: str,
        content_bytes: bytes,
        mime_type: str = "application/octet-stream",
    ) -> Dict[str, Any]:
        """Uploads and indexes a document. Avoids aggressive retries to prevent duplicate submissions."""
        files = {"file": (filename, content_bytes, mime_type)}
        res = self._execute_with_retry("POST", "/api/v1/documents/upload", files=files, timeout=120.0, max_retries=1)
        self.clear_cache("documents")
        self.clear_cache("health")
        return res.json()

    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Deletes a document and its chunks from all vector and keyword indexes."""
        res = self._execute_with_retry("DELETE", f"/api/v1/documents/{document_id}", timeout=20.0, max_retries=2)
        self.clear_cache("documents")
        self.clear_cache("health")
        return res.json()

    def retrieve(
        self,
        query: str,
        strategy: str = "full_advanced",
        top_k: int = 5,
        machine_id_filter: Optional[str] = None,
        incident_type_filter: Optional[str] = None,
        enable_rewriting: Optional[bool] = None,
        document_id_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Performs multi-stage retrieval across knowledge base."""
        payload = {
            "query": query,
            "strategy": strategy,
            "top_k": top_k,
            "machine_id_filter": machine_id_filter or None,
            "incident_type_filter": incident_type_filter or None,
            "enable_rewriting": enable_rewriting,
            "document_id_filter": document_id_filter or None,
        }
        return self._post("/api/v1/retrieve", payload, timeout=40.0)

    def investigate(
        self,
        query: str,
        strategy: str = "full_advanced",
        top_k: int = 5,
        machine_id_filter: Optional[str] = None,
        incident_type_filter: Optional[str] = None,
        enable_rewriting: Optional[bool] = None,
        document_id_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Executes full incident investigation and root cause synthesis."""
        payload = {
            "query": query,
            "strategy": strategy,
            "top_k": top_k,
            "machine_id_filter": machine_id_filter or None,
            "incident_type_filter": incident_type_filter or None,
            "enable_rewriting": enable_rewriting,
            "document_id_filter": document_id_filter or None,
        }
        return self._post("/api/v1/investigate", payload, timeout=60.0)

    def analyze_document(self, document_id: str, timeout: float = 60.0) -> Dict[str, Any]:
        """Requests structured forensic analysis of a specific document (document-driven workflow)."""
        return self._post(f"/api/v1/documents/{document_id}/analyze", {}, timeout=timeout)

    def run_evaluation(
        self,
        strategies: List[str],
        top_k: int = 5,
        run_ragas: bool = False,
    ) -> Dict[str, Any]:
        """Executes live quantitative benchmark across ground-truth dataset."""
        payload = {
            "strategies": strategies,
            "top_k": top_k,
            "run_ragas": run_ragas,
        }
        data = self._post("/api/v1/evaluate", payload, timeout=180.0, max_retries=1)
        self.clear_cache("evaluations")
        return data

    def list_evaluations(self, ttl: float = 60.0) -> List[Dict[str, Any]]:
        """Returns benchmark reports history with 60s caching."""
        now = time.time()
        cache_key = "evaluations_list"
        if cache_key in self._cache and (now - self._cache_timestamps.get(cache_key, 0.0)) < ttl:
            return self._cache[cache_key]

        data = self._get("/api/v1/evaluations")
        self._cache[cache_key] = data
        self._cache_timestamps[cache_key] = now
        return data

    def get_evaluation(self, run_id: str, ttl: float = 300.0) -> Dict[str, Any]:
        """Returns evaluation report for run_id with 5-minute caching (immutable reports)."""
        now = time.time()
        cache_key = f"eval_report_{run_id}"
        if cache_key in self._cache and (now - self._cache_timestamps.get(cache_key, 0.0)) < ttl:
            return self._cache[cache_key]

        data = self._get(f"/api/v1/evaluations/{run_id}")
        self._cache[cache_key] = data
        self._cache_timestamps[cache_key] = now
        return data


# Global singleton client instance
api_client = APIClient()
