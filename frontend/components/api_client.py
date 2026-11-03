"""HTTP API client connecting the Streamlit frontend to the FastAPI backend with connection pooling and caching."""

import os
import time
from typing import Any, Dict, List, Optional
import httpx

def get_configured_backend_url() -> str:
    """Dynamically resolves backend API URL from Streamlit secrets, environment variables, or fallback."""
    # 1. Streamlit Secrets (for Streamlit Community Cloud and local .streamlit/secrets.toml)
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

    # 2. Environment Variables (.env, container env, system)
    for key in ("BACKEND_URL", "backend_url", "API_URL", "api_url"):
        val = os.environ.get(key)
        if val and val.strip():
            return val.strip().rstrip("/")

    # 3. Default fallback for local development
    return "http://127.0.0.1:8000"


DEFAULT_BACKEND_URL = get_configured_backend_url()


class APIClient:
    """High-performance client with connection pooling and intelligent caching."""

    def __init__(self, base_url: Optional[str] = None):
        self._explicit_base_url = base_url.rstrip("/") if base_url else None
        self._client: Optional[httpx.Client] = None
        self._active_url: Optional[str] = None
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}

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
                timeout=httpx.Timeout(connect=15.0, read=120.0, write=60.0, pool=15.0),
                limits=httpx.Limits(max_keepalive_connections=15, max_connections=25, keepalive_expiry=60.0),
            )
            self._active_url = target_url
        return self._client

    def _get(self, endpoint: str, timeout: float = 10.0) -> Dict[str, Any]:
        """Executes GET request using pooled HTTP client with auto-recovery."""
        client = self._get_client()
        try:
            res = client.get(endpoint, timeout=timeout)
            res.raise_for_status()
            return res.json()
        except httpx.ConnectTimeout:
            self._client = None
            raise ConnectionError(
                f"Connection timed out while reaching backend at {self.base_url}. "
                f"If hosted on a free cloud platform (e.g. Render free tier), the instance may be waking up from sleep. "
                f"Please wait a few seconds and retry."
            )
        except httpx.ConnectError:
            self._client = None
            raise ConnectionError(
                f"Unable to connect to backend at {self.base_url}. "
                f"If running on Streamlit Cloud, verify that the FastAPI backend is deployed and "
                f"BACKEND_URL is set in your Streamlit Cloud app secrets."
            )
        except httpx.HTTPStatusError as e:
            raise ValueError(f"Backend error ({e.response.status_code}): {e.response.text}")
        except Exception:
            self._client = None
            raise

    def _post(self, endpoint: str, json_data: Dict[str, Any], timeout: float = 60.0) -> Dict[str, Any]:
        """Executes POST request using pooled HTTP client with auto-recovery."""
        client = self._get_client()
        try:
            res = client.post(endpoint, json=json_data, timeout=timeout)
            res.raise_for_status()
            return res.json()
        except httpx.ConnectTimeout:
            self._client = None
            raise ConnectionError(
                f"Connection timed out while reaching backend at {self.base_url}. "
                f"If hosted on a free cloud platform (e.g. Render free tier), the instance may be waking up from sleep. "
                f"Please wait a few seconds and retry."
            )
        except httpx.ConnectError:
            self._client = None
            raise ConnectionError(
                f"Unable to connect to backend at {self.base_url}. "
                f"If running on Streamlit Cloud, verify that the FastAPI backend is deployed and "
                f"BACKEND_URL is set in your Streamlit Cloud app secrets."
            )
        except httpx.HTTPStatusError as e:
            raise ValueError(f"Backend error ({e.response.status_code}): {e.response.text}")
        except Exception:
            self._client = None
            raise

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

        data = self._get("/health", timeout=3.0)
        self._cache[cache_key] = data
        self._cache_timestamps[cache_key] = now
        return data

    def health_check(self, ttl: float = 10.0) -> Dict[str, Any]:
        """Alias for check_health to support callers using health_check()."""
        return self.check_health(ttl=ttl)

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
        return self._get(f"/api/v1/documents/{document_id}")

    def upload_document(self, filename: str, content_bytes: bytes, mime_type: str = "application/octet-stream") -> Dict[str, Any]:
        client = self._get_client()
        files = {"file": (filename, content_bytes, mime_type)}
        try:
            res = client.post("/api/v1/documents/upload", files=files, timeout=60.0)
            res.raise_for_status()
            self.clear_cache("documents")
            self.clear_cache("health")
            return res.json()
        except httpx.ConnectError:
            self._client = None
            raise ConnectionError(f"Unable to connect to backend at {self.base_url}.")
        except httpx.HTTPStatusError as e:
            raise ValueError(f"Backend upload error ({e.response.status_code}): {e.response.text}")
        except Exception:
            self._client = None
            raise

    def delete_document(self, document_id: str) -> Dict[str, Any]:
        client = self._get_client()
        try:
            res = client.delete(f"/api/v1/documents/{document_id}", timeout=10.0)
            res.raise_for_status()
            self.clear_cache("documents")
            self.clear_cache("health")
            return res.json()
        except httpx.ConnectError:
            self._client = None
            raise ConnectionError(f"Unable to connect to backend at {self.base_url}.")
        except httpx.HTTPStatusError as e:
            raise ValueError(f"Backend error ({e.response.status_code}): {e.response.text}")
        except Exception:
            self._client = None
            raise

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
        payload = {
            "query": query,
            "strategy": strategy,
            "top_k": top_k,
            "machine_id_filter": machine_id_filter or None,
            "incident_type_filter": incident_type_filter or None,
            "enable_rewriting": enable_rewriting,
            "document_id_filter": document_id_filter or None,
        }
        return self._post("/api/v1/retrieve", payload, timeout=30.0)

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
        payload = {
            "query": query,
            "strategy": strategy,
            "top_k": top_k,
            "machine_id_filter": machine_id_filter or None,
            "incident_type_filter": incident_type_filter or None,
            "enable_rewriting": enable_rewriting,
            "document_id_filter": document_id_filter or None,
        }
        return self._post("/api/v1/investigate", payload, timeout=45.0)

    def analyze_document(self, document_id: str, timeout: float = 45.0) -> Dict[str, Any]:
        """Requests structured forensic analysis of a specific document (document-driven workflow)."""
        return self._post(f"/api/v1/documents/{document_id}/analyze", {}, timeout=timeout)


    def run_evaluation(
        self,
        strategies: List[str],
        top_k: int = 5,
        run_ragas: bool = False,
    ) -> Dict[str, Any]:
        payload = {
            "strategies": strategies,
            "top_k": top_k,
            "run_ragas": run_ragas,
        }
        data = self._post("/api/v1/evaluate", payload, timeout=120.0)
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
