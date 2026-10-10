"""HTTP API client connecting the Streamlit frontend to the FastAPI backend."""

import os
from typing import Any, Dict, List, Optional
import httpx

DEFAULT_BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")


class APIClient:
    """Client for making HTTP requests to the Industrial Incident Investigator API."""

    def __init__(self, base_url: str = DEFAULT_BACKEND_URL):
        self.base_url = base_url.rstrip("/")

    def _get(self, endpoint: str, timeout: float = 10.0) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        try:
            with httpx.Client(timeout=timeout) as client:
                res = client.get(url)
                res.raise_for_status()
                return res.json()
        except httpx.ConnectError:
            raise ConnectionError(f"Unable to connect to backend at {self.base_url}. Ensure FastAPI is running.")
        except httpx.HTTPStatusError as e:
            raise ValueError(f"Backend error ({e.response.status_code}): {e.response.text}")

    def _post(self, endpoint: str, json_data: Dict[str, Any], timeout: float = 60.0) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        try:
            with httpx.Client(timeout=timeout) as client:
                res = client.post(url, json=json_data)
                res.raise_for_status()
                return res.json()
        except httpx.ConnectError:
            raise ConnectionError(f"Unable to connect to backend at {self.base_url}. Ensure FastAPI is running.")
        except httpx.HTTPStatusError as e:
            raise ValueError(f"Backend error ({e.response.status_code}): {e.response.text}")

    def check_health(self) -> Dict[str, Any]:
        return self._get("/health")

    def get_settings(self) -> Dict[str, Any]:
        return self._get("/api/v1/settings")

    def list_documents(self) -> List[Dict[str, Any]]:
        return self._get("/api/v1/documents")

    def get_document(self, document_id: str) -> Dict[str, Any]:
        return self._get(f"/api/v1/documents/{document_id}")

    def upload_document(self, filename: str, content_bytes: bytes, mime_type: str = "application/octet-stream") -> Dict[str, Any]:
        url = f"{self.base_url}/api/v1/documents/upload"
        files = {"file": (filename, content_bytes, mime_type)}
        try:
            with httpx.Client(timeout=60.0) as client:
                res = client.post(url, files=files)
                res.raise_for_status()
                return res.json()
        except httpx.ConnectError:
            raise ConnectionError(f"Unable to connect to backend at {self.base_url}.")
        except httpx.HTTPStatusError as e:
            raise ValueError(f"Backend upload error ({e.response.status_code}): {e.response.text}")

    def delete_document(self, document_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/api/v1/documents/{document_id}"
        try:
            with httpx.Client(timeout=10.0) as client:
                res = client.delete(url)
                res.raise_for_status()
                return res.json()
        except httpx.ConnectError:
            raise ConnectionError(f"Unable to connect to backend at {self.base_url}.")
        except httpx.HTTPStatusError as e:
            raise ValueError(f"Backend error ({e.response.status_code}): {e.response.text}")

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
        return self._post("/api/v1/evaluate", payload, timeout=120.0)

    def list_evaluations(self) -> List[Dict[str, Any]]:
        return self._get("/api/v1/evaluations")

    def get_evaluation(self, run_id: str) -> Dict[str, Any]:
        return self._get(f"/api/v1/evaluations/{run_id}")


# Global client instance
api_client = APIClient()
