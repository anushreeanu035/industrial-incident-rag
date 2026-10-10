"""Tests for frontend API client communication with FastAPI backend."""

from unittest.mock import MagicMock, patch
import pytest
from frontend.components.api_client import APIClient


def test_api_client_health():
    client = APIClient(base_url="http://127.0.0.1:8000")
    with patch("httpx.Client.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "healthy", "indexed_documents": 5}
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        health = client.check_health()
        assert health["status"] == "healthy"
        assert health["indexed_documents"] == 5


def test_api_client_retrieve():
    client = APIClient(base_url="http://127.0.0.1:8000")
    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "query": "HP-17 trip",
            "strategy": "hybrid_rrf",
            "results": [{"chunk_id": "c1", "text": "Bearing wiped"}],
            "execution_time_ms": 25.0,
        }
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        res = client.retrieve("HP-17 trip", strategy="hybrid_rrf", top_k=3)
        assert res["strategy"] == "hybrid_rrf"
        assert len(res["results"]) == 1


def test_api_client_investigate():
    client = APIClient(base_url="http://127.0.0.1:8000")
    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "executive_summary": "Turbine tripped on high vibration.",
            "potential_root_causes": [{"hypothesis": "Cooler seal degraded", "confidence_level": "High"}],
            "citations": [{"citation_id": "[CIT-1]"}],
            "is_sufficient_evidence": True,
        }
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        inv = client.investigate("Why did HP-17 trip?")
        assert inv["is_sufficient_evidence"] is True
        assert len(inv["citations"]) == 1
