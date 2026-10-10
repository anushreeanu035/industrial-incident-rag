"""Tests for FastAPI endpoints using TestClient."""

from fastapi.testclient import TestClient
import pytest
from incident_rag.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "indexed_documents" in data
    assert "embedding_model" in data


def test_settings_endpoint(client: TestClient):
    resp = client.get("/api/v1/settings")
    assert resp.status_code == 200
    data = resp.json()
    assert data["embedding_model"] == "all-MiniLM-L6-v2"
    assert "dense_top_k" in data


def test_list_and_upload_document(client: TestClient):
    # Check seeded documents exist
    list_resp = client.get("/api/v1/documents")
    assert list_resp.status_code == 200
    initial_docs = list_resp.json()
    assert len(initial_docs) >= 1

    # Upload new test document
    test_content = b"# INCIDENT TEST\nMachine ID: TEST-99\nPump failure."
    upload_resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("test_incident.md", test_content, "text/markdown")},
    )
    assert upload_resp.status_code == 200
    up_data = upload_resp.json()
    assert up_data["status"] == "indexed"
    assert up_data["filename"] == "test_incident.md"
    doc_id = up_data["document_id"]

    # Retrieve single document
    get_resp = client.get(f"/api/v1/documents/{doc_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["document_id"] == doc_id

    # Delete test document
    del_resp = client.delete(f"/api/v1/documents/{doc_id}")
    assert del_resp.status_code == 200

    # 404 after deletion
    del_404 = client.get(f"/api/v1/documents/{doc_id}")
    assert del_404.status_code == 404


def test_retrieve_endpoint(client: TestClient):
    req_body = {
        "query": "steam turbine vibration HP-17",
        "strategy": "hybrid_rrf",
        "top_k": 3,
    }
    resp = client.post("/api/v1/retrieve", json=req_body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["strategy"] == "hybrid_rrf"
    assert len(data["results"]) >= 1
    assert "stage_results" in data


def test_investigate_endpoint(client: TestClient):
    req_body = {
        "query": "What caused the bearing failure on turbine HP-17?",
        "strategy": "full_advanced",
        "top_k": 3,
    }
    resp = client.post("/api/v1/investigate", json=req_body)
    assert resp.status_code == 200
    data = resp.json()
    assert "executive_summary" in data
    assert "potential_root_causes" in data
    assert len(data["citations"]) >= 1
    assert data["is_sufficient_evidence"] is True


def test_validation_errors(client: TestClient):
    # Empty query
    resp = client.post("/api/v1/investigate", json={"query": "   "})
    assert resp.status_code == 400

    # Unsupported upload format
    bad_upload = client.post(
        "/api/v1/documents/upload",
        files={"file": ("image.png", b"fake-png-data", "image/png")},
    )
    assert bad_upload.status_code == 400


def test_evaluate_and_history_endpoints(client: TestClient):
    eval_req = {
        "strategies": ["bm25"],
        "top_k": 3,
        "run_ragas": False,
    }
    resp = client.post("/api/v1/evaluate", json=eval_req)
    assert resp.status_code == 200
    data = resp.json()
    run_id = data["run_id"]
    assert len(data["results"]) == 1
    assert data["results"][0]["strategy"] == "bm25"

    # List evaluations
    list_resp = client.get("/api/v1/evaluations")
    assert list_resp.status_code == 200
    assert any(r["run_id"] == run_id for r in list_resp.json())

    # Get single evaluation
    get_resp = client.get(f"/api/v1/evaluations/{run_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["run_id"] == run_id
