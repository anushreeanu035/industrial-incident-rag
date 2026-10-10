"""Comprehensive tests for user document upload and isolated analysis feature."""

import pytest
from fastapi.testclient import TestClient
import pymupdf

from incident_rag.api import app
from incident_rag.dependencies import get_rag_pipeline, get_ingestion_service, get_document_store
from incident_rag.schemas import InvestigationRequest, RetrievalRequest, RetrievalStrategy
from frontend.components.api_client import APIClient
from frontend.components.evidence_cards import generate_report_markdown


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def _create_sample_pdf_bytes(title: str, body: str) -> bytes:
    """Helper to generate a minimal in-memory PDF."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), f"{title}\n\n{body}", fontsize=11)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# ---------------------------------------------------------------------------
# 1. Supported File Uploads & Ingestion
# ---------------------------------------------------------------------------

def test_upload_supported_markdown_document(client: TestClient):
    md_content = (
        "# INCIDENT REPORT: INC-2026-901\n"
        "Machine ID: EXTRUDER-X1\n"
        "Date: 2026-05-12\n"
        "Incident Type: Overheating\n\n"
        "Severe barrel temperature runaway reached 310 deg C due to heater relay welding shut.\n"
        "Root Cause: Solid state relay SSR-3 contact fusion.\n"
        "Corrective Action: Replace SSR with dual-redundant contactor interlock."
    ).encode("utf-8")

    resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("INC-2026-901_extruder.md", md_content, "text/markdown")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "indexed"
    assert data["filename"] == "INC-2026-901_extruder.md"
    assert data["chunks_created"] >= 1
    doc_id = data["document_id"]

    # Clean up after test
    del_resp = client.delete(f"/api/v1/documents/{doc_id}")
    assert del_resp.status_code == 200


def test_upload_supported_text_document(client: TestClient):
    txt_content = (
        "INCIDENT REPORT: INC-2026-902\n"
        "Machine ID: CHILLER-C4\n"
        "Date: 2026-06-15\n"
        "Refrigerant leak detected in condenser coil.\n"
        "Root Cause: Stress corrosion cracking at copper joint brazing.\n"
        "Corrective Action: Re-braze with silver alloy and pressure test to 25 bar."
    ).encode("utf-8")

    resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("INC-2026-902_chiller.txt", txt_content, "text/plain")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "indexed"
    assert data["chunks_created"] >= 1
    doc_id = data["document_id"]

    client.delete(f"/api/v1/documents/{doc_id}")


def test_upload_supported_pdf_document(client: TestClient):
    pdf_bytes = _create_sample_pdf_bytes(
        title="SAFETY BULLETIN: INC-2026-903",
        body=(
            "Machine ID: BOILER-B2\n"
            "Date: 2026-07-20\n"
            "Burner flameout event caused emergency steam header isolation.\n"
            "Root Cause: Flame scanner optical sensor soot accumulation.\n"
            "Corrective Action: Install purge air blower on UV flame detector."
        ),
    )

    resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("INC-2026-903_boiler.pdf", pdf_bytes, "application/pdf")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "indexed"
    assert data["chunks_created"] >= 1
    doc_id = data["document_id"]

    client.delete(f"/api/v1/documents/{doc_id}")


# ---------------------------------------------------------------------------
# 2. Validation Errors: Unsupported, Empty, & Oversized Files
# ---------------------------------------------------------------------------

def test_upload_invalid_file_extension(client: TestClient):
    resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("payload.exe", b"MZbinarycontent", "application/octet-stream")},
    )
    assert resp.status_code == 400
    assert "Unsupported file format" in resp.json()["detail"]


def test_upload_empty_document(client: TestClient):
    resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("empty_report.txt", b"", "text/plain")},
    )
    assert resp.status_code == 400
    assert "is empty" in resp.json()["detail"]


def test_upload_oversized_file(client: TestClient):
    # Create fake oversized content exceeding 25MB (25MB + 100KB)
    oversized = b"0" * (25 * 1024 * 1024 + 102400)
    resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("massive_report.txt", oversized, "text/plain")},
    )
    assert resp.status_code == 413
    assert "exceeds maximum permitted limit of 25MB" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 3. Document-Level Retrieval Isolation & Analysis
# ---------------------------------------------------------------------------

def test_document_retrieval_isolation(client: TestClient):
    # Upload Document Alpha with unique token ALPHA_UNIQUE_PUMP_77
    content_a = (
        "INCIDENT REPORT ALPHA\n"
        "Machine ID: ALPHA_PUMP_77\n"
        "Vibration reading spiked to 48.2 mm/s due to broken impeller vane ALPHA_UNIQUE_PUMP_77.\n"
        "Root Cause: Foreign object damage lodging into vane 3.\n"
        "Corrective Action: Inspect suction strainer ALPHA-STRAIN."
    ).encode("utf-8")

    # Upload Document Beta with unique token BETA_UNIQUE_TURBINE_88
    content_b = (
        "INCIDENT REPORT BETA\n"
        "Machine ID: BETA_TURBINE_88\n"
        "Lube oil temperature elevated to 94 deg C on BETA_UNIQUE_TURBINE_88.\n"
        "Root Cause: Heat exchanger tube bio-fouling and scale buildup.\n"
        "Corrective Action: Acid wash cooler tubes quarterly."
    ).encode("utf-8")

    resp_a = client.post(
        "/api/v1/documents/upload",
        files={"file": ("doc_alpha.md", content_a, "text/markdown")},
    )
    assert resp_a.status_code == 200
    doc_a_id = resp_a.json()["document_id"]

    resp_b = client.post(
        "/api/v1/documents/upload",
        files={"file": ("doc_beta.md", content_b, "text/markdown")},
    )
    assert resp_b.status_code == 200
    doc_b_id = resp_b.json()["document_id"]

    try:
        # 1. Retrieve with doc_a isolation filter
        req_a = {
            "query": "What caused the vibration failure?",
            "document_id_filter": doc_a_id,
            "strategy": "full_advanced",
            "top_k": 5,
        }
        res_a = client.post("/api/v1/retrieve", json=req_a)
        assert res_a.status_code == 200
        items_a = res_a.json()["results"]
        assert len(items_a) > 0
        # All items must belong ONLY to doc_a
        for item in items_a:
            assert item["document_id"] == doc_a_id
            assert "BETA" not in item["text"]

        # 2. Retrieve with doc_b isolation filter
        req_b = {
            "query": "What caused the lube oil temperature problem?",
            "document_id_filter": doc_b_id,
            "strategy": "full_advanced",
            "top_k": 5,
        }
        res_b = client.post("/api/v1/retrieve", json=req_b)
        assert res_b.status_code == 200
        items_b = res_b.json()["results"]
        assert len(items_b) > 0
        # All items must belong ONLY to doc_b
        for item in items_b:
            assert item["document_id"] == doc_b_id
            assert "ALPHA" not in item["text"]

        # 3. Investigate with doc_a isolation filter
        inv_req_a = {
            "query": "What was the root cause of impeller failure?",
            "document_id_filter": doc_a_id,
            "strategy": "full_advanced",
            "top_k": 5,
        }
        inv_res_a = client.post("/api/v1/investigate", json=inv_req_a)
        assert inv_res_a.status_code == 200
        inv_data_a = inv_res_a.json()
        assert inv_data_a["retrieval_metadata"]["document_id_filter"] == doc_a_id
        # All citations must point to doc_a
        for cit in inv_data_a["citations"]:
            assert cit["document_id"] == doc_a_id
            assert cit["source_name"] == "doc_alpha.md"

    finally:
        # Clean up
        client.delete(f"/api/v1/documents/{doc_a_id}")
        client.delete(f"/api/v1/documents/{doc_b_id}")


# ---------------------------------------------------------------------------
# 4. Report Markdown Generation & Traceability
# ---------------------------------------------------------------------------

def test_generate_report_markdown_structure():
    mock_response = {
        "executive_summary": "Bearing babbitt wiped on turbine HP-17.",
        "documented_observations": [
            "Radial vibration peaked at 14.2 mm/s [CIT-1]",
            "Bearing temperature reached 118 deg C [CIT-2]",
        ],
        "potential_root_causes": [
            {
                "hypothesis": "Lube oil starvation caused boundary friction.",
                "confidence_level": "High (Documented Fact)",
                "supporting_evidence": ["Oil pressure dropped below 0.8 bar"],
                "citation_ids": ["[CIT-1]"],
            }
        ],
        "corrective_actions": [
            "Replace journal bearing shells [CIT-1]",
            "Install redundant differential pressure switch",
        ],
        "missing_evidence_and_uncertainties": [
            "Oil sample laboratory particulate count not yet available."
        ],
        "citations": [
            {
                "citation_id": "[CIT-1]",
                "document_id": "doc_test123",
                "chunk_id": "chunk_001",
                "source_name": "HP17_Turbine_Log.pdf",
                "page_number": 3,
                "excerpt": "Vibration sensor XY-102 registered 14.2 mm/s RMS...",
                "relevance_reason": "Contains telemetry readings",
            }
        ],
    }

    report = generate_report_markdown(mock_response, "What caused bearing failure on HP-17?")
    assert "# Industrial Incident Investigation Report" in report
    assert "## 1. Executive Summary" in report
    assert "Bearing babbitt wiped on turbine HP-17." in report
    assert "## 2. Documented Observations & Telemetry" in report
    assert "14.2 mm/s" in report
    assert "## 3. Potential Root Causes & Hypotheses" in report
    assert "High (Documented Fact)" in report
    assert "## 4. Mandated Corrective & Preventive Actions" in report
    assert "## 5. Missing Evidence & Forensic Uncertainties" in report
    assert "## 6. Traceable Evidence Citations" in report
    assert "[CIT-1] — HP17_Turbine_Log.pdf (Page 3)" in report


# ---------------------------------------------------------------------------
# 5. Frontend API Client Document Filter Integration
# ---------------------------------------------------------------------------

def test_api_client_forwards_document_id_filter():
    client_instance = APIClient()
    # Check that investigate signature accepts document_id_filter
    import inspect
    sig_inv = inspect.signature(client_instance.investigate)
    assert "document_id_filter" in sig_inv.parameters
    assert sig_inv.parameters["document_id_filter"].default is None

    # Check that retrieve signature accepts document_id_filter
    sig_ret = inspect.signature(client_instance.retrieve)
    assert "document_id_filter" in sig_ret.parameters
    assert sig_ret.parameters["document_id_filter"].default is None
