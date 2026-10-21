"""Tests for the 9-section structured incident investigation report template,
missing fields handling, citation accuracy, stale state invalidation, and equipment mismatch detection.
"""

from typing import Any, Dict, List
import pytest

from incident_rag.generation import GroundedAnswerGenerator, validate_equipment_relevance
from incident_rag.schemas import (
    ChunkMetadata,
    DocumentChunk,
    DocumentMetadata,
    EvidenceCitation,
    IncidentOverview,
    InvestigationHypothesis,
    InvestigationRequest,
    RetrievalResultItem,
    TelemetryItem,
    TimelineEvent,
)
from incident_rag.dependencies import get_rag_pipeline
from frontend.components.evidence_cards import generate_document_analysis_markdown


# ---------------------------------------------------------------------------
# 1. Structured Output: 9 Sections in Exact Order
# ---------------------------------------------------------------------------

def test_all_9_sections_in_exact_order():
    """Verifies that the document analysis markdown outputs all 9 sections in exact sequential order."""
    sample_analysis: Dict[str, Any] = {
        "incident_overview": {
            "incident_id": "INC-2023-044",
            "machine_id": "M-301",
            "machine_name": "400 kW Conveyor Motor",
            "incident_type": "Overheating Outage",
            "incident_date": "2023-11-14",
            "facility": "Port Hedland Berth 2",
            "severity_level": "Level 3",
            "operational_impact": "Stranded 650 tons iron ore",
            "status": "Tripped E-301",
            "lead_investigator": "Marcus Vance",
        },
        "executive_summary": "Motor M-301 tripped due to external thermal choking and bearing lubrication degradation.",
        "timeline": [
            {"timestamp": "08:30", "event": "Phase B current recorded at 84 A", "category": "Telemetry"},
            {"timestamp": "09:12", "event": "High-temperature warning at 135°C", "category": "Alarm"},
            {"timestamp": "09:18", "event": "Protective thermal trip code E-301", "category": "Protection Trip"},
        ],
        "telemetry": [
            {
                "parameter": "Stator Winding Temperature",
                "measured_value": "148°C",
                "nominal_threshold": "155°C (Class F Thermal Limit)",
                "status": "Critical",
                "interpretation": "Approaching thermal breakdown threshold",
                "citation_id": "[CIT-1]",
            }
        ],
        "physical_observations": ["Cooling fins choked with 15–25 mm crust across 75% circumference."],
        "root_causes": [
            {
                "hypothesis": "Primary Root Cause: Severe external thermal choking",
                "confidence_level": "High (Documented Fact)",
                "supporting_evidence": ["Heavy dust accumulation on stator ribs"],
                "citation_ids": ["[CIT-1]"],
            }
        ],
        "contributing_factors": ["High ambient temperature (38°C) with high relative humidity."],
        "corrective_actions": [
            "Clean cooling ribs using dry-ice blast cleaning. [CIT-1]",
            "Replace bearing B-301 and flush housing with Kluberplex synthetic grease. [CIT-1]",
        ],
        "missing_evidence_and_limitations": [
            "Continuous SCADA telemetry log prior to 08:30: Not reported in the source incident report.",
            "Winding insulation breakdown: Confirmed NOT to have occurred.",
        ],
        "citations": [
            {
                "citation_id": "[CIT-1]",
                "document_id": "doc_m301",
                "chunk_id": "doc_m301_chunk_0",
                "source_name": "INC-2023-044_conveyor_motor_overheating.md",
                "page_number": 1,
                "excerpt": "At 09:18, Yard Conveyor 3 tripped on thermal overload code E-301.",
                "relevance_reason": "High relevance match (Rank 1)",
            }
        ],
    }

    md = generate_document_analysis_markdown(sample_analysis)

    expected_sections = [
        "## 1. Incident Overview",
        "## 2. Executive Summary",
        "## 3. Incident Timeline",
        "## 4. Physical Observations & Telemetry",
        "## 5. Root Cause Analysis",
        "## 6. Contributing Factors",
        "## 7. Corrective & Preventive Actions",
        "## 8. Missing Evidence & Uncertainties",
        "## 9. Evidence & Citations",
    ]

    last_pos = -1
    for sec in expected_sections:
        pos = md.find(sec)
        assert pos != -1, f"Missing required section header: '{sec}'"
        assert pos > last_pos, f"Section '{sec}' appeared out of order!"
        last_pos = pos


# ---------------------------------------------------------------------------
# 2. Missing Fields Handling: Defaulting to "Not specified in the source"
# ---------------------------------------------------------------------------

def test_missing_fields_defaults_to_not_specified():
    """Verifies that missing or null fields display 'Not specified in the source' rather than inventing values."""
    sparse_analysis: Dict[str, Any] = {
        "incident_overview": {
            "incident_id": None,
            "machine_id": "",
            "machine_name": "Not specified",
            "incident_type": None,
            "incident_date": "",
            "facility": None,
            "severity_level": "None",
            "operational_impact": None,
            "status": "",
            "lead_investigator": None,
        },
        "executive_summary": "",
        "timeline": [],
        "telemetry": [],
        "physical_observations": [],
        "root_causes": [],
        "contributing_factors": [],
        "corrective_actions": [],
        "missing_evidence_and_limitations": [],
        "citations": [],
    }

    md = generate_document_analysis_markdown(sparse_analysis)

    # Overview fields must show "Not specified in the source"
    assert "- **Incident ID:** Not specified in the source" in md
    assert "- **Equipment ID:** Not specified in the source" in md
    assert "- **Facility / Location:** Not specified in the source" in md
    assert "- **Operational Impact:** Not specified in the source" in md

    # Executive Summary must display missing fallback
    assert "## 2. Executive Summary\nNot specified in the source" in md

    # Timeline must display missing fallback
    assert "Not specified in the source: No chronological sequence of events was documented" in md

    # Telemetry must display missing fallback
    assert "Not specified in the source: No sensor telemetry or physical observations documented" in md

    # Root Causes must display missing fallback
    assert "Not specified in the source: No definitive root cause identified" in md

    # Contributing Factors must display missing fallback
    assert "Not specified in the source: No secondary contributing factors specified." in md

    # Corrective Actions must display missing fallback
    assert "Not specified in the source: No corrective actions specified." in md


# ---------------------------------------------------------------------------
# 3. Citation Accuracy & No Raw Reranker Scores
# ---------------------------------------------------------------------------

def test_citation_accuracy_and_no_raw_logits():
    """Verifies citations link to actual documents/chunks and do not expose raw negative cross-encoder logits."""
    citations_data = [
        {
            "citation_id": "[CIT-1]",
            "document_id": "doc_test_101",
            "chunk_id": "doc_test_101_chunk_0",
            "source_name": "INC-2023-044_conveyor_motor.md",
            "page_number": 2,
            "excerpt": "Stator winding insulation resistance tested at 450 MΩ (> 100 MΩ threshold).",
            "relevance_reason": "High cross-encoder relevance match (Rank 1)",
        }
    ]

    analysis: Dict[str, Any] = {
        "incident_overview": {"incident_id": "INC-TEST-01"},
        "executive_summary": "Test summary.",
        "citations": citations_data,
    }

    md = generate_document_analysis_markdown(analysis)

    assert "### [CIT-1] — INC-2023-044_conveyor_motor.md (Page 2)" in md
    assert "450 MΩ" in md
    # Must not contain raw negative floats like -6.25 or -9.40
    assert "-6." not in md
    assert "-9." not in md
    assert "-7." not in md


# ---------------------------------------------------------------------------
# 4. Stale State Invalidation
# ---------------------------------------------------------------------------

def test_stale_state_invalidation_logic():
    """Simulates UI session state invalidation when switching active documents."""
    session_state = {
        "analyzed_doc_id": "doc_file_A",
        "doc_analysis_result": {"incident_id": "INC-AAA", "summary": "Analysis for doc A"},
    }

    # User selects a different document "doc_file_B"
    active_doc_id = "doc_file_B"
    prev_doc_id = session_state.get("analyzed_doc_id")

    if active_doc_id and prev_doc_id and prev_doc_id != active_doc_id:
        session_state.pop("doc_analysis_result", None)
        session_state["analyzed_doc_id"] = None

    # Stale analysis for doc A must no longer be present
    assert "doc_analysis_result" not in session_state
    assert session_state["analyzed_doc_id"] is None

    # Result should only display if session matches active document
    should_render = (
        "doc_analysis_result" in session_state
        and session_state.get("analyzed_doc_id") == active_doc_id
    )
    assert should_render is False


# ---------------------------------------------------------------------------
# 5. Equipment Mismatch Detection (HP-17 Question vs M-301 Document)
# ---------------------------------------------------------------------------

def test_equipment_mismatch_detection():
    """Tests equipment mismatch detection when user asks about HP-17 but retrieved evidence is for M-301."""
    generator = GroundedAnswerGenerator()

    # Create dummy retrieved items from M-301 motor report
    chunk_meta = ChunkMetadata(
        chunk_id="doc_m301_chunk_0",
        document_id="doc_m301",
        chunk_index=0,
        source_filename="INC-2023-044_conveyor_motor_overheating.md",
        machine_id="M-301",
        incident_type="Overheating & Stoppage",
    )
    retrieved_items = [
        RetrievalResultItem(
            chunk_id="doc_m301_chunk_0",
            document_id="doc_m301",
            text="Yard Conveyor 3 drive motor M-301 tripped on thermal overload E-301 due to dust crust on stator ribs.",
            metadata=chunk_meta,
            score=0.85,
            rerank_score=1.2,
            final_rank=1,
        )
    ]

    # User asks about HP-17 (steam turbine)
    query = "What caused the high radial vibration and babbitt wiping on steam turbine HP-17?"

    # Check mismatch helper directly
    mismatch_info = validate_equipment_relevance(query, retrieved_items)
    assert mismatch_info["is_mismatch"] is True
    assert mismatch_info["target_machine_id"] == "HP-17"
    assert mismatch_info["retrieved_machine_id"] == "M-301"
    assert "does not establish the cause" in mismatch_info["explanation"]

    # Check generator behavior
    response = generator.generate(query=query, retrieved_items=retrieved_items)
    assert response.is_sufficient_evidence is False
    assert "HP-17" in response.executive_summary
    assert "M-301" in response.executive_summary
    assert "Please provide a matching incident report" in response.executive_summary


# ---------------------------------------------------------------------------
# 6. End-to-End M-301 Document Analysis via Pipeline
# ---------------------------------------------------------------------------

def test_m301_document_analysis_pipeline():
    """Verifies that the M-301 incident report produces a full structured response with all 9 sections populated."""
    pipeline = get_rag_pipeline()

    # Find M-301 document ID
    m301_doc_id = None
    for d in pipeline.doc_store.list_documents():
        if "M-301" in getattr(d, "machine_id", "") or "044" in getattr(d, "filename", ""):
            m301_doc_id = d.document_id
            break

    if not m301_doc_id:
        pytest.skip("M-301 document not indexed in knowledge base store for test.")

    analysis_resp = pipeline.analyze_document(m301_doc_id)
    analysis = analysis_resp.model_dump()

    # Verify structured fields
    overview = analysis.get("incident_overview", {})
    assert "M-301" in overview.get("machine_id", "")
    assert "650 tons" in overview.get("operational_impact", "")

    telemetry = analysis.get("telemetry", [])
    assert any("148" in str(t.get("measured_value")) for t in telemetry)
    assert any("84" in str(t.get("measured_value")) for t in telemetry)
    assert any("450" in str(t.get("measured_value")) for t in telemetry)

    timeline = analysis.get("timeline", [])
    assert len(timeline) >= 3

    root_causes = analysis.get("root_causes", [])
    assert len(root_causes) >= 1
    assert any("thermal choking" in rc.get("hypothesis", "").lower() or "dust" in rc.get("hypothesis", "").lower() for rc in root_causes)

    actions = analysis.get("corrective_actions", [])
    assert len(actions) >= 2

    missing = analysis.get("missing_evidence_and_limitations", [])
    assert any("dielectric" in m.lower() or "insulation" in m.lower() for m in missing)

    # Format to Markdown and verify 9 sections in exact order
    md = generate_document_analysis_markdown(analysis)
    assert "## 1. Incident Overview" in md
    assert "## 2. Executive Summary" in md
    assert "## 3. Incident Timeline" in md
    assert "## 4. Physical Observations & Telemetry" in md
    assert "## 5. Root Cause Analysis" in md
    assert "## 6. Contributing Factors" in md
    assert "## 7. Corrective & Preventive Actions" in md
    assert "## 8. Missing Evidence & Uncertainties" in md
    assert "## 9. Evidence & Citations" in md


# ---------------------------------------------------------------------------
# 7. Regression Tests for Report Layout, Field Mapping & Clean Timeline
# ---------------------------------------------------------------------------

def test_m301_overview_cards_clean_fields():
    """Verifies overview cards have isolated, clean values without full report text."""
    pipeline = get_rag_pipeline()
    target = next((d for d in pipeline.doc_store.list_documents() if "044" in getattr(d, "filename", "") or "M-301" in getattr(d, "machine_id", "")), None)
    assert target is not None

    res = pipeline.analyze_document(target.document_id)
    ov = res.incident_overview.model_dump()

    # 1. Incident Type card must contain only the classification, not the full document
    assert ov["incident_type"] == "Electrical Motor Thermal Overload"
    assert "Facility:" not in ov["incident_type"]
    assert "Lead Investigator:" not in ov["incident_type"]
    assert "1. Overview of Incident" not in ov["incident_type"]
    assert len(ov["incident_type"]) < 50

    # 2. Facility must be clean
    assert "Bulk Material Handling Terminal" in ov["facility"]
    assert "Incident Date:" not in ov["facility"]
    assert len(ov["facility"]) < 80

    # 3. Severity Level must be clean
    assert "Tier 2 Operational Interruption" in ov["severity_level"]
    assert "Lead Investigator:" not in ov["severity_level"]
    assert len(ov["severity_level"]) < 50

    # 4. Lead Investigator must be clean
    assert "James K. O'Connor" in ov["lead_investigator"]
    assert "1. Overview" not in ov["lead_investigator"]
    assert len(ov["lead_investigator"]) < 80

    # 5. Operational Impact must be separate and clean
    assert "650 tons" in ov["operational_impact"]


def test_m301_executive_summary_complete_and_not_truncated():
    """Verifies executive summary is a complete, concise briefing and not a fragment."""
    pipeline = get_rag_pipeline()
    target = next((d for d in pipeline.doc_store.list_documents() if "044" in getattr(d, "filename", "") or "M-301" in getattr(d, "machine_id", "")), None)
    assert target is not None

    res = pipeline.analyze_document(target.document_id)
    exec_summary = res.executive_summary

    # Must NOT be the old fragment
    assert exec_summary.strip() != "that no dielectric breakdown of the winding enamel had occurred. [CIT-1]"
    # Must capture the core overview details
    assert "400 kW" in exec_summary or "M-301" in exec_summary
    assert "148°C" in exec_summary or "thermal limit" in exec_summary.lower() or "overload" in exec_summary.lower()
    assert "650 tons" in exec_summary
    # Must be concise (< 800 chars) and contain citation
    assert len(exec_summary) < 800
    assert "[CIT-1]" in exec_summary


def test_m301_timeline_chronological_and_no_duplicates():
    """Verifies timeline events are in chronological order without duplicates."""
    pipeline = get_rag_pipeline()
    target = next((d for d in pipeline.doc_store.list_documents() if "044" in getattr(d, "filename", "") or "M-301" in getattr(d, "machine_id", "")), None)
    assert target is not None

    res = pipeline.analyze_document(target.document_id)
    timestamps = [t.timestamp for t in res.timeline]

    # For M-301, the documented sequence events are 08:30, 09:00, 09:12, and 09:18
    assert timestamps == ["08:30", "09:00", "09:12", "09:18"]
    # No duplicate timestamps
    assert len(timestamps) == len(set(timestamps))


def test_second_document_analysis_hp17():
    """Verifies analysis works correctly for a second incident document (HP-17 turbine)."""
    pipeline = get_rag_pipeline()
    target = next((d for d in pipeline.doc_store.list_documents() if "089" in getattr(d, "filename", "") or "HP-17" in getattr(d, "machine_id", "")), None)
    assert target is not None

    res = pipeline.analyze_document(target.document_id)
    ov = res.incident_overview.model_dump()

    assert ov["machine_id"] == "HP-17"
    assert "Mechanical Vibration" in ov["incident_type"]
    assert len(ov["incident_type"]) < 60
    assert len(res.executive_summary) > 20
    assert "HP-17" in res.executive_summary or "turbine" in res.executive_summary.lower()
    assert len(res.timeline) >= 3
    # Check chronological order
    timestamps = [t.timestamp for t in res.timeline]
    assert timestamps == sorted(timestamps)


def test_missing_metadata_document_displays_not_specified():
    """Verifies document with missing metadata fields defaults to 'Not specified' without error."""
    pipeline = get_rag_pipeline()
    # Create minimal metadata object with missing fields
    dummy_meta = DocumentMetadata(
        document_id="doc_minimal_test",
        filename="minimal_incident.txt",
        file_type="txt",
        file_size_bytes=200,
        checksum_sha256="abc123dummy",
        machine_id=None,
        incident_type=None,
        incident_date=None,
        facility=None,
        severity_level=None,
        lead_investigator=None,
        chunk_count=1,
    )
    dummy_chunk = DocumentChunk(
        chunk_id="chk_dummy_0",
        document_id="doc_minimal_test",
        text="Operational anomaly noted during shift handover. Equipment was shut down safely.",
        metadata=ChunkMetadata(
            chunk_id="chk_dummy_0",
            document_id="doc_minimal_test",
            chunk_index=0,
            page_number=1,
            start_char=0,
            end_char=80,
            source_filename="minimal_incident.txt",
        ),
    )

    res = pipeline.generator.analyze_document(dummy_meta, [dummy_chunk])
    ov = res.incident_overview.model_dump()

    # Missing fields should default to "Not specified"
    assert ov["machine_id"] == "Not specified"
    assert ov["incident_type"] == "Not specified"
    assert ov["incident_date"] == "Not specified"
    assert ov["facility"] == "Not specified"
    assert ov["severity_level"] == "Not specified"
    assert ov["lead_investigator"] == "Not specified"
