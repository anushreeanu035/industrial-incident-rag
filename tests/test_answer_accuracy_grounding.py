"""Unit and integration tests for RAG answer accuracy, evidence grounding, citations, and formatting."""

import pytest
from incident_rag.generation import (
    GroundedAnswerGenerator,
    deduplicate_findings,
    deduplicate_investigation_response,
    extract_equipment_identifiers,
    split_sentences_safely,
    validate_equipment_relevance,
)
from incident_rag.query_rewriting import QueryRewriter
from incident_rag.schemas import (
    ChunkMetadata,
    EvidenceCitation,
    InvestigationHypothesis,
    InvestigationResponse,
    RetrievalResultItem,
)
from frontend.components.evidence_cards import generate_report_markdown


@pytest.fixture
def compressor_chunk_item() -> RetrievalResultItem:
    text = (
        "INCIDENT INVESTIGATION REPORT: INC-2024-201\n"
        "Machine ID: K-102\n"
        "Incident Type: Aerodynamic Compressor Surge & Tripping\n"
        "Executive Summary: Dynamic differential pressure sensors recorded surge pressure swings of 3.8 bar peak-to-peak at 1.8 Hz, placing severe axial thrust loads on active thrust bearing assembly TB-102.\n"
        "Physical Inspection: Rotor axial displacement sensor registered 0.42 mm deflection (normal < 0.15 mm). Inspection of active thrust shoes on TB-102 confirmed babbitt fatigue micro-cracking and minor scuffing due to repetitive transient surge loading.\n"
        "Direct Cause: Spurious closing of bypass valve FV-102 starved compressor K-102 of adequate mass flow, driving the operating point over the surge margin.\n"
        "Root Cause: Failure of desiccant air dryer D-10 switching valve SV-4 allowed high dew-point air into outdoor instrument tubing, causing icing and pneumatic solenoid failure.\n"
        "Systemic Factor: Anti-surge valve ASV-101 actuator had an overly conservative booster volume setting, resulting in a 1.4-second slew time when the design specification called for under 750 milliseconds.\n"
        "Corrective Actions: Upgraded pneumatic booster on anti-surge valve ASV-101, reducing full-stroke opening time from 1.4 seconds to 0.58 seconds."
    )
    meta = ChunkMetadata(
        chunk_id="doc_compressor_chk_0",
        document_id="doc_compressor_id",
        chunk_index=0,
        page_number=1,
        machine_id="K-102",
        incident_type="Aerodynamic Compressor Surge",
        source_filename="INC-2024-201_compressor_surge_shutdown.md",
    )
    return RetrievalResultItem(
        chunk_id="doc_compressor_chk_0",
        document_id="doc_compressor_id",
        text=text,
        metadata=meta,
        rerank_score=0.92,
        final_rank=1,
    )


@pytest.fixture
def turbine_chunk_item() -> RetrievalResultItem:
    text = (
        "INCIDENT INVESTIGATION REPORT: INC-2024-089\n"
        "Machine ID: HP-17\n"
        "Incident Type: Mechanical Vibration & Bearing Failure\n"
        "Executive Summary: High-pressure steam turbine HP-17 tripped automatically on high radial shaft vibration on non-drive end journal bearing B-101, spiking from 2.1 mm/s RMS to 14.2 mm/s RMS within 90 seconds (trip threshold 9.0 mm/s).\n"
        "Physical Inspection: Disassembly of bearing B-101 revealed severe wiping and smearing of the lead-tin-copper babbitt lining on lower tilting pads 3 and 4.\n"
        "Direct Cause: Hydrodynamic lubrication collapse due to severe oil starvation at orifice valve V-204.\n"
        "Root Cause: Seal degradation in oil cooler E-201 allowed cooling water and glycol to contaminate the turbine lubrication circuit.\n"
        "Corrective Actions: Replaced journal bearing set B-101 with precision-machined tilting pad assemblies."
    )
    meta = ChunkMetadata(
        chunk_id="doc_turbine_chk_0",
        document_id="doc_turbine_id",
        chunk_index=0,
        page_number=1,
        machine_id="HP-17",
        incident_type="Mechanical Vibration & Bearing Failure",
        source_filename="INC-2024-089_turbine_bearing_failure.md",
    )
    return RetrievalResultItem(
        chunk_id="doc_turbine_chk_0",
        document_id="doc_turbine_id",
        text=text,
        metadata=meta,
        rerank_score=0.95,
        final_rank=1,
    )


def test_equipment_mismatch_detection(compressor_chunk_item):
    """Verifies that asking about steam turbine HP-17 against compressor K-102 evidence triggers mismatch validation."""
    query = "What caused the high vibration and babbitt wiping on steam turbine HP-17?"
    mismatch_info = validate_equipment_relevance(query, [compressor_chunk_item])

    assert mismatch_info["is_mismatch"] is True
    assert mismatch_info["target_machine_id"] == "HP-17"
    assert mismatch_info["retrieved_machine_id"] == "K-102"
    assert "steam turbine HP-17" in mismatch_info["explanation"]
    assert "compressor K-102" in mismatch_info["explanation"]
    assert "not establish" in mismatch_info["explanation"].lower()


def test_equipment_match_detection(turbine_chunk_item):
    """Verifies that asking about steam turbine HP-17 against HP-17 evidence confirms valid match."""
    query = "What caused the high vibration and babbitt wiping on steam turbine HP-17?"
    mismatch_info = validate_equipment_relevance(query, [turbine_chunk_item])

    assert mismatch_info["is_mismatch"] is False


def test_generic_query_without_equipment_id_not_rejected(compressor_chunk_item):
    """Verifies that questions omitting equipment IDs are not rejected as mismatches."""
    query = "What caused the babbitt micro-cracking and surge shutdown?"
    mismatch_info = validate_equipment_relevance(query, [compressor_chunk_item])

    assert mismatch_info["is_mismatch"] is False


def test_generator_mismatch_response(compressor_chunk_item):
    """Verifies GroundedAnswerGenerator produces an explicit mismatch explanation and sets is_sufficient_evidence=False."""
    generator = GroundedAnswerGenerator(api_key="")  # Extractive deterministic mode
    query = "What caused the high vibration and babbitt wiping on steam turbine HP-17?"

    resp = generator.generate(query, [compressor_chunk_item])

    assert resp.is_sufficient_evidence is False
    assert "K-102" in resp.executive_summary
    assert "HP-17" in resp.executive_summary
    assert "not establish" in resp.executive_summary.lower()
    assert len(resp.potential_root_causes) == 0  # Does not assign compressor causes to turbine
    assert len(resp.missing_evidence_and_uncertainties) > 0


def test_technical_measurements_preserved_exactly(compressor_chunk_item):
    """Verifies that numbers, decimals, and comparison operators are not truncated."""
    generator = GroundedAnswerGenerator(api_key="")
    query = "Analyze the surge telemetry and thrust bearing measurements for compressor K-102."

    resp = generator.generate(query, [compressor_chunk_item])

    obs_combined = " ".join(resp.documented_observations)

    # 1. Surge pressure swings
    assert "3.8 bar peak-to-peak at 1.8 Hz" in obs_combined
    # 2. Deflection and normal threshold
    assert "0.42 mm deflection" in obs_combined
    assert "< 0.15 mm" in obs_combined or "normal < 0.15 mm" in obs_combined
    # 3. No truncated patterns ending in decimal points
    import re
    assert not re.search(r"at 1\.(?!\d)", obs_combined)
    assert not re.search(r"< 0\.(?!\d)", obs_combined)


def test_split_sentences_safely_preserves_decimals_and_units():
    """Verifies that sentence tokenization never breaks on decimals or technical operators."""
    sample_text = (
        "Sensors recorded surge pressure swings of 3.8 bar peak-to-peak at 1.8 Hz. "
        "Rotor deflection was 0.42 mm (normal < 0.15 mm). "
        "Opening slew time was 1.4 seconds compared to design < 750 ms. "
        "Upgraded actuator reduced time to 0.58 seconds."
    )
    sentences = split_sentences_safely(sample_text)
    assert len(sentences) == 4
    assert sentences[0] == "Sensors recorded surge pressure swings of 3.8 bar peak-to-peak at 1.8 Hz."
    assert sentences[1] == "Rotor deflection was 0.42 mm (normal < 0.15 mm)."
    assert sentences[2] == "Opening slew time was 1.4 seconds compared to design < 750 ms."
    assert sentences[3] == "Upgraded actuator reduced time to 0.58 seconds."


def test_deduplication_of_findings():
    """Verifies that redundant sentences across sections are eliminated."""
    hypotheses = [
        InvestigationHypothesis(
            hypothesis="Failure of desiccant air dryer D-10 switching valve SV-4",
            confidence_level="High (Documented Fact)",
            supporting_evidence=["Found stuck mid-stroke [CIT-1]"],
            citation_ids=["[CIT-1]"],
        ),
        InvestigationHypothesis(
            hypothesis="Failure of desiccant air dryer D-10 switching valve SV-4",  # Duplicate
            confidence_level="High (Documented Fact)",
            supporting_evidence=["Found stuck mid-stroke [CIT-1]"],
            citation_ids=["[CIT-1]"],
        ),
    ]

    obs = [
        "Failure of desiccant air dryer D-10 switching valve SV-4 [CIT-1]",  # Redundant with root cause
        "Dynamic differential pressure sensors recorded surge pressure swings of 3.8 bar peak-to-peak at 1.8 Hz [CIT-1]",
    ]

    actions = [
        "Replaced switching valve SV-4 and recharged desiccant alumina bed [CIT-1]",
        "Replaced switching valve SV-4 and recharged desiccant alumina bed [CIT-1]",  # Duplicate
    ]

    resp = InvestigationResponse(
        executive_summary="Incident analysis for K-102.",
        documented_observations=obs,
        potential_root_causes=hypotheses,
        corrective_actions=actions,
        missing_evidence_and_uncertainties=[],
        citations=[],
        is_sufficient_evidence=True,
    )

    clean_resp = deduplicate_investigation_response(resp)

    # Hypotheses deduplicated to 1
    assert len(clean_resp.potential_root_causes) == 1
    # Observation matching root cause removed
    assert len(clean_resp.documented_observations) == 1
    assert "3.8 bar peak-to-peak at 1.8 Hz" in clean_resp.documented_observations[0]
    # Actions deduplicated to 1
    assert len(clean_resp.corrective_actions) == 1


def test_query_rewriter_preserves_equipment_id():
    """Verifies that QueryRewriter always retains machine IDs in expanded queries."""
    rewriter = QueryRewriter(api_key="")
    res = rewriter.rewrite("What caused the vibration and babbitt failure on HP-17?")

    assert len(res.rewritten_queries) > 0
    for q in res.rewritten_queries:
        assert "HP-17" in q


def test_generate_report_markdown_clean_structure():
    """Verifies that generated Markdown report follows the 6-part structure without empty bullets or broken syntax."""
    resp_dict = {
        "executive_summary": "On May 18, 2024, compressor K-102 tripped due to repeated surge cycles.",
        "documented_observations": [
            "Surge pressure swings of 3.8 bar peak-to-peak at 1.8 Hz [CIT-1]",
            "Rotor axial deflection registered 0.42 mm (normal < 0.15 mm) [CIT-1]",
        ],
        "potential_root_causes": [
            {
                "hypothesis": "Direct Cause: Spurious closing of bypass valve FV-102",
                "confidence_level": "High (Documented Fact)",
                "supporting_evidence": ["Suction flow collapsed below surge margin [CIT-1]"],
                "citation_ids": ["[CIT-1]"],
            }
        ],
        "corrective_actions": [
            "Upgraded pneumatic booster on anti-surge valve ASV-101 [CIT-1]"
        ],
        "missing_evidence_and_uncertainties": [],
        "citations": [
            {
                "citation_id": "[CIT-1]",
                "source_name": "INC-2024-201_compressor_surge_shutdown.md",
                "page_number": 1,
                "excerpt": "Dynamic differential pressure sensors recorded surge pressure swings...",
                "relevance_reason": "Rank #1",
            }
        ],
        "is_sufficient_evidence": True,
    }

    report = generate_report_markdown(resp_dict, "What caused compressor K-102 trip?")

    assert "1. Executive Summary: Incident Summary" in report
    assert "2. Documented Observations & Telemetry (Equipment & Operating Context)" in report
    assert "3. Potential Root Causes & Hypotheses (Root Cause Analysis)" in report
    assert "4. Mandated Corrective & Preventive Actions" in report
    assert "5. Missing Evidence & Forensic Uncertainties (Evidence Limitations and Gaps)" in report
    assert "6. Traceable Evidence Citations" in report
    assert "3.8 bar peak-to-peak at 1.8 Hz" in report
    assert "-  " not in report  # No empty bullets
