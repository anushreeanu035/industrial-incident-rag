"""Tests for configuration, schemas, and benchmark dataset validity."""

import json
from pathlib import Path
import pytest
from incident_rag.config import Settings, get_settings
from incident_rag.schemas import (
    ChunkMetadata,
    DocumentChunk,
    DocumentMetadata,
    InvestigationResponse,
    InvestigationHypothesis,
    EvidenceCitation,
    RetrievalRequest,
    RetrievalStrategy,
)


def test_settings_load_and_resolve_path():
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
    assert settings.DENSE_TOP_K == 10
    assert settings.RRF_K == 60

    resolved_benchmark = settings.resolve_path(settings.BENCHMARK_DATASET_PATH)
    assert resolved_benchmark.exists(), f"Benchmark dataset missing at {resolved_benchmark}"


def test_schemas_chunk_and_document():
    chunk_meta = ChunkMetadata(
        chunk_id="doc1_chk_0",
        document_id="doc1",
        chunk_index=0,
        page_number=1,
        start_char=0,
        end_char=100,
        machine_id="HP-17",
        incident_type="Mechanical Vibration",
        incident_date="2024-03-14",
        source_filename="test.md",
    )
    chunk = DocumentChunk(
        chunk_id="doc1_chk_0",
        document_id="doc1",
        text="Vibration spiked to 14.2 mm/s on bearing B-101.",
        metadata=chunk_meta,
    )
    assert chunk.metadata.machine_id == "HP-17"
    assert chunk.text.startswith("Vibration")

    doc_meta = DocumentMetadata(
        document_id="doc1",
        filename="test.md",
        file_type="md",
        file_size_bytes=1024,
        checksum_sha256="abcdef123456",
        machine_id="HP-17",
        status="indexed",
    )
    assert doc_meta.status == "indexed"


def test_investigation_response_validation():
    citation = EvidenceCitation(
        citation_id="[CIT-1]",
        document_id="doc1",
        chunk_id="doc1_chk_0",
        source_name="INC-2024-089.md",
        page_number=1,
        excerpt="Vibration spiked to 14.2 mm/s.",
        relevance_reason="Confirms trip vibration magnitude.",
    )
    hypothesis = InvestigationHypothesis(
        hypothesis="Lube oil starvation caused babbitt wiping",
        confidence_level="High (Documented Fact)",
        supporting_evidence=["V-204 was 80% occluded by carbonized sludge."],
        citation_ids=["[CIT-1]"],
    )
    resp = InvestigationResponse(
        executive_summary="Turbine HP-17 tripped due to bearing failure.",
        documented_observations=["Vibration spiked to 14.2 mm/s RMS."],
        potential_root_causes=[hypothesis],
        corrective_actions=["Flush oil header", "Replace B-101"],
        missing_evidence_and_uncertainties=[],
        citations=[citation],
        is_sufficient_evidence=True,
    )
    assert resp.is_sufficient_evidence is True
    assert len(resp.potential_root_causes) == 1
    assert resp.potential_root_causes[0].citation_ids == ["[CIT-1]"]


def test_benchmark_dataset_integrity():
    settings = get_settings()
    benchmark_path = settings.resolve_path(settings.BENCHMARK_DATASET_PATH)
    assert benchmark_path.exists()

    with open(benchmark_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "test_cases" in data
    assert len(data["test_cases"]) >= 10

    for tc in data["test_cases"]:
        assert "query_id" in tc
        assert "query" in tc
        assert "target_filename" in tc
        assert "reference_answer" in tc
        assert "ground_truth_passages" in tc
        assert len(tc["ground_truth_passages"]) > 0

        # Check raw file exists
        raw_file = settings.resolve_path(f"data/raw/{tc['target_filename']}")
        assert raw_file.exists(), f"Target file {tc['target_filename']} does not exist in data/raw/"
