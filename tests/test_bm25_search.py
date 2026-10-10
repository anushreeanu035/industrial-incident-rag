"""Tests for BM25 keyword search engine and technical tokenization."""

from pathlib import Path
from incident_rag.retrieval.bm25_search import BM25SearchEngine, tokenize_technical_text
from incident_rag.schemas import ChunkMetadata, DocumentChunk


def test_technical_tokenization():
    text = "Machine ID: HP-17 tripped on code E-301. Vibration 14.2 mm/s, valve V-204 blocked."
    tokens = tokenize_technical_text(text)
    assert "hp-17" in tokens
    assert "e-301" in tokens
    assert "14.2" in tokens
    assert "v-204" in tokens


def test_bm25_indexing_and_search(tmp_path: Path):
    index_path = tmp_path / "bm25.pkl"
    engine = BM25SearchEngine(index_path=index_path)

    chunk1 = DocumentChunk(
        chunk_id="c1",
        document_id="d1",
        text="High-pressure steam turbine HP-17 bearing B-101 failure and V-204 orifice clog.",
        metadata=ChunkMetadata(
            chunk_id="c1",
            document_id="d1",
            chunk_index=0,
            machine_id="HP-17",
            source_filename="d1.md",
        ),
    )
    chunk2 = DocumentChunk(
        chunk_id="c2",
        document_id="d2",
        text="Batch reactor RX-404 runaway reaction with burst disk PRD-12 rupture.",
        metadata=ChunkMetadata(
            chunk_id="c2",
            document_id="d2",
            chunk_index=0,
            machine_id="RX-404",
            source_filename="d2.txt",
        ),
    )

    engine.add_chunks([chunk1, chunk2])
    assert len(engine.chunks) == 2

    # Query for technical tag V-204
    results = engine.search("V-204", top_k=2)
    assert len(results) >= 1
    assert results[0].chunk_id == "c1"
    assert results[0].bm25_score is not None
    assert results[0].bm25_score > 0

    # Test metadata filter
    filtered = engine.search("failure", top_k=2, machine_id_filter="RX-404")
    assert len(filtered) == 0  # "failure" only in c1 (HP-17)

    # Test removing document
    engine.remove_document_chunks("d1")
    assert len(engine.chunks) == 1
    assert engine.chunks[0].chunk_id == "c2"
