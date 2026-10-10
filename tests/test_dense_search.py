"""Tests for dense embedding encoder and vector search using Qdrant in-memory mode."""

import pytest
from incident_rag.embeddings.encoder import get_embedding_encoder
from incident_rag.schemas import ChunkMetadata, DocumentChunk
from incident_rag.storage.vector_store import VectorStore
from incident_rag.retrieval.dense_search import DenseSearchEngine


def test_embedding_encoder():
    encoder = get_embedding_encoder()
    assert encoder.embedding_dim == 384

    doc_embs = encoder.encode_documents(["Turbine vibration spiked on bearing B-101."])
    assert len(doc_embs) == 1
    assert len(doc_embs[0]) == 384

    q_emb = encoder.encode_query("What caused the bearing failure?")
    assert len(q_emb) == 384


def test_vector_store_in_memory_indexing_and_search():
    encoder = get_embedding_encoder()
    store = VectorStore(collection_name="test_dense_search", in_memory=True, vector_dim=384)

    chunk1 = DocumentChunk(
        chunk_id="c1",
        document_id="doc1",
        text="High-pressure steam turbine HP-17 experienced catastrophic bearing vibration.",
        metadata=ChunkMetadata(
            chunk_id="c1",
            document_id="doc1",
            chunk_index=0,
            page_number=1,
            machine_id="HP-17",
            incident_type="Mechanical Vibration",
            source_filename="doc1.md",
        ),
    )
    chunk2 = DocumentChunk(
        chunk_id="c2",
        document_id="doc2",
        text="Batch polymerization reactor RX-404 runaway reaction during methyl methacrylate charging.",
        metadata=ChunkMetadata(
            chunk_id="c2",
            document_id="doc2",
            chunk_index=0,
            page_number=1,
            machine_id="RX-404",
            incident_type="Thermal Runaway",
            source_filename="doc2.txt",
        ),
    )

    embs = encoder.encode_documents([chunk1.text, chunk2.text])
    store.upsert_chunks([chunk1, chunk2], embs)
    assert store.count_chunks() == 2

    engine = DenseSearchEngine(vector_store=store, encoder=encoder)

    # Search for turbine
    results = engine.search("steam turbine bearing vibration", top_k=2)
    assert len(results) >= 1
    assert results[0].chunk_id == "c1"
    assert results[0].dense_score is not None
    assert results[0].dense_score > 0.4
    assert results[0].metadata.machine_id == "HP-17"

    # Search with filter
    filtered_results = engine.search("reaction runaway", top_k=2, machine_id_filter="RX-404")
    assert len(filtered_results) == 1
    assert filtered_results[0].chunk_id == "c2"

    # Delete document
    store.delete_document("doc1")
    assert store.count_chunks() == 1
