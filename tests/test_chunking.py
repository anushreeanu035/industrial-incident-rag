"""Tests for structure-aware document chunking and metadata preservation."""

from pathlib import Path
from incident_rag.ingestion.chunker import chunk_document, split_into_paragraphs
from incident_rag.ingestion.loader import load_file


def test_paragraph_splitting():
    text = "Paragraph 1 line A.\nParagraph 1 line B.\n\nParagraph 2 line A.\n\n# Header Section\nParagraph 3."
    paras = split_into_paragraphs(text)
    assert len(paras) == 3
    assert paras[0].startswith("Paragraph 1")
    assert paras[1].startswith("Paragraph 2")
    assert "Header Section" in paras[2]


def test_chunking_metadata_and_ids():
    raw_md = Path("data/raw/INC-2024-089_turbine_bearing_failure.md")
    assert raw_md.exists()
    doc = load_file(raw_md)

    extracted_meta = {
        "machine_id": "HP-17",
        "incident_type": "Mechanical Vibration",
        "incident_date": "2024-03-14",
    }
    chunks = chunk_document(doc, chunk_size=100, chunk_overlap=20, extracted_meta=extracted_meta)

    assert len(chunks) > 1
    for idx, chk in enumerate(chunks):
        assert chk.chunk_id == f"{doc.document_id}_chk_{idx}"
        assert chk.document_id == doc.document_id
        assert chk.metadata.chunk_index == idx
        assert chk.metadata.machine_id == "HP-17"
        assert chk.metadata.incident_type == "Mechanical Vibration"
        assert chk.metadata.source_filename == doc.filename
        assert chk.metadata.page_number == 1
        assert len(chk.text) > 0


def test_chunking_pdf_pages():
    raw_pdf = Path("data/raw/INC-2024-500_transformer_explosion_safety_bulletin.pdf")
    doc = load_file(raw_pdf)

    chunks = chunk_document(doc, chunk_size=150, chunk_overlap=30)
    page_numbers = {c.metadata.page_number for c in chunks}
    assert 1 in page_numbers
    assert 2 in page_numbers
