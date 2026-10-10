"""Tests for document loading, text cleaning, and metadata extraction."""

from pathlib import Path
import pytest
from incident_rag.ingestion.loader import clean_text, load_bytes, load_file
from incident_rag.ingestion.metadata import extract_metadata_from_text


def test_clean_text():
    dirty = "Line 1   \t\n\n\n\nLine 2 with   spaces   \r\n\r\nLine 3"
    cleaned = clean_text(dirty)
    assert "Line 1\n\nLine 2 with spaces\n\nLine 3" == cleaned


def test_load_markdown_file(tmp_path: Path):
    md_file = tmp_path / "report.md"
    content = "# Incident Report\n\nMachine ID: HP-17\nVibration spiked to 14.2 mm/s."
    md_file.write_text(content, encoding="utf-8")

    doc = load_file(md_file)
    assert doc.file_type == "md"
    assert doc.filename == "report.md"
    assert len(doc.pages) == 1
    assert "HP-17" in doc.full_text
    assert doc.document_id.startswith("doc_")


def test_load_txt_file(tmp_path: Path):
    txt_file = tmp_path / "log.txt"
    content = "INCIDENT INVESTIGATION REPORT: INC-2024-112\nMachine ID: RX-404\nReactor runaway."
    txt_file.write_text(content, encoding="utf-8")

    doc = load_file(txt_file)
    assert doc.file_type == "txt"
    assert len(doc.pages) == 1
    assert "RX-404" in doc.full_text


def test_load_pdf_file():
    raw_pdf = Path("data/raw/INC-2024-500_transformer_explosion_safety_bulletin.pdf")
    assert raw_pdf.exists()

    doc = load_file(raw_pdf)
    assert doc.file_type == "pdf"
    assert len(doc.pages) == 2
    assert doc.pages[0].page_number == 1
    assert doc.pages[1].page_number == 2
    assert "TR-200" in doc.full_text


def test_unsupported_file_format():
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_bytes(b"some content", "image.png")


def test_empty_file():
    with pytest.raises(ValueError, match="is empty"):
        load_bytes(b"", "empty.txt")


def test_metadata_extraction():
    text = """
    # INCIDENT REPORT
    Machine ID: HP-17
    Incident Type: Mechanical Vibration & Bearing Failure
    Incident Date: 2024-03-14
    """
    meta = extract_metadata_from_text(text, "INC-2024-089.md")
    assert meta["machine_id"] == "HP-17"
    assert "Mechanical Vibration" in meta["incident_type"]
    assert meta["incident_date"] == "2024-03-14"
