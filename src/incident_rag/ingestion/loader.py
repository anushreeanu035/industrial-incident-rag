"""Document loader supporting PDF, Markdown, and TXT files with text cleaning."""

import hashlib
import re
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional
import pymupdf
from incident_rag.logging_config import get_logger

logger = get_logger(__name__)


class ExtractedPage(NamedTuple):
    page_number: int
    text: str


class LoadedDocument(NamedTuple):
    document_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    checksum_sha256: str
    pages: List[ExtractedPage]
    full_text: str


def clean_text(raw_text: str) -> str:
    """Cleans malformed whitespace while preserving paragraph structure, lists, and technical symbols."""
    # Replace carriage returns
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace tabs with 4 spaces
    text = text.replace("\t", "    ")
    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple inline spaces/tabs to single space, but keep newlines intact
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n")]
    cleaned = "\n".join(lines).strip()
    return cleaned


def calculate_sha256(content_bytes: bytes) -> str:
    """Computes SHA-256 hex digest for given byte content."""
    return hashlib.sha256(content_bytes).hexdigest()


def load_file(file_path: Path) -> LoadedDocument:
    """Loads a document from disk and extracts text and metadata."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    content_bytes = file_path.read_bytes()
    return load_bytes(content_bytes, file_path.name)


def load_bytes(content_bytes: bytes, filename: str) -> LoadedDocument:
    """Loads document from raw bytes."""
    suffix = Path(filename).suffix.lower()
    checksum = calculate_sha256(content_bytes)
    file_size = len(content_bytes)
    document_id = f"doc_{checksum[:16]}"
    pages: List[ExtractedPage] = []

    if suffix == ".pdf":
        try:
            doc = pymupdf.open(stream=content_bytes, filetype="pdf")
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_text = clean_text(page.get_text("text"))
                if page_text:
                    pages.append(ExtractedPage(page_number=page_idx + 1, text=page_text))
            doc.close()
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {filename}: {e}")
            raise ValueError(f"Corrupt or unreadable PDF document '{filename}': {str(e)}")

        if not pages:
            raise ValueError(f"PDF document '{filename}' contains no extractable text.")

        file_type = "pdf"
        full_text = "\n\n".join(p.text for p in pages)

    elif suffix in [".md", ".markdown"]:
        text = content_bytes.decode("utf-8", errors="replace")
        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError(f"Markdown document '{filename}' is empty.")
        pages.append(ExtractedPage(page_number=1, text=cleaned))
        file_type = "md"
        full_text = cleaned

    elif suffix in [".txt", ".text"]:
        text = content_bytes.decode("utf-8", errors="replace")
        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError(f"Text document '{filename}' is empty.")
        pages.append(ExtractedPage(page_number=1, text=cleaned))
        file_type = "txt"
        full_text = cleaned

    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Supported formats: .pdf, .md, .txt")

    return LoadedDocument(
        document_id=document_id,
        filename=filename,
        file_type=file_type,
        file_size_bytes=file_size,
        checksum_sha256=checksum,
        pages=pages,
        full_text=full_text,
    )
