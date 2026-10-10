"""Structure-aware and metadata-preserving text chunker for industrial incident reports."""

import re
from typing import Dict, List, Optional
from incident_rag.ingestion.loader import LoadedDocument
from incident_rag.schemas import ChunkMetadata, DocumentChunk


def split_into_paragraphs(text: str) -> List[str]:
    """Splits text into coherent paragraphs and section blocks."""
    # Split on double newlines or markdown headings
    parts = re.split(r"(?:\n\s*\n|(?=^#{1,4}\s+))", text, flags=re.MULTILINE)
    return [p.strip() for p in parts if p.strip()]


def split_paragraph_into_sentences(para: str) -> List[str]:
    """Splits a paragraph into sentences without breaking technical IDs like HP-17 or 14.2 mm/s."""
    # Split on periods followed by whitespace and capital letter, not preceded by single letters or digits
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", para)
    return [s.strip() for s in sentences if s.strip()]


def chunk_document(
    doc: LoadedDocument,
    chunk_size: int = 400,
    chunk_overlap: int = 80,
    extracted_meta: Optional[Dict[str, Optional[str]]] = None,
) -> List[DocumentChunk]:
    """Splits a LoadedDocument into structured, overlapping chunks with stable IDs and metadata."""
    if extracted_meta is None:
        extracted_meta = {}

    machine_id = extracted_meta.get("machine_id")
    incident_type = extracted_meta.get("incident_type")
    incident_date = extracted_meta.get("incident_date")

    chunks: List[DocumentChunk] = []
    seen_texts = set()
    chunk_idx = 0

    global_offset = 0

    for page in doc.pages:
        page_num = page.page_number
        paragraphs = split_into_paragraphs(page.text)

        current_words: List[str] = []
        current_start_offset = global_offset

        for para in paragraphs:
            para_words = para.split()
            if not para_words:
                continue

            # If adding this paragraph exceeds target chunk_size and we already have words
            if len(current_words) + len(para_words) > chunk_size and current_words:
                chunk_text = " ".join(current_words).strip()
                if chunk_text and chunk_text not in seen_texts:
                    seen_texts.add(chunk_text)
                    chunk_id = f"{doc.document_id}_chk_{chunk_idx}"
                    end_offset = current_start_offset + len(chunk_text)

                    meta = ChunkMetadata(
                        chunk_id=chunk_id,
                        document_id=doc.document_id,
                        chunk_index=chunk_idx,
                        page_number=page_num,
                        start_char=current_start_offset,
                        end_char=end_offset,
                        machine_id=machine_id,
                        incident_type=incident_type,
                        incident_date=incident_date,
                        source_filename=doc.filename,
                    )
                    chunks.append(DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=doc.document_id,
                        text=chunk_text,
                        metadata=meta,
                    ))
                    chunk_idx += 1

                # Implement overlap by keeping the trailing words
                overlap_count = min(chunk_overlap, len(current_words))
                current_words = current_words[-overlap_count:] + para_words
                current_start_offset += len(chunk_text) - len(" ".join(current_words[:overlap_count]))
            else:
                # If single paragraph is larger than chunk_size, split by sentences
                if len(para_words) > chunk_size:
                    sentences = split_paragraph_into_sentences(para)
                    for sent in sentences:
                        sent_words = sent.split()
                        if len(current_words) + len(sent_words) > chunk_size and current_words:
                            chunk_text = " ".join(current_words).strip()
                            if chunk_text and chunk_text not in seen_texts:
                                seen_texts.add(chunk_text)
                                chunk_id = f"{doc.document_id}_chk_{chunk_idx}"
                                end_offset = current_start_offset + len(chunk_text)

                                meta = ChunkMetadata(
                                    chunk_id=chunk_id,
                                    document_id=doc.document_id,
                                    chunk_index=chunk_idx,
                                    page_number=page_num,
                                    start_char=current_start_offset,
                                    end_char=end_offset,
                                    machine_id=machine_id,
                                    incident_type=incident_type,
                                    incident_date=incident_date,
                                    source_filename=doc.filename,
                                )
                                chunks.append(DocumentChunk(
                                    chunk_id=chunk_id,
                                    document_id=doc.document_id,
                                    text=chunk_text,
                                    metadata=meta,
                                ))
                                chunk_idx += 1

                            overlap_count = min(chunk_overlap, len(current_words))
                            current_words = current_words[-overlap_count:] + sent_words
                        else:
                            current_words.extend(sent_words)
                else:
                    current_words.extend(para_words)

        # Flush any remaining words on the page
        if current_words:
            chunk_text = " ".join(current_words).strip()
            if chunk_text and chunk_text not in seen_texts:
                seen_texts.add(chunk_text)
                chunk_id = f"{doc.document_id}_chk_{chunk_idx}"
                end_offset = current_start_offset + len(chunk_text)

                meta = ChunkMetadata(
                    chunk_id=chunk_id,
                    document_id=doc.document_id,
                    chunk_index=chunk_idx,
                    page_number=page_num,
                    start_char=current_start_offset,
                    end_char=end_offset,
                    machine_id=machine_id,
                    incident_type=incident_type,
                    incident_date=incident_date,
                    source_filename=doc.filename,
                )
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=doc.document_id,
                    text=chunk_text,
                    metadata=meta,
                ))
                chunk_idx += 1

        global_offset += len(page.text)

    return chunks
