"""Context construction and citation mapping with prompt injection neutralization."""

import re
from typing import List, Tuple
from incident_rag.schemas import EvidenceCitation, RetrievalResultItem

# Defense against prompt injection in untrusted technical documents
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
    re.compile(r"system\s*prompt\s*override", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+a\s+different\s+ai", re.IGNORECASE),
    re.compile(r"secret\s*key\s*leak", re.IGNORECASE),
]


def sanitize_document_text(text: str) -> str:
    """Neutralizes common prompt injection patterns while preserving technical facts."""
    sanitized = text
    for pattern in INJECTION_PATTERNS:
        sanitized = pattern.sub("[REDACTED UNTRUSTED INSTRUCTION]", sanitized)
    return sanitized


def build_investigation_context(
    retrieved_items: List[RetrievalResultItem],
) -> Tuple[str, List[EvidenceCitation]]:
    """Formats retrieved chunks into a numbered evidence context block and builds citation objects.

    Returns:
        Tuple of (formatted_context_string, list_of_citations)
    """
    if not retrieved_items:
        return "NO RELEVANT RETRIEVED EVIDENCE AVAILABLE.", []

    context_lines = []
    citations: List[EvidenceCitation] = []

    for idx, item in enumerate(retrieved_items, start=1):
        citation_id = f"[CIT-{idx}]"
        meta = item.metadata
        page_str = f"Page {meta.page_number}" if meta.page_number else "Page N/A"
        machine_str = f"Machine: {meta.machine_id}" if meta.machine_id else "Machine: Unknown"

        cleaned_text = sanitize_document_text(item.text)

        header = f"--- EVIDENCE ITEM {citation_id} | Source: {meta.source_filename} | {page_str} | {machine_str} | Chunk: {item.chunk_id} ---"
        context_lines.append(header)
        context_lines.append(cleaned_text)
        context_lines.append("")

        # Create citation object
        # Take first 180 chars as representative excerpt
        excerpt = cleaned_text[:200] + ("..." if len(cleaned_text) > 200 else "")
        citations.append(
            EvidenceCitation(
                citation_id=citation_id,
                document_id=item.document_id,
                chunk_id=item.chunk_id,
                source_name=meta.source_filename,
                page_number=meta.page_number,
                excerpt=excerpt,
                relevance_reason=f"Retrieved rank #{item.final_rank} (Rerank score: {item.rerank_score or 'N/A'})",
            )
        )

    formatted_context = "\n".join(context_lines)
    return formatted_context, citations
