"""Metadata extraction heuristics for industrial incident documents."""

import re
from typing import Dict, Optional, Tuple


# Regex patterns for common industrial metadata fields
MACHINE_ID_PATTERN = re.compile(
    r"(?:Machine\s*(?:ID|Identifier|No\.?)|Equipment\s*(?:ID|Tag|No\.?)|Tag\s*No\.?)\s*[:=\-]?\s*([A-Z0-9]{1,6}(?:-[A-Z0-9]{1,6})*)",
    re.IGNORECASE
)

INCIDENT_TYPE_PATTERN = re.compile(
    r"(?:Incident\s*Type|Failure\s*Mode|Classification|Category)\s*[:=\-]?\s*([^\n\r\t]+)",
    re.IGNORECASE
)

INCIDENT_DATE_PATTERN = re.compile(
    r"(?:Incident\s*Date|Event\s*Date|Date\s*of\s*Occurrence|Date)\s*[:=\-]?\s*(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|[A-Za-z]+ \d{1,2},? \d{4})",
    re.IGNORECASE
)

# Fallback regex for equipment codes like HP-17, RX-404, M-301, K-102, BFP-02, TR-200 in text
EQUIPMENT_CODE_PATTERN = re.compile(r"\b([A-Z]{1,4}-[0-9]{2,4}[A-Z]?)\b")


def extract_metadata_from_text(text: str, filename: str) -> Dict[str, Optional[str]]:
    """Extracts machine_id, incident_type, and incident_date from document text or filename."""
    machine_id = None
    incident_type = None
    incident_date = None

    # Check explicit headers
    m_match = MACHINE_ID_PATTERN.search(text)
    if m_match:
        machine_id = m_match.group(1).strip()
    else:
        # Check filename or fallback pattern
        fn_match = EQUIPMENT_CODE_PATTERN.search(filename)
        if fn_match:
            machine_id = fn_match.group(1)
        else:
            text_match = EQUIPMENT_CODE_PATTERN.search(text[:2000])
            if text_match:
                machine_id = text_match.group(1)

    t_match = INCIDENT_TYPE_PATTERN.search(text)
    if t_match:
        raw_type = t_match.group(1).strip()
        # Clean markdown asterisks or quotes
        incident_type = raw_type.replace("*", "").replace("#", "").strip()

    d_match = INCIDENT_DATE_PATTERN.search(text)
    if d_match:
        incident_date = d_match.group(1).strip()

    return {
        "machine_id": machine_id,
        "incident_type": incident_type,
        "incident_date": incident_date,
    }
