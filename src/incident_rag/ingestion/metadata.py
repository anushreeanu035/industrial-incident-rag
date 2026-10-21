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


FACILITY_PATTERN = re.compile(
    r"(?:Facility|Location|Plant|Site|Area)\s*[:=\-]?\s*([^\n\r\t]+)",
    re.IGNORECASE,
)

SEVERITY_PATTERN = re.compile(
    r"(?:Severity(?:\s*Level)?|Tier|Priority)\s*[:=\-]?\s*([^\n\r\t]+)",
    re.IGNORECASE,
)

INVESTIGATOR_PATTERN = re.compile(
    r"(?:Lead\s*Investigator|Investigator|Author|Reported\s*By)\s*[:=\-]?\s*([^\n\r\t]+)",
    re.IGNORECASE,
)

INCIDENT_ID_PATTERN = re.compile(
    r"(?:^|[^A-Za-z0-9])(INC-\d{4}-\d{2,4})(?=[^A-Za-z0-9]|$)",
    re.IGNORECASE,
)


def extract_metadata_from_text(text: str, filename: str) -> Dict[str, Optional[str]]:
    """Extracts machine_id, incident_type, incident_date, and forensic metadata from document text or filename."""
    machine_id = None
    incident_type = None
    incident_date = None
    facility = None
    severity_level = None
    lead_investigator = None
    incident_id = None

    # Clean text of markdown formatting for reliable matching
    clean_text = re.sub(r"[*_#`]", "", text)

    # 1. Incident ID extraction
    id_match = INCIDENT_ID_PATTERN.search(clean_text) or INCIDENT_ID_PATTERN.search(filename)
    if id_match:
        incident_id = id_match.group(1).strip()

    # 2. Machine ID extraction
    m_match = MACHINE_ID_PATTERN.search(clean_text)
    if m_match and not m_match.group(1).upper().startswith("INC-"):
        machine_id = m_match.group(1).strip()
    else:
        # Check filename or fallback pattern in text (excluding incident IDs like INC-2023)
        fn_matches = [m for m in EQUIPMENT_CODE_PATTERN.findall(filename) if not m.upper().startswith("INC-")]
        if fn_matches:
            machine_id = fn_matches[0]
        else:
            text_matches = [m for m in EQUIPMENT_CODE_PATTERN.findall(clean_text[:3000]) if not m.upper().startswith("INC-")]
            if text_matches:
                machine_id = text_matches[0]

    # 3. Incident Type extraction
    t_match = INCIDENT_TYPE_PATTERN.search(clean_text)
    if t_match:
        incident_type = t_match.group(1).strip()

    # 4. Incident Date extraction
    d_match = INCIDENT_DATE_PATTERN.search(clean_text)
    if d_match:
        incident_date = d_match.group(1).strip()

    # 5. Facility extraction
    fac_match = FACILITY_PATTERN.search(clean_text)
    if fac_match:
        facility = fac_match.group(1).strip()

    # 6. Severity Level extraction
    sev_match = SEVERITY_PATTERN.search(clean_text)
    if sev_match:
        severity_level = sev_match.group(1).strip()

    # 7. Lead Investigator extraction
    inv_match = INVESTIGATOR_PATTERN.search(clean_text)
    if inv_match:
        lead_investigator = inv_match.group(1).strip()

    return {
        "machine_id": machine_id,
        "incident_type": incident_type,
        "incident_date": incident_date,
        "incident_id": incident_id,
        "facility": facility,
        "severity_level": severity_level,
        "lead_investigator": lead_investigator,
    }

