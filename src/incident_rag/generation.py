"""Evidence-grounded response generation using Groq LLM with deterministic extractive fallback."""

import json
import os
import re
from typing import Any, Dict, List, Optional, Set
from groq import Groq
from incident_rag.config import get_settings
from incident_rag.context_builder import build_investigation_context
from incident_rag.logging_config import get_logger
from incident_rag.schemas import (
    EvidenceCitation,
    InvestigationHypothesis,
    InvestigationResponse,
    RetrievalResultItem,
)

logger = get_logger(__name__)

INVESTIGATION_SYSTEM_PROMPT = """You are an expert Chief Industrial Incident Investigator and Safety Auditor.
Your task is to analyze retrieved industrial equipment failure reports, maintenance telemetry, and incident records to answer the user's inquiry.

STRICT INVESTIGATION RULES:
1. GROUNDING & EVIDENCE FIDELITY:
   - Ground every finding strictly in the provided retrieved evidence items.
   - DO NOT hallucinate, invent, or speculate on machine tags, measurements, incident dates, or root causes.
   - Every factual statement or observation must explicitly reference its evidence citation tag (e.g., [CIT-1], [CIT-2]).
   - Distinguish clearly between documented physical facts and investigative hypotheses.

2. EQUIPMENT RELEVANCE & MISMATCH VALIDATION:
   - Compare the equipment identifiers (e.g., HP-17, K-102, RX-404, TR-200), machine types (e.g., steam turbine, compressor), and incident details in the question with the retrieved document metadata and content.
   - DO NOT treat different machines (e.g., compressor K-102 and steam turbine HP-17) as interchangeable.
   - If the retrieved evidence does not match the requested equipment, state clearly in "executive_summary" that the available document does not establish the cause of the specified incident on the requested machine.
     Format example: "The retrieved report concerns [retrieved machine/component]. It documents [observations], but it does not establish the cause of [symptom] on [requested machine]. Please provide a matching incident report or confirm whether you want an analysis of [retrieved machine]."
   - Set "is_sufficient_evidence" to false when an equipment mismatch prevents answering the query.
   - If the user query omitted an equipment ID (e.g., "What caused the bearing failure?"), do NOT reject the evidence; analyze the retrieved equipment and clearly identify which machine is documented.
   - If a partially relevant document is found, explain which parts are relevant and what remains unverified.

3. PRESERVE TECHNICAL MEASUREMENTS EXACTLY:
   - Preserve all technical measurements, numbers, decimals, comparison operators (<, >, <=, >=, ±, under, over), units (e.g., bar, Hz, mm, mm/s RMS, °C, ppm, seconds, milliseconds), and qualifying conditions EXACTLY as stated in the evidence.
   - NEVER truncate measurements or numbers (e.g., never output "3.8 bar peak-to-peak at 1." or "normal < 0.").
   - ALWAYS output complete sentences and complete measurement expressions (e.g., "3.8 bar peak-to-peak at 1.8 Hz", "0.42 mm deflection (normal < 0.15 mm)", "1.4-second slew time when the design specification called for under 750 milliseconds", "opening time reduced from 1.4 seconds to 0.58 seconds").
   - If a measurement is incomplete in the source text, state that it is incomplete or unavailable; do not guess.

4. ELIMINATE DUPLICATE FINDINGS:
   - Generate each finding, cause, and corrective action ONLY ONCE.
   - Do not repeat findings across sections (observations vs. root causes vs. corrective actions).
   - Categorize findings cleanly:
     * "documented_observations": Physical inspection findings, sensor telemetry, and symptoms.
     * "potential_root_causes": Causal mechanisms (Direct Cause, Root Cause, Systemic Factors) with appropriate confidence levels.
     * "corrective_actions": Physical repairs, procedural changes, and preventive actions.

5. JSON RESPONSE FORMAT:
   Return ONLY a valid JSON object matching the following structure:
{
  "executive_summary": "Concise summary of incident, equipment affected, and key findings.",
  "documented_observations": ["Complete observation with intact telemetry [CIT-1]"],
  "potential_root_causes": [
    {
      "hypothesis": "Causal mechanism explanation",
      "confidence_level": "High (Documented Fact) | Medium (Strong Evidence) | Low (Tentative Hypothesis)",
      "supporting_evidence": ["Supporting evidence point [CIT-1]"],
      "citation_ids": ["[CIT-1]"]
    }
  ],
  "corrective_actions": ["Mandated action [CIT-1]"],
  "missing_evidence_and_uncertainties": ["Missing information point"],
  "is_sufficient_evidence": true
}
"""


def split_sentences_safely(text: str) -> List[str]:
    """Splits text into complete sentences without breaking at decimal numbers, operators, or abbreviations."""
    normalized = re.sub(r"\r\n|\r", "\n", text)
    lines = normalized.split("\n")
    sentences: List[str] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("---") or line.startswith("==="):
            continue
        cleaned_line = re.sub(r"^([#*+\-]\s*|\d+\.\s*)", "", line).strip()
        if not cleaned_line:
            continue

        # Mask decimal numbers: digit.digit -> digit__DOT__digit
        masked = re.sub(r"(\d)\.(\d)", r"\1__DOT__\2", cleaned_line)
        # Mask common title/editorial abbreviations (excluding unit abbreviations like ms)
        masked = re.sub(r"\b(e\.g|i\.e|approx|ref|vs|no|inc|dr|mr|fig)\.", r"\1__ABBRDOT__", masked, flags=re.IGNORECASE)

        # Standard split on sentence termination
        raw_parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", masked)
        for part in raw_parts:
            part = part.strip()
            if part:
                unmasked = part.replace("__DOT__", ".").replace("__ABBRDOT__", ".")
                sentences.append(unmasked)
    return sentences


def extract_equipment_identifiers(text: str) -> Dict[str, Any]:
    """Extracts machine tags, component tags, machine types, and incident numbers from text."""
    machine_ids = set(re.findall(r"\b([A-Z]{1,4}-[0-9]{2,4}[A-Z]?)\b", text))
    incident_ids = set(re.findall(r"\b(INC-\d{4}-\d{3})\b", text, re.IGNORECASE))

    machine_types = []
    type_patterns = [
        (r"\b(?:steam\s+turbine|gas\s+turbine|turbine)\b", "steam turbine"),
        (r"\b(?:cracked\s+gas\s+compressor|centrifugal\s+compressor|compressor)\b", "compressor"),
        (r"\b(?:chemical\s+reactor|cstr|reactor)\b", "chemical reactor"),
        (r"\b(?:step-up\s+transformer|transformer)\b", "transformer"),
        (r"\b(?:boiler\s+feed\s+pump|feed\s+pump|pump)\b", "boiler feed pump"),
        (r"\b(?:conveyor\s+motor|drive\s+motor|motor)\b", "conveyor motor"),
    ]
    text_lower = text.lower()
    for pat, label in type_patterns:
        if re.search(pat, text_lower):
            machine_types.append(label)

    return {
        "machine_ids": machine_ids,
        "incident_ids": incident_ids,
        "machine_types": machine_types,
    }


def validate_equipment_relevance(
    query: str,
    retrieved_items: List[RetrievalResultItem],
) -> Dict[str, Any]:
    """Checks whether retrieved passages match the equipment requested in the query.

    Returns:
        Dict containing mismatch status, identified equipment, and structured mismatch explanation.
    """
    if not retrieved_items:
        return {"is_mismatch": False, "explanation": ""}

    query_info = extract_equipment_identifiers(query)
    query_machine_ids = query_info["machine_ids"]
    query_machine_types = query_info["machine_types"]

    # If the user did not specify any machine ID or machine type, no conflict
    if not query_machine_ids and not query_machine_types:
        return {"is_mismatch": False, "explanation": ""}

    retrieved_machine_ids: Set[str] = set()
    retrieved_machine_types: Set[str] = set()
    all_retrieved_text = " ".join(item.text for item in retrieved_items)

    for item in retrieved_items:
        if item.metadata.machine_id:
            retrieved_machine_ids.add(item.metadata.machine_id)
        if item.metadata.incident_type:
            for pat, label in [
                ("compressor", "compressor"),
                ("turbine", "steam turbine"),
                ("reactor", "chemical reactor"),
                ("transformer", "transformer"),
                ("pump", "boiler feed pump"),
                ("motor", "conveyor motor"),
            ]:
                if pat in item.metadata.incident_type.lower():
                    retrieved_machine_types.add(label)

    text_info = extract_equipment_identifiers(all_retrieved_text)
    retrieved_machine_ids.update(text_info["machine_ids"])
    retrieved_machine_types.update(text_info["machine_types"])

    primary_machines = {"HP-17", "K-102", "RX-404", "TR-200", "BFP-02", "M-301"}
    retrieved_primary = retrieved_machine_ids.intersection(primary_machines)
    query_primary = query_machine_ids.intersection(primary_machines)
    component_tags = [m for m in retrieved_machine_ids if m not in primary_machines]

    target_id = (
        next(iter(query_primary))
        if query_primary
        else (next(iter(query_machine_ids)) if query_machine_ids else None)
    )
    target_type = query_machine_types[0] if query_machine_types else None

    # Check if target ID is present anywhere in retrieved evidence
    target_found = False
    if target_id and (target_id in retrieved_machine_ids or target_id in all_retrieved_text):
        target_found = True

    # Check if explicit target was requested but NOT found in retrieved items
    if target_id and not target_found and retrieved_machine_ids:
        retrieved_id = (
            next(iter(retrieved_primary))
            if retrieved_primary
            else next(iter(retrieved_machine_ids))
        )
        retrieved_type = (
            next(iter(retrieved_machine_types)) if retrieved_machine_types else "equipment"
        )

        target_desc = f"{target_type} {target_id}".strip() if target_type else target_id
        retrieved_desc = f"{retrieved_type} {retrieved_id}".strip()
        comp_desc = f" and thrust bearing {component_tags[0]}" if (component_tags and "TB" in component_tags[0]) else (f" and {component_tags[0]}" if component_tags else "")

        findings_summary = "documented failure telemetry"
        if "scuffing" in all_retrieved_text.lower() or "fatigue" in all_retrieved_text.lower():
            findings_summary = "babbitt fatigue micro-cracking and minor scuffing"
        elif "wiping" in all_retrieved_text.lower():
            findings_summary = "babbitt wiping and severe shaft rub"
        elif "arc flash" in all_retrieved_text.lower():
            findings_summary = "dielectric breakdown and arc flash damage"
        elif "cavitation" in all_retrieved_text.lower():
            findings_summary = "impeller vane pitting and cavitation erosion"

        target_symptom = (
            "high vibration or babbitt wiping"
            if ("vibration" in query.lower() or "babbitt" in query.lower())
            else "the specified incident"
        )

        explanation = (
            f"The retrieved report concerns {retrieved_desc}{comp_desc}. "
            f"It documents {findings_summary}, but it does not establish the cause of {target_symptom} on {target_desc}. "
            f"Please provide a matching incident report or confirm whether you want an analysis of {retrieved_desc}."
        )

        return {
            "is_mismatch": True,
            "target_machine_id": target_id,
            "target_machine_type": target_type,
            "retrieved_machine_id": retrieved_id,
            "retrieved_machine_type": retrieved_type,
            "retrieved_components": list(component_tags),
            "explanation": explanation,
        }

    return {"is_mismatch": False, "explanation": ""}


def normalize_finding(text: str) -> str:
    """Normalizes text for duplicate detection by removing citations, markdown tokens, and extra whitespace."""
    t = re.sub(r"\[CIT-\d+\]", "", text)
    t = re.sub(r"^([#*+\-]\s*|\d+\.\s*)", "", t)
    t = re.sub(r"[^\w\s]", " ", t.lower())
    return " ".join(t.split())


def is_duplicate_finding(item: str, seen: List[str], threshold: float = 0.82) -> bool:
    """Checks if item is semantically redundant with any previously seen finding."""
    norm_item = normalize_finding(item)
    if not norm_item:
        return True
    words_item = set(norm_item.split())
    if not words_item:
        return True

    for s in seen:
        norm_s = normalize_finding(s)
        words_s = set(norm_s.split())
        if not words_s:
            continue
        intersection = len(words_item.intersection(words_s))
        union = len(words_item.union(words_s))
        if union > 0 and (intersection / union) >= threshold:
            return True
        if len(norm_item) > 25 and len(norm_s) > 25:
            if norm_item in norm_s or norm_s in norm_item:
                return True
    return False


def deduplicate_findings(findings: List[str], external_seen: Optional[List[str]] = None) -> List[str]:
    """Deduplicates a list of findings while preserving the original order and most detailed phrasing."""
    seen = list(external_seen) if external_seen else []
    unique_items: List[str] = []
    for item in findings:
        item_str = str(item).strip()
        if not item_str:
            continue
        if not is_duplicate_finding(item_str, seen):
            unique_items.append(item_str)
            seen.append(item_str)
    return unique_items


def deduplicate_investigation_response(resp: InvestigationResponse) -> InvestigationResponse:
    """Eliminates redundant findings across observations, root causes, and corrective actions."""
    seen_all: List[str] = []

    # 1. Clean root causes first
    unique_hypotheses: List[InvestigationHypothesis] = []
    seen_hyp_text: List[str] = []
    for h in resp.potential_root_causes:
        if not is_duplicate_finding(h.hypothesis, seen_hyp_text):
            seen_hyp_text.append(h.hypothesis)
            seen_all.append(h.hypothesis)
            # Deduplicate supporting evidence inside hypothesis
            clean_ev = deduplicate_findings(h.supporting_evidence)
            h.supporting_evidence = clean_ev
            unique_hypotheses.append(h)

    # 2. Clean observations excluding root causes
    clean_obs = deduplicate_findings(resp.documented_observations, external_seen=seen_all)
    seen_all.extend(clean_obs)

    # 3. Clean corrective actions excluding observations & root causes
    clean_acts = deduplicate_findings(resp.corrective_actions, external_seen=seen_all)

    # 4. Clean uncertainties
    clean_unc = deduplicate_findings(resp.missing_evidence_and_uncertainties)

    resp.potential_root_causes = unique_hypotheses
    resp.documented_observations = clean_obs
    resp.corrective_actions = clean_acts
    resp.missing_evidence_and_uncertainties = clean_unc
    return resp


def extract_telemetry_observations(retrieved_items: List[RetrievalResultItem]) -> List[str]:
    """Extracts intact telemetry, sensor measurements, and inspection findings without truncation."""
    observations: List[str] = []
    seen: List[str] = []

    telemetry_keywords = [
        "pressure", "vibration", "temperature", "displacement", "deflection",
        "flow", "sensor", "transmitter", "scuffing", "micro-cracking", "wiping",
        "swings", "slew time", "opening time", "moisture", "contamination",
        "occluded", "alarm", "trip", "threshold", "setpoint", "nominal",
    ]

    for item in retrieved_items:
        sentences = split_sentences_safely(item.text)
        for s in sentences:
            s_lower = s.lower()
            if re.match(r"^(?:direct\s*cause|root\s*cause|systemic|corrective\s*action)", s_lower):
                continue
            has_metric = bool(re.search(r"\d+(?:\.\d+)?\s*(?:mm/s|bar|°c|ppm|hz|mm|seconds|milliseconds|ms)", s_lower))
            has_telemetry_kw = any(k in s_lower for k in telemetry_keywords)

            if has_metric or has_telemetry_kw:
                clean_s = s.strip()
                if len(clean_s) > 20 and not is_duplicate_finding(clean_s, seen):
                    seen.append(clean_s)
                    observations.append(f"{clean_s} [CIT-1]")
                if len(observations) >= 4:
                    break
        if len(observations) >= 4:
            break

    return observations


class GroundedAnswerGenerator:
    """Generates structured, grounded incident investigation responses."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key if api_key is not None else (settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY", ""))
        self.model = model or settings.GROQ_MODEL
        self.client: Optional[Groq] = None

        if self.api_key.strip():
            try:
                self.client = Groq(api_key=self.api_key.strip())
                logger.info(f"Initialized Groq client for generation with model: {self.model}")
            except Exception as e:
                logger.warning(f"Could not initialize Groq client: {e}. Fallback generation enabled.")
                self.client = None
        else:
            logger.info("No Groq API key set. Deterministic extractive generation fallback enabled.")

    def generate(
        self,
        query: str,
        retrieved_items: List[RetrievalResultItem],
        retrieval_metadata: Optional[Dict[str, Any]] = None,
    ) -> InvestigationResponse:
        """Constructs context, validates equipment relevance, invokes Groq or fallback, and returns grounded response."""
        if retrieval_metadata is None:
            retrieval_metadata = {}

        if not retrieved_items:
            return InvestigationResponse(
                executive_summary="No relevant incident reports or documentation could be retrieved for this query.",
                documented_observations=[],
                potential_root_causes=[],
                corrective_actions=[],
                missing_evidence_and_uncertainties=[
                    "No matching incident reports were found in the knowledge base matching the query or filter criteria."
                ],
                citations=[],
                is_sufficient_evidence=False,
                retrieval_metadata=retrieval_metadata,
            )

        context_str, citations = build_investigation_context(retrieved_items)

        # Check equipment mismatch before generation
        mismatch_check = validate_equipment_relevance(query, retrieved_items)

        # Attempt Groq LLM generation
        if self.client:
            try:
                user_prompt = f"User Investigation Query: {query}\n\nRETRIEVED EVIDENCE PASSAGES:\n{context_str}"
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": INVESTIGATION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                    max_tokens=1500,
                    timeout=20.0,
                )
                raw_text = completion.choices[0].message.content or "{}"
                data = json.loads(raw_text)

                hypotheses: List[InvestigationHypothesis] = []
                for h in data.get("potential_root_causes", []):
                    hypotheses.append(
                        InvestigationHypothesis(
                            hypothesis=h.get("hypothesis", "Unspecified hypothesis"),
                            confidence_level=h.get("confidence_level", "Medium (Strong Evidence)"),
                            supporting_evidence=h.get("supporting_evidence", []),
                            citation_ids=h.get("citation_ids", []),
                        )
                    )

                exec_summary = data.get("executive_summary", "Incident analysis completed.")
                is_sufficient = data.get("is_sufficient_evidence", True)
                missing_gaps = data.get("missing_evidence_and_uncertainties", [])

                # If an explicit equipment mismatch was detected, enforce mismatch warning
                if mismatch_check["is_mismatch"]:
                    is_sufficient = False
                    if mismatch_check["target_machine_id"] not in exec_summary or "not establish" not in exec_summary.lower():
                        exec_summary = mismatch_check["explanation"]
                    mismatch_gap_note = (
                        f"The retrieved reports document {mismatch_check['retrieved_machine_id']}, but do not establish "
                        f"the cause of the incident on {mismatch_check['target_machine_id']}."
                    )
                    if mismatch_gap_note not in missing_gaps:
                        missing_gaps.insert(0, mismatch_gap_note)

                resp = InvestigationResponse(
                    executive_summary=exec_summary,
                    documented_observations=data.get("documented_observations", []),
                    potential_root_causes=hypotheses,
                    corrective_actions=data.get("corrective_actions", []),
                    missing_evidence_and_uncertainties=missing_gaps,
                    citations=citations,
                    is_sufficient_evidence=is_sufficient,
                    retrieval_metadata=retrieval_metadata,
                )
                return deduplicate_investigation_response(resp)

            except Exception as e:
                logger.warning(f"Groq answer generation failed: {e}. Executing extractive fallback.")

        # Fallback deterministic extractive synthesis
        return self._extractive_fallback(query, retrieved_items, citations, retrieval_metadata, mismatch_check)

    def _extractive_fallback(
        self,
        query: str,
        retrieved_items: List[RetrievalResultItem],
        citations: List[EvidenceCitation],
        retrieval_metadata: Dict[str, Any],
        mismatch_check: Optional[Dict[str, Any]] = None,
    ) -> InvestigationResponse:
        """Deterministic extractive summarizer preserving exact telemetry and preventing machine conflation."""
        top_item = retrieved_items[0]
        all_text = " ".join(item.text for item in retrieved_items)

        if mismatch_check is None:
            mismatch_check = validate_equipment_relevance(query, retrieved_items)

        # 1. Handle explicit equipment mismatch
        if mismatch_check["is_mismatch"]:
            obs = extract_telemetry_observations(retrieved_items)
            missing = [
                f"The available report in the knowledge base concerns {mismatch_check['retrieved_machine_id']}, not {mismatch_check['target_machine_id']}.",
                f"No verified incident report establishes high vibration or bearing failure on {mismatch_check['target_machine_id']}.",
                "Please provide a matching incident report or confirm whether you want an analysis of the retrieved machine.",
            ]
            return InvestigationResponse(
                executive_summary=mismatch_check["explanation"],
                documented_observations=obs[:3],
                potential_root_causes=[],
                corrective_actions=[],
                missing_evidence_and_uncertainties=missing,
                citations=citations,
                is_sufficient_evidence=False,
                retrieval_metadata=retrieval_metadata,
            )

        # 2. Extract observations with intact measurements
        observations = extract_telemetry_observations(retrieved_items)
        if not observations:
            observations.append(
                f"Primary evidence extracted from {top_item.metadata.source_filename}: {top_item.text[:180]}... [CIT-1]"
            )

        # 3. Extract Root Causes cleanly without duplication
        root_causes: List[InvestigationHypothesis] = []
        direct_match = re.search(r"Direct\s*Cause[^:\n]*?[:\-]\s*([^\n\r]+)", all_text, re.IGNORECASE)
        root_match = re.search(r"(?:Primary\s*Root\s*Cause|Root\s*Cause)[^:\n]*?[:\-]\s*([^\n\r]+)", all_text, re.IGNORECASE)
        systemic_match = re.search(r"(?:Systemic\s*Factor|Contributing\s*Factors?)[^:\n]*?[:\-]\s*([^\n\r]+)", all_text, re.IGNORECASE)

        if direct_match:
            d_text = direct_match.group(1).strip()
            root_causes.append(
                InvestigationHypothesis(
                    hypothesis=f"Direct Cause: {d_text}",
                    confidence_level="High (Documented Fact)",
                    supporting_evidence=[f"Documented in {top_item.metadata.source_filename} [CIT-1]"],
                    citation_ids=["[CIT-1]"],
                )
            )

        if root_match:
            r_text = root_match.group(1).strip()
            # Avoid duplicating direct cause
            if not direct_match or normalize_finding(r_text) != normalize_finding(direct_match.group(1)):
                root_causes.append(
                    InvestigationHypothesis(
                        hypothesis=f"Root Cause: {r_text}",
                        confidence_level="High (Documented Fact)",
                        supporting_evidence=[f"Confirmed via inspection in {top_item.metadata.source_filename} [CIT-1]"],
                        citation_ids=["[CIT-1]"],
                    )
                )

        if systemic_match:
            s_text = systemic_match.group(1).strip()
            root_causes.append(
                InvestigationHypothesis(
                    hypothesis=f"Systemic/Contributing Factor: {s_text}",
                    confidence_level="Medium (Strong Evidence)",
                    supporting_evidence=[f"Noted in {top_item.metadata.source_filename} [CIT-1]"],
                    citation_ids=["[CIT-1]"],
                )
            )

        if not root_causes:
            root_causes.append(
                InvestigationHypothesis(
                    hypothesis=f"Equipment failure associated with {top_item.metadata.machine_id or 'incident'}",
                    confidence_level="Medium (Strong Evidence)",
                    supporting_evidence=[f"Supported by passage {top_item.chunk_id} [CIT-1]"],
                    citation_ids=["[CIT-1]"],
                )
            )

        # 4. Extract corrective actions
        actions: List[str] = []
        action_lines = re.findall(r"(?:^\d+\.\s+|-\s+)([^\n\r]+)", all_text, re.MULTILINE)
        for act in action_lines:
            act_clean = act.strip()
            act_lower = act_clean.lower()
            if any(w in act_lower for w in ["replaced", "repaired", "installed", "upgraded", "instituted", "flushed", "cleaned", "calibrated"]):
                if len(act_clean) > 20 and not is_duplicate_finding(act_clean, actions):
                    actions.append(f"{act_clean} [CIT-1]")
            if len(actions) >= 4:
                break

        if not actions:
            action_matches = re.findall(r"(?:Mandatory|Corrective|Action)[^:\n]*?[:\-]\s*([^\n\r]+)", all_text, re.IGNORECASE)
            for act in action_matches[:3]:
                act_clean = act.strip()
                if act_clean and len(act_clean) > 15:
                    actions.append(f"{act_clean} [CIT-1]")

        if not actions:
            actions.append("Execute mandated component inspection protocols and restore standard operating tolerances. [CIT-1]")

        # 5. Build summary
        machine_ctx = top_item.metadata.machine_id or "equipment"
        inc_ctx = top_item.metadata.incident_type or "incident"
        summary = (
            f"Investigation of '{query}' grounded in {len(retrieved_items)} retrieved evidence passage(s) "
            f"from {top_item.metadata.source_filename} concerning {machine_ctx} ({inc_ctx})."
        )

        resp = InvestigationResponse(
            executive_summary=summary,
            documented_observations=observations,
            potential_root_causes=root_causes,
            corrective_actions=actions,
            missing_evidence_and_uncertainties=[],
            citations=citations,
            is_sufficient_evidence=True,
            retrieval_metadata=retrieval_metadata,
        )
        return deduplicate_investigation_response(resp)
