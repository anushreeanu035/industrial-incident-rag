"""Evidence-grounded response generation using Groq LLM with extractive fallback."""

import json
import os
import re
from typing import Any, Dict, List, Optional
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
1. Ground every finding strictly in the provided retrieved evidence items.
2. DO NOT hallucinate, invent, or speculate on machine tags, measurements, incident dates, or root causes.
3. Every factual statement or observation must explicitly reference its evidence citation tag (e.g., [CIT-1], [CIT-2]).
4. Distinguish clearly between documented physical facts and investigative hypotheses.
5. If the retrieved evidence is insufficient to answer the query with high confidence, set "is_sufficient_evidence" to false and state what evidence is missing.
6. Never recommend bypassing safety interlocks or violating OSHA/ISO/IEC safety procedures.
7. Return ONLY a valid JSON object matching the following structure:
{
  "executive_summary": "Concise summary of incident, machine affected, and key findings.",
  "documented_observations": ["Observation 1 [CIT-1]", "Observation 2 [CIT-2]"],
  "potential_root_causes": [
    {
      "hypothesis": "Root cause explanation",
      "confidence_level": "High (Documented Fact) | Medium (Strong Evidence) | Low (Tentative Hypothesis)",
      "supporting_evidence": ["Evidence point 1 [CIT-1]"],
      "citation_ids": ["[CIT-1]"]
    }
  ],
  "corrective_actions": ["Mandated action 1 [CIT-1]", "Action 2 [CIT-2]"],
  "missing_evidence_and_uncertainties": ["Missing information point 1"],
  "is_sufficient_evidence": true
}
"""


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
        """Constructs context, invokes Groq LLM or deterministic fallback, and returns structured InvestigationResponse."""
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

                return InvestigationResponse(
                    executive_summary=data.get("executive_summary", "Incident analysis completed."),
                    documented_observations=data.get("documented_observations", []),
                    potential_root_causes=hypotheses,
                    corrective_actions=data.get("corrective_actions", []),
                    missing_evidence_and_uncertainties=data.get("missing_evidence_and_uncertainties", []),
                    citations=citations,
                    is_sufficient_evidence=data.get("is_sufficient_evidence", True),
                    retrieval_metadata=retrieval_metadata,
                )
            except Exception as e:
                logger.warning(f"Groq answer generation failed: {e}. Executing extractive fallback.")

        # Fallback deterministic extractive synthesis
        return self._extractive_fallback(query, retrieved_items, citations, retrieval_metadata)

    def _extractive_fallback(
        self,
        query: str,
        retrieved_items: List[RetrievalResultItem],
        citations: List[EvidenceCitation],
        retrieval_metadata: Dict[str, Any],
    ) -> InvestigationResponse:
        """Deterministic extractive summarizer for offline operation and automated testing."""
        top_item = retrieved_items[0]
        all_text = " ".join(item.text for item in retrieved_items)

        # Extract observations
        observations = []
        # Find lines with numbers, metrics, or telemetry
        metric_matches = re.findall(
            r"([A-Z0-9\-]+(?:[^.\n]*?(?:\d+(?:\.\d+)?\s*(?:mm/s|bar|°C|deg C|ppm|A|kV|t/h|dB|m))[^.\n]*?)\.)",
            all_text,
            re.IGNORECASE,
        )
        for m in metric_matches[:4]:
            m_clean = m.strip()
            if m_clean and len(m_clean) > 20:
                observations.append(f"{m_clean} [CIT-1]")

        if not observations:
            observations.append(f"Primary evidence extracted from {top_item.metadata.source_filename}: {top_item.text[:150]}... [CIT-1]")

        # Extract root cause
        root_causes: List[InvestigationHypothesis] = []
        rc_match = re.search(r"(?:Root\s*Cause|Direct\s*Cause)[^:\n]*?[:\-]\s*([^\n\r]+)", all_text, re.IGNORECASE)
        if rc_match:
            cause_text = rc_match.group(1).strip()
            root_causes.append(
                InvestigationHypothesis(
                    hypothesis=cause_text,
                    confidence_level="High (Documented Fact)",
                    supporting_evidence=[f"Documented in {top_item.metadata.source_filename} [CIT-1]"],
                    citation_ids=["[CIT-1]"],
                )
            )
        else:
            root_causes.append(
                InvestigationHypothesis(
                    hypothesis=f"Equipment failure associated with {top_item.metadata.machine_id or 'incident'}",
                    confidence_level="Medium (Strong Evidence)",
                    supporting_evidence=[f"Supported by passage {top_item.chunk_id} [CIT-1]"],
                    citation_ids=["[CIT-1]"],
                )
            )

        # Extract actions
        actions = []
        action_matches = re.findall(r"(?:Mandatory|Corrective|Action)[^:\n]*?[:\-]\s*([^\n\r]+)", all_text, re.IGNORECASE)
        for act in action_matches[:3]:
            act_clean = act.strip()
            if act_clean and len(act_clean) > 15:
                actions.append(f"{act_clean} [CIT-1]")
        if not actions:
            actions.append("Follow standard operating procedure maintenance reviews and component inspection protocols. [CIT-1]")

        summary = (
            f"Investigation of '{query}' grounded in {len(retrieved_items)} retrieved evidence passage(s) "
            f"from {top_item.metadata.source_filename}."
        )

        return InvestigationResponse(
            executive_summary=summary,
            documented_observations=observations,
            potential_root_causes=root_causes,
            corrective_actions=actions,
            missing_evidence_and_uncertainties=[],
            citations=citations,
            is_sufficient_evidence=True,
            retrieval_metadata=retrieval_metadata,
        )
