"""Query rewriting module powered by Groq with safe fallback and technical entity preservation."""

import json
import os
import re
from typing import List, Optional
from groq import Groq
from incident_rag.config import get_settings
from incident_rag.logging_config import get_logger
from incident_rag.schemas import QueryRewriteResponse

logger = get_logger(__name__)

REWRITE_SYSTEM_PROMPT = """You are an expert industrial maintenance and reliability engineer.
Your task is to rewrite a user query about an industrial equipment incident into alternative, highly specific search queries.

Requirements:
1. Generate between 1 and {num_rewrites} alternative search queries.
2. PRESERVE exact equipment tags, machine IDs, component numbers (e.g. HP-17, RX-404, M-301, B-101, V-204, LT-501, E-301).
3. Expand colloquial descriptions into domain terminology (e.g., "shaking" -> "vibration spectrum spike", "heating" -> "thermal overload winding limit").
4. Keep the queries concise and focused on failure mechanisms, root causes, and sensor telemetry.
5. Return ONLY a valid JSON object matching this schema:
   {{
     "original_query": "<user query>",
     "rewritten_queries": ["<query 1>", "<query 2>", ...]
   }}
"""


class QueryRewriter:
    """Generates technical query variants using Groq LLM with offline fallback."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key if api_key is not None else (settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY", ""))
        self.model = model or settings.GROQ_MODEL
        self.num_rewrites = settings.NUM_QUERY_REWRITES
        self.client: Optional[Groq] = None

        if self.api_key.strip():
            try:
                self.client = Groq(api_key=self.api_key.strip())
                logger.info(f"Initialized Groq client for query rewriting with model: {self.model}")
            except Exception as e:
                logger.warning(f"Could not initialize Groq client: {e}. Fallback rewriting will be used.")
                self.client = None
        else:
            logger.info("No Groq API key configured. Offline fallback query expansion enabled.")

    def rewrite(self, query: str, num_rewrites: Optional[int] = None) -> QueryRewriteResponse:
        """Generates alternative search queries, falling back safely if Groq is unavailable."""
        query = query.strip()
        if not query:
            return QueryRewriteResponse(original_query="", rewritten_queries=[])

        n_rewrites = num_rewrites or self.num_rewrites

        # Attempt Groq API rewrite if client is ready
        if self.client:
            try:
                prompt = REWRITE_SYSTEM_PROMPT.format(num_rewrites=n_rewrites)
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"User Query: {query}"},
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"},
                    max_tokens=400,
                    timeout=10.0,
                )
                raw_content = completion.choices[0].message.content or "{}"
                data = json.loads(raw_content)
                rewritten = data.get("rewritten_queries", [])

                # Clean and deduplicate queries, strictly preserving machine ID if present
                clean_rewrites = []
                # Extract any machine tag from original query
                machine_match = re.search(r"\b([A-Z]{1,4}-[0-9]{2,4}[A-Z]?)\b", query)
                machine_id = machine_match.group(1) if machine_match else ""

                for q in rewritten:
                    q_str = str(q).strip()
                    if q_str and q_str != query:
                        if machine_id and machine_id.lower() not in q_str.lower():
                            q_str = f"{machine_id} {q_str}"
                        if q_str not in clean_rewrites:
                            clean_rewrites.append(q_str)

                if clean_rewrites:
                    logger.info(f"Groq generated {len(clean_rewrites)} rewritten queries for: '{query}'")
                    return QueryRewriteResponse(
                        original_query=query,
                        rewritten_queries=clean_rewrites[:n_rewrites],
                    )
            except Exception as e:
                logger.warning(f"Groq query rewriting request failed: {e}. Engaging rule-based fallback.")

        # Safe offline rule-based fallback
        fallback_queries = self._rule_based_fallback(query, n_rewrites)
        return QueryRewriteResponse(
            original_query=query,
            rewritten_queries=fallback_queries,
        )

    def _rule_based_fallback(self, query: str, limit: int) -> List[str]:
        """Deterministic heuristic query expansion preserving technical codes and synonyms."""
        variants = []

        # Find machine IDs (e.g. HP-17, RX-404, K-102)
        machine_match = re.search(r"\b([A-Z]{1,4}-[0-9]{2,4}[A-Z]?)\b", query)
        machine_id = machine_match.group(1) if machine_match else ""

        # Synonym replacements
        synonyms = [
            ("vibration", "radial shaft vibration amplitude bearing rub"),
            ("bearing", "tilting pad journal bearing lubrication hydrodynamic"),
            ("overheating", "thermal overload winding temperature ETD"),
            ("runaway", "exothermic excursion runaway reaction burst disk"),
            ("cavitation", "hydraulic cavitation impeller vane erosion NPSHa"),
            ("surge", "aerodynamic compressor surge anti-surge bypass"),
            ("explosion", "arc flash transformer bushing dielectric breakdown"),
        ]

        for trigger, expansion in synonyms:
            if trigger in query.lower():
                if machine_id:
                    v = f"{machine_id} {expansion}"
                else:
                    v = f"{query} {expansion}"
                if v not in variants and v != query:
                    variants.append(v)
                if len(variants) >= limit:
                    break

        if not variants:
            # Default variant appending domain terms
            if machine_id and machine_id.lower() not in query.lower():
                variants.append(f"{machine_id} {query} root cause failure analysis corrective action")
            else:
                variants.append(f"{query} root cause failure analysis corrective action")

        return variants[:limit]
