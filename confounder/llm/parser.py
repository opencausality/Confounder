"""Parse LLM JSON output into candidate objects."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConfounderCandidate:
    """A proposed confounding variable from the LLM."""

    name: str
    description: str
    causes_treatment_because: str
    causes_outcome_because: str
    severity: str  # "high", "medium", "low"
    
    @property
    def is_plausible(self) -> bool:
        """Basic validation."""
        return bool(self.name and self.causes_treatment_because and self.causes_outcome_because)

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "causes_treatment_because": self.causes_treatment_because,
            "causes_outcome_because": self.causes_outcome_because,
            "severity": self.severity,
        }


def parse_candidates(llm_response: str) -> list[ConfounderCandidate]:
    """
    Parse the JSON response from the LLM into ConfounderCandidate objects.
    Handles potential markdown code blocks wrapping the JSON.
    """
    # 1. Clean markdown formatting if present
    cleaned = llm_response.strip()
    if cleaned.startswith("```"):
        # Match ```json or just ```
        match = re.search(r"```(?:json)?\s+([\s\S]*?)\s+```", cleaned)
        if match:
            cleaned = match.group(1)
        else:
             # Fallback: strip leading/trailing ticks
             cleaned = cleaned.strip("`").strip()

    # 2. Parse JSON
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse LLM response as JSON: %s\nResponse: %s", e, cleaned[:200])
        raise ValueError("LLM returned malformed JSON.") from e

    # 3. Extract standard format
    candidates_data = data.get("candidates", [])
    if not isinstance(candidates_data, list):
         # Try direct array
         if isinstance(data, list):
             candidates_data = data
         else:
             raise ValueError("JSON does not contain a 'candidates' array.")

    # 4. Convert to objects
    results = []
    for item in candidates_data:
        try:
            candidate = ConfounderCandidate(
                name=str(item.get("name", "")).strip().lower().replace(" ", "_"),
                description=str(item.get("description", "")).strip(),
                causes_treatment_because=str(item.get("causes_treatment_because", "")).strip(),
                causes_outcome_because=str(item.get("causes_outcome_because", "")).strip(),
                severity=str(item.get("severity", "medium")).strip().lower(),
            )
            if candidate.is_plausible:
                results.append(candidate)
            else:
                logger.warning("Skipping implausible candidate (missing fields): %s", item.get("name"))
        except Exception as e:
            logger.warning("Failed to parse candidate item: %s. Error: %s", item, e)

    logger.info("Parsed %d valid confounder candidates from LLM response", len(results))
    return results
