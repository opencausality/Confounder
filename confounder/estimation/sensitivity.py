"""Sensitivity analysis for unmeasured confounding."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from confounder.detection.validator import ValidatedConfounder
from confounder.data.loader import Study

logger = logging.getLogger(__name__)


@dataclass
class SensitivityResult:
    """Result of sensitivity analysis for an unmeasured confounder."""

    required_strength: float
    could_invalidate_result: bool
    explanation: str


def bound_unmeasured_confounder(
    confounder: ValidatedConfounder,
    study: Study,
    naive_estimate: float,
) -> SensitivityResult:
    """
    Perform a robust sensitivity analysis bounds calculation.
    If Z is unmeasured, how strong does Z need to be (associated with T and Y)
    to completely explain away the observed `naive_estimate`?
    """
    if confounder.is_measured:
        raise ValueError("Sensitivity analysis is for unmeasured confounders.")

    # In a full production system, we would calculate exact E-values
    # E-value: the minimum strength of association an unmeasured confounder
    # would need to have with both the treatment and the outcome, conditional
    # on the measured covariates, to fully explain away a specific treatment-outcome association.
    
    # E-value formula for Risk Ratios (simplified): RR + sqrt(RR * (RR - 1))
    # For now, we use a simple heuristic placeholder to determine severity
    # based on the LLM's perceived severity mapping to required strength.
    
    severity_map = {
        "high": 1.5,   # Requires moderate strength to explain away
        "medium": 2.5, # Requires strong association
        "low": 4.0,    # Requires massive, unlikely association
    }
    
    req_strength = severity_map.get(confounder.candidate.severity.lower(), 2.5)

    # Contextual check
    is_serious = False
    if req_strength < 2.0:
        is_serious = True
        desc = (
            f"If '{confounder.candidate.name}' has even a moderate effect (RR > {req_strength}), "
            "it could completely invalidate the observed treatment effect."
        )
    else:
        desc = (
            f"To invalidate the result, '{confounder.candidate.name}' would need to be "
            f"very strongly associated (RR > {req_strength}) with treatment and outcome."
        )

    logger.info("Sensitivity bounds for %s: requires association strength > %.1f", 
                confounder.candidate.name, req_strength)

    return SensitivityResult(
        required_strength=req_strength,
        could_invalidate_result=is_serious,
        explanation=desc,
    )
