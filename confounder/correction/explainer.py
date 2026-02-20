"""Generate comprehensive bias reports from analysis results."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from confounder.data.loader import Study
from confounder.detection.validator import ValidatedConfounder
from confounder.correction.suggester import suggest_corrections, CorrectionStrategy
from confounder.estimation.sensitivity import bound_unmeasured_confounder

logger = logging.getLogger(__name__)


@dataclass
class RankedConfounder:
    confounder: ValidatedConfounder
    severity: str    # "Critical", "Moderate", "Minor"
    priority: int    # 1 (highest) to 5


@dataclass
class BiasReport:
    """Full confounder analysis report ready for display."""
    
    study: Study
    naive_estimate: float | None
    adjusted_estimate: float | None
    ranked_confounders: list[RankedConfounder]
    recommendations: dict[str, list[CorrectionStrategy]]
    
    @property
    def has_critical_confounders(self) -> bool:
        return any(rc.severity == "Critical" for rc in self.ranked_confounders)


def rank_confounders(
    validated: list[ValidatedConfounder]
) -> list[RankedConfounder]:
    """Rank confounders by their bias magnitude or theoretical threat level."""
    ranked = []
    
    for v in validated:
        priority = 5
        severity = "Minor"
        
        if v.is_measured and v.is_statistically_significant:
            bias_pct = abs(v.bias_percentage or 0.0)
            if bias_pct > 25.0:
                priority = 1
                severity = "Critical"
            elif bias_pct > 10.0:
                priority = 2
                severity = "Moderate"
            else:
                priority = 4
                severity = "Minor"
        elif not v.is_measured:
            # Unmeasured: rely on LLM severity
            if v.candidate.severity.lower() == "high":
                priority = 1
                severity = "Critical"
            elif v.candidate.severity.lower() == "medium":
                priority = 3
                severity = "Moderate"
            else:
                priority = 5
                severity = "Minor"
        else:
             # Measured but not sig
             continue
             
        ranked.append(RankedConfounder(confounder=v, severity=severity, priority=priority))
        
    # Sort by priority (1 is highest)
    ranked.sort(key=lambda x: x.priority)
    return ranked


def generate_report(
    study: Study,
    validated_confounders: list[ValidatedConfounder],
    naive_estimate: float | None = None,
    final_adjusted_estimate: float | None = None,
) -> BiasReport:
    """Synthesizes all analysis phases into a final report object."""
    ranked = rank_confounders(validated_confounders)
    recs = suggest_corrections([r.confounder for r in ranked])
    
    report = BiasReport(
        study=study,
        naive_estimate=naive_estimate,
        adjusted_estimate=final_adjusted_estimate,
        ranked_confounders=ranked,
        recommendations=recs
    )
    
    logger.info("Generated report with %d relevant confounders", len(ranked))
    return report
