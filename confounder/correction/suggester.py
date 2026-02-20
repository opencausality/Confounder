"""Suggest strategies to correct or mitigate detected confounders."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from confounder.detection.validator import ValidatedConfounder
from confounder.estimation.sensitivity import bound_unmeasured_confounder

logger = logging.getLogger(__name__)


@dataclass
class CorrectionStrategy:
    """A recommended strategy to handle a specific confounder."""
    
    action_type: str  # e.g., "control", "iv", "sensitivity"
    description: str
    code_example: str | None = None


def suggest_corrections(
    confounders: list[ValidatedConfounder],
) -> dict[str, list[CorrectionStrategy]]:
    """Generates actionable recommendations for each confounder."""
    
    recommendations: dict[str, list[CorrectionStrategy]] = {}
    
    for conf in confounders:
        name = conf.candidate.name
        strats = []
        
        if conf.is_measured and conf.is_statistically_significant:
            strats.append(CorrectionStrategy(
                action_type="control",
                description=f"Include '{conf.matched_column}' as a covariate in your regression model or propensity score matching.",
                code_example=f"sm.OLS(Y, sm.add_constant(df[['Treatment', '{conf.matched_column}']]))"
            ))
            
            # If bias is very high, maybe stratification is better
            if conf.bias_percentage and abs(conf.bias_percentage) > 25:
                 strats.append(CorrectionStrategy(
                    action_type="stratify",
                    description=f"Bias from '{conf.matched_column}' is substantial (>25%). Consider analyzing treatment effects entirely separately for different levels/strata of '{conf.matched_column}'.",
                 ))
                 
        elif not conf.is_measured:
            strats.append(CorrectionStrategy(
                action_type="sensitivity",
                description=(
                    f"'{name}' is unmeasured. Run a sensitivity analysis (e.g., Rosenbaum bounds) "
                    "in your final paper to prove your result is robust to it."
                )
            ))
            strats.append(CorrectionStrategy(
                action_type="study_design",
                description=f"Can you find a proxy variable for '{name}' in your existing data? If not, future studies MUST measure it."
            ))
            
        if strats:
             recommendations[name] = strats
             
    return recommendations
