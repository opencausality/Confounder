"""Quantify the bias introduced by a confounder using statsmodels."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
import numpy as np
import statsmodels.api as sm

from confounder.data.loader import Study
from confounder.detection.validator import ValidatedConfounder

logger = logging.getLogger(__name__)


@dataclass
class BiasEstimationResult:
    """Result of bias quantification for a measured confounder."""

    naive_estimate: float
    adjusted_estimate: float
    bias_magnitude: float
    bias_percentage: float
    is_problematic: bool


def _estimate_effect(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    controls: list[str]
) -> float:
    """Helper to estimate treatment effect using OLS."""
    df = data.dropna(subset=[treatment, outcome] + controls)
    if len(df) < len(controls) + 5:
        raise ValueError("Insufficient data for regression after dropping NaNs.")
    
    Y = df[outcome]
    X = df[[treatment] + controls]
    X = sm.add_constant(X)
    
    model = sm.OLS(Y, X)
    result = model.fit(disp=0)
    return float(result.params[treatment])


def estimate_bias(
    confounder: ValidatedConfounder,
    study: Study,
    bias_threshold: float = 0.1,
) -> BiasEstimationResult:
    """
    Quantify how much a measured confounder biases the treatment effect.
    Uses OLS to estimate the naive effect vs the (partially) adjusted effect.
    """
    if not confounder.is_measured or not confounder.matched_column:
        raise ValueError("Cannot quantify precise bias for unmeasured confounders.")

    z_col = confounder.matched_column
    t_col = study.treatment
    y_col = study.outcome

    # 1. Estimate Naive Effect (no controls)
    # Graph: T -> Y
    try:
        estimate_naive = _estimate_effect(study.data, t_col, y_col, [])
    except Exception as e:
        logger.warning("Failed to compute naive effect using OLS: %s", e)
        # Fallback to simple mean difference
        if set(study.data[t_col].unique()) <= {0, 1, 0.0, 1.0}:
             t_0 = study.data[study.data[t_col] == 0][y_col].mean()
             t_1 = study.data[study.data[t_col] == 1][y_col].mean()
             estimate_naive = float(t_1 - t_0)
        else:
            raise ValueError("Naive estimation failed on continuous treatment") from e

    # 2. Estimate Adjusted Effect (controlling for the confounder Z)
    # Graph: Z -> T, Z -> Y, T -> Y
    try:
        estimate_adj = _estimate_effect(study.data, t_col, y_col, [z_col])
    except Exception as e:
        logger.error("Failed to compute adjusted effect controlling for %s: %s", z_col, e)
        return BiasEstimationResult(estimate_naive, estimate_naive, 0.0, 0.0, False)

    # 3. Compute bias
    bias_mag = estimate_naive - estimate_adj
    
    denom = abs(estimate_adj) if abs(estimate_adj) > 1e-10 else abs(estimate_naive)
    
    if denom > 1e-10:
        bias_pct = (bias_mag / denom) * 100.0
    else:
        bias_pct = 0.0

    is_prob = abs(bias_pct) > (bias_threshold * 100)

    # Update candidate directly
    confounder.bias_magnitude = bias_mag
    confounder.bias_percentage = bias_pct

    logger.info(
        "Bias computed for '%s': Naive=%.3f, Adjusted=%.3f, Bias=%.3f (%.1f%%)",
        z_col, estimate_naive, estimate_adj, bias_mag, bias_pct
    )

    return BiasEstimationResult(
        naive_estimate=float(estimate_naive),
        adjusted_estimate=float(estimate_adj),
        bias_magnitude=float(bias_mag),
        bias_percentage=float(bias_pct),
        is_problematic=is_prob,
    )
