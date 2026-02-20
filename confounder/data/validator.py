"""Data quality validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from confounder.data.loader import Study

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]


def validate_study(
    study: Study,
    min_samples: int = 100,
) -> ValidationResult:
    """
    Validate that a dataset is suitable for causal inference.
    
    Checks:
    - Sufficient sample size
    - Variation in treatment
    - Variation in outcome
    - No NaN values in treatment/outcome
    """
    errors = []
    warnings = []
    
    df = study.data
    t = study.treatment
    y = study.outcome

    # 1. Sample size
    if len(df) < min_samples:
        errors.append(f"Insufficient samples: {len(df)} < {min_samples} required for reliable causal inference.")
    elif len(df) < min_samples * 2:
        warnings.append(f"Small sample size ({len(df)}). Causal estimates may have wide confidence intervals.")

    # 2. Missing values in core columns
    if df[t].isna().any():
        errors.append(f"Treatment column '{t}' contains missing values.")
    if df[y].isna().any():
        errors.append(f"Outcome column '{y}' contains missing values.")

    # 3. Variance in treatment
    if not df[t].isna().any():
        unique_t = df[t].nunique()
        if unique_t < 2:
            errors.append(f"Treatment column '{t}' has no variation (only {unique_t} unique value).")
        # Is it binary or continuous?
        if unique_t == 2 and not set(df[t].unique()) <= {0, 1, 0.0, 1.0, True, False}:
             warnings.append(f"Treatment '{t}' is binary but not 0/1. Consider recoding for cleaner interpretation.")
             
    # 4. Variance in outcome
    if not df[y].isna().any():
        if df[y].nunique() < 2:
            errors.append(f"Outcome column '{y}' has no variation.")
            
    # 5. Missing values in covariates
    for cov in study.measured_covariates:
        null_pct = df[cov].isna().mean()
        if null_pct > 0.5:
            warnings.append(f"Covariate '{cov}' is missing in {null_pct:.0%} of rows.")
        elif null_pct > 0:
            warnings.append(f"Covariate '{cov}' has missing values. Statistical detection metrics may drop these rows.")

    is_valid = len(errors) == 0
    if not is_valid:
        logger.error("Study validation failed with %d errors", len(errors))
    elif warnings:
        logger.warning("Study validation passed with %d warnings", len(warnings))
    else:
        logger.info("Study validation passed.")

    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)
