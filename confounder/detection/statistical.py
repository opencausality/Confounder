"""Statistical tests to evaluate if a variable is a confounder."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import statsmodels.api as sm
from scipy import stats

logger = logging.getLogger(__name__)


def check_association(
    var_x: pd.Series,
    var_y: pd.Series,
    alpha: float = 0.05,
) -> tuple[float, bool]:
    """
    Test unconditional association between X and Y.
    Returns (p_value, is_significant).
    """
    # Simple correlation test for continuous variables
    # (In a real system, we'd check types to use Chi-square for categorical)
    
    # Drop NaNs
    mask = ~var_x.isna() & ~var_y.isna()
    x_clean = var_x[mask]
    y_clean = var_y[mask]

    if len(x_clean) < 5:
        return 1.0, False

    try:
         r, p_val = stats.pearsonr(x_clean, y_clean)
         return float(p_val), p_val < alpha
    except Exception as e:
         logger.warning("Association test failed: %s", e)
         return 1.0, False


def check_conditional_association(
    target: str,
    predictor: str,
    controls: list[str],
    data: pd.DataFrame,
    alpha: float = 0.05,
    is_binary_target: bool = False,
) -> tuple[float, bool, dict[str, Any]]:
    """
    Test association between predictor and target, controlling for covariates.
    Uses OLS for continuous targets, Logit for binary targets.
    
    Returns (p_value, is_significant, regression_details).
    """
    cols = [target, predictor] + controls
    df = data[cols].dropna()

    if len(df) < len(cols) + 5:
        return 1.0, False, {"error": "Insufficient data after dropping NaNs"}

    Y = df[target]
    
    # Determine model type if not explicitly set
    if is_binary_target or set(Y.unique()) <= {0, 1, 0.0, 1.0, True, False}:
        model_class = sm.Logit
        # Ensure Y is numeric 0/1 for statsmodels
        Y = Y.astype(float)
    else:
        model_class = sm.OLS

    X = df[[predictor] + controls]
    X = sm.add_constant(X)

    try:
        model = model_class(Y, X)
        result = model.fit(disp=0)  # disp=0 suppresses convergence messages
        
        # Get p-value for the predictor
        p_val = result.pvalues[predictor]
        coef = result.params[predictor]
        
        details = {
            "coefficient": float(coef),
            "std_err": float(result.bse[predictor]),
            "model_type": "Logit" if model_class == sm.Logit else "OLS",
            "r_squared": float(getattr(result, "rsquared", getattr(result, "prsquared", 0.0)))
        }
        
        return float(p_val), float(p_val) < alpha, details

    except Exception as e:
        logger.warning("Conditional association test failed for %s ~ %s: %s", target, predictor, e)
        return 1.0, False, {"error": str(e)}


def check_confounding_criteria(
    z_col: str,
    t_col: str,
    y_col: str,
    covariates: list[str],
    data: pd.DataFrame,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Test the two structural criteria for Z being a measured confounder:
    1. Z causes T (Z is associated with T, possibly controlling for covariates)
    2. Z causes Y (Z is associated with Y, controlling for T and covariates)
    """
    # Test 1: Z -> T
    # Is Z associated with treatment (controlling for other covariates)?
    p_t, sig_t, det_t = check_conditional_association(
        target=t_col,
        predictor=z_col,
        controls=covariates,
        data=data,
        alpha=alpha
    )

    # Test 2: Z -> Y
    # Is Z associated with outcome (controlling for treatment and other covariates)?
    p_y, sig_y, det_y = check_conditional_association(
        target=y_col,
        predictor=z_col,
        controls=[t_col] + covariates,
        data=data,
        alpha=alpha
    )

    is_confounder = sig_t and sig_y

    return {
        "is_statistical_confounder": is_confounder,
        "causes_treatment": {
            "p_value": p_t,
            "is_significant": sig_t,
            "details": det_t
        },
        "causes_outcome": {
            "p_value": p_y,
            "is_significant": sig_y,
            "details": det_y
        }
    }
