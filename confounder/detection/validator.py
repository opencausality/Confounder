"""Combine LLM candidates and statistical tests."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from confounder.data.loader import Study
from confounder.detection.statistical import check_confounding_criteria
from confounder.llm.parser import ConfounderCandidate

logger = logging.getLogger(__name__)


@dataclass
class ValidatedConfounder:
    """A statistically or theoretically evaluated confounder."""

    candidate: ConfounderCandidate
    is_measured: bool
    matched_column: str | None
    
    # Statistical evidence (if measured)
    causes_treatment_pval: float | None = None
    causes_outcome_pval: float | None = None
    is_statistically_significant: bool = False
    
    # Bias evidence (populated later by estimation module)
    bias_magnitude: float | None = None
    bias_percentage: float | None = None
    
    @property
    def validation_status(self) -> str:
        if not self.is_measured:
            return "unmeasured_theoretical"
        if self.is_statistically_significant:
            return "measured_confirmed"
        return "measured_rejected"


def match_candidate_to_column(
    candidate: ConfounderCandidate,
    columns: list[str],
) -> str | None:
    """Attempt to find the candidate variable in the dataset columns."""
    c_name = candidate.name.lower()
    
    # Exact match
    if c_name in [col.lower() for col in columns]:
        return next(col for col in columns if col.lower() == c_name)
        
    # Partial match (e.g. "age" matches "user_age")
    for col in columns:
        if c_name in col.lower() or col.lower() in c_name:
            if len(c_name) >= 3:  # avoid matching short strings like "id"
                logger.info("Fuzzy matched LLM candidate '%s' to column '%s'", candidate.name, col)
                return col
                
    return None


def validate_candidates(
    candidates: list[ConfounderCandidate],
    study: Study,
    alpha: float = 0.05,
) -> list[ValidatedConfounder]:
    """
    Take LLM candidates and run statistical tests on them if they exist in the data.
    """
    validated = []
    
    # We can check among covariates, but also check if the LLM hallucinated
    # and proposed the treatment/outcome themselves
    available_cols = study.data.columns.tolist()

    for cand in candidates:
        matched_col = match_candidate_to_column(cand, available_cols)
        
        # If the LLM just regurgitated the treatment or outcome... skip it
        if matched_col in (study.treatment, study.outcome):
            logger.warning("LLM proposed treatment/outcome '%s' as a confounder. Ignoring.", matched_col)
            continue
            
        if not matched_col:
            # Unmeasured confounder
            logger.info("Candidate '%s' is UNMEASURED in this dataset.", cand.name)
            validated.append(ValidatedConfounder(
                candidate=cand,
                is_measured=False,
                matched_column=None,
            ))
            continue

        # It's measured! Run statistical validation
        logger.info("Candidate '%s' MATCHED to column '%s'. Running stats...", cand.name, matched_col)
        
        # We test Z against T and Y, controlling for standard measured covariates (excluding Z itself)
        controls = [c for c in study.measured_covariates if c != matched_col]
        
        stats = check_confounding_criteria(
            z_col=matched_col,
            t_col=study.treatment,
            y_col=study.outcome,
            covariates=controls,
            data=study.data,
            alpha=alpha
        )
        
        is_sig = stats["is_statistical_confounder"]
        p_t = stats["causes_treatment"]["p_value"]
        p_y = stats["causes_outcome"]["p_value"]
        
        if is_sig:
             logger.info("✅ '%s' statistically CONFIRMED as a confounder! (p_T=%.3f, p_Y=%.3f)", matched_col, p_t, p_y)
        else:
             logger.info("❌ '%s' REJECTED by stats. Not a confounder in this data.", matched_col)

        validated.append(ValidatedConfounder(
            candidate=cand,
            is_measured=True,
            matched_column=matched_col,
            causes_treatment_pval=p_t,
            causes_outcome_pval=p_y,
            is_statistically_significant=is_sig,
        ))

    return validated
