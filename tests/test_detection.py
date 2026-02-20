"""Tests for statistical detection module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from confounder.detection.statistical import check_association, check_conditional_association, check_confounding_criteria
from confounder.detection.validator import match_candidate_to_column, validate_candidates
from confounder.llm.parser import ConfounderCandidate


class TestStatisticalFoundations:
    """Test core stat methods."""

    def test_association_correlated(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        y = x * 0.8 + rng.normal(0, 0.2, 100)
        p1, sig1 = check_association(pd.Series(x), pd.Series(y))
        assert sig1 == True
        assert p1 < 0.05

    def test_association_uncorrelated(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        z = rng.normal(0, 1, 100)
        p2, sig2 = check_association(pd.Series(x), pd.Series(z))
        assert sig2 == False
        assert p2 > 0.05

    def test_conditional_association_unconditional(self):
        rng = np.random.default_rng(42)
        n = 500
        z = rng.normal(0, 1, n)
        x = z * 0.8 + rng.normal(0, 0.2, n)
        y = x * 0.8 + rng.normal(0, 0.2, n)
        
        df = pd.DataFrame({"X": x, "Y": y, "Z": z})
        p1, sig1, _ = check_conditional_association("Y", "Z", [], df)
        assert sig1 == True

    def test_conditional_association_conditional(self):
        rng = np.random.default_rng(42)
        n = 500
        z = rng.normal(0, 1, n)
        x = z * 0.8 + rng.normal(0, 0.2, n)
        y = x * 0.8 + rng.normal(0, 0.2, n)
        
        df = pd.DataFrame({"X": x, "Y": y, "Z": z})
        p2, sig2, _ = check_conditional_association("Y", "Z", ["X"], df)
        assert sig2 == False
        assert p2 > 0.05

    def test_confounding_criteria_real(self, scenario_1_study):
        """Test full confounding criteria on known confounder."""
        result = check_confounding_criteria(
            z_col="student_age",
            t_col="received_tutoring",
            y_col="test_score",
            covariates=[],
            data=scenario_1_study.data,
            alpha=0.05,
        )
        assert result["is_statistical_confounder"] == True
        assert result["causes_treatment"]["is_significant"] == True
        assert result["causes_outcome"]["is_significant"] == True

    def test_confounding_criteria_noise(self, scenario_1_study):
        """Test that random noise column is NOT identified as a confounder."""
        result = check_confounding_criteria(
            z_col="school_size",
            t_col="received_tutoring",
            y_col="test_score",
            covariates=[],
            data=scenario_1_study.data,
            alpha=0.05,
        )
        assert result["is_statistical_confounder"] == False


class TestValidator:
    """Test mapping and validating LLM candidates."""

    def test_match_exact(self):
        cols = ["student_age", "income", "received_treatment"]
        cand = ConfounderCandidate("income", "", "", "", "high")
        assert match_candidate_to_column(cand, cols) == "income"

    def test_match_fuzzy(self):
        cols = ["student_age", "income", "received_treatment"]
        cand = ConfounderCandidate("age", "", "", "", "high")
        assert match_candidate_to_column(cand, cols) == "student_age"
        
    def test_match_none(self):
        cols = ["student_age", "income", "received_treatment"]
        cand = ConfounderCandidate("genetics", "", "", "", "high")
        assert match_candidate_to_column(cand, cols) is None
        
    def test_match_short_string_blocked(self):
        cand = ConfounderCandidate("id", "", "", "", "high")
        assert match_candidate_to_column(cand, ["video_id", "user_id"]) is None

    def test_validate_candidates_scenario_1(self, scenario_1_study):
        """Test if validation catches the true confounder and rejects noise."""
        cand_real = ConfounderCandidate("student_age", "", "", "", "high")
        cand_fake = ConfounderCandidate("school_size", "", "", "", "low")
        cand_unmeasured = ConfounderCandidate("motivation", "", "", "", "medium")
        
        validated = validate_candidates([cand_real, cand_fake, cand_unmeasured], scenario_1_study)
        
        assert len(validated) == 3
        
        val_real = next(v for v in validated if v.candidate.name == "student_age")
        assert val_real.is_measured == True
        assert val_real.is_statistically_significant == True
        
        val_fake = next(v for v in validated if v.candidate.name == "school_size")
        assert val_fake.is_measured == True
        assert val_fake.is_statistically_significant == False
        
        val_unm = next(v for v in validated if v.candidate.name == "motivation")
        assert val_unm.is_measured == False
        assert val_unm.is_statistically_significant == False
