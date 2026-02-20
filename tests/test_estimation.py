"""Tests for bias estimation and sensitivity bounds."""

from __future__ import annotations

import pytest

from confounder.detection.validator import ValidatedConfounder
from confounder.estimation.bias import estimate_bias
from confounder.estimation.sensitivity import bound_unmeasured_confounder
from confounder.llm.parser import ConfounderCandidate


class TestBiasEstimation:

    def test_estimate_bias_measured(self, scenario_1_study):
        """Test computing the bias introduced by the known confounder."""
        cand = ConfounderCandidate("student_age", "", "", "", "high")
        conf = ValidatedConfounder(
            candidate=cand,
            is_measured=True,
            matched_column="student_age",
            is_statistically_significant=True,
        )
        
        res = estimate_bias(conf, scenario_1_study, bias_threshold=0.05)
        
        # Naive and adjusted should differ because age confounds the relationship
        assert res.naive_estimate != res.adjusted_estimate
        assert abs(res.bias_percentage) > 0.0
        # The true effect is 5. Adjusted should be closer to 5 than naive.
        assert abs(res.adjusted_estimate - 5.0) < abs(res.naive_estimate - 5.0)
        assert res.is_problematic == True

    def test_estimate_bias_unmeasured_fails(self, scenario_1_study):
        cand = ConfounderCandidate("genetics", "", "", "", "high")
        conf = ValidatedConfounder(cand, False, None)
        
        with pytest.raises(ValueError, match="Cannot quantify"):
            estimate_bias(conf, scenario_1_study)


class TestSensitivity:

    def test_bound_high_severity(self, scenario_1_study):
        cand = ConfounderCandidate("genetics", "", "", "", "high")
        conf = ValidatedConfounder(cand, False, None)
        
        res = bound_unmeasured_confounder(conf, scenario_1_study, 2.5)
        assert res.required_strength == 1.5
        assert res.could_invalidate_result == True

    def test_bound_low_severity(self, scenario_1_study):
        cand = ConfounderCandidate("weather", "", "", "", "low")
        conf = ValidatedConfounder(cand, False, None)
        
        res = bound_unmeasured_confounder(conf, scenario_1_study, 2.5)
        assert res.required_strength == 4.0
        assert res.could_invalidate_result == False

    def test_bound_medium_severity(self, scenario_1_study):
        cand = ConfounderCandidate("diet", "", "", "", "medium")
        conf = ValidatedConfounder(cand, False, None)
        
        res = bound_unmeasured_confounder(conf, scenario_1_study, 2.5)
        assert res.required_strength == 2.5
        assert res.could_invalidate_result == False
