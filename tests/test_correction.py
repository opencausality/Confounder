"""Tests for report generation and correction strategies."""

from __future__ import annotations

from confounder.correction.suggester import suggest_corrections
from confounder.correction.explainer import rank_confounders, generate_report
from confounder.detection.validator import ValidatedConfounder
from confounder.llm.parser import ConfounderCandidate


class TestSuggestions:
    
    def test_suggest_corrections_measured(self):
        cand = ConfounderCandidate("age", "desc", "cx", "cy", "high")
        conf = ValidatedConfounder(cand, True, "age", 0.01, 0.01, True, 2.5, 30.0)
        
        recs = suggest_corrections([conf])
        assert "age" in recs
        strats = recs["age"]
        actions = [s.action_type for s in strats]
        assert "control" in actions
        assert "stratify" in actions

    def test_suggest_corrections_unmeasured(self):
        cand = ConfounderCandidate("genetics", "desc", "cx", "cy", "high")
        conf = ValidatedConfounder(cand, False, None)
        
        recs = suggest_corrections([conf])
        assert "genetics" in recs
        strats = recs["genetics"]
        actions = [s.action_type for s in strats]
        assert "sensitivity" in actions
        assert "study_design" in actions

    def test_suggest_corrections_measured_low_bias(self):
        """Low bias should NOT trigger stratification."""
        cand = ConfounderCandidate("height", "desc", "cx", "cy", "low")
        conf = ValidatedConfounder(cand, True, "height", 0.04, 0.03, True, 0.5, 5.0)
        
        recs = suggest_corrections([conf])
        assert "height" in recs
        strats = recs["height"]
        actions = [s.action_type for s in strats]
        assert "control" in actions
        assert "stratify" not in actions


class TestRanking:

    def test_rank_confounders(self):
        c1 = ValidatedConfounder(
            ConfounderCandidate("c1", "", "", "", "high"), True, "c1",
            is_statistically_significant=True, bias_percentage=30.0
        )
        c2 = ValidatedConfounder(
            ConfounderCandidate("c2", "", "", "", "low"), True, "c2",
            is_statistically_significant=True, bias_percentage=5.0
        )
        c3 = ValidatedConfounder(
            ConfounderCandidate("c3", "", "", "", "medium"), False, None
        )
        
        ranked = rank_confounders([c1, c2, c3])
        assert len(ranked) == 3
        # c1 is critical (priority 1)
        assert ranked[0].confounder.candidate.name == "c1"
        assert ranked[0].severity == "Critical"

    def test_generate_report(self, scenario_1_study):
        c1 = ValidatedConfounder(
            ConfounderCandidate("c1", "", "", "", "high"), True, "c1",
            is_statistically_significant=True, bias_percentage=30.0
        )
        
        report = generate_report(scenario_1_study, [c1], 5.0, 3.0)
        
        assert report.study == scenario_1_study
        assert report.naive_estimate == 5.0
        assert report.adjusted_estimate == 3.0
        assert report.has_critical_confounders == True
        assert len(report.recommendations) == 1
