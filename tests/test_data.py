"""Tests for the data loader and validator."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from confounder.data.loader import Study, load_study
from confounder.data.validator import validate_study


class TestDataLoader:
    def test_load_study_csv(self, tmp_path: Path):
        csv_path = tmp_path / "test.csv"
        pd.DataFrame({
            "age": [25, 30, 45],
            "treatment": [1, 0, 1],
            "outcome": [10.5, 8.0, 15.2],
            "id": ["A", "B", "C"]
        }).to_csv(csv_path, index=False)

        study = load_study(
            data_path=csv_path,
            treatment="treatment",
            outcome="outcome",
            research_question="Does it work?"
        )
        
        assert study.n_samples == 3
        # ID should be stripped
        assert study.measured_covariates == ["age"]
        assert study.treatment == "treatment"
        assert study.outcome == "outcome"

    def test_load_study_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_study("nonexistent.csv", "t", "o", "q")

    def test_load_study_missing_cols(self, tmp_path: Path):
        csv_path = tmp_path / "test.csv"
        pd.DataFrame({"x": [1]}).to_csv(csv_path, index=False)
        
        with pytest.raises(ValueError, match="Treatment column"):
            load_study(csv_path, "t", "o", "q")

        pd.DataFrame({"t": [1]}).to_csv(csv_path, index=False)
        with pytest.raises(ValueError, match="Outcome column"):
            load_study(csv_path, "t", "o", "q")


class TestDataValidator:
    def test_validate_valid_study(self, scenario_1_study):
        res = validate_study(scenario_1_study, min_samples=100)
        assert res.is_valid is True
        assert not res.errors
        
    def test_validate_insufficient_samples(self, scenario_1_study):
        scenario_1_study.data = scenario_1_study.data.iloc[:50]
        res = validate_study(scenario_1_study, min_samples=100)
        assert res.is_valid is False
        assert any("Insufficient samples" in e for e in res.errors)

    def test_validate_missing_treatment(self, tmp_path: Path):
        df = pd.DataFrame({"t": [1, None, 0], "o": [1, 2, 3]})
        study = Study(df, "t", "o", [], "q")
        res = validate_study(study, min_samples=2)
        assert res.is_valid is False
        assert any("missing values" in e for e in res.errors)

    def test_validate_no_variance(self):
        df = pd.DataFrame({"t": [1, 1, 1], "o": [1, 2, 3]})
        study = Study(df, "t", "o", [], "q")
        res = validate_study(study, min_samples=2)
        assert res.is_valid is False
        assert any("no variation" in e for e in res.errors)
