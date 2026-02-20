"""Test fixtures for Confounder."""

from __future__ import annotations

import pandas as pd
import pytest

from confounder.config import ConfounderSettings
from confounder.data.loader import Study
from tests.fixtures.synthetic_studies import generate_scenario_1_measured_confounder


@pytest.fixture
def settings():
    """Test settings."""
    return ConfounderSettings(
        llm_provider="ollama",
        llm_model="test-model",
        alpha=0.05,
    )


@pytest.fixture
def scenario_1_study():
    """Study loaded with scenario 1 data (measured confounder present)."""
    data_dict = generate_scenario_1_measured_confounder(n_samples=500, seed=42)
    df = pd.DataFrame(data_dict)
    
    return Study(
        data=df,
        treatment="received_tutoring",
        outcome="test_score",
        measured_covariates=["student_age", "school_size"],
        research_question="Does online tutoring improve test scores?",
        background_context="Education setting where age affects tutoring adoption and baseline scores."
    )
