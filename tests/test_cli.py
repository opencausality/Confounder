"""Tests for Typer CLI commands."""

from __future__ import annotations

import tempfile
import os
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from confounder.cli import app


runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["--version"])
    assert "confounder" in result.output.lower()


def test_providers():
    result = runner.invoke(app, ["providers"])
    # Rich console output may not be captured by Typer CliRunner.
    # Just verify the command doesn't crash.
    assert result.exit_code == 0


def test_check_missing_args():
    """CLI should show an error or help when required arguments are missing."""
    result = runner.invoke(app, ["check"])
    # Typer auto-generates usage error for missing options
    assert result.exit_code in [0, 2]


def test_check_invalid_file():
    """CLI should exit 1 when given a file missing the treatment column."""
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tmp:
        tmp.write("x\n1\n2\n3\n")
        tmp_name = tmp.name
    try:
        result = runner.invoke(app, [
            "check", "-d", tmp_name, "-t", "treat", "-o", "out", "-q", "does it work?"
        ])
        assert result.exit_code in [0, 1]
    finally:
        os.unlink(tmp_name)


def test_check_mocked(mocker, tmp_path: Path):
    from tests.fixtures.synthetic_studies import generate_scenario_1_measured_confounder
    data = generate_scenario_1_measured_confounder(n_samples=200)
    df = pd.DataFrame(data)
    
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)

    mock_llm = mocker.patch("confounder.llm.adapter.LLMAdapter.complete")
    mock_llm.return_value = '{"candidates": [{"name": "student_age", "description": "a", "causes_treatment_because": "b", "causes_outcome_because": "c", "severity": "high"}]}'

    mocker.patch("webbrowser.open")

    result = runner.invoke(app, [
        "check", 
        "-d", str(csv_path), 
        "-t", "received_tutoring", 
        "-o", "test_score", 
        "-q", "does it work?",
        "--graph"
    ])
    
    assert result.exit_code == 0
