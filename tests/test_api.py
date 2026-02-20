"""Tests for the API routes and server."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from confounder.api.server import app


client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_get_providers():
    response = client.get("/providers")
    assert response.status_code == 200
    data = response.json()
    assert "providers" in data
    assert "ollama" in data["providers"]
    assert "active_provider" in data


# We mock the LLM adapter so we don't make real network calls during test
def test_check_study_api(mocker):
    # Mock LLM Adapter
    mock_llm = mocker.patch("confounder.llm.adapter.LLMAdapter.complete")
    mock_llm.return_value = '{"candidates": [{"name": "age", "description": "a", "causes_treatment_because": "b", "causes_outcome_because": "c", "severity": "high"}]}'

    payload = {
        "dataset_records": [
            {"treatment": 1, "outcome": 10, "age": 20},
            {"treatment": 0, "outcome": 5, "age": 40},
            {"treatment": 1, "outcome": 12, "age": 22},
            {"treatment": 0, "outcome": 6, "age": 45},
            {"treatment": 1, "outcome": 11, "age": 25},
            {"treatment": 0, "outcome": 7, "age": 38},
        ],
        "treatment": "treatment",
        "outcome": "outcome",
        "research_question": "Does it work?",
        "min_samples": 5 # low threshold so test passes
    }

    response = client.post("/check", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "ranked_confounders" in data
