"""Tests for LLM candidates and formatters."""

from __future__ import annotations

import pytest

from confounder.llm.parser import ConfounderCandidate, parse_candidates
from confounder.llm.prompts import format_generation_prompt


class TestLLMParser:
    def test_parse_valid_json(self):
        json_str = """
        {
          "candidates": [
            {
              "name": "age",
              "description": "User age",
              "causes_treatment_because": "X",
              "causes_outcome_because": "Y",
              "severity": "high"
            }
          ]
        }
        """
        cands = parse_candidates(json_str)
        assert len(cands) == 1
        assert cands[0].name == "age"
        assert cands[0].severity == "high"

    def test_parse_markdown_json(self):
        json_str = """```json
        {
          "candidates": [
            {
              "name": "Income Bracket",
              "description": "User income",
              "causes_treatment_because": "X",
              "causes_outcome_because": "Y",
              "severity": "medium"
            }
          ]
        }
        ```"""
        cands = parse_candidates(json_str)
        assert len(cands) == 1
        # Tests that space normalization works
        assert cands[0].name == "income_bracket"

    def test_parse_malformed_json_raises(self):
        with pytest.raises(ValueError, match="malformed JSON"):
            parse_candidates("{ not json }")

    def test_invalid_schema_skipped(self):
        # Missing required causal explanation fields
        json_str = """
        {
          "candidates": [
            {
              "name": "age",
              "description": "User age"
            }
          ]
        }
        """
        cands = parse_candidates(json_str)
        assert len(cands) == 0


def test_format_generation_prompt():
    prompt = format_generation_prompt(
        "Q", "T", "O", ["C1"], "Ctx"
    )
    assert "Q" in prompt
    assert "T" in prompt
    assert "C1" in prompt
    assert "Ctx" in prompt
