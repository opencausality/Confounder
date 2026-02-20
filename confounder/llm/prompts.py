"""Prompt templates for confounder candidate generation."""

from __future__ import annotations

SYSTEM_PROMPT = """You are a causal inference expert reviewing an observational study design.
Your task is to identify critical hidden confounders that researchers often miss.
You think strictly in terms of causal mechanisms, not just loose correlations."""

CANDIDATE_GENERATION_PROMPT = """Review the following observational study setup.

Research Question: {research_question}
Treatment Variable: {treatment}
Outcome Variable: {outcome}
Measured Covariates: {covariates}

Background Context:
{context}

Based on domain knowledge and causal theory, propose up to 8 candidate confounding variables that:
1. Plausibly cause BOTH the treatment selection AND the outcome.
2. Are fundamentally distinct from the measured covariates (do not just rename them).
3. Would bias the treatment effect estimate if left completely unmeasured.

You must output your response ONLY as a JSON object with a single key "candidates" containing a list of objects exactly matching this format:

{{
  "candidates": [
    {{
      "name": "snake_case_variable_name",
      "description": "Clear 1-sentence definition of what this variable measures",
      "causes_treatment_because": "Mechanistic reason why this causes X",
      "causes_outcome_because": "Mechanistic reason why this causes Y independent of X",
      "severity": "high"  // "low", "medium", or "high"
    }}
  ]
}}

Output ONLY the JSON, with no markdown formatting or conversational text."""

def format_generation_prompt(
    research_question: str,
    treatment: str,
    outcome: str,
    covariates: list[str],
    context: str | None,
) -> str:
    """Format the generation prompt with study details."""
    covs = ", ".join(covariates) if covariates else "None listed"
    ctx = context or "No background context provided. Rely on general domain knowledge."
    
    return CANDIDATE_GENERATION_PROMPT.format(
        research_question=research_question,
        treatment=treatment,
        outcome=outcome,
        covariates=covs,
        context=ctx,
    )
