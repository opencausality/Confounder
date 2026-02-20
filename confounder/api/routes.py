"""FastAPI routes for Confounder."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from confounder.config import LLMProvider, get_settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Confounder API"])


class ProvidersResponse(BaseModel):
    """Response containing available LLM providers."""
    providers: list[str]
    active_provider: str
    active_model: str


@router.get("/providers", response_model=ProvidersResponse)
def get_providers() -> ProvidersResponse:
    """List all supported LLM providers and current configuration."""
    settings = get_settings()
    return ProvidersResponse(
        providers=[p.value for p in LLMProvider],
        active_provider=settings.llm_provider.value,
        active_model=settings.resolved_model,
    )


# Note: In a real production system, the `/check` endpoint would
# accept file uploads using FastAPI's UploadFile, or a JSON payload
# representing a study, and return the BiasReport object serialized.
# For this WhyNet-parity version, we stub the endpoint architecture.

class CheckRequest(BaseModel):
    """Request payload to check a dataset (JSON format)."""
    dataset_records: list[dict[str, Any]]
    treatment: str
    outcome: str
    research_question: str
    context: str | None = None
    min_samples: int = 100
    
    model_config = ConfigDict(
         json_schema_extra={
             "example": {
                 "dataset_records": [
                     {"treatment": 1, "outcome": 10.5, "age": 25},
                     {"treatment": 0, "outcome": 8.0, "age": 40}
                 ],
                 "treatment": "treatment",
                 "outcome": "outcome",
                 "research_question": "Does treatment improve outcome?"
             }
         }
    )

@router.post("/check")
def check_study(request: CheckRequest) -> dict[str, Any]:
    """
    Run full Confounder analysis on a dataset provided via JSON.
    """
    import pandas as pd
    from confounder.data.loader import Study
    from confounder.data.validator import validate_study
    from confounder.llm.adapter import LLMAdapter
    from confounder.llm.prompts import SYSTEM_PROMPT, format_generation_prompt
    from confounder.llm.parser import parse_candidates
    from confounder.detection.validator import validate_candidates
    from confounder.estimation.bias import estimate_bias
    from confounder.correction.explainer import generate_report

    if not request.dataset_records:
        raise HTTPException(status_code=400, detail="dataset_records cannot be empty")
        
    df = pd.DataFrame(request.dataset_records)
    
    # Very basic Study assembly mimicking the loader
    exclude = {"id", "index", "timestamp", "date", request.treatment, request.outcome}
    covariates = [c for c in df.columns if c.lower() not in exclude]
    
    study = Study(
        data=df,
        treatment=request.treatment,
        outcome=request.outcome,
        measured_covariates=covariates,
        research_question=request.research_question,
        background_context=request.context
    )
    
    val_res = validate_study(study, min_samples=request.min_samples)
    if not val_res.is_valid:
        raise HTTPException(status_code=400, detail={"errors": val_res.errors, "warnings": val_res.warnings})

    llm = LLMAdapter()
    prompt = format_generation_prompt(
        research_question=study.research_question,
        treatment=study.treatment,
        outcome=study.outcome,
        covariates=study.measured_covariates,
        context=study.background_context
    )

    try:
         # Note: For an API, this could take standard timeout/retry params or async
         response = llm.complete(prompt, system=SYSTEM_PROMPT, format_json=True)
         candidates = parse_candidates(response)
    except Exception as e:
         raise HTTPException(status_code=502, detail=f"LLM failure: {e}")

    validated = validate_candidates(candidates, study)
    
    naive_est = None
    for v in validated:
        if v.is_measured and v.is_statistically_significant:
            res = estimate_bias(v, study)
            if naive_est is None:
                naive_est = res.naive_estimate

    report = generate_report(study, validated, naive_estimate=naive_est)

    # Simplified serialization
    ranked_out = []
    for rc in report.ranked_confounders:
         v = rc.confounder
         c = v.candidate
         ranked_out.append({
             "name": c.name,
             "is_measured": v.is_measured,
             "is_significant": v.is_statistically_significant,
             "severity": rc.severity,
             "bias_magnitude": v.bias_magnitude,
             "bias_percentage": v.bias_percentage,
             "causes_treatment_because": c.causes_treatment_because,
             "causes_outcome_because": c.causes_outcome_because,
             "recommendations": [r.description for r in report.recommendations.get(c.name, [])]
         })

    return {
        "status": "success",
        "research_question": report.study.research_question,
        "naive_estimate": report.naive_estimate,
        "critical_confounders_found": report.has_critical_confounders,
        "ranked_confounders": ranked_out
    }
