"""Example: A/B Testing with potential confounding."""

import pandas as pd
import numpy as np
from confounder.data.loader import Study
from confounder.llm.parser import ConfounderCandidate
from confounder.detection.validator import validate_candidates
from confounder.estimation.bias import estimate_bias
from confounder.correction.explainer import generate_report

def run_example():
    # 1. Generate fake A/B test data where "user_activity" confounds the result
    rng = np.random.default_rng(42)
    n = 2000
    
    # Highly active users are more likely to find/use the new feature (treatment)
    user_activity = rng.normal(50, 15, n)
    prob_t = 1 / (1 + np.exp(-(user_activity - 50) / 10))
    treatment = rng.binomial(1, prob_t)
    
    # Highly active users also naturally spend more (outcome)
    # The true treatment effect is $2.00
    outcome = 10.0 + 2.0 * treatment + 0.5 * user_activity + rng.normal(0, 5, n)
    
    df = pd.DataFrame({
        "user_activity": user_activity,
        "device_type": rng.choice(["mobile", "desktop"], n),
        "saw_new_feature": treatment,
        "spend_amount": outcome
    })
    
    study = Study(
        data=df,
        treatment="saw_new_feature",
        outcome="spend_amount",
        measured_covariates=["user_activity", "device_type"],
        research_question="Does the new feature increase user spend?",
        background_context="E-commerce app testing a new recommended products carousel."
    )
    
    print("\n--- Confounder A/B Test Example ---")
    print(f"Naive effect (means): {df[df['saw_new_feature']==1]['spend_amount'].mean() - df[df['saw_new_feature']==0]['spend_amount'].mean():.2f}")
    print(f"True causal effect: 2.00")
    
    # 2. Simulate LLM Candidates (In reality, we'd call the LLMAdapter here)
    candidates = [
        ConfounderCandidate(
            name="user_activity",
            description="Overall engagement level of the user prior to the test.",
            causes_treatment_because="Active users navigate more pages and are likelier to encounter the feature.",
            causes_outcome_because="Active users naturally buy more items.",
            severity="high"
        ),
        ConfounderCandidate(
            name="device_type",
            description="Whether the user is on mobile or desktop.",
            causes_treatment_because="Feature might render differently or be more prominent on desktop.",
            causes_outcome_because="Desktop users tend to have larger basket sizes.",
            severity="medium"
        )
    ]
    
    # 3. Statistical Validation
    validated = validate_candidates(candidates, study)
    
    # 4. Bias Quantification
    naive = None
    for v in validated:
        if v.is_measured and v.is_statistically_significant:
            res = estimate_bias(v, study)
            if naive is None:
                naive = res.naive_estimate
                
    # 5. Reporting
    report = generate_report(study, validated, naive_estimate=naive)
    
    print("\n--- Final Generated Report ---")
    for rc in report.ranked_confounders:
         v = rc.confounder
         if v.is_statistically_significant:
             print(f"\nCONFIRMED: {v.candidate.name}")
             print(f"Bias Magnitude: {v.bias_magnitude:+.2f} ({v.bias_percentage:+.1f}%)")
             
             recs = report.recommendations.get(v.candidate.name, [])
             if recs:
                 print(f"Recommendation: {recs[0].description}")
                 
if __name__ == "__main__":
    run_example()
