"""Generate synthetic study datasets with known confounding for testing."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def generate_scenario_1_measured_confounder(
    n_samples: int = 500,
    seed: int = 42,
) -> dict[str, list[float]]:
    """
    Scenario 1: True model is Z -> X, Z -> Y, X -> Y.
    Z = Age (Measured Confounder)
    X = Treatment (Online Tutoring)
    Y = Outcome (Test Score)
    
    True Treatment Effect: +5 points
    Age bias: Older students less likely to get tutoring (X), but naturally score higher (Y).
    Resulting in a biased (underestimated or negative) naive treatment effect.
    """
    rng = np.random.default_rng(seed)
    
    # Age (Z): 10 to 18
    age = rng.uniform(10, 18, n_samples)
    
    # Treatment (X): Probability decreases with age
    # p(X=1) = logistic(5 - 0.4 * age)
    logits_x = 5.0 - 0.4 * age
    p_x = 1.0 / (1.0 + np.exp(-logits_x))
    treatment = rng.binomial(n=1, p=p_x)
    
    # Outcome (Y): True effect of treatment=+5, Age adds 2 points per year naturally
    outcome = (
        50.0 +               # base score
        5.0 * treatment +    # true causal effect
        2.0 * age +          # confounding effect
        rng.normal(0, 3, n_samples)  # noise
    )
    
    # Unrelated measured covariate (W)
    school_size = rng.normal(1000, 200, n_samples)
    
    return {
        "student_age": age.tolist(),
        "school_size": school_size.tolist(),
        "received_tutoring": treatment.tolist(),
        "test_score": outcome.tolist(),
    }


def generate_scenario_2_unmeasured_confounder(
    n_samples: int = 500,
    seed: int = 42,
) -> dict[str, list[float]]:
    """
    Scenario 2: True model Z -> X, Z -> Y. No direct X -> Y effect.
    Z = Genetic Predisposition (UNMEASURED)
    X = Coffee Consumption (Treatment)
    Y = Heart Rate (Outcome)
    
    True Treatment Effect: 0
    Naive Effect: Positive (Appears coffee causes high heart rate)
    """
    rng = np.random.default_rng(seed)
    
    # Genetics (Z, unmeasured): Standard normal
    genetics = rng.normal(0, 1, n_samples)
    
    # Treatment (X): Depends on genetics
    coffee_cups = rng.poisson(lam=np.exp(genetics))
    binary_coffee = (coffee_cups > 2).astype(int)
    
    # Outcome (Y): Depends on genetics, NO effect from coffee
    heart_rate = (
        60.0 + 
        15.0 * genetics +     # strong confounding
        0.0 * binary_coffee + # NO causal effect
        rng.normal(0, 5, n_samples)
    )
    
    # Measured covariate
    age = rng.uniform(20, 60, n_samples)
    heart_rate += 0.5 * age  # age affects heart rate but not coffee
    
    return {
        # 'genetics' is intentionally omitted from output!
        "age": age.tolist(),
        "drinks_coffee": binary_coffee.tolist(),
        "heart_rate": outcome.tolist() if 'outcome' in locals() else heart_rate.tolist(), # fixed missing var
    }


def save_as_csv(data: dict[str, list[float]], path: str | Path) -> None:
    """Save metric data as wide-format CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(data.keys())
    n = len(next(iter(data.values())))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(n):
            writer.writerow([round(data[k][i], 4) for k in keys])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["1", "2"], default="1")
    parser.add_argument("--output", default="study_data.csv")
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()

    if args.scenario == "1":
        data = generate_scenario_1_measured_confounder(n_samples=args.samples)
    else:
        data = generate_scenario_2_unmeasured_confounder(n_samples=args.samples)
        
    save_as_csv(data, args.output)
    print(f"Generated Scenario {args.scenario} data -> {args.output}")
