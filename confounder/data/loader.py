"""Dataset loading and representation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Study:
    """An observational study or experiment dataset."""

    data: pd.DataFrame
    treatment: str
    outcome: str
    measured_covariates: list[str]
    research_question: str
    background_context: str | None = None

    @property
    def n_samples(self) -> int:
        return len(self.data)

    def select(self, columns: list[str]) -> pd.DataFrame:
        """Return subset of dataframe ignoring unmeasured columns."""
        available = [c for c in columns if c in self.data.columns]
        return self.data[available]


def load_study(
    data_path: str | Path,
    treatment: str,
    outcome: str,
    research_question: str,
    context_path: str | Path | None = None,
) -> Study:
    """
    Load a study dataset from CSV.
    
    All columns other than treatment and outcome are assumed to be
    measured covariates unless they are explicitly dropped later.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    # Load data
    logger.info("Loading dataset from %s", path.name)
    df = pd.read_csv(path)
    
    # Validate core columns
    if treatment not in df.columns:
        raise ValueError(f"Treatment column '{treatment}' not found in data")
    if outcome not in df.columns:
        raise ValueError(f"Outcome column '{outcome}' not found in data")

    # Drop any obvious metadata columns that shouldn't be covariates
    exclude = {"id", "index", "timestamp", "date", treatment, outcome}
    covariates = [c for c in df.columns if c.lower() not in exclude]

    logger.info("Loaded dataset: %d rows, %d standard covariates", len(df), len(covariates))

    # Load context if provided
    context_text = None
    if context_path:
        ctx_p = Path(context_path)
        if ctx_p.exists():
            context_text = ctx_p.read_text(encoding="utf-8")
            logger.info("Loaded background context from %s", ctx_p.name)
        else:
            logger.warning("Context file not found: %s", ctx_p)

    return Study(
        data=df,
        treatment=treatment,
        outcome=outcome,
        measured_covariates=covariates,
        research_question=research_question,
        background_context=context_text,
    )
