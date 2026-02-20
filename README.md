<div align="center">

# 🕵️ Confounder

**Measure what confounds, or know the limits of what you claim.**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/your-username/confounder/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/confounder/actions)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Confounder uses **LLMs + causal statistics** to detect hidden confounders in observational studies *before* they invalidate your results.

[The Problem](#-the-core-problem) · [Quick Start](#-quick-start) · [How It Works](#-how-it-works) · [Example Output](#-example-output) · [LLM vs Causal](#-llm-vs-causal) · [API](#-api)

</div>

---

## 🧠 Philosophy

- 🏠 **Local-first** — Ollama is the default. Your data never leaves your machine.
- 🔬 **Scientifically rigorous** — Every LLM hypothesis is validated with real statistical tests.
- 🧠 **Knowledge-augmented** — LLMs propose confounders that pure statistics can't conceive of.
- ✅ **Actionable** — Concrete study design corrections, not generic warnings.
- 🚫 **No telemetry** — All analysis happens locally. Zero data collection.

---

## 📖 The Core Problem

You're running an observational study. You find:

> *"Treatment X increases outcome Y by 15%"*

But did it? Or is there a hidden confounder Z that causes both X and Y?

```
Naive model:     X ──→ Y           ← "Treatment causes outcome"

Reality:         Z ──→ X
                 Z ──→ Y           ← Z is doing the work.
                 X ··→ Y (weak)       Your 15% is spurious.
```

**Why current approaches fail:**

| Approach | Limitation |
|---|---|
| 📉 **Pure statistics** | Can detect measured confounders, but *cannot* detect unmeasured ones |
| 🧑‍🔬 **Domain experts** | Miss structural variables outside their immediate expertise |
| 🤖 **Pure LLMs** | Hallucinate confounders without statistical grounding |

---

## 🚀 Quick Start

```bash
git clone https://github.com/your-username/confounder.git
cd confounder
pip install -e ".[dev]"
cp .env.example .env    # Configure LLM provider (default: Ollama)
```

Run a full confounder audit:

```bash
confounder check \
  --data study.csv \
  --treatment received_treatment \
  --outcome health_score \
  --question "Does the treatment improve health scores?" \
  --context background.md \
  --graph
```

---

## 🏗️ How It Works

```
Dataset + Research Question + Background Context
                    │
                    ▼
           ┌────────────────┐
           │  LLM Candidate │  ← Expert-prompted LLM proposes 5-10
           │   Generation   │    mechanistic confounder hypotheses
           └────────────────┘
                    │
                    ▼
           ┌────────────────┐
           │   Statistical  │  ← OLS/Logit conditional independence
           │   Validation   │    tests prove or disprove each one
           └────────────────┘
                    │
                    ▼
           ┌────────────────┐
           │     Bias       │  ← Naive vs. adjusted effect estimation
           │  Quantifier    │    calculates exact bias %
           └────────────────┘
                    │
                    ▼
           ┌────────────────┐
           │   Correction   │  ← Control, stratify, sensitivity
           │   Suggester    │    bounds, or study redesign
           └────────────────┘
                    │
                    ▼
              Audit Report
          + Interactive DAG
```

1. **LLM Candidate Brainstorming** — Feeds your research question, column names, and background context to an expert-prompted LLM. It thinks *mechanistically*: which variables strictly cause both treatment and outcome?

2. **Statistical Validation** — Fuzzy-matches the LLM's proposals against your dataset. If a candidate is measured, runs conditional independence tests (OLS/Logit) to mathematically prove or disprove the hypothesis with your actual data.

3. **Exact Bias Quantification** — Estimates the naive treatment effect and compares it to the adjusted estimate, calculating the exact percentage of bias introduced by each confirmed confounder.

4. **Sensitivity Bounds** — If the LLM proposes an *unmeasured* confounder, runs E-value sensitivity analysis to determine how strong that hidden variable would need to be to completely explain away your observed effect.

5. **Interactive DAGs** — Auto-generates `pyvis` network graphs highlighting exactly where the structural breaks are in your causal model.

---

## 📊 Example Output

```console
$ confounder check -d data.csv -t saw_feature -o spend -q "Does the new feature increase spend?"

📂 Loading study data from data.csv...
   1847 rows | Treatment: saw_feature | Outcome: spend

🧠 Querying LLM for candidate confounders...

🔬 Statistically validating 4 proposed confounders...
   ✅ 'user_activity' CONFIRMED as a confounder (p_T=0.003, p_Y<0.001)
   ❌ 'device_type' REJECTED. Not a confounder in this data.
   ❌ 'signup_source' REJECTED. Not a confounder in this data.

🧮 Quantifying bias...

==================================================
=== Confounder Analysis Report ===
==================================================

Research Question: Does the new feature increase spend?

Naive Estimate: +12.3041

CRITICAL CONFOUNDERS DETECTED:

1. USER_ACTIVITY (MEASURED & CONFIRMED)
   Mechanism: Active users navigate more → likelier to see feature.
              Active users naturally spend more.
   Estimated bias: +8.2014 (+66.7% of full effect)
   Evidence: p_treatment=0.0034, p_outcome<0.0001
   Recommendation: Include 'user_activity' as a covariate.

2. PARENTAL_INCOME (UNMEASURED)
   Mechanism: High income → better devices → more exposure.
              High income → higher baseline spend.
   Sensitivity: Would need RR > 2.5 to invalidate result.
   Recommendation: Run sensitivity analysis in final paper.

✓ DAG saved to confounder_dag.html
```

---

## 🔬 LLM vs Causal

This is the core philosophical question Confounder answers: **what happens when you add rigorous causal inference on top of LLM reasoning?**

| | Raw LLM | Confounder (LLM + Causal) |
|---|---|---|
| **Proposes confounders** | ✅ Often 20+ candidates | ✅ Focused 5-10 with mechanistic reasoning |
| **Validates against data** | ❌ Cannot test hypotheses | ✅ OLS/Logit conditional independence |
| **Quantifies bias** | ❌ No numerical output | ✅ Exact % of naive effect explained |
| **Rejects false positives** | ❌ Everything "could be" a confounder | ✅ Statistical significance filtering |
| **Handles unmeasured** | ❌ "You should measure it" | ✅ Sensitivity bounds (E-values) |
| **Actionable output** | ⚠️ Generic paragraphs | ✅ Specific corrections per confounder |
| **Reproducible** | ❌ Different answer each time | ✅ Same data → same statistical result |

An LLM can tell you *"age might be a confounder."* Confounder tells you *"age IS a confounder — it explains 66.7% of your observed effect, and here's how to correct for it."*

**The gap between hypothesis and proof is where bad science lives. Confounder closes that gap.**

---

## 🌐 API

### CLI Commands

```bash
confounder check -d study.csv -t treatment -o outcome -q "Research question?" --graph
confounder providers    # Show available LLM providers
confounder --version
```

### REST API

```bash
uvicorn confounder.api.server:app --reload
```

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check + version |
| `GET` | `/providers` | Available LLM providers |
| `POST` | `/check` | Full confounder audit |

### Python SDK

```python
from confounder.data.loader import load_study
from confounder.detection.validator import validate_candidates
from confounder.estimation.bias import estimate_bias

study = load_study("data.csv", "treatment", "outcome", "Does it work?")
# ... → validate → estimate → report
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full API documentation.

---

## ⚙️ Supported Providers

| Provider | Config | Notes |
|---|---|---|
| `ollama` | Default | Local, private, free |
| `openai` | `OPENAI_API_KEY` | GPT-4o recommended |
| `anthropic` | `ANTHROPIC_API_KEY` | Claude 3.5 Sonnet |
| `groq` | `GROQ_API_KEY` | Fast inference |
| `mistral` | `MISTRAL_API_KEY` | Open-weight models |
| `together` | `TOGETHER_API_KEY` | Llama, Mixtral |

---

## 🧪 Testing

41 tests across 7 modules:

| Module | Coverage |
|---|---|
| `test_data.py` | CSV loading, missing columns, min samples, variance |
| `test_llm.py` | JSON parsing, markdown stripping, schema validation |
| `test_detection.py` | Association tests, conditional independence, fuzzy matching |
| `test_estimation.py` | OLS bias calculation, sensitivity bounds |
| `test_correction.py` | Strategy generation, ranking, report creation |
| `test_cli.py` | Version, providers, invalid data, mocked E2E |
| `test_api.py` | Health check, providers, mocked /check |

```bash
pytest tests/ -v
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

*"Measure what confounds, or know the limits of what you claim."*

</div>
