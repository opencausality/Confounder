"""CLI interface — Typer commands with Rich output."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from confounder import __version__
from confounder.config import configure_logging

app = typer.Typer(
    name="confounder",
    help="🕵️  Confounder — Detect hidden confounders in observational studies.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"confounder {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", "-V", callback=version_callback, is_eager=True),
) -> None:
    """Confounder — Measure what confounds, or know the limits of what you claim."""


@app.command()
def check(
    data: str = typer.Option(..., "--data", "-d", help="Path to study CSV"),
    treatment: str = typer.Option(..., "--treatment", "-t", help="Treatment column name"),
    outcome: str = typer.Option(..., "--outcome", "-o", help="Outcome column name"),
    research_question: str = typer.Option(..., "--question", "-q", help="E.g., Does tutoring improve test scores?"),
    context: Optional[str] = typer.Option(None, "--context", "-c", help="Path to background domain doc (txt/md)"),
    min_samples: int = typer.Option(100, help="Minimum required sample size"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    show_graph: bool = typer.Option(False, "--graph", "-g", help="Open DAG visualization in browser")
) -> None:
    """Run full confounder audit: LLM Generation → Stats Validation → Bias Report."""
    configure_logging("DEBUG" if verbose else "INFO")

    # 1. Load Data
    from confounder.data.loader import load_study
    from confounder.data.validator import validate_study

    console.print(f"\n📂 Loading study data from [bold]{data}[/bold]...")
    try:
         study = load_study(data, treatment, outcome, research_question, context_path=context)
    except Exception as e:
         console.print(f"[red]Failed to load study: {e}[/red]")
         raise typer.Exit(1)

    console.print(f"   [dim]{study.n_samples} rows | Treatment: {treatment} | Outcome: {outcome}[/dim]")

    # 2. Validate Data
    val_res = validate_study(study, min_samples=min_samples)
    if not val_res.is_valid:
        for err in val_res.errors:
            console.print(f"❌ [red]{err}[/red]")
        raise typer.Exit(1)
    for warn in val_res.warnings:
         console.print(f"⚠️  [yellow]{warn}[/yellow]")

    # 3. LLM Candidate Generation
    from confounder.llm.adapter import LLMAdapter
    from confounder.llm.prompts import SYSTEM_PROMPT, format_generation_prompt
    from confounder.llm.parser import parse_candidates

    console.print("\n🧠 Querying LLM for candidate confounders based on domain knowledge...")
    llm = LLMAdapter()
    prompt = format_generation_prompt(
        research_question=study.research_question,
        treatment=study.treatment,
        outcome=study.outcome,
        covariates=study.measured_covariates,
        context=study.background_context
    )

    try:
         response = llm.complete(prompt, system=SYSTEM_PROMPT, format_json=True)
         candidates = parse_candidates(response)
    except Exception as e:
         console.print(f"[red]LLM candidate generation failed: {e}[/red]")
         raise typer.Exit(1)
         
    if not candidates:
         console.print("ℹ️  [cyan]LLM found no obvious mechanistic confounders.[/cyan]")
         raise typer.Exit(0)

    # 4. Statistical Validation
    from confounder.detection.validator import validate_candidates
    
    console.print(f"\n🔬 Statistically validating {len(candidates)} proposed confounders...")
    validated = validate_candidates(candidates, study)

    # 5. Bias Quantification
    from confounder.estimation.bias import estimate_bias
    from confounder.estimation.sensitivity import bound_unmeasured_confounder
    
    console.print("🧮 Quantifying bias introduced by confirmed confounders...")
    
    naive_est = None
    for v in validated:
        if v.is_measured and v.is_statistically_significant:
            res = estimate_bias(v, study)
            if naive_est is None:
                naive_est = res.naive_estimate
        elif not v.is_measured:
             # Just bound it generally (stub if naive doesn't exist)
             pass

    # 6. Report Generation
    from confounder.correction.explainer import generate_report
    
    report = generate_report(study, validated, naive_estimate=naive_est)
    
    _render_report(report)
    
    if show_graph:
        from confounder.graph.visualizer import render_confounder_dag
        path = render_confounder_dag(study, report.ranked_confounders)
        console.print(f"✓ DAG saved to [bold]{path}[/bold]")
        try:
             import webbrowser
             webbrowser.open(str(path))
        except Exception:
             pass


@app.command()
def providers() -> None:
    """Show available LLM providers and current config."""
    from confounder.config import get_settings, LLMProvider

    settings = get_settings()
    table = Table(title="LLM Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Model")

    for provider in LLMProvider:
        is_current = provider == settings.llm_provider
        status = "✓ active" if is_current else ""
        model = settings.resolved_model if is_current else ""
        table.add_row(provider.value, status, model)

    console.print(table)


def _render_report(report) -> None:
    """Render the BiasReport to the console."""
    console.print("\n" + "="*50)
    console.print(f"=== [bold]Confounder Analysis Report[/bold] ===")
    console.print("="*50 + "\n")
    
    console.print(f"Research Question: [cyan]{report.study.research_question}[/cyan]")
    
    if report.naive_estimate is not None:
         console.print(f"\n[bold]Naive Estimate:[/bold] {report.naive_estimate:+.4f}")
    
    if not report.ranked_confounders:
         console.print("\n✅ [green]No critical confounders detected. Study design appears robust.[/green]")
         return
         
    console.print("\n[bold red]CRITICAL CONFOUNDERS DETECTED:[/bold red]" if report.has_critical_confounders else "\n[bold yellow]CONFOUNDERS DETECTED:[/bold yellow]")
    
    for i, rc in enumerate(report.ranked_confounders, 1):
         conf = rc.confounder
         cand = conf.candidate
         
         title_str = f"{i}. {cand.name.upper()}"
         if not conf.is_measured:
             title_str += " [bold magenta](UNMEASURED)[/bold magenta]"
         elif conf.is_statistically_significant:
             title_str += " [bold green](MEASURED & CONFIRMED)[/bold green]"
         else:
             title_str += " (Measured but rejected)"
             
         console.print(f"\n{title_str}")
         console.print(f"   [dim]Mechanism:[/dim] {cand.causes_treatment_because}")
         console.print(f"              {cand.causes_outcome_because}")
         
         if not conf.is_measured:
              from confounder.estimation.sensitivity import bound_unmeasured_confounder
              sens = bound_unmeasured_confounder(conf, report.study, report.naive_estimate or 0.0)
              console.print(f"   [dim]Estimated bias:[/dim] {sens.explanation}")
              console.print(f"   [dim]Evidence:[/dim] LLM mechanistic proposition (Severity: {cand.severity})")
         elif conf.is_statistically_significant:
              console.print(f"   [dim]Estimated bias:[/dim] {conf.bias_magnitude:+.4f} ({conf.bias_percentage:+.1f}% of full effect)")
              console.print(f"   [dim]Evidence:[/dim] Correlation with treatment (p={conf.causes_treatment_pval:.4f}), outcome (p={conf.causes_outcome_pval:.4f})")
         
         # Recommendations
         recs = report.recommendations.get(cand.name, [])
         if recs:
              rec_text = " | ".join(r.description for r in recs)
              console.print(f"   [bold]Recommendation:[/bold] {rec_text}")
              
    console.print()


if __name__ == "__main__":
    app()
