"""DAG visualization for observational studies with confounders."""

from __future__ import annotations

import logging
from pathlib import Path

from confounder.data.loader import Study
from confounder.correction.explainer import RankedConfounder

logger = logging.getLogger(__name__)


def render_confounder_dag(
    study: Study,
    ranked_confounders: list[RankedConfounder],
    output_path: str | Path = "confounder_dag.html",
) -> Path:
    """
    Render an interactive DAG showing the treatment, outcome, and all 
    detected/proposed confounders with severity coloring.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        logger.warning("pyvis not installed — skipping DAG visualization")
        return Path(output_path)

    output_path = Path(output_path)
    net = Network(
        height="600px",
        width="100%",
        directed=True,
        bgcolor="#1E1E1E",  # dark mode by default
        font_color="white",
        heading="Causal Graph (Confounder Audit)",
    )

    # Core nodes
    net.add_node(
        study.treatment,
        label=f"Treatment: {study.treatment}",
        color={"background": "#4a90d9", "border": "#007BFF"},
        shape="box",
        borderWidth=2,
        level=2,
    )
    net.add_node(
        study.outcome,
        label=f"Outcome: {study.outcome}",
        color={"background": "#e94560", "border": "#FF4136"},
        shape="box",
        borderWidth=2,
        level=2,
    )

    # Core relationship
    net.add_edge(study.treatment, study.outcome, color="#FFFFFF", width=2, title="Apparent Effect")

    # Add confounders
    for i, rc in enumerate(ranked_confounders):
        c_name = rc.confounder.candidate.name
        
        # Color by severity
        if rc.severity == "Critical":
            bg_color = "#FF851B" # Orange
            border_color = "#FF4136"
        elif rc.severity == "Moderate":
            bg_color = "#FFDC00" # Yellow
            border_color = "#FF851B"
        else:
            bg_color = "#AAAAAA" # Gray
            border_color = "#777777"
            
        label = f"{c_name}\n"
        if not rc.confounder.is_measured:
             label += "(Unmeasured)"
        elif rc.confounder.is_statistically_significant:
             label += f"(Bias: {rc.confounder.bias_percentage:.1f}%)"
             
        net.add_node(
            c_name,
            label=label,
            color={"background": bg_color, "border": border_color},
            shape="ellipse",
            borderWidth=2,
            level=1, # Layer above treatment/outcome
        )
        
        # Confounder paths
        # Z -> T
        net.add_edge(
            c_name, study.treatment,
            color=border_color,
            width=2,
            title=rc.confounder.candidate.causes_treatment_because
        )
        # Z -> Y
        net.add_edge(
            c_name, study.outcome,
            color=border_color,
            width=2,
            title=rc.confounder.candidate.causes_outcome_because
        )

    # Physics layout
    net.set_options("""
    {
        "layout": {
            "hierarchical": {
                "enabled": true,
                "direction": "UD",
                "sortMethod": "directed"
            }
        },
        "physics": {"enabled": false},
        "edges": {"arrows": {"to": {"enabled": true, "scaleFactor": 1.0}}}
    }
    """)

    net.save_graph(str(output_path))
    logger.info("Saved Confounder DAG to %s", output_path)
    return output_path
