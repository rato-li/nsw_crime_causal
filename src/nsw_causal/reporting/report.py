"""Assemble the final markdown report from pipeline artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def _dag_from_json(path: Path) -> nx.DiGraph | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return nx.node_link_graph(data, edges="links")


def _plot_dag(g: nx.DiGraph, outcome: str, out: Path) -> None:
    pos = nx.spring_layout(g, seed=42, k=1.6 / np.sqrt(max(g.number_of_nodes(), 1)))
    fig, ax = plt.subplots(figsize=(11, 8))
    solid = [(u, v) for u, v, d in g.edges(data=True)
             if d.get("source") not in ("prior", "prior-conflict")]
    prior = [(u, v) for u, v, d in g.edges(data=True)
             if d.get("source") in ("prior", "prior-conflict")]
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=650,
                           node_color=["#d62728" if n == outcome else "#1f77b4"
                                       for n in g])
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=7)
    nx.draw_networkx_edges(g, pos, edgelist=solid, ax=ax,
                           edge_color="#333333", arrows=True, width=1.2)
    nx.draw_networkx_edges(g, pos, edgelist=prior, ax=ax, edge_color="#1f77b4",
                           style="dashed", arrows=True, width=1.0)
    ax.set_title("Discovered causal graph (dashed = LLM/literature prior)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_effects(effects: pd.DataFrame, out: Path) -> None:
    df = effects.dropna(subset=["ate"]).head(12).iloc[::-1]
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(df) + 2))
    y = np.arange(len(df))
    ax.errorbar(df["ate"], y,
                xerr=[df["ate"] - df["ci_low"].fillna(df["ate"]),
                      df["ci_high"].fillna(df["ate"]) - df["ate"]],
                fmt="o", color="#1f77b4", capsize=3)
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_yticks(y, df["treatment"])
    ax.set_xlabel("ATE on crime rate (standardised units)")
    ax.set_title("Causal effect estimates (backdoor-adjusted)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _md_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_no results_\n"
    return df.to_markdown(index=False) + "\n"


def build_report(artifacts_dir: Path, reports_dir: Path, settings) -> Path:
    artifacts_dir = Path(artifacts_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figs = reports_dir / "figures"
    figs.mkdir(exist_ok=True)

    parts: list[str] = [
        "# NSW Crime Causal Model — Run Report",
        f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
        f"| outcome: `{settings.outcome_factor_id}` | seed {settings.seed}_",
    ]

    # --- catalog --------------------------------------------------------
    catalog_path = artifacts_dir / "catalog.json"
    if catalog_path.exists():
        from ..factors.schema import FactorCatalog

        catalog = FactorCatalog.load(catalog_path)
        cat_df = pd.DataFrame([{
            "factor": f.factor_id, "category": f.category,
            "expected sign": f.expected_sign, "evidence": f.evidence,
            "primary proxy": (f.proxies[0].dataset + ": " + f.proxies[0].variable)
            if f.proxies else "",
        } for f in catalog.factors])
        parts += ["## 1. Candidate causal factors (LLM-discovered + curated)",
                  _md_table(cat_df)]

    # --- embeddings -----------------------------------------------------
    emb_path = artifacts_dir / "embeddings.npz"
    if emb_path.exists():
        data = np.load(emb_path, allow_pickle=True)
        nodes = [str(n) for n in data["nodes"]]
        z = data["embeddings"]
        clusters = {}
        if (artifacts_dir / "clusters.json").exists():
            clusters = json.loads((artifacts_dir / "clusters.json")
                                  .read_text(encoding="utf-8"))
        parts += ["## 2. GNN message-passing factor embeddings",
                  f"- {len(nodes)} factor nodes embedded to {z.shape[1]} dims "
                  f"(self-supervised link prediction on the factor graph)",
                  ""]
        if clusters.get("clusters"):
            cl_df = (pd.DataFrame({"factor": list(clusters["clusters"].keys()),
                                   "cluster": list(clusters["clusters"].values())})
                     .sort_values("cluster"))
            reps = clusters.get("representatives", {})
            parts += [_md_table(cl_df),
                      f"Cluster representatives (adjustment-set members): "
                      f"`{', '.join(reps.values())}`"]

    # --- DAG -------------------------------------------------------------
    g = _dag_from_json(artifacts_dir / "dag.json")
    if g is not None:
        dag_png = figs / "dag.png"
        _plot_dag(g, settings.outcome_factor_id, dag_png)
        method = g.graph.get("method", "?")
        parts += ["## 3. Causal structure discovery",
                  f"- method: `{method}`",
                  f"- edges: {g.number_of_edges()}",
                  "![causal DAG](figures/dag.png)", ""]
        if (artifacts_dir / "dag.dot").exists():
            parts.append("```dot\n"
                         + (artifacts_dir / "dag.dot").read_text(encoding="utf-8")
                         + "\n```\n")

    # --- effects ----------------------------------------------------------
    effects_path = artifacts_dir / "effects.csv"
    if effects_path.exists():
        effects = pd.read_csv(effects_path)
        eff_png = figs / "effects.png"
        _plot_effects(effects, eff_png)
        parts += ["## 4. Causal effect estimates",
                  _md_table(effects.drop(
                      columns=[c for c in ["abs_ate"] if c in effects.columns])),
                  "![effects](figures/effects.png)", ""]

    # --- refutation --------------------------------------------------------
    ref_path = artifacts_dir / "refutation.csv"
    if ref_path.exists():
        ref = pd.read_csv(ref_path)
        parts += ["## 5. Refutation / falsification tests", _md_table(ref)]

    parts += [
        "## 6. Threats to validity",
        "- **Ecological data**: factors and outcome are area-level aggregates; "
        "individual-level offender attributes (age, gender, education) are NOT "
        "used, so effects are contextual, not individual-level causal claims.",
        "- **Reporting propensity**: recorded crime depends on policing and "
        "reporting behaviour, not only offending.",
        "- **LLM priors**: catalog and prior edges come from a local LLM and "
        "must be reviewed by a domain expert before any decision use.",
        "- **Synthetic demo data**: unless real ABS/NSW SBS panels were loaded, "
        "numbers illustrate the pipeline only.",
        "- **Unmeasured confounding**: no refutation test can rule it out; "
        "interpret magnitudes alongside sensitivity analyses.",
    ]

    out = reports_dir / "report.md"
    out.write_text("\n\n".join(parts), encoding="utf-8")
    return out
