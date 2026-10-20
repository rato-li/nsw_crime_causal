"""nswcausal command-line interface.

Typical full run (offline demo, no DB / no LLM):

    nswcausal run-all --synthetic

Typical production run:

    nswcausal init-db
    nswcausal discover-factors --use-llm
    nswcausal load-data --csv data/raw/panel.csv --use-db
    nswcausal embed --use-db
    nswcausal discover-causal
    nswcausal estimate --refute
    nswcausal report
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from .config import Settings, get_settings

app = typer.Typer(help="NSW crime causal-model pipeline", no_args_is_help=True)
console = Console()

CATALOG = "catalog.json"
PANEL = "panel.csv"
WIDE = "wide.csv"
DAG_JSON = "dag.json"
DAG_DOT = "dag.dot"
EFFECTS = "effects.csv"
REFUTATION = "refutation.csv"


def _settings() -> Settings:
    return get_settings()


def _art() -> Path:
    s = _settings()
    p = s.resolve(s.artifacts_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_catalog() -> object:
    from .factors.schema import FactorCatalog
    from .llm.factor_discovery import load_seed_catalog

    path = _art() / CATALOG
    if path.exists():
        return FactorCatalog.load(path)
    console.print("[yellow]no discovered catalog; using curated seed catalog[/]")
    return load_seed_catalog()


def _wide_from_artifacts() -> pd.DataFrame:
    art = _art()
    wide_path = art / WIDE
    if not wide_path.exists():
        raise typer.Exit("run load-data first (missing artifacts/wide.csv)")
    return pd.read_csv(wide_path, index_col=[0, 1])


def _db_connect(settings: Settings):
    from .db import connect

    return connect(settings)


# ---------------------------------------------------------------------------
@app.command("init-db")
def init_db():
    """Create the Data Vault 2.0 schema (hubs/links/satellites/views) in MSSQL."""
    settings = _settings()
    from .db import init_schema

    console.print(f"connecting to {settings.mssql_host}:{settings.mssql_port}"
                  f"/{settings.mssql_database} ...")
    conn = _db_connect(settings)
    try:
        init_schema(conn)
        console.print("[green]Data Vault schema ready.[/]")
    finally:
        conn.close()


@app.command("discover-factors")
def discover_factors(use_llm: bool = False,
                     max_factors: int = 30,
                     seed_catalog: Path | None = None):
    """Build the factor catalog (LLM multi-angle discovery, or curated seed)."""
    settings = _settings()
    if use_llm:
        from .llm.factor_discovery import discover_catalog

        console.print(f"querying local LLM {settings.llm_model} @ "
                      f"{settings.llm_base_url} ...")
        catalog = discover_catalog(settings, max_factors=max_factors)
    else:
        from .llm.factor_discovery import load_seed_catalog

        catalog = load_seed_catalog(seed_catalog)
        console.print("using curated seed catalog "
                      "(pass --use-llm to query the local model)")
    path = catalog.save(_art() / CATALOG)
    console.print(f"[green]{len(catalog.factors)} factors, "
                  f"{len(catalog.edges)} prior edges -> {path}[/]")


@app.command("load-data")
def load_data(csv: Path | None = None,
              synthetic: bool = False,
              use_db: bool = False):
    """Load the factor panel (CSV or synthetic demo) and optionally push to MSSQL."""
    settings = _settings()
    from .factors.collect import load_panel_csv, panel_to_wide, synthetic_panel

    catalog = _load_catalog()
    if csv is not None:
        panel = load_panel_csv(csv)
        console.print(f"loaded {len(panel)} rows from {csv}")
    elif synthetic:
        panel = synthetic_panel(catalog, settings)
        console.print(f"generated synthetic panel: {len(panel)} rows "
                      f"(demo only - planted effects)")
    else:
        raise typer.Exit("pass --csv PATH or --synthetic")

    art = _art()
    panel.to_csv(art / PANEL, index=False)
    wide = panel_to_wide(panel)
    wide.to_csv(art / WIDE)
    console.print(f"panel: {wide.shape[0]} region-period rows x "
                  f"{wide.shape[1]} factors")

    if use_db:
        from .db import insert_observations, upsert_factor, upsert_relation

        conn = _db_connect(settings)
        try:
            for f in catalog.factors:
                upsert_factor(conn, f)
            for e in catalog.edges:
                upsert_relation(conn, e)
            n = insert_observations(conn, panel)
            console.print(f"[green]wrote {n} observations to Data Vault[/]")
        finally:
            conn.close()


@app.command("embed")
def embed(use_db: bool = False):
    """GNN message-passing embeddings of every factor + proxy clustering."""
    settings = _settings()
    from .gnn.embed import embed_factors

    catalog = _load_catalog()
    wide = _wide_from_artifacts()
    console.print("training message-passing GNN on the factor graph ...")
    result = embed_factors(catalog, wide, settings, save_to_db=use_db)
    result.save(_art())
    console.print(f"[green]{len(result.nodes)} embeddings "
                  f"({result.embeddings.shape[1]} dims), "
                  f"{len(set(result.clusters.values()))} clusters[/]")


@app.command("discover-causal")
def discover_causal(alpha: float | None = None):
    """Causal discovery: PC algorithm merged with catalog prior edges."""
    settings = _settings()
    from .causal.discovery import dag_to_dot, discover_dag

    catalog = _load_catalog()
    wide = _wide_from_artifacts()
    g, method = discover_dag(wide, settings.outcome_factor_id,
                             prior_edges=catalog.edges,
                             alpha=alpha or settings.pc_alpha)
    art = _art()
    (art / DAG_JSON).write_text(
        json.dumps(nx.node_link_data(g, edges="links"), indent=1), encoding="utf-8")
    (art / DAG_DOT).write_text(dag_to_dot(g, settings.outcome_factor_id),
                               encoding="utf-8")
    console.print(f"[green]DAG via {method}: {g.number_of_nodes()} nodes, "
                  f"{g.number_of_edges()} edges -> {art / DAG_DOT}[/]")


@app.command("estimate")
def estimate(econml: bool = True, refute: bool = False, top_k: int = 5):
    """Estimate causal effects of factors adjacent to the outcome (DoWhy/EconML)."""
    settings = _settings()
    from .causal.discovery import candidate_treatments
    from .causal.estimation import estimate_effects

    wide = _wide_from_artifacts()
    art = _art()
    if not (art / DAG_JSON).exists():
        raise typer.Exit("run discover-causal first")
    g = nx.node_link_graph(json.loads((art / DAG_JSON).read_text()), edges="links")

    treatments = candidate_treatments(g, settings.outcome_factor_id)
    console.print(f"{len(treatments)} candidate treatments: {treatments}")
    effects = estimate_effects(wide, g, settings.outcome_factor_id, treatments,
                               settings, use_econml=econml)
    effects.to_csv(art / EFFECTS, index=False)
    console.print(f"[green]effects -> {art / EFFECTS}[/]")

    if refute:
        from .causal.refutation import refute_effects

        top = effects.dropna(subset=["ate"]).head(top_k)["treatment"].tolist()
        ref = refute_effects(wide, g, settings.outcome_factor_id, top, settings)
        ref.to_csv(art / REFUTATION, index=False)
        console.print(f"[green]refutation tests -> {art / REFUTATION}[/]")


@app.command("report")
def report():
    """Render reports/report.md with tables and figures."""
    from .reporting import build_report

    settings = _settings()
    out = build_report(settings.resolve(settings.artifacts_dir),
                       settings.resolve(settings.reports_dir), settings)
    console.print(f"[green]report -> {out}[/]")


@app.command("show-effects")
def show_effects():
    """Print the effects table."""
    path = _art() / EFFECTS
    if not path.exists():
        raise typer.Exit("no effects yet")
    console.print(Table.from_dataframe(pd.read_csv(path).head(20)))


@app.command("run-all")
def run_all(synthetic: bool = False, csv: Path | None = None,
            use_llm: bool = False, use_db: bool = False):
    """Full pipeline: discover -> load -> embed -> discover-causal -> estimate -> report."""
    if use_db:
        try:
            init_db()
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]MSSQL unreachable ({exc}); continuing offline[/]")
            use_db = False

    discover_factors(use_llm=use_llm)
    load_data(csv=csv, synthetic=synthetic and csv is None, use_db=use_db)
    embed(use_db=use_db)
    discover_causal()
    estimate(refute=False)
    report()
    console.print("[bold green]pipeline complete[/]")


if __name__ == "__main__":
    app()
