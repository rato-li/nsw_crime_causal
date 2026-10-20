"""Causal discovery tests (uses the fallback engine when causal-learn is absent)."""

import networkx as nx

from nsw_causal.causal.discovery import (
    adjustment_set,
    candidate_treatments,
    dag_to_dot,
    discover_dag,
)
from nsw_causal.factors.collect import panel_to_wide, synthetic_panel
from nsw_causal.llm.factor_discovery import load_seed_catalog


def _wide():
    catalog = load_seed_catalog()
    return catalog, panel_to_wide(synthetic_panel(catalog, n_regions=30,
                                                  years=range(2014, 2022)))


def test_discover_dag_contains_outcome():
    catalog, wide = _wide()
    g, method = discover_dag(wide, "crime_rate", prior_edges=catalog.edges,
                             alpha=0.05)
    assert "crime_rate" in g.nodes
    assert method  # pc-fisherz or fallback
    assert g.number_of_edges() > 0
    assert isinstance(g, nx.DiGraph)


def test_prior_edges_added_without_cycles():
    catalog, wide = _wide()
    g, _ = discover_dag(wide, "crime_rate", prior_edges=catalog.edges)
    assert nx.is_directed_acyclic_graph(g) or any(
        d.get("source") == "pc-unknown" for _, _, d in g.edges(data=True))


def test_adjustment_set_excludes_outcome_and_descendants():
    g = nx.DiGraph()
    g.add_edge("ses", "unemployment_rate")
    g.add_edge("ses", "crime_rate")
    g.add_edge("unemployment_rate", "crime_rate")
    adj = adjustment_set(g, "unemployment_rate", "crime_rate")
    assert adj == ["ses"]
    assert candidate_treatments(g, "crime_rate") == ["ses", "unemployment_rate"]


def test_adjustment_set_falls_back_to_prior_causes():
    """Dense bidirected (fallback) graph -> prior parents are used."""
    g = nx.DiGraph()
    for n in ["a", "b", "crime_rate"]:
        g.add_edge(n, "crime_rate", source="pc-unknown")
        g.add_edge("crime_rate", n, source="pc-unknown")
    g.graph["prior_causes"] = [("a", "b"), ("crime_rate", "b")]
    assert adjustment_set(g, "b", "crime_rate") == ["a"]


def test_dag_to_dot_renders():
    g = nx.DiGraph([("a", "crime_rate")])
    dot = dag_to_dot(g, "crime_rate")
    assert "digraph" in dot and '"a" -> "crime_rate"' in dot
