"""Synthetic panel + factor graph tests."""

import numpy as np

from nsw_causal.config import get_settings
from nsw_causal.factors.collect import panel_to_wide, synthetic_panel
from nsw_causal.gnn.graph import build_factor_graph, factor_node_features, normalized_adjacency
from nsw_causal.llm.factor_discovery import load_seed_catalog


def _panel():
    catalog = load_seed_catalog()
    settings = get_settings()
    panel = synthetic_panel(catalog, settings, n_regions=25,
                            years=range(2015, 2021))
    return catalog, panel_to_wide(panel), panel


def test_synthetic_panel_shape_and_columns():
    catalog, wide, panel = _panel()
    assert set(panel.columns) == {"factor_id", "region", "period", "value",
                                  "dataset", "quality"}
    assert "crime_rate" in wide.columns
    assert wide.shape[0] == 25 * 6


def test_factor_graph_has_edges_and_outcome():
    catalog, wide, _ = _panel()
    g = build_factor_graph(catalog, wide, corr_threshold=0.35)
    assert "crime_rate" in g.nodes
    assert g.number_of_edges() > 0
    provenance = {d.get("provenance") for _, _, d in g.edges(data=True)}
    assert "llm" in provenance or "correlation" in provenance


def test_node_features_finite_and_adjacency_symmetric():
    catalog, wide, _ = _panel()
    nodes, x = factor_node_features(catalog, wide)
    assert np.isfinite(x).all()
    g = build_factor_graph(catalog, wide)
    keep_nodes = [n for n in nodes if n in g]
    a = normalized_adjacency(g, keep_nodes)
    assert a.shape == (len(keep_nodes), len(keep_nodes))
    assert np.allclose(a, a.T, atol=1e-5)
