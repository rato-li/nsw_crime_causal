"""Build the factor graph the GNN runs message passing over.

Nodes  = causal factors (including the outcome node).
Edges  = union of
  * LLM / literature prior relations from the catalog (weighted by confidence)
  * empirical edges where |Spearman rho| between factor panels exceeds a
    threshold (weighted by |rho|).
Node features = standardised panel summary statistics + correlation profile
with the outcome + categorical metadata.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..factors.schema import FactorCatalog

CATEGORY_ORDER = ["economics", "demographics", "housing", "health",
                  "environment", "infrastructure", "policing", "social", "outcome"]


def build_factor_graph(catalog: FactorCatalog, wide: pd.DataFrame,
                       corr_threshold: float = 0.35,
                       outcome: str = "crime_rate") -> nx.Graph:
    """Nodes=factor ids; edges from catalog priors + empirical correlation."""
    g = nx.Graph()
    for f in catalog.factors:
        g.add_node(f.factor_id, category=f.category, expected_sign=f.expected_sign,
                   name=f.name)
    if outcome not in g:
        g.add_node(outcome, category="outcome", expected_sign="?", name="Crime rate")

    for e in catalog.edges:
        if e.source in g and e.target in g:
            g.add_edge(e.source, e.target, weight=float(e.confidence),
                       relation=e.relation, provenance="llm")

    cols = [c for c in g.nodes if c in wide.columns]
    if len(cols) >= 2:
        rho = pd.DataFrame(spearmanr(wide[cols].to_numpy()).statistic,
                           index=cols, columns=cols)
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                r = float(rho.loc[a, b])
                if np.isfinite(r) and abs(r) >= corr_threshold:
                    if g.has_edge(a, b):
                        g[a][b]["corr"] = r
                    else:
                        g.add_edge(a, b, weight=abs(r), relation="correlates",
                                   provenance="correlation", corr=r)
    return g


def factor_node_features(catalog: FactorCatalog, wide: pd.DataFrame,
                         outcome: str = "crime_rate") -> tuple[list[str], np.ndarray]:
    """Feature matrix per factor node: summary stats + outcome correlation."""
    nodes = [f.factor_id for f in catalog.factors]
    if outcome in wide.columns and outcome not in nodes:
        nodes.append(outcome)

    y = wide[outcome].to_numpy(dtype=float) if outcome in wide.columns else None
    feats = []
    for fid in nodes:
        if fid not in wide.columns:
            feats.append(np.zeros(6 + len(CATEGORY_ORDER) + 1))
            continue
        x = wide[fid].to_numpy(dtype=float)
        mask = np.isfinite(x) & (np.isfinite(y) if y is not None else True)
        corr = 0.0
        if y is not None and mask.sum() > 3 and np.std(x[mask]) > 1e-12:
            corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
        cat = catalog.get(fid)
        cat_idx = CATEGORY_ORDER.index(cat.category) if cat and cat.category in CATEGORY_ORDER else -1
        sign = {"+": 1.0, "-": -1.0}.get(cat.expected_sign if cat else "?", 0.0)
        row = [
            np.nanmean(x), np.nanstd(x),
            float(pd.Series(x).skew()) if np.isfinite(x).sum() > 2 else 0.0,
            float(np.isnan(x).mean()), corr, sign,
        ]
        onehot = [0.0] * len(CATEGORY_ORDER)
        if cat_idx >= 0:
            onehot[cat_idx] = 1.0
        feats.append(np.array(row + onehot, dtype=float))

    X = np.vstack(feats)
    # standardise numeric columns (all but the one-hot block)
    n_num = X.shape[1] - len(CATEGORY_ORDER)
    mu = np.nanmean(X[:, :n_num], axis=0)
    sd = np.nanstd(X[:, :n_num], axis=0)
    sd[sd < 1e-9] = 1.0
    X[:, :n_num] = (X[:, :n_num] - mu) / sd
    return nodes, np.nan_to_num(X)


def normalized_adjacency(g: nx.Graph, nodes: list[str]) -> np.ndarray:
    """Symmetric-normalised adjacency with self-loops: D^-1/2 (A+I) D^-1/2."""
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    a = np.eye(n, dtype=np.float32)
    for u, v in g.edges:
        if u in idx and v in idx:
            w = float(g[u][v].get("weight", 1.0))
            a[idx[u], idx[v]] += w
            a[idx[v], idx[u]] += w
    deg = a.sum(1)
    dinv = np.power(np.maximum(deg, 1e-12), -0.5)
    return (dinv[:, None] * a * dinv[None, :]).astype(np.float32)
