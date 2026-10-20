"""Causal structure discovery: PC algorithm merged with LLM/literature priors.

Primary engine: causal-learn's PC with Fisher-Z conditional independence
tests. If causal-learn is unavailable, a transparent Spearman-skeleton
fallback is used (edges only, orientation from priors/outcome) and the
result is flagged so reports can caveat it.

Prior merging rule: a prior "causes" edge is added only when PC left that
pair unconnected AND adding it cannot create a cycle. PC-oriented edges are
never overridden; conflicts are recorded as attributes for review.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..factors.schema import FactorEdge


def _pc_edges(wide: pd.DataFrame, alpha: float) -> tuple[list[tuple[str, str]],
                                                          list[tuple[str, str]]]:
    """Run PC; return (directed_edges, undirected_or_bidirected_edges)."""
    from causallearn.search.ConstraintBased.PC import pc

    cols = list(wide.columns)
    cg = pc(wide.to_numpy(dtype=float), alpha=alpha, indep_test="fisherz",
            show_progress=False)
    g = cg.G.graph  # g[j,i]==1 and g[i,j]==-1  =>  i --> j
    directed: list[tuple[str, str]] = []
    undirected: list[tuple[str, str]] = []
    n = len(cols)
    for i in range(n):
        for j in range(i + 1, n):
            if g[j, i] == 1 and g[i, j] == -1:
                directed.append((cols[i], cols[j]))
            elif g[i, j] == 1 and g[j, i] == -1:
                directed.append((cols[j], cols[i]))
            elif (g[i, j] == -1 and g[j, i] == -1) or (g[i, j] == 1 and g[j, i] == 1):
                undirected.append((cols[i], cols[j]))
    return directed, undirected


def _correlation_skeleton(wide: pd.DataFrame,
                          threshold: float) -> list[tuple[str, str]]:
    cols = list(wide.columns)
    if len(cols) < 2:
        return []
    rho = np.abs(spearmanr(wide.to_numpy(dtype=float)).statistic)
    rho = np.atleast_2d(rho)
    return [(cols[i], cols[j])
            for i in range(len(cols)) for j in range(i + 1, len(cols))
            if np.isfinite(rho[i, j]) and rho[i, j] >= threshold]


def discover_dag(wide: pd.DataFrame, outcome: str,
                 prior_edges: list[FactorEdge] | None = None,
                 alpha: float = 0.05,
                 fallback_threshold: float = 0.25) -> tuple[nx.DiGraph, str]:
    """Return (DAG, method). Nodes are factor ids; outcome node always present."""
    prior_edges = prior_edges or []
    wide = wide.loc[:, ~wide.columns.duplicated()].dropna(axis=1, how="all")
    if outcome not in wide.columns:
        raise ValueError(f"outcome column '{outcome}' missing from panel")

    try:
        directed, undirected = _pc_edges(wide, alpha)
        method = f"pc-fisherz(alpha={alpha})"
    except Exception as exc:  # noqa: BLE001 - optional heavy dependency
        print(f"  warning: PC unavailable ({exc}); using correlation skeleton")
        directed, undirected = [], _correlation_skeleton(wide, fallback_threshold)
        method = "correlation-skeleton(fallback)"

    g = nx.DiGraph()
    g.add_nodes_from(wide.columns)
    for u, v in directed:
        g.add_edge(u, v, source="pc")
    for u, v in undirected:
        # store both directions, flagged: orientation unknown
        g.add_edge(u, v, source="pc-unknown")
        g.add_edge(v, u, source="pc-unknown")

    conflicts: list[tuple[str, str]] = []
    for e in prior_edges:
        if e.relation != "causes" or e.source not in g or e.target not in g:
            continue
        if g.has_edge(e.target, e.source) and not g.has_edge(e.source, e.target):
            conflicts.append((e.source, e.target))
            continue
        if not g.has_edge(e.source, e.target) and not nx.has_path(g, e.target, e.source):
            g.add_edge(e.source, e.target, source="prior",
                       confidence=e.confidence)
    for u, v in conflicts:
        g.add_edge(u, v, source="prior-conflict")

    g.graph["method"] = method
    g.graph["prior_conflicts"] = conflicts
    g.graph["prior_causes"] = [(e.source, e.target) for e in prior_edges
                               if e.relation == "causes"]
    return g, method


def candidate_treatments(g: nx.DiGraph, outcome: str) -> list[str]:
    """Factors adjacent to the outcome (either orientation) = effect candidates."""
    if outcome not in g:
        return []
    nbrs = set(g.predecessors(outcome)) | set(g.successors(outcome))
    return sorted(n for n in nbrs if n != outcome)


def adjustment_set(g: nx.DiGraph, treatment: str, outcome: str) -> list[str]:
    """Backdoor adjustment set: parents of treatment that are not descendants
    of the outcome (standard backdoor restriction; conservative under
    pc-unknown bidirected edges because both endpoints count as parents).

    When the graph yields no adjustment variables (typical of the dense
    correlation-skeleton fallback), the catalog's prior "causes" parents of
    the treatment are used instead, so estimation is never unadjusted by
    accident.
    """
    if treatment not in g or outcome not in g:
        return []
    parents = set(g.predecessors(treatment))
    desc_out = set(nx.descendants(g, outcome)) | {outcome}
    adj = (parents - desc_out) - {treatment}
    if not adj:
        prior = [s for (s, t) in g.graph.get("prior_causes", []) if t == treatment]
        adj = set(prior) - {outcome, treatment}
    return sorted(adj)


def dag_to_dot(g: nx.DiGraph, outcome: str) -> str:
    lines = ["digraph causal {", '  rankdir=LR;', '  node [shape=box, fontsize=10];']
    for n in g.nodes:
        style = ' [shape=doublecircle, style=bold];' if n == outcome else ";"
        lines.append(f'  "{n}"{style}')
    for u, v, data in g.edges(data=True):
        src = data.get("source", "pc")
        style = {"prior": ' [style=dashed, color=blue]',
                 "prior-conflict": ' [style=dashed, color=red]',
                 "pc-unknown": ' [dir=none, color=gray]'}.get(src, "")
        lines.append(f'  "{u}" -> "{v}"{style};')
    lines.append("}")
    return "\n".join(lines)
