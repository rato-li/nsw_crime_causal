"""Orchestrate factor embedding: graph -> message passing -> clusters -> store."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Settings
from ..factors.schema import FactorCatalog
from .graph import build_factor_graph, factor_node_features, normalized_adjacency


@dataclass
class EmbedResult:
    nodes: list[str]
    embeddings: np.ndarray            # [n_nodes, dim]
    clusters: dict[str, int]          # factor_id -> cluster label
    representatives: dict[int, str]   # cluster label -> medoid factor_id
    graph_edges: list[tuple[str, str]] = field(default_factory=list)

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "embeddings.npz",
                            embeddings=self.embeddings,
                            nodes=np.array(self.nodes))
        (out_dir / "clusters.json").write_text(
            json.dumps({"clusters": self.clusters,
                        "representatives": {str(k): v for k, v in
                                            self.representatives.items()}},
                       indent=2), encoding="utf-8")


def cluster_embeddings(z: np.ndarray, nodes: list[str],
                       max_clusters: int = 6) -> tuple[dict[str, int], dict[int, str]]:
    """K-means on embeddings; medoid of each cluster = adjustment-set representative."""
    from sklearn.cluster import KMeans

    k = max(2, min(max_clusters, len(nodes) // 3))
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(z)

    clusters = {n: int(lab) for n, lab in zip(nodes, labels, strict=True)}
    reps: dict[int, str] = {}
    for c in range(k):
        members = [i for i, lab in enumerate(labels) if lab == c]
        if not members:
            continue
        centroid = km.cluster_centers_[c]
        medoid = min(members, key=lambda i: float(np.linalg.norm(z[i] - centroid)))
        reps[c] = nodes[medoid]
    return clusters, reps


def embed_factors(catalog: FactorCatalog, wide: pd.DataFrame,
                  settings: Settings, save_to_db: bool = False) -> EmbedResult:
    """Full embedding pass; optionally persists vectors to SAT_FACTOR_EMBEDDING."""
    g = build_factor_graph(catalog, wide,
                           corr_threshold=settings.corr_edge_threshold,
                           outcome=settings.outcome_factor_id)
    nodes, x = factor_node_features(catalog, wide, outcome=settings.outcome_factor_id)
    # keep only nodes present in the graph
    keep = [i for i, n in enumerate(nodes) if n in g]
    nodes = [nodes[i] for i in keep]
    x = x[keep]
    a_hat = normalized_adjacency(g, nodes)

    idx = {n: i for i, n in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in g.edges if u in idx and v in idx]

    from .model import train_embeddings

    z = train_embeddings(
        x, a_hat, edges,
        dim=settings.gnn_dim, hidden=settings.gnn_hidden, layers=settings.gnn_layers,
        epochs=settings.gnn_epochs, lr=settings.gnn_lr, neg_ratio=settings.gnn_neg_ratio,
        seed=settings.seed, device=settings.gnn_device, verbose=True,
    )
    clusters, reps = cluster_embeddings(z, nodes)
    result = EmbedResult(nodes=nodes, embeddings=z, clusters=clusters,
                         representatives=reps,
                         graph_edges=[(u, v) for u, v in g.edges])

    if save_to_db:
        from ..db import connect, save_embeddings

        conn = connect(settings)
        try:
            n = save_embeddings(conn, {n: list(map(float, z[i]))
                                       for i, n in enumerate(nodes)},
                                settings.embedding_model_version)
            print(f"  wrote {n} embedding rows to SAT_FACTOR_EMBEDDING")
        finally:
            conn.close()
    return result
