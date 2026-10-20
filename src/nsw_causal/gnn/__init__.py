"""Graph Neural Network utilities: factor graph + message-passing embeddings."""

from .graph import build_factor_graph, factor_node_features, normalized_adjacency

__all__ = ["build_factor_graph", "factor_node_features", "normalized_adjacency"]
