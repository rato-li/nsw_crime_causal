"""GNN message-passing tests (require torch; skipped otherwise)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nsw_causal.gnn.model import train_embeddings  # noqa: E402


def _toy_graph(n=12, d=7, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)
    a = np.eye(n, dtype=np.float32)
    edges = []
    for i in range(n - 1):
        a[i, i + 1] = a[i + 1, i] = 1.0
        edges.append((i, i + 1))
    deg = a.sum(1)
    dinv = deg ** -0.5
    return x, (dinv[:, None] * a * dinv[None, :]).astype(np.float32), edges


def test_train_embeddings_shape_and_finite():
    x, a, edges = _toy_graph()
    z = train_embeddings(x, a, edges, dim=8, hidden=16, layers=2,
                         epochs=30, seed=1, device="cpu")
    assert z.shape == (12, 8)
    assert np.isfinite(z).all()


def test_link_prediction_learns_edges():
    """After training, connected pairs should score higher than random pairs."""
    x, a, edges = _toy_graph(n=14, seed=2)
    z = train_embeddings(x, a, edges, dim=16, hidden=32, layers=2,
                         epochs=200, lr=1e-2, seed=2, device="cpu")
    pos = np.array([float(z[i] @ z[j]) for i, j in edges])
    rng = np.random.default_rng(0)
    neg = np.array([float(z[i] @ z[j])
                    for i, j in zip(rng.integers(0, 14, 50), rng.integers(0, 14, 50),
                                    strict=True)
                    if abs(i - j) > 1])
    assert pos.mean() > neg.mean()
