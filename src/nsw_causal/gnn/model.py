"""Message-passing GNN over the factor graph (pure PyTorch, no PyG needed).

Design (consistent with the CFR-GNN notes in this workspace):
  * one node per causal factor; dense symmetric-normalised adjacency
    (factor graphs are tiny - tens of nodes - so dense propagation is exact
    and simpler than neighbour sampling);
  * GraphSAGE-style layer:  h' = LayerNorm(W_self h + W_nbr (A_hat h));
  * self-supervised training objective: LINK PREDICTION on the observed
    factor graph (LLM priors + empirical correlation edges) via a dot-product
    decoder, plus a feature-reconstruction term so embeddings also retain
    each factor's own statistics.

The resulting embeddings place factors with similar causal/semantic graph
context close together; downstream they are used to (a) cluster proxy factors
so adjustment sets are not padded with near-duplicates, and (b) provide
regularised node features for causal discovery and estimation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MPLayer(nn.Module):
    """One round of message passing: self transform + aggregated neighbour messages."""

    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.lin_self = nn.Linear(d_in, d_out)
        self.lin_nbr = nn.Linear(d_in, d_out, bias=False)
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        out = self.norm(self.lin_self(h) + self.lin_nbr(a_hat @ h))
        return self.drop(F.relu(out))


class FactorGNN(nn.Module):
    """Stacked message-passing layers + linear decoder for feature reconstruction."""

    def __init__(self, d_in: int, hidden: int = 64, dim: int = 32,
                 layers: int = 2, dropout: float = 0.1):
        super().__init__()
        dims = [d_in] + [hidden] * (layers - 1) + [dim]
        self.mp = nn.ModuleList(
            [MPLayer(dims[i], dims[i + 1], dropout) for i in range(layers)])
        self.residual = nn.Linear(d_in, dim, bias=False)
        self.decode = nn.Linear(dim, d_in)

    def encode(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.mp:
            h = layer(h, a_hat)
        return h + self.residual(x)  # residual keeps node identity visible

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor):
        z = self.encode(x, a_hat)
        return z, self.decode(z)


def _sample_negatives(pos: torch.Tensor, n_nodes: int, ratio: int,
                      generator: torch.Generator) -> torch.Tensor:
    """Random non-edge pairs (excluding self-loops and positives)."""
    pos_set = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in pos.tolist()}
    neg: list[tuple[int, int]] = []
    want = max(pos.shape[0] * ratio, 1)
    tries = 0
    while len(neg) < want and tries < 50 * want:
        tries += 1
        i = int(torch.randint(n_nodes, (1,), generator=generator).item())
        j = int(torch.randint(n_nodes, (1,), generator=generator).item())
        if i == j or (min(i, j), max(i, j)) in pos_set:
            continue
        neg.append((i, j))
    return torch.tensor(neg, dtype=torch.long) if neg else torch.zeros((0, 2), dtype=torch.long)


def train_embeddings(x: np.ndarray, a_hat: np.ndarray, edges: list[tuple[int, int]],
                     dim: int = 32, hidden: int = 64, layers: int = 2,
                     epochs: int = 300, lr: float = 5e-3, neg_ratio: int = 2,
                     seed: int = 42, device: str = "auto",
                     verbose: bool = False) -> np.ndarray:
    """Self-supervised link-prediction training; returns [n_nodes, dim] embeddings."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    dev = torch.device(device)
    x_t = torch.tensor(x, dtype=torch.float32, device=dev)
    a_t = torch.tensor(a_hat, dtype=torch.float32, device=dev)
    pos = torch.tensor(edges, dtype=torch.long, device=dev) if edges else \
        torch.zeros((0, 2), dtype=torch.long, device=dev)
    if pos.shape[0] == 0:
        raise ValueError("factor graph has no edges; cannot train link prediction")

    model = FactorGNN(x.shape[1], hidden, dim, layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        neg = _sample_negatives(pos, x.shape[0], neg_ratio, gen).to(dev)
        idx = torch.cat([pos[:, 0], neg[:, 0]]) if neg.shape[0] else pos[:, 0]
        jdx = torch.cat([pos[:, 1], neg[:, 1]]) if neg.shape[0] else pos[:, 1]
        labels = torch.cat([torch.ones(pos.shape[0], device=dev),
                            torch.zeros(neg.shape[0], device=dev)])

        z, x_hat = model(x_t, a_t)
        logits = (z[idx] * z[jdx]).sum(-1)
        link_loss = F.binary_cross_entropy_with_logits(logits, labels)
        recon_loss = F.mse_loss(x_hat, x_t)
        loss = link_loss + 0.1 * recon_loss

        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch + 1:4d}  link_loss={link_loss.item():.4f} "
                  f"recon={recon_loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        z, _ = model(x_t, a_t)
        return z.detach().cpu().numpy()
