"""Small per-interval MLP used by the neural retrieval predictor and FPCA snaps."""
from __future__ import annotations

import numpy as np


def train_mlp(
    X: np.ndarray,
    Y: np.ndarray,
    hidden_dim: int = 64,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
):
    """Fit a small MLP via PyTorch. Returns a callable that predicts on (n, d_in)."""
    import torch
    import torch.nn as nn
    # Avoid OpenMP conflict on macOS when libomp (from XGBoost) is system-installed.
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(seed)
    d_in = X.shape[1]
    d_out = Y.shape[1]
    model = nn.Sequential(
        nn.Linear(d_in, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, d_out),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    n = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = Xt[idx], Yt[idx]
            pred = model(xb)
            loss = ((pred - yb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    def _predict(X_new: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.tensor(X_new, dtype=torch.float32)).numpy()
    return _predict


_train_mlp = train_mlp


class _MLPEstimator:
    def __init__(self, **mlp_kwargs):
        self._kwargs = mlp_kwargs
        self._fn = None

    def fit(self, X, Y):
        self._fn = train_mlp(X.astype(np.float32), Y.astype(np.float32), **self._kwargs)
        return self

    def predict(self, X):
        return self._fn(np.asarray(X, dtype=np.float32))
