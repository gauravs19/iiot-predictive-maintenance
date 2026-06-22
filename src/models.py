"""PyTorch models: an LSTM regressor for RUL and an autoencoder for anomaly scoring.

Both are intentionally compact, hand-written nn.Modules with small training
loops so the mechanics are visible. They train in minutes on CPU and faster on
a Colab GPU.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# LSTM — Remaining Useful Life regression
# --------------------------------------------------------------------------- #
class LSTMRegressor(nn.Module):
    """Two-layer LSTM -> dense head producing a single RUL value."""

    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, num_layers=layers,
            batch_first=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)  # last timestep -> RUL


def train_lstm(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_split: float = 0.1,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """Train an LSTM regressor with MSE loss; return train/val loss history."""
    device = get_device()
    model.to(device)

    n_val = int(len(X) * val_split)
    perm = np.random.permutation(len(X))
    tr, va = perm[n_val:], perm[:n_val]

    def loader(idx, shuffle):
        ds = TensorDataset(torch.tensor(X[idx]), torch.tensor(y[idx], dtype=torch.float32))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    tr_dl, va_dl = loader(tr, True), loader(va, False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = {"train": [], "val": []}

    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        history["train"].append(tot / len(tr))

        model.eval()
        vtot = 0.0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(device), yb.to(device)
                vtot += loss_fn(model(xb), yb).item() * len(xb)
        history["val"].append(vtot / max(len(va), 1))

        if verbose and (ep == 1 or ep % 5 == 0 or ep == epochs):
            print(f"epoch {ep:3d}  train_mse={history['train'][-1]:8.2f}  "
                  f"val_mse={history['val'][-1]:8.2f}")
    return history


def predict_lstm(model: nn.Module, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    device = get_device()
    model.to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i : i + batch_size]).to(device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------------- #
# Autoencoder — reconstruction-error anomaly detection
# --------------------------------------------------------------------------- #
class AutoEncoder(nn.Module):
    """Symmetric MLP autoencoder. High reconstruction error => anomalous."""

    def __init__(self, n_features: int, latent: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, latent), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def train_autoencoder(
    model: nn.Module,
    X: np.ndarray,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    verbose: bool = True,
) -> list[float]:
    """Train AE only on *healthy* data so anomalies reconstruct poorly later."""
    device = get_device()
    model.to(device)
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for (xb,) in dl:
            xb = xb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        history.append(tot / len(X))
        if verbose and (ep == 1 or ep % 10 == 0 or ep == epochs):
            print(f"epoch {ep:3d}  recon_mse={history[-1]:.5f}")
    return history


def reconstruction_error(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Per-sample mean squared reconstruction error = anomaly score."""
    device = get_device()
    model.to(device).eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32).to(device)
        recon = model(xb)
        return ((recon - xb) ** 2).mean(dim=1).cpu().numpy()
