"""Shared metrics & plotting helpers used across the notebooks."""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# RUL metrics
# --------------------------------------------------------------------------- #
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def cmapss_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Official C-MAPSS asymmetric score.

    Penalises *late* predictions (predicting more life than remains) more than
    early ones — under-maintenance is costlier than over-maintenance. Lower is
    better.
    """
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


# --------------------------------------------------------------------------- #
# Standardisation (fit on train, apply everywhere) — avoids sklearn coupling
# --------------------------------------------------------------------------- #
class Standardizer:
    """Minimal mean/std scaler so the transform is explicit and portable."""

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_loss(history, title: str = "Training loss"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    if isinstance(history, dict):
        ax.plot(history["train"], label="train")
        if history.get("val"):
            ax.plot(history["val"], label="val")
        ax.legend()
    else:
        ax.plot(history, label="loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    return ax


def plot_rul_scatter(y_true: np.ndarray, y_pred: np.ndarray):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.5, s=15)
    lim = max(np.max(y_true), np.max(y_pred))
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    ax.set_xlabel("true RUL")
    ax.set_ylabel("predicted RUL")
    ax.set_title("Predicted vs true RUL")
    return ax
