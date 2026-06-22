"""Feature engineering for both datasets.

Kept deliberately small and explicit so the transformations are easy to read
and reason about — the point of this project is to *understand* the pipeline,
not hide it behind a framework.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Sensors that are essentially constant in FD001 and carry no signal.
CMAPSS_DEAD_SENSORS = [
    "sensor_1", "sensor_5", "sensor_6", "sensor_10",
    "sensor_16", "sensor_18", "sensor_19",
]


# --------------------------------------------------------------------------- #
# AI4I 2020
# --------------------------------------------------------------------------- #
def prepare_ai4i(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for AI4I machine-failure classification.

    - Drops identifier columns and the per-failure-mode flags (they leak the label).
    - One-hot encodes the machine ``Type`` (L/M/H quality variants).
    - ``y`` is the binary 'Machine failure' target.
    """
    df = df.copy()
    target = "Machine failure"

    drop_cols = [
        "UID", "Product ID",
        "TWF", "HDF", "PWF", "OSF", "RNF",  # individual failure modes -> leakage
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    y = df.pop(target).astype(int)

    if "Type" in df.columns:
        df = pd.get_dummies(df, columns=["Type"], prefix="type")

    # Tidy the long UCI column names into snake_case.
    df.columns = (
        df.columns.str.replace(r"\[.*?\]", "", regex=True)
        .str.strip()
        .str.replace(" ", "_")
        .str.lower()
    )
    return df, y


# --------------------------------------------------------------------------- #
# C-MAPSS — RUL labels + rolling features + sequence windows
# --------------------------------------------------------------------------- #
def add_rul(train: pd.DataFrame, clip: int | None = 125) -> pd.DataFrame:
    """Add a per-row Remaining-Useful-Life column to run-to-failure training data.

    RUL = (max cycle for that unit) - (current cycle). Optionally clipped, since
    degradation is only observable near end-of-life; a piecewise-linear RUL with
    a cap is the standard C-MAPSS convention and trains far better.
    """
    out = train.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = max_cycle - out["cycle"]
    if clip is not None:
        out["rul"] = out["rul"].clip(upper=clip)
    return out


def feature_columns(df: pd.DataFrame, drop_dead: bool = True) -> list[str]:
    """Sensor + op-setting columns to use as model inputs."""
    cols = [c for c in df.columns if c.startswith(("sensor_", "op_setting_"))]
    if drop_dead:
        cols = [c for c in cols if c not in CMAPSS_DEAD_SENSORS]
    return cols


def add_rolling_features(df: pd.DataFrame, cols: list[str], window: int = 5) -> pd.DataFrame:
    """Add per-unit rolling mean & std — cheap degradation-trend features."""
    out = df.sort_values(["unit", "cycle"]).copy()
    g = out.groupby("unit")[cols]
    out[[f"{c}_rmean" for c in cols]] = g.transform(
        lambda s: s.rolling(window, min_periods=1).mean()
    )
    out[[f"{c}_rstd" for c in cols]] = g.transform(
        lambda s: s.rolling(window, min_periods=1).std().fillna(0)
    )
    return out


def make_sequences(
    df: pd.DataFrame, cols: list[str], seq_len: int = 30, label: str = "rul"
) -> tuple[np.ndarray, np.ndarray]:
    """Build (N, seq_len, n_features) windows + their end-of-window label.

    Used to feed the LSTM. Units shorter than ``seq_len`` are left-padded with
    their first reading so every unit contributes at least one sequence.
    """
    X, y = [], []
    for _, unit in df.sort_values(["unit", "cycle"]).groupby("unit"):
        vals = unit[cols].to_numpy(dtype="float32")
        labels = unit[label].to_numpy(dtype="float32")
        if len(unit) < seq_len:
            pad = np.repeat(vals[:1], seq_len - len(unit), axis=0)
            vals = np.vstack([pad, vals])
            labels = np.concatenate([np.repeat(labels[:1], seq_len - len(unit)), labels])
        for i in range(len(vals) - seq_len + 1):
            X.append(vals[i : i + seq_len])
            y.append(labels[i + seq_len - 1])
    return np.asarray(X), np.asarray(y)


def last_sequence_per_unit(
    df: pd.DataFrame, cols: list[str], seq_len: int = 30
) -> np.ndarray:
    """Final window for each unit — used for test-set RUL prediction."""
    seqs = []
    for _, unit in df.sort_values(["unit", "cycle"]).groupby("unit"):
        vals = unit[cols].to_numpy(dtype="float32")
        if len(vals) < seq_len:
            pad = np.repeat(vals[:1], seq_len - len(vals), axis=0)
            vals = np.vstack([pad, vals])
        seqs.append(vals[-seq_len:])
    return np.asarray(seqs)
