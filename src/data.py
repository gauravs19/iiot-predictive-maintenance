"""Data loading for the two open datasets used in this project.

Datasets
--------
1. AI4I 2020 Predictive Maintenance Dataset (UCI id=601)
   - 10,000 rows of synthetic milling-machine telemetry with failure labels.
   - Used for *supervised* failure classification.

2. NASA C-MAPSS Turbofan Engine Degradation (FD001 subset)
   - Multivariate run-to-failure sensor time series.
   - Used for Remaining-Useful-Life (RUL) regression and unsupervised anomaly detection.

Everything downloads at runtime into ./data/raw so nothing large lives in git.
Designed to work identically on a laptop and on Google Colab.
"""
from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

# Resolve ./data/raw relative to the repo root (parent of src/), works in Colab too.
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# C-MAPSS column schema: unit id, time cycle, 3 operational settings, 21 sensors.
CMAPSS_COLS = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Community mirror that hosts the raw C-MAPSS .txt files directly (the original
# NASA repository URL changes often). We fetch the three FD00x files we need.
_CMAPSS_RAW_BASE = "https://raw.githubusercontent.com/edwardzjl/CMAPSSData/master"


# --------------------------------------------------------------------------- #
# AI4I 2020 — supervised failure classification
# --------------------------------------------------------------------------- #
def load_ai4i() -> pd.DataFrame:
    """Return the AI4I 2020 dataset as a single tidy DataFrame.

    Tries the `ucimlrepo` package first (cleanest), then falls back to the
    public UCI CSV mirror so the notebook still runs without that dependency.
    """
    try:
        from ucimlrepo import fetch_ucirepo

        ds = fetch_ucirepo(id=601)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    except Exception as exc:  # pragma: no cover - network/dependency fallback
        print(f"[data] ucimlrepo unavailable ({exc}); using CSV mirror.")
        url = (
            "https://archive.ics.uci.edu/static/public/601/"
            "ai4i+2020+predictive+maintenance+dataset.zip"
        )
        raw = urllib.request.urlopen(url, timeout=60).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            with zf.open(csv_name) as fh:
                df = pd.read_csv(fh)

    # Normalise column names so both sources agree: strip unit suffixes like
    # "Air temperature [K]" -> "Air temperature". (ucimlrepo omits them; the CSV
    # mirror includes them.) Units are documented in the notebook instead.
    df.columns = [re.sub(r"\s*\[.*?\]", "", c).strip() for c in df.columns]
    return df


# --------------------------------------------------------------------------- #
# NASA C-MAPSS — RUL regression + anomaly detection
# --------------------------------------------------------------------------- #
def download_cmapss(subset: str = "FD001") -> Path:
    """Ensure C-MAPSS text files for a subset exist locally; return the data dir.

    Idempotent: files already present are skipped. Fetches the three raw files
    (train/test/RUL) individually from the community mirror.
    """
    files = [f"train_{subset}.txt", f"test_{subset}.txt", f"RUL_{subset}.txt"]
    if all((RAW_DIR / f).exists() for f in files):
        return RAW_DIR

    for fname in files:
        dest = RAW_DIR / fname
        if dest.exists():
            continue
        url = f"{_CMAPSS_RAW_BASE}/{fname}"
        try:
            print(f"[data] downloading {fname} ...")
            raw = urllib.request.urlopen(url, timeout=120).read()
            dest.write_bytes(raw)
        except Exception as exc:
            raise RuntimeError(
                f"Could not download {fname} from {url} ({exc}). Manually place "
                f"the C-MAPSS {subset} files into {RAW_DIR} "
                "(search 'NASA C-MAPSS Turbofan dataset')."
            ) from exc

    print(f"[data] C-MAPSS {subset} ready in {RAW_DIR}")
    return RAW_DIR


def load_cmapss(subset: str = "FD001") -> dict[str, pd.DataFrame]:
    """Load a C-MAPSS subset into train/test/rul DataFrames.

    Returns
    -------
    dict with keys ``train`` (run-to-failure), ``test`` (truncated), and
    ``rul`` (true remaining cycles for each test unit's final reading).
    """
    download_cmapss(subset)

    def _read(fname: str) -> pd.DataFrame:
        df = pd.read_csv(RAW_DIR / fname, sep=r"\s+", header=None)
        df = df.dropna(axis=1, how="all")  # trailing whitespace makes phantom cols
        df.columns = CMAPSS_COLS[: df.shape[1]]
        return df

    rul = pd.read_csv(RAW_DIR / f"RUL_{subset}.txt", sep=r"\s+", header=None)
    rul.columns = ["rul"]

    return {
        "train": _read(f"train_{subset}.txt"),
        "test": _read(f"test_{subset}.txt"),
        "rul": rul,
    }
