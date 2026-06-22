"""iiot-predictive-maintenance: reusable ML pipeline for PdM + anomaly detection.

Notebooks import from this package so logic stays versioned and testable.

Submodules are imported explicitly by the caller (e.g. ``from src import data``)
rather than eagerly here, so the data/feature steps don't drag in torch.
"""

