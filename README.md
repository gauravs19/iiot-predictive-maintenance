# IIoT Predictive Maintenance & Anomaly Detection

A notebook-based, end-to-end **machine-learning** demo for two core Industrial-IoT /
manufacturing problems, built from scratch on open datasets:

1. **Predictive maintenance** — *will this machine fail, and how soon?*
2. **Anomaly detection** — *is this machine behaving abnormally right now?*

Every notebook explains **each step in detail** (the *what*, the *why*, and how to
read the output), so it doubles as a learning resource — not just runnable code.

### ▶ Run the whole thing in one click
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gauravs19/iiot-predictive-maintenance/blob/main/notebooks/run_all.ipynb)
&nbsp; **[`run_all.ipynb`](notebooks/run_all.ipynb)** runs the entire pipeline
(data → features → PdM → anomaly detection) end-to-end. In Colab: `Runtime → Run all`.
Prefer the detailed step-by-step version? Open notebooks `00`–`03` in order.

> **Scope:** this project is the **ML core**. The GenAI layer (LLM-generated
> work-orders + RAG over a vector DB of incident signatures) lives in a separate
> sibling project, **`iiot-ai-rag`**, which *consumes* this project's model outputs.

---

## The ML → capability ladder

| Notebook | Capability | Techniques | Dataset |
|---|---|---|---|
| [`run_all`](notebooks/run_all.ipynb) | **Everything, end-to-end** | full pipeline in one notebook | both |
| [`00_setup_and_data`](notebooks/00_setup_and_data.ipynb) | Data ingestion | UCI + NASA loaders, validation | AI4I 2020, C-MAPSS |
| [`01_eda_and_features`](notebooks/01_eda_and_features.ipynb) | Feature engineering | RUL labels, rolling stats, sequence windows | C-MAPSS |
| [`02_predictive_maintenance`](notebooks/02_predictive_maintenance.ipynb) | Predictive maintenance | Random Forest + SHAP, **LSTM** RUL regression | AI4I + C-MAPSS |
| [`03_anomaly_detection`](notebooks/03_anomaly_detection.ipynb) | Anomaly detection | Isolation Forest, **Autoencoder** (reconstruction error) | C-MAPSS |

### Headline results (reproducible on a laptop CPU)
- **Failure classification (AI4I):** ROC-AUC ≈ **0.96**
- **Anomaly detection (C-MAPSS):** Isolation Forest AUC ≈ **1.00**, Autoencoder AUC ≈ **0.83**
- **RUL regression (C-MAPSS LSTM):** RMSE ≈ **15–20 cycles**

---

## Datasets (both open, downloaded at runtime)

- **AI4I 2020 Predictive Maintenance** — UCI ML Repository (id 601). 10,000 rows of
  synthetic milling-machine telemetry with failure labels. *Supervised classification.*
- **NASA C-MAPSS Turbofan Engine Degradation (FD001)** — run-to-failure multivariate
  sensor time series. *RUL regression + unsupervised anomaly detection.*

No data is committed to git; the loaders in `src/data.py` fetch and cache it.

---

## Run it

### Option A — Google Colab (zero setup, free GPU)
Click the **"Open in Colab"** badge at the top of any notebook. The first cell clones
this repo and installs dependencies automatically.

### Option B — Locally
```bash
git clone https://github.com/gauravs19/iiot-predictive-maintenance.git
cd iiot-predictive-maintenance
python -m venv .venv && . .venv/Scripts/activate   # Windows; use bin/activate on macOS/Linux
pip install -r requirements.txt
jupyter notebook            # then open notebooks/ in order 00 → 03
```

Run the notebooks **in order** (00 → 03); each is self-contained but they build on
the same concepts.

---

## Project layout

```
iiot-predictive-maintenance/
├── notebooks/                 # 00–03, heavily annotated, Colab-ready
├── src/                       # reusable, tested pipeline code
│   ├── data.py                #   dataset download + loading
│   ├── features.py            #   RUL labels, rolling features, sequence windows
│   ├── models.py              #   PyTorch LSTM regressor + autoencoder
│   └── utils.py               #   metrics (RMSE, C-MAPSS score), scaler, plots
├── data/                      # downloaded datasets (git-ignored)
├── artifacts/                 # saved models/outputs (git-ignored)
├── requirements.txt
└── README.md
```

Logic lives in `src/` (versioned, reusable, testable); notebooks stay thin and
explanatory.

---

## Where this fits a reference IIoT architecture

```
 Edge sensors ─▶ Ingest/Store ─▶ Feature eng. ─▶ ML models ─▶ Serving/Action
 (turbofan,      (nb 00)         (nb 01)         (nb 02/03)    └─▶ iiot-ai-rag
  milling)                                                         (LLM + RAG)
```

The **autoencoder's bottleneck** produces a compact vector "signature" of a machine's
state, and notebook 02 yields failure probabilities / RUL. Those are exactly the
inputs the sibling **`iiot-ai-rag`** project will embed into a vector DB (e.g.
Supabase + pgvector) to retrieve similar past incidents and generate maintenance
work-orders with an LLM.

---

## License & data attribution
Code: MIT (add a LICENSE file if publishing). Datasets belong to their original
authors (UCI ML Repository; NASA Prognostics Center of Excellence). Cite accordingly.
