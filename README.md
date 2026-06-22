# IIoT Predictive Maintenance & Anomaly Detection

A notebook-based, end-to-end **machine-learning** demo for two core Industrial-IoT /
manufacturing problems, built from scratch on open datasets:

1. **Predictive maintenance** — *will this machine fail, and how soon?*
2. **Anomaly detection** — *is this machine behaving abnormally right now?*

The two use-case notebooks are written for **beginners** — they assume minimal
Machine-Learning knowledge and explain every concept, abbreviation, library, and chart
as they go (each ends with a glossary). They are **independent**: read either on its own.

### ▶ Open in Colab
| Notebook | What it teaches | |
|---|---|---|
| **[`predictive_maintenance.ipynb`](notebooks/predictive_maintenance.ipynb)** | *Will it fail, and how soon?* — supervised ML: Random Forest classification + LSTM RUL regression | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gauravs19/iiot-predictive-maintenance/blob/main/notebooks/predictive_maintenance.ipynb) |
| **[`anomaly_detection.ipynb`](notebooks/anomaly_detection.ipynb)** | *Is it behaving abnormally now?* — unsupervised ML: Isolation Forest + Autoencoder | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gauravs19/iiot-predictive-maintenance/blob/main/notebooks/anomaly_detection.ipynb) |
| [`run_all.ipynb`](notebooks/run_all.ipynb) | both pipelines, condensed, no explanations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gauravs19/iiot-predictive-maintenance/blob/main/notebooks/run_all.ipynb) |

In Colab, `Runtime → Run all` runs everything. The first cell clones the repo and
installs dependencies automatically.

> **Scope:** this project is the **ML core**. The GenAI layer (LLM-generated
> work-orders + RAG over a vector DB of incident signatures) lives in a separate
> sibling project, **`iiot-ai-rag`**, which *consumes* this project's model outputs.

---

## The two use cases

| Use case | Learning type | Question | Techniques | Dataset |
|---|---|---|---|---|
| **Predictive maintenance** | supervised | will it fail / how soon? | Random Forest + SHAP, **LSTM** RUL regression | AI4I 2020 + C-MAPSS |
| **Anomaly detection** | unsupervised | is it abnormal now? | Isolation Forest, **Autoencoder** (reconstruction error) | C-MAPSS |

Both share the data loaders and feature engineering in `src/`, but each notebook is
self-contained — it loads and explains the data it needs from scratch.

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
jupyter notebook            # then open either notebook in notebooks/
```

Open **`predictive_maintenance.ipynb`** or **`anomaly_detection.ipynb`** and run top
to bottom, or **`run_all.ipynb`** for the condensed version of both.

---

## Project layout

```
iiot-predictive-maintenance/
├── notebooks/                 # 2 beginner notebooks (one per use case) + run_all, Colab-ready
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
