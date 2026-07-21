# DataPulse — Analytics Backend

A Flask-based analytics API that turns raw retail sales data into actionable insight.
Built for **CEWIT 2024** on the classic [Superstore](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final) dataset (~10k orders), it exposes four machine-learning services behind a small REST interface, with Redis caching in front of every expensive computation.

> This repository is the **backend**. It serves JSON (mostly [Plotly](https://plotly.com/python/) figure specs) that a companion frontend renders as interactive charts.

---

## Features

| Endpoint | What it does | Under the hood |
|---|---|---|
| `POST /create_rfm` | **Customer / segment clustering.** Computes Recency–Frequency–Monetary metrics for a chosen dimension and groups them into behavioural tiers (Diamond → Iron). | `KMeans` with automatic *k* selection via the elbow method (`kneed`) |
| `POST /predict_sales` | **Sales forecasting.** Projects future sales for the next *N* days, optionally filtered by category and/or state. | Facebook `Prophet` time-series model |
| `POST /similar_products` | **Product recommendations.** Returns products most similar to a given one, as a Plotly network graph. | GloVe word embeddings + cosine similarity, laid out with `networkx` |
| `POST /generate_text` | **Conversational text generation.** A lightweight assistant response for a given prompt. | Google `gemma-2b`, 4-bit quantized (`bitsandbytes`) |
| `GET /` | Health check. | — |

Every response is cached in Redis for 7 days, keyed by the request parameters, so repeated queries are served instantly.

---

## Architecture

```
                        ┌──────────────────────────────┐
   HTTP / JSON  ─────▶  │  run.py  (Flask + CORS)       │
                        │  · routing                    │
                        │  · Redis read-through cache   │
                        └───────────────┬──────────────┘
                                        │
                ┌───────────────────────┼───────────────────────┐
                ▼                       ▼                       ▼
        analytics/k_means.py   analytics/timeseries.py   analytics/similar_
        (RFM + clustering)     (Prophet forecasting)     product_prediction.py
                                                         + analytics/gemma.py
                                        │
                                        ▼
                            dataset/*.csv  (pre-built)
```

- **`run.py`** — the Flask app: routes, JSON serialization, and the Redis cache layer.
- **`serve.py`** — production entrypoint that boots `gunicorn` (gevent workers) reading config from env vars.
- **`analytics/`** — one module per ML feature. These run at request time against the pre-built CSVs in `dataset/`.
- **`utils/`** — the **offline data pipeline** that builds those CSVs from the raw `storesales.csv` and GloVe embeddings (see below). You only need these to regenerate the datasets.
- **`constants.py`** — the tier label map used to name customer segments.

---

## Data pipeline (`utils/`)

The files in `dataset/` are already generated, so the API runs out of the box. The `utils/` scripts document (and reproduce) how they were built:

| Script | Produces |
|---|---|
| `wordnet.py` | Embeds product text with GloVe and computes each product's top-similar SKUs → `products.csv` |
| `products.py` | Alternate similarity build using `gensim` cosine similarity |
| `clean_products.py` | Trims embedding columns → `clean_products.csv` |
| `clean_sales.py` | Drops unused columns from raw sales → `clean_sales.csv` |
| `preprocess_sales.py` | Encodes categoricals & engineers RFM features → `prod.csv` |
| `geo_data.py` → `geo_json_to_csv.py` | Aggregates sales by state/city/date/category → `geo_data.json` → `geo_data.csv` |
| `similar_products.py` | Groups SKUs by category/sub-category → `similar_product_data.json` |

> Regenerating similarity data requires the GloVe vectors (`glove.6B.100d.txt`), which are **not** committed (`.gitignore`d). Download them from [Stanford NLP](https://nlp.stanford.edu/projects/glove/) into `dataset/`.

---

## Getting started

### Prerequisites
- Python 3.10+
- A Redis instance (local or hosted)
- *(Optional)* an NVIDIA GPU with CUDA for the `/generate_text` Gemma endpoint

### Setup

```bash
# 1. Clone
git clone https://github.com/jlinnnn/DataPulse-Backed.git
cd DataPulse-Backed

# 2. Virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Dependencies
pip install -r requirements.txt

# 4. Configuration
cp .env.example .env      # then edit with your Redis credentials
```

The app reads `REDIS_HOST`, `REDIS_PORT`, and `REDIS_PASSWORD` from the environment
(defaults: `localhost:6379`, no password).

### Run

```bash
# Development
python run.py                 # http://localhost:5000

# Production (gunicorn via the service shell)
python serve.py
```

---

## API examples

```bash
# Cluster customers by their RFM behaviour
curl -X POST http://localhost:5000/create_rfm \
  -H "Content-Type: application/json" \
  -d '{"option": "customer_id"}'
# option ∈ {customer_id, state, city, segment, ship_mode, category, sub_category}

# Forecast the next 90 days of Furniture sales in California
curl -X POST http://localhost:5000/predict_sales \
  -H "Content-Type: application/json" \
  -d '{"period": 90, "category": "Furniture", "state": "California"}'

# Find products similar to a given one
curl -X POST http://localhost:5000/similar_products \
  -H "Content-Type: application/json" \
  -d '{"product": "Xerox 1967"}'

# Generate a short assistant reply
curl -X POST http://localhost:5000/generate_text \
  -H "Content-Type: application/json" \
  -d '{"text": "Summarize why customer retention matters."}'
```

Responses for the first three endpoints are Plotly figure JSON, ready to hand to `Plotly.newPlot` on the frontend.

---

## Tech stack

**Web** — Flask · Flask-CORS · gunicorn / gevent · waitress
**ML / Data** — scikit-learn · Prophet · gensim (GloVe) · Transformers + bitsandbytes (Gemma-2b) · pandas · NumPy · NetworkX · kneed
**Viz** — Plotly
**Cache** — Redis

---

## Project layout

```
.
├── run.py                 # Flask app + routes + Redis cache
├── serve.py               # gunicorn production entrypoint
├── constants.py           # segment tier labels
├── requirements.txt
├── analytics/             # request-time ML services
│   ├── k_means.py         # RFM + KMeans clustering
│   ├── timeseries.py      # Prophet forecasting
│   ├── similar_product_prediction.py
│   └── gemma.py           # Gemma-2b text generation
├── utils/                 # offline data-prep pipeline
├── scripts/               # deployment helpers (systemd + nginx)
└── dataset/               # pre-built CSV/JSON inputs
```

---

## Notes

- This began as a hackathon project; the `analytics/` and `utils/` folders are the original submission, tidied up here for readability.
- `scripts/install_app.sh` / `run_server.sh` target an Ubuntu + systemd + nginx deployment and assume a fixed install path — treat them as a reference rather than a turnkey installer.
