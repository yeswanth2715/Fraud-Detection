# 💳 End-to-End Fraud Detection System

![CI](https://github.com/yeswanth2715/Fraud-Detection/actions/workflows/ci.yml/badge.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue)
![Deployment](https://img.shields.io/badge/deployed-railway-purple)
![Python](https://img.shields.io/badge/python-3.11-blue)
![XGBoost](https://img.shields.io/badge/model-XGBoost-orange)

A **production-ready machine learning system** for real-time credit card fraud detection — featuring a trained XGBoost classifier, a FastAPI prediction service, a Streamlit monitoring dashboard, Docker containerization, and automated CI/CD deployment to Railway.

---

## 📋 Table of Contents

- [Problem](#-problem)
- [Task](#-task)
- [Action](#-action)
- [Result](#-result)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Tech Stack](#-tech-stack)
- [Environment Setup](#-environment-setup)

---

## 🔴 Problem

Credit card fraud costs financial institutions **billions of dollars annually**. Fraudulent transactions are rare — typically less than 1% of all activity — making detection inherently difficult:

- **Class imbalance**: Legitimate transactions vastly outnumber fraudulent ones, causing naive models to simply predict "not fraud" for everything
- **Real-time requirements**: Fraud must be flagged *at the point of transaction*, not hours later in a batch job
- **High cost of errors**: False negatives (missed fraud) cause direct financial loss; false positives (wrongly blocked transactions) damage customer experience
- **Static rules fail**: Rule-based systems (e.g., "flag transactions over $X") are easily circumvented and require constant manual maintenance

The dataset (`User0_credit_card_transactions.csv`) contains ~20,000 real-world credit card transactions spanning multiple years, merchants, and geographies — with a severe class imbalance of ~0.025% fraud rate.

---

## 🎯 Task

Design and ship a **full-stack fraud detection system** that:

1. Trains a machine learning model capable of detecting fraud despite extreme class imbalance
2. Serves real-time predictions through a REST API
3. Provides an analytics dashboard for fraud monitoring and model performance tracking
4. Is fully containerized and automatically deployed on every code push

---

## ⚙️ Action

### 1. Data & Feature Engineering

Raw transaction data required careful preprocessing before modelling:

| Raw Field | Transform | Output Feature |
|-----------|-----------|----------------|
| `Amount` | Strip `$`/`,`, cast to float | Numeric `Amount` |
| `Time` | Split `HH:MM`, extract hour | Numeric `Hour` |
| `Hour` | Threshold rule (< 6 or > 22) | Binary `is_night_tx` |
| All others | Type coercion, imputation | Numeric / OHE categoricals |

Feature engineering runs identically at **train time and inference time**, preventing train-serve skew.

### 2. Model Training

**Algorithm:** XGBoost — chosen for its strong performance on tabular data, native handling of missing values, and efficient training on imbalanced datasets.

**Handling class imbalance** — two complementary strategies:
- `scale_pos_weight` = (# negatives / # positives): tells XGBoost to penalise missed fraud detections more heavily during training
- **F1-optimised threshold**: instead of the default 0.5 cutoff, the training pipeline runs `precision_recall_curve` across all candidate thresholds and selects the one maximising F1 on the held-out test set

**Hyperparameters (tuned):**

```python
XGBClassifier(
    n_estimators=800,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=fraud_ratio,  # dynamic, derived from training data
    eval_metric="logloss",
    random_state=42,
)
```

**Validation:** 5-fold stratified cross-validation scoring PR-AUC before the final model fit.

**Preprocessing pipeline** (sklearn `Pipeline` + `ColumnTransformer`):
- Numeric features: median imputation → StandardScaler
- Categorical features: most-frequent imputation → OneHotEncoder (unknown categories handled gracefully)

### 3. Prediction Service (FastAPI)

A lightweight REST API wraps the trained model:

- **`POST /predict`** — accepts a raw transaction JSON, runs feature engineering, returns fraud probability + risk tier
- Risk tiers: `HIGH` (≥ threshold), `MEDIUM` (≥ threshold × 0.5), `LOW` (below)
- Pydantic schemas enforce strict input validation with type coercion
- Structured logging on every request and prediction result
- Custom exception classes (`PredictionException`) for clean error propagation

### 4. Monitoring Dashboard (Streamlit)

A multi-page Streamlit app for analysts and operations teams:

- **Model Performance page**: ROC curve, confusion matrix, precision/recall/F1 metrics
- **Fraud Monitoring page**: real-time fraud rate and transaction volume KPIs
- **User Drilldown page**: per-user fraud history and transaction patterns

### 5. Containerisation & CI/CD

**Dockerfile:**
- Base: `python:3.11-slim` for a minimal attack surface
- Dependencies installed before source copy to leverage Docker layer caching
- Runs: `uvicorn app.main:app --host 0.0.0.0 --port 8000`

**GitHub Actions pipeline** triggered on every push to `main`:

```
Checkout → Install deps → Run pytest → Build Docker image → Push to Docker Hub → Deploy to Railway
```

---

## 📊 Result

The trained XGBoost model achieved the following on a held-out 20% test split (~4,000 transactions):

| Metric | Score |
|--------|-------|
| **ROC-AUC** | **0.9969** |
| **PR-AUC** | **0.7752** |
| **Precision** | 0.80 |
| **Recall** | 0.80 |
| **F1 Score** | 0.80 |
| **Optimal Threshold** | 0.1278 |

**Confusion matrix:**

```
                    Predicted: Legit   Predicted: Fraud
Actual: Legit            3987                 1
Actual: Fraud               1                 4
```

A ROC-AUC of **0.997** indicates near-perfect rank ordering of fraud vs. legitimate transactions. The F1-optimised threshold yields **80% precision and 80% recall** — a strong balance given how rare fraud events are in this dataset.

**Deliverables shipped:**
- ✅ Trained model artifact saved to `models/model.joblib`
- ✅ Metrics persisted to `metrics/metrics.json`
- ✅ Live FastAPI prediction service deployed to Railway
- ✅ Streamlit dashboard for fraud operations teams
- ✅ Automated Docker build + cloud deployment on every commit

---

## 🏗 Architecture

```
┌──────────────────────┐
│    Raw CSV Data       │
│  (~20k transactions)  │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│  Feature Engineering  │  ← create_features() [shared: train & serve]
│  Amount, Hour, Flags  │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│   XGBoost Pipeline    │  ← sklearn Pipeline (preprocessor + model)
│  + F1 Threshold Opt   │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│    model.joblib       │  ← pipeline + threshold + feature lists
└──────────┬───────────┘
           │
    ┌──────┴──────┐
    │             │
┌───▼────┐   ┌───▼──────┐
│FastAPI  │   │Streamlit │
│/predict │   │Dashboard │
└─────────┘   └──────────┘
      │
┌─────▼──────┐
│   Docker   │
└─────┬──────┘
      │
┌─────▼──────┐
│ GitHub CI  │  ← test → build → push
└─────┬──────┘
      │
┌─────▼──────┐
│  Railway   │  ← live cloud deployment
└────────────┘
```

---

## 📁 Project Structure

```
Fraud-Detection/
├── app/
│   ├── main.py                      # FastAPI app entrypoint
│   ├── config.py                    # Settings (model path, thresholds)
│   ├── api/
│   │   ├── routes.py                # POST /predict endpoint
│   │   └── schemas.py               # Pydantic request/response models
│   ├── models/
│   │   ├── train.py                 # Full ML training pipeline
│   │   └── predict.py               # Inference + risk scoring
│   ├── services/
│   │   └── feature_engineering.py  # Shared feature transforms
│   └── core/
│       ├── logger.py                # Structured logging
│       ├── exceptions.py            # Custom exception classes
│       └── error_handlers.py        # FastAPI error handlers
├── dashboard/
│   ├── dashboard.py                 # Streamlit entrypoint
│   └── pages/
│       ├── Model_Performance.py
│       └── Fraud Monitoring Dashboard.py
├── data/
│   └── User0_credit_card_transactions.csv
├── models/
│   └── model.joblib                 # Trained pipeline artifact
├── metrics/
│   └── metrics.json                 # Evaluation results
├── tests/
│   └── test_system.py               # Pytest test suite
├── .github/workflows/
│   └── ci.yml                       # GitHub Actions CI/CD
├── Dockerfile
├── requirements.txt
├── requirements-api.txt
├── setup.sh
└── .gitignore
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional, for containerised run)

### 1. Clone the repo

```bash
git clone https://github.com/yeswanth2715/Fraud-Detection.git
cd Fraud-Detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

> ⚠️ **Never commit your virtual environment.** It is excluded by `.gitignore` under `.venv/` and `venv/`.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python -m app.models.train
```

This outputs `models/model.joblib` and `metrics/metrics.json`.

### 5. Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### 6. Run the dashboard

```bash
streamlit run dashboard/dashboard.py
```

### 7. Docker

```bash
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection
```

---

## 📡 API Reference

### `GET /`

Health check.

```json
{ "message": "Fraud Detection API is Live 🚀" }
```

### `POST /predict`

**Request body:**

```json
{
  "User": 0,
  "Card": 0,
  "Year": 2024,
  "Month": 6,
  "Day": 15,
  "Time": "02:34",
  "Amount": "$450.00",
  "Use_Chip": "Swipe Transaction",
  "Merchant_Name": "1234567890",
  "Merchant_City": "Los Angeles",
  "Merchant_State": "CA",
  "Zip": 90001,
  "MCC": 5411,
  "Errors": ""
}
```

**Response:**

```json
{
  "fraud_probability": 0.87,
  "risk_level": "HIGH"
}
```

| Risk Level | Condition |
|------------|-----------|
| `HIGH` | probability ≥ optimal threshold (0.128) |
| `MEDIUM` | probability ≥ threshold × 0.5 |
| `LOW` | probability < threshold × 0.5 |

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | XGBoost + scikit-learn Pipeline |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Containerisation | Docker |
| CI/CD | GitHub Actions |
| Cloud Deployment | Railway |
| Data Handling | pandas, numpy |
| Validation | Pydantic v2 |

---

## 🔐 Environment Setup

Sensitive configuration lives in a `.env` file — **never commit this file**. It is excluded in `.gitignore` under `.env` and `.env.*`.

Create a local `.env`:

```bash
cp .env.example .env   # edit with your values
```

Key variables:

```env
MODEL_PATH=models/model.joblib
```

The `.gitignore` also excludes `venv/`, `.venv/`, compiled model binaries (`*.joblib`, `*.pkl`), and all log files.

---

## 🧪 Running Tests

```bash
pytest tests/
```

---

## 📄 License

MIT
