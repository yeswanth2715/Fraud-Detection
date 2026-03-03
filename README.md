
---

```markdown
![CI](https://github.com/yeswanth2715/Fraud-Detection/actions/workflows/ci.yml/badge.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue)
![Deployment](https://img.shields.io/badge/deployed-railway-purple)
```




# 🚀 End-to-End Fraud Detection System

A production-ready machine learning system for fraud detection with automated CI/CD, Docker containerization, and live cloud deployment.


---

## 🏗 Architecture Overview

                ┌──────────────┐
                │   Dataset    │
                └──────┬───────┘
                       │
                ┌──────▼───────┐
                │  Training    │
                │  Pipeline    │
                └──────┬───────┘
                       │
                ┌──────▼───────┐
                │ Model Artifact│
                └──────┬───────┘
                       │
            ┌──────────▼──────────┐
            │ FastAPI + Streamlit │
            └──────────┬──────────┘
                       │
                 ┌─────▼─────┐
                 │  Docker   │
                 └─────┬─────┘
                       │
                ┌──────▼──────┐
                │ GitHub CI   │
                └──────┬──────┘
                       │
                ┌──────▼──────┐
                │  Railway    │
                └─────────────┘

---

## 🧠 Model

- Algorithm: XGBoost (Goldilocks tuned)
- Optimized Threshold Selection
- ROC-AUC Evaluation
- Confusion Matrix Analysis
- Precision / Recall Optimization

---

## 📊 Dashboard Features

- Executive KPI Overview
- Fraud Rate Monitoring
- Confusion Matrix
- ROC Curve Visualization

---

## ⚙️ Tech Stack

- Python
- FastAPI
- Streamlit
- XGBoost
- Scikit-learn
- Docker
- GitHub Actions (CI/CD)
- Railway (Cloud Deployment)

---

## 🔁 CI/CD Pipeline

On every push to `main`:

1. Run automated tests
2. Build Docker image
3. Push to Docker Hub
4. Auto-deploy to cloud

---

## 🐳 Docker

Build locally:

```bash
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection
