import sys
import os
sys.path.append(os.path.abspath("."))

import streamlit as st
import pandas as pd
import joblib
import json
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ------------------------
# PATH FIX (No Errno 2)
# ------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models/model.joblib")
METRICS_PATH = os.path.join(PROJECT_ROOT, "metrics/metrics.json")
DATA_PATH = os.path.join(PROJECT_ROOT, "data/User0_credit_card_transactions.csv")

st.set_page_config(layout="wide")
st.title("🥇 Goldilocks Model Performance")

# ------------------------
# Load Model + Data
# ------------------------

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["is_fraud"])
y_true = df["is_fraud"]

# ------------------------
# Threshold Slider
# ------------------------

threshold = st.slider("Adjust Decision Threshold", 0.0, 1.0, 0.5, 0.01)

probs = model.predict_proba(X)[:, 1]
y_pred = (probs >= threshold).astype(int)

# ------------------------
# Metrics
# ------------------------

accuracy = (y_pred == y_true).mean()
precision = ((y_pred & y_true).sum()) / max(y_pred.sum(), 1)
recall = ((y_pred & y_true).sum()) / max(y_true.sum(), 1)
f1 = 2 * precision * recall / max((precision + recall), 1e-9)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy:.2%}")
col2.metric("Precision", f"{precision:.2%}")
col3.metric("Recall", f"{recall:.2%}")
col4.metric("F1 Score", f"{f1:.2%}")

st.success("🏆 Goldilocks Model: Balanced Precision & Recall")

st.divider()

# ------------------------
# Confusion Matrix
# ------------------------

cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

st.pyplot(fig)