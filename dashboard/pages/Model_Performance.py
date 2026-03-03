import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score
)
import joblib
import os

st.set_page_config(page_title="Model Performance", layout="wide")

st.title("🧠 Goldilocks Model Performance")

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():
    file_path = os.path.join("data", "User0_credit_card_transactions.csv")
    if not os.path.exists(file_path):
        st.error("Dataset not found.")
        st.stop()
    return pd.read_csv(file_path)

df = load_data()

# =====================================================
# DETECT FRAUD COLUMN
# =====================================================

possible_targets = ["Is Fraud?", "is_fraud", "fraud", "Class", "target"]
target_col = next((col for col in possible_targets if col in df.columns), None)

if target_col is None:
    st.error("Fraud target column not found.")
    st.stop()

# Convert Yes/No if needed
if df[target_col].dtype == object:
    df[target_col] = df[target_col].map({"Yes": 1, "No": 0})

X = df.drop(columns=[target_col])
y = df[target_col]

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    model_path = os.path.join("models", "model.joblib")
    if not os.path.exists(model_path):
        st.error("Model file not found.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

if not hasattr(model, "predict_proba"):
    st.error("Model does not support probability prediction.")
    st.stop()

y_prob = model.predict_proba(X)[:, 1]

# =====================================================
# THRESHOLD SLIDER
# =====================================================

st.sidebar.header("Threshold Control")

threshold = st.sidebar.slider(
    "Fraud Probability Threshold",
    0.0, 1.0, 0.5, 0.01
)

y_pred = (y_prob >= threshold).astype(int)

# =====================================================
# METRICS
# =====================================================

precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
roc_auc = roc_auc_score(y, y_prob)

st.markdown("### Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Precision", f"{precision:.3f}")
col2.metric("Recall", f"{recall:.3f}")
col3.metric("F1 Score", f"{f1:.3f}")
col4.metric("ROC-AUC", f"{roc_auc:.3f}")

st.markdown("---")

# =====================================================
# CONFUSION MATRIX HEATMAP
# =====================================================

st.subheader("Confusion Matrix")

cm = confusion_matrix(y, y_pred)

fig_cm = px.imshow(
    cm,
    text_auto=True,
    labels=dict(x="Predicted", y="Actual"),
    x=["Non-Fraud", "Fraud"],
    y=["Non-Fraud", "Fraud"],
    color_continuous_scale="Blues"
)

st.plotly_chart(fig_cm, use_container_width=True)

# =====================================================
# ROC CURVE
# =====================================================

st.subheader("ROC Curve")

fpr, tpr, _ = roc_curve(y, y_prob)

fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Model"))
fig_roc.add_trace(go.Scatter(
    x=[0,1], y=[0,1],
    mode="lines",
    name="Random",
    line=dict(dash="dash")
))

fig_roc.update_layout(
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    height=400
)

st.plotly_chart(fig_roc, use_container_width=True)

# =====================================================
# PRECISION-RECALL CURVE
# =====================================================

st.subheader("Precision-Recall Curve")

precision_vals, recall_vals, _ = precision_recall_curve(y, y_prob)

fig_pr = go.Figure()
fig_pr.add_trace(go.Scatter(
    x=recall_vals,
    y=precision_vals,
    mode="lines",
    name="PR Curve"
))

fig_pr.update_layout(
    xaxis_title="Recall",
    yaxis_title="Precision",
    height=400
)

st.plotly_chart(fig_pr, use_container_width=True)

st.markdown("---")
st.success("Goldilocks model evaluation complete.")