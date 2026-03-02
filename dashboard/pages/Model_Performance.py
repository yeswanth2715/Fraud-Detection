import streamlit as st
import pandas as pd
import plotly.express as px
import json

st.set_page_config(layout="wide")
st.title("🥇 Goldilocks Model Performance")

# ======================
# Load Metrics
# ======================

with open("metrics/metrics.json", "r") as f:
    metrics = json.load(f)

accuracy = metrics["accuracy"]
precision = metrics["precision"]
recall = metrics["recall"]
f1 = metrics["f1_score"]
roc_auc = metrics["roc_auc"]

# ======================
# KPI CARDS
# ======================

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Accuracy", f"{accuracy:.2%}")
col2.metric("Precision", f"{precision:.2%}")
col3.metric("Recall", f"{recall:.2%}")
col4.metric("F1 Score", f"{f1:.2%}")
col5.metric("ROC AUC", f"{roc_auc:.2%}")

st.success("🏆 Goldilocks Model: Balanced Precision & Recall")

st.divider()

# ======================
# Metric Visualization
# ======================

metric_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
    "Score": [accuracy, precision, recall, f1, roc_auc]
})

fig = px.bar(
    metric_df,
    x="Metric",
    y="Score",
    color="Metric",
    text_auto=True,
    title="Model Performance Overview"
)

fig.update_layout(template="plotly_dark", height=500)

st.plotly_chart(fig, use_container_width=True)