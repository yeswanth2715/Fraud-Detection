import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report
import joblib

st.set_page_config(page_title="Goldilocks Model Performance", layout="wide")

st.title("🥇 Goldilocks Model Performance")

# ==============================
# Load Data
# ==============================

@st.cache_data
def load_data():
    return pd.read_csv("finance-risk-engine/data/User0_credit_card_transactions.csv")  # adjust if needed

df = load_data()

st.write("Available Columns:", df.columns.tolist())

# ==============================
# Detect Target Column
# ==============================

possible_targets = ["is_fraud", "fraud", "Class", "target", "label"]

target_column = None
for col in possible_targets:
    if col in df.columns:
        target_column = col
        break

if target_column is None:
    st.error("No fraud target column found in dataset.")
    st.stop()

X = df.drop(columns=[target_column])
y = df[target_column]

# ==============================
# Load Model
# ==============================

@st.cache_resource
def load_model():
    return joblib.load("models/model.joblib")

model = load_model()

# ==============================
# Predict Probabilities
# ==============================

y_prob = model.predict_proba(X)[:, 1]

# ==============================
# Threshold Slider
# ==============================

st.sidebar.header("⚙️ Threshold Tuning")

threshold = st.sidebar.slider(
    "Fraud Probability Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)

y_pred = (y_prob >= threshold).astype(int)

# ==============================
# KPI Section
# ==============================

col1, col2, col3 = st.columns(3)

fraud_rate = (y.sum() / len(y)) * 100
predicted_fraud_rate = (y_pred.sum() / len(y_pred)) * 100
accuracy = (y_pred == y).mean() * 100

col1.metric("Actual Fraud Rate %", f"{fraud_rate:.2f}%")
col2.metric("Predicted Fraud Rate %", f"{predicted_fraud_rate:.2f}%")
col3.metric("Model Accuracy %", f"{accuracy:.2f}%")

# ==============================
# Confusion Matrix
# ==============================

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

# ==============================
# Classification Report
# ==============================

st.subheader("Classification Report")

report = classification_report(y, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df)

# ==============================
# Probability Distribution
# ==============================

st.subheader("Fraud Probability Distribution")

fig_dist = px.histogram(
    y_prob,
    nbins=50,
    title="Distribution of Fraud Probabilities",
    labels={"value": "Fraud Probability"}
)

st.plotly_chart(fig_dist, use_container_width=True)

# ==============================
# Goldilocks Highlight
# ==============================

st.markdown("---")
st.success(
    f"""
    🥇 **Goldilocks Threshold Active:** {threshold}

    This threshold balances:
    - False Positives
    - False Negatives
    - Business Risk
    """
)