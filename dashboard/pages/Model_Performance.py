import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import os

st.set_page_config(page_title="Goldilocks Model Performance", layout="wide")

st.title("🥇 Goldilocks Model Performance")

# ==========================================
# SAFE DATA LOADING (DEPLOYMENT READY)
# ==========================================

@st.cache_data
def load_data():
    file_path = os.path.join("data", "User0_credit_card_transactions.csv")
    if not os.path.exists(file_path):
        st.error(f"Data file not found at {file_path}")
        st.stop()
    return pd.read_csv(file_path)

df = load_data()

st.write("Available Columns:", df.columns.tolist())

# ==========================================
# TARGET COLUMN DETECTION
# ==========================================

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

# ==========================================
# SAFE MODEL LOADING
# ==========================================

@st.cache_resource
def load_model():
    model_path = os.path.join("models", "model.joblib")
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# ==========================================
# PREDICT PROBABILITIES
# ==========================================

if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X)[:, 1]
else:
    st.error("Model does not support predict_proba.")
    st.stop()

# ==========================================
# THRESHOLD SLIDER
# ==========================================

st.sidebar.header("⚙️ Threshold Tuning")

threshold = st.sidebar.slider(
    "Fraud Probability Threshold",
    0.0,
    1.0,
    0.5,
    0.01
)

y_pred = (y_prob >= threshold).astype(int)

# ==========================================
# KPI SECTION
# ==========================================

col1, col2, col3 = st.columns(3)

actual_fraud_rate = (y.sum() / len(y)) * 100
predicted_fraud_rate = (y_pred.sum() / len(y_pred)) * 100
accuracy = (y_pred == y).mean() * 100

col1.metric("Actual Fraud Rate %", f"{actual_fraud_rate:.2f}%")
col2.metric("Predicted Fraud Rate %", f"{predicted_fraud_rate:.2f}%")
col3.metric("Model Accuracy %", f"{accuracy:.2f}%")

# ==========================================
# CONFUSION MATRIX
# ==========================================

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

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

st.subheader("Classification Report")

report = classification_report(y, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df, use_container_width=True)

# ==========================================
# PROBABILITY DISTRIBUTION
# ==========================================

st.subheader("Fraud Probability Distribution")

fig_dist = px.histogram(
    y_prob,
    nbins=50,
    title="Distribution of Fraud Probabilities",
    labels={"value": "Fraud Probability"}
)

st.plotly_chart(fig_dist, use_container_width=True)

# ==========================================
# GOLDILOCKS HIGHLIGHT
# ==========================================

st.markdown("---")

st.success(
    f"""
    🥇 **Goldilocks Threshold Active:** {threshold}

    This threshold balances:
    • False Positives  
    • False Negatives  
    • Business Risk  
    """
)