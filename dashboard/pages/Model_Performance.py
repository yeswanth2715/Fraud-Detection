import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    confusion_matrix,
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
    df = pd.read_csv(file_path)
    
    # Cleaning Amount for consistent features
    if 'Amount' in df.columns and df['Amount'].dtype == object:
        df['Amount'] = df['Amount'].replace(r'[\$,]', '', regex=True).astype(float)
    
    return df

df = load_data()

# =====================================================
# DETECT FRAUD COLUMN & PREP FEATURES
# =====================================================
possible_targets = ["Is Fraud?", "is_fraud", "fraud", "Class", "target"]
target_col = next((col for col in possible_targets if col in df.columns), None)

if target_col is None:
    st.error("Fraud target column not found.")
    st.stop()

# Convert target to numeric
if df[target_col].dtype == object:
    df[target_col] = df[target_col].map({"Yes": 1, "No": 0, "1": 1, "0": 0})
df[target_col] = df[target_col].fillna(0)

# IMPORTANT: Drop non-numeric columns like 'Date' or 'Merchant' 
# if your model wasn't trained on them
X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
y = df[target_col]

# =====================================================
# LOAD MODEL & PROBABILITY HANDLING
# =====================================================
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "model.joblib")
    if not os.path.exists(model_path):
        st.error("Model file not found.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# Handle models that don't support predict_proba naturally
if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X)[:, 1]
elif hasattr(model, "decision_function"):
    # Fallback for SVM/Linear models: convert decision scores to 0-1 range
    scores = model.decision_function(X)
    y_prob = (scores - scores.min()) / (scores.max() - scores.min())
else:
    st.error("This model type does not support probability scoring. Please retrain with 'probability=True' (for SVM) or use a Logistic/Tree-based model.")
    st.stop()

# =====================================================
# THRESHOLD SLIDER
# =====================================================
st.sidebar.header("Threshold Control")
threshold = st.sidebar.slider(
    "Fraud Probability Threshold",
    0.0, 1.0, 0.5, 0.01,
    help="Higher threshold reduces False Positives but may miss actual fraud."
)

y_pred = (y_prob >= threshold).astype(int)

# =====================================================
# METRICS
# =====================================================
precision = precision_score(y, y_pred, zero_division=0)
recall = recall_score(y, y_pred, zero_division=0)
f1 = f1_score(y, y_pred, zero_division=0)
roc_auc = roc_auc_score(y, y_prob)

st.markdown("### Performance Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Precision", f"{precision:.3f}")
col2.metric("Recall", f"{recall:.3f}")
col3.metric("F1 Score", f"{f1:.3f}")
col4.metric("ROC-AUC", f"{roc_auc:.3f}")

st.markdown("---")

# =====================================================
# VISUALIZATIONS
# =====================================================
c1, c2 = st.columns(2)

with c1:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig_cm = px.imshow(
        cm, text_auto=True,
        labels=dict(x="Predicted", y="Actual"),
        x=["Non-Fraud", "Fraud"], y=["Non-Fraud", "Fraud"],
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

with c2:
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y, y_prob)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC: {roc_auc:.2f}"))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash"), name="Random"))
    fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig_roc, use_container_width=True)

st.subheader("Precision-Recall Curve")
p_vals, r_vals, _ = precision_recall_curve(y, y_prob)
fig_pr = px.line(x=r_vals, y=p_vals, labels={'x':'Recall', 'y':'Precision'}, title="PR Curve")
st.plotly_chart(fig_pr, use_container_width=True)

st.success("Evaluation complete.")