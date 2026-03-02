import sys
import os
sys.path.append(os.path.abspath("."))

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from app.services.feature_engineering import create_features

# -----------------------------
# Load Model
# -----------------------------
artifact = joblib.load("models/model.joblib")
pipeline = artifact["pipeline"]
threshold = artifact["threshold"]

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("data/User0_credit_card_transactions.csv")

df["Is Fraud?"] = df["Is Fraud?"].map({"Yes": 1, "No": 0})

df = create_features(df)

# -----------------------------
# Predict Risk for All Transactions
# -----------------------------
proba = pipeline.predict_proba(df.drop(columns=["Is Fraud?"]))[:, 1]
df["fraud_probability"] = proba

def assign_risk(prob):
    if prob >= threshold:
        return "HIGH"
    elif prob >= threshold * 0.5:
        return "MEDIUM"
    else:
        return "LOW"

df["risk_level"] = df["fraud_probability"].apply(assign_risk)

# -----------------------------
# DASHBOARD UI
# -----------------------------

st.title("💳 Fraud Risk Monitoring Dashboard")

total_tx = len(df)
fraud_tx = df["Is Fraud?"].sum()
fraud_rate = fraud_tx / total_tx

high_risk = (df["risk_level"] == "HIGH").sum()
medium_risk = (df["risk_level"] == "MEDIUM").sum()
low_risk = (df["risk_level"] == "LOW").sum()

col1, col2, col3 = st.columns(3)

col1.metric("Total Transactions", total_tx)
col2.metric("Fraud Transactions", fraud_tx)
col3.metric("Fraud Rate", f"{fraud_rate:.4f}")

st.subheader("Risk Distribution")

risk_counts = df["risk_level"].value_counts()

fig1, ax1 = plt.subplots()
risk_counts.plot(kind="bar", ax=ax1)
ax1.set_ylabel("Count")
st.pyplot(fig1)

st.subheader("Top High Risk Users")

high_risk_users = (
    df[df["risk_level"] == "HIGH"]
    .groupby("User")
    .size()
    .sort_values(ascending=False)
    .head(10)
)

st.dataframe(high_risk_users)

st.subheader("Top Risky Merchants")

top_merchants = (
    df[df["risk_level"] == "HIGH"]
    .groupby("Merchant Name")
    .size()
    .sort_values(ascending=False)
    .head(10)
)

st.dataframe(top_merchants)

st.subheader("Transaction Table")

st.dataframe(
    df[[
        "User",
        "Amount",
        "Merchant Name",
        "fraud_probability",
        "risk_level"
    ]].sort_values(by="fraud_probability", ascending=False).head(50)
)