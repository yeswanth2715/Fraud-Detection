import sys
import os
sys.path.append(os.path.abspath("."))

import streamlit as st
import pandas as pd
import plotly.express as px
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "data/User0_credit_card_transactions.csv")

st.set_page_config(layout="wide")
st.title("📊 Fraud Monitoring Dashboard")

df = pd.read_csv(DATA_PATH)

# ------------------------
# Sidebar Filters
# ------------------------

st.sidebar.header("Filters")
risk_filter = st.sidebar.multiselect(
    "Risk Level",
    df["risk_level"].unique(),
    default=df["risk_level"].unique()
)

df = df[df["risk_level"].isin(risk_filter)]

# ------------------------
# KPI Cards
# ------------------------

total = len(df)
high = len(df[df["risk_level"] == "HIGH"])
medium = len(df[df["risk_level"] == "MEDIUM"])
low = len(df[df["risk_level"] == "LOW"])
fraud_rate = (high / total) * 100 if total > 0 else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Transactions", f"{total:,}")
c2.metric("High Risk", f"{high:,}")
c3.metric("Medium Risk", f"{medium:,}")
c4.metric("Low Risk", f"{low:,}")
c5.metric("Fraud Rate %", f"{fraud_rate:.2f}%")

st.divider()

# ------------------------
# Risk Distribution
# ------------------------

risk_counts = df["risk_level"].value_counts().reset_index()
risk_counts.columns = ["Risk Level", "Transaction Count"]

fig = px.pie(
    risk_counts,
    names="Risk Level",
    values="Transaction Count",
    title="Risk Distribution"
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------
# Anomaly Time Series
# ------------------------

st.subheader("📈 High Risk Trend")

trend = (
    df.groupby("transaction_date")["risk_level"]
    .apply(lambda x: (x == "HIGH").sum())
    .reset_index(name="High Risk Count")
)

fig2 = px.line(
    trend,
    x="transaction_date",
    y="High Risk Count",
    title="High Risk Transactions Over Time"
)

st.plotly_chart(fig2, use_container_width=True)

# ------------------------
# Top Merchants
# ------------------------

st.subheader("🏪 Top High Risk Merchants")

top_merchants = (
    df[df["risk_level"] == "HIGH"]
    .groupby("merchant_name")
    .size()
    .reset_index(name="High Risk Transactions")
    .sort_values("High Risk Transactions", ascending=False)
    .head(10)
)

fig3 = px.bar(
    top_merchants,
    x="High Risk Transactions",
    y="merchant_name",
    orientation="h"
)

st.plotly_chart(fig3, use_container_width=True)

st.dataframe(df, use_container_width=True)