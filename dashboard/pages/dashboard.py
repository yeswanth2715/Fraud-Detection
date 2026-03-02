import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
st.title("📈 Fraud Risk Monitoring Dashboard")

# ======================
# Load Data
# ======================

df = pd.read_csv("data/User0_credit_card_transactions.csv")
df.columns = df.columns.str.strip().str.replace("_", " ")

# ======================
# Sidebar Filters
# ======================

st.sidebar.header("🔍 Filters")

risk_filter = st.sidebar.multiselect(
    "Select Risk Level",
    options=df["risk level"].unique(),
    default=df["risk level"].unique()
)

df = df[df["risk level"].isin(risk_filter)]

# ======================
# KPI CARDS
# ======================

total = len(df)
high = len(df[df["risk level"] == "HIGH"])
medium = len(df[df["risk level"] == "MEDIUM"])
low = len(df[df["risk level"] == "LOW"])

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Transactions", f"{total:,}")
col2.metric("High Risk", f"{high:,}")
col3.metric("Medium Risk", f"{medium:,}")
col4.metric("Low Risk", f"{low:,}")

st.divider()

# ======================
# Risk Distribution
# ======================

risk_counts = df["risk level"].value_counts().reset_index()
risk_counts.columns = ["Risk Level", "Transaction Count"]

fig = px.pie(
    risk_counts,
    names="Risk Level",
    values="Transaction Count",
    title="Risk Distribution"
)

fig.update_layout(template="plotly_dark")

st.plotly_chart(fig, use_container_width=True)

# ======================
# Top High Risk Merchants
# ======================

st.subheader("🏪 Top High Risk Merchants")

top_merchants = (
    df[df["risk level"] == "HIGH"]
    .groupby("merchant name")
    .size()
    .reset_index(name="High Risk Transactions")
    .sort_values("High Risk Transactions", ascending=False)
    .head(10)
)

fig2 = px.bar(
    top_merchants,
    x="High Risk Transactions",
    y="merchant name",
    orientation="h",
    title="Top 10 High Risk Merchants"
)

fig2.update_layout(template="plotly_dark", height=500)

st.plotly_chart(fig2, use_container_width=True)

# ======================
# Transaction Table
# ======================

st.subheader("📋 Transaction Details")

st.dataframe(df, use_container_width=True)