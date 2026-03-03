import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Executive Overview", layout="wide")

st.title("🏢 Fraud Intelligence Executive Dashboard")

# =====================================================
# LOAD DATA (SAFE FOR DEPLOYMENT)
# =====================================================

@st.cache_data
def load_data():
    file_path = os.path.join("data", "User0_credit_card_transactions.csv")
    if not os.path.exists(file_path):
        st.error("Dataset not found in data/ folder.")
        st.stop()
    return pd.read_csv(file_path)

df = load_data()

# =====================================================
# AUTO DETECT IMPORTANT COLUMNS
# =====================================================

# Fraud column detection
possible_fraud_cols = ["Is Fraud?", "is_fraud", "fraud", "Class", "target"]
fraud_col = next((col for col in possible_fraud_cols if col in df.columns), None)

# Amount column detection
possible_amount_cols = ["Amount", "amount", "transaction_amount"]
amount_col = next((col for col in possible_amount_cols if col in df.columns), None)

# Date column detection
possible_date_cols = ["Date", "date", "timestamp", "transaction_date"]
date_col = next((col for col in possible_date_cols if col in df.columns), None)

# Risk column detection (optional)
possible_risk_cols = ["risk_level", "Risk", "prediction"]
risk_col = next((col for col in possible_risk_cols if col in df.columns), None)

# Convert fraud column if Yes/No
if fraud_col and df[fraud_col].dtype == object:
    df[fraud_col] = df[fraud_col].map({"Yes": 1, "No": 0})

# =====================================================
# KPI CALCULATIONS
# =====================================================

total_tx = len(df)

fraud_tx = df[fraud_col].sum() if fraud_col else 0
fraud_rate = (fraud_tx / total_tx) * 100 if fraud_col else 0

avg_amount = df[amount_col].mean() if amount_col else 0

high_risk_tx = 0
if risk_col:
    high_risk_tx = len(df[df[risk_col] == "HIGH"])

# =====================================================
# KPI DISPLAY (POWERBI STYLE)
# =====================================================

st.markdown("### Key Business Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Transactions", f"{total_tx:,}")
col2.metric("Fraud Transactions", f"{int(fraud_tx):,}")
col3.metric("Fraud Rate %", f"{fraud_rate:.2f}%")
col4.metric("High Risk Transactions", f"{high_risk_tx:,}")
col5.metric("Avg Transaction Amount", f"${avg_amount:,.2f}")

st.markdown("---")

# =====================================================
# FRAUD TREND OVER TIME
# =====================================================

if date_col and fraud_col:
    st.subheader("📈 Fraud Trend Over Time")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    trend_df = df.groupby(df[date_col].dt.date)[fraud_col].sum().reset_index()

    fig_trend = px.line(
        trend_df,
        x=date_col,
        y=fraud_col,
        markers=True,
        title="Daily Fraud Count",
    )

    fig_trend.update_layout(height=400)
    st.plotly_chart(fig_trend, use_container_width=True)

# =====================================================
# RISK DISTRIBUTION
# =====================================================

if risk_col:
    st.subheader("🍩 Risk Level Distribution")

    risk_counts = df[risk_col].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Count"]

    fig_risk = px.pie(
        risk_counts,
        names="Risk Level",
        values="Count",
        hole=0.5,
        color="Risk Level",
        color_discrete_map={
            "LOW": "#2ca02c",
            "MEDIUM": "#ff7f0e",
            "HIGH": "#d62728"
        }
    )

    fig_risk.update_layout(height=400)
    st.plotly_chart(fig_risk, use_container_width=True)

st.markdown("---")
st.success("Executive view showing high-level fraud intelligence metrics.")