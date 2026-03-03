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
    
    df = pd.read_csv(file_path)
    
    # -------------------------------------------------
    # DATA CLEANING & TYPE CONVERSION
    # -------------------------------------------------
    
    # 1. Auto Detect Columns
    fraud_cols = ["Is Fraud?", "is_fraud", "fraud", "Class", "target"]
    amt_cols = ["Amount", "amount", "transaction_amount"]
    date_cols = ["Date", "date", "timestamp", "transaction_date"]
    
    f_col = next((c for c in fraud_cols if c in df.columns), None)
    a_col = next((c for c in amt_cols if c in df.columns), None)
    d_col = next((c for c in date_cols if c in df.columns), None)

    # 2. Fix Amount (The Error Fix)
    if a_col:
        # Remove '$' and ',' then convert to numeric
        df[a_col] = df[a_col].replace(r'[\$,]', '', regex=True)
        df[a_col] = pd.to_numeric(df[a_col], errors='coerce').fillna(0)

    # 3. Fix Fraud column if it's Yes/No or String
    if f_col:
        if df[f_col].dtype == object:
            df[f_col] = df[f_col].map({"Yes": 1, "No": 0, "1": 1, "0": 0})
        df[f_col] = df[f_col].fillna(0).astype(int)

    # 4. Fix Date Column
    if d_col:
        df[d_col] = pd.to_datetime(df[d_col], errors="coerce")

    return df, f_col, a_col, d_col

# Initialize Data
df, fraud_col, amount_col, date_col = load_data()

# Risk column detection (stays outside cache as it's simpler)
possible_risk_cols = ["risk_level", "Risk", "prediction"]
risk_col = next((col for col in possible_risk_cols if col in df.columns), None)

# =====================================================
# KPI CALCULATIONS
# =====================================================

total_tx = len(df)

# Calculate Fraud Metrics
fraud_tx = df[fraud_col].sum() if fraud_col else 0
fraud_rate = (fraud_tx / total_tx) * 100 if total_tx > 0 else 0

# Calculate Average Amount (This now works because amount is numeric)
avg_amount = df[amount_col].mean() if amount_col else 0

# Calculate High Risk
high_risk_tx = 0
if risk_col:
    high_risk_tx = len(df[df[risk_col].str.upper() == "HIGH"])

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

    # Grouping by date for the trend line
    trend_df = df.groupby(df[date_col].dt.date)[fraud_col].sum().reset_index()

    fig_trend = px.line(
        trend_df,
        x=date_col,
        y=fraud_col,
        markers=True,
        title="Daily Fraud Count",
        labels={fraud_col: "Number of Fraud Cases", date_col: "Date"}
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