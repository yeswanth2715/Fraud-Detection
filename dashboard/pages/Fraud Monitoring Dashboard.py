import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os

# --- Page Config ---
st.set_page_config(page_title="Fraud Intelligence Command Center", layout="wide")

# --- Custom Styling for Power BI Look ---
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 32px; color: #0078D4; font-weight: bold; }
    .main { background-color: #f8f9fa; }
    div.stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_all_assets():
    # 1. Load Data
    data_path = os.path.join("data", "User0_credit_card_transactions.csv")
    df = pd.read_csv(data_path)
    
    # 2. Fix KeyError: Robust Column Detection
    # These find the right column even if the name varies (e.g., 'date' vs 'Date')
    d_col = next((c for c in ["Date", "date", "timestamp"] if c in df.columns), None)
    f_col = next((c for c in ["Is Fraud?", "is_fraud", "fraud", "Class"] if c in df.columns), None)
    a_col = next((c for c in ["Amount", "amount", "transaction_amount"] if c in df.columns), None)
    
    # 3. Clean and Format Data
    if a_col:
        df[a_col] = df[a_col].replace(r'[\$,]', '', regex=True).astype(float)
    if d_col:
        df[d_col] = pd.to_datetime(df[d_col], errors='coerce')
    if f_col and df[f_col].dtype == object:
        df[f_col] = df[f_col].map({"Yes": 1, "No": 0})

    # 4. Load Metrics from metrics.json (Avoids Probability Error)
    metrics_path = os.path.join("metrics", "metrics.json")
    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)
    
    return df, metrics_data, d_col, f_col, a_col

# Initialize Assets
try:
    df, metrics, date_col, fraud_col, amount_col = load_all_assets()
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()

# --- HEADER ---
st.title("🏢 Fraud Intelligence Command Center")
st.markdown("Real-time Model Performance & Transaction Analysis")
st.divider()

# --- ROW 1: KPI CARDS ---
col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.metric("Total Transactions", f"{len(df):,}")
with col2: st.metric("Fraud Transactions", f"{int(df[fraud_col].sum()):,}")
with col3: st.metric("Model ROC-AUC", f"{metrics['roc_auc']:.3f}")
with col4: st.metric("Avg Amount", f"${df[amount_col].mean():.2f}")
with col5: st.metric("Optimal Threshold", f"{metrics['best_threshold']:.3f}")

st.divider()

# --- ROW 2: TRENDS AND CATEGORIES ---
row2_left, row2_right = st.columns([2, 1])

with row2_left:
    st.subheader("📈 Fraud & Volume Trend")
    # Aggregate daily stats for the Stacked Area Graph
    daily = df.groupby(df[date_col].dt.date).agg({amount_col: 'count', fraud_col: 'sum'}).reset_index()
    daily.columns = ['Date', 'Total Volume', 'Fraud Cases']
    
    fig_trend = px.area(daily, x='Date', y=['Total Volume', 'Fraud Cases'],
                        title="Daily Activity vs Fraud Frequency",
                        color_discrete_sequence=['#0078D4', '#D13438'],
                        template="plotly_white")
    st.plotly_chart(fig_trend, use_container_width=True)

with row2_right:
    st.subheader("🍩 Risk Distribution")
    # Pie chart showing Merchant Risk concentration
    top_merch = df['Merchant Name'].value_counts().head(5)
    fig_pie = px.pie(values=top_merch.values, names=top_merch.index, hole=0.5,
                     title="Top 5 Merchant Volume",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_pie, use_container_width=True)

# --- ROW 3: PERFORMANCE DETAILS ---
row3_left, row3_right = st.columns(2)

with row3_left:
    st.subheader("📊 Confusion Matrix")
    cm = np.array(metrics['confusion_matrix'])
    fig_cm = px.imshow(cm, text_auto=True,
                       x=['Predicted Safe', 'Predicted Fraud'],
                       y=['Actual Safe', 'Actual Fraud'],
                       color_continuous_scale="Blues")
    st.plotly_chart(fig_cm, use_container_width=True)

with row3_right:
    st.subheader("📉 Transaction Distribution")
    # Stacked histogram for value analysis
    fig_hist = px.histogram(df, x=amount_col, color=fraud_col, 
                            nbins=40, barmode='stack',
                            color_discrete_map={0: "#0078D4", 1: "#D13438"},
                            title="Transaction Amounts by Fraud Status")
    st.plotly_chart(fig_hist, use_container_width=True)

st.success(f"Dashboard synchronized with Goldilocks Model (Threshold: {metrics['best_threshold']:.4f})")