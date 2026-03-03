import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os

# --- Page Config & Styling ---
st.set_page_config(page_title="Fraud Command Center", layout="wide")

st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 32px; color: #0078D4; font-weight: bold; }
    .main { background-color: #f8f9fa; }
    div.stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_all_assets():
    # 1. Load Transaction Data
    df = pd.read_csv(os.path.join("data", "User0_credit_card_transactions.csv"))
    
    # 2. Robust Column Detection (Fixes KeyError)
    date_col = next((c for c in ["Date", "date", "timestamp"] if c in df.columns), None)
    target_col = next((c for c in ["Is Fraud?", "is_fraud", "Class"] if c in df.columns), None)
    amt_col = next((c for c in ["Amount", "amount"] if c in df.columns), None)
    
    # Data Cleaning
    if amt_col:
        df[amt_col] = df[amt_col].replace(r'[\$,]', '', regex=True).astype(float)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    if target_col and df[target_col].dtype == object:
        df[target_col] = df[target_col].map({"Yes": 1, "No": 0})
    
    # 3. Load Metrics (Fixes Probability Error)
    with open(os.path.join("metrics", "metrics.json"), "r") as f:
        metrics = json.load(f)
    
    return df, metrics, date_col, target_col, amt_col

df, metrics, date_col, target_col, amt_col = load_all_assets()

# --- Dashboard Header ---
st.title("🏢 Fraud Intelligence Command Center")
st.markdown("Power BI-Style Performance & Operational Overview")

# --- Row 1: KPI Scorecards ---
col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.metric("Total Transactions", f"{len(df):,}")
with col2: st.metric("Detected Fraud", f"{int(df[target_col].sum()):,}")
with col3: st.metric("Model ROC-AUC", f"{metrics['roc_auc']:.3f}")
with col4: st.metric("F1 Score", f"{metrics['f1_score']:.3f}")
with col5: st.metric("Avg Amount", f"${df[amt_col].mean():.2f}")

st.divider()

# --- Row 2: Trend & Distribution ---
left_chart, right_chart = st.columns([2, 1])

with left_chart:
    st.subheader("📈 Fraud & Volume Trend")
    # Aggregating daily data
    daily = df.groupby(df[date_col].dt.date).agg({amt_col: 'count', target_col: 'sum'}).reset_index()
    daily.columns = ['Date', 'Volume', 'Fraud']
    
    fig_trend = px.area(daily, x='Date', y=['Volume', 'Fraud'], 
                        color_discrete_sequence=['#0078D4', '#D13438'],
                        title="Daily Transaction Activity vs Fraud Detection")
    st.plotly_chart(fig_trend, use_container_width=True)

with right_chart:
    st.subheader("🍩 Merchant Risk")
    # Top merchants by transaction count
    top_merch = df['Merchant Name'].value_counts().head(5)
    fig_pie = px.pie(values=top_merch.values, names=top_merch.index, hole=0.5,
                     title="Top 5 Merchant Volume",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_pie, use_container_width=True)

# --- Row 3: Model Health ---
matrix_col, hist_col = st.columns(2)

with matrix_col:
    st.subheader("📊 Confusion Matrix")
    cm = np.array(metrics['confusion_matrix'])
    fig_cm = px.imshow(cm, text_auto=True,
                       x=['Safe', 'Fraud'], y=['Safe', 'Fraud'],
                       labels=dict(x="Predicted", y="Actual"),
                       color_continuous_scale="Blues")
    st.plotly_chart(fig_cm, use_container_width=True)

with hist_col:
    st.subheader("💵 Value Distribution")
    fig_hist = px.histogram(df, x=amt_col, color=target_col, 
                            nbins=40, barmode='stack',
                            color_discrete_map={0: "#0078D4", 1: "#D13438"},
                            title="Transaction Amounts by Class")
    st.plotly_chart(fig_hist, use_container_width=True)

st.success(f"Dashboard model v1.0. Optimal Threshold: {metrics['best_threshold']:.4f}")