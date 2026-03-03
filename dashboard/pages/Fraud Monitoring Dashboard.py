import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os

# --- Page Config ---
st.set_page_config(page_title="Fraud Intelligence Command Center", layout="wide")

##Dashboard for monitoring fraud patterns and model performance in real-time. Pulls in pre-calculated metrics and transaction data to provide actionable insights.
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 28px; color: #0078D4; }
    .main { background-color: #f0f2f6; }
    div.stButton > button { width: 100%; background-color: #0078D4; color: white; }
    .plot-container { border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_all_data():
    # Load Main Data
    df = pd.read_csv(os.path.join("data", "User0_credit_card_transactions.csv"))
    df['Amount'] = df['Amount'].replace(r'[\$,]', '', regex=True).astype(float)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Is Fraud?'] = df['Is Fraud?'].map({"Yes": 1, "No": 0})
    
    # Load Model Metrics
    with open(os.path.join("metrics", "metrics.json"), "r") as f:
        metrics = json.load(f)
    
    return df, metrics

df, metrics = load_all_data()

# --- Header Section ---
st.title("🏢 Fraud Intelligence Command Center")
st.markdown("Real-time Model Performance & Transaction Analysis")
st.divider()

# --- Top Row: KPIs (The Scorecard) ---
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Transactions", f"{len(df):,}")
col2.metric("Fraud Detected", f"{int(df['Is Fraud?'].sum()):,}")
col3.metric("Model ROC-AUC", f"{metrics['roc_auc']:.3f}") # From metrics.json
col4.metric("Avg Transaction", f"${df['Amount'].mean():.2f}")
col5.metric("Optimal Threshold", f"{metrics['best_threshold']:.3f}") # From metrics.json

st.divider()

# --- Middle Row: Trends and Distribution ---
row2_left, row2_right = st.columns([2, 1])

with row2_left:
    st.subheader("📈 Fraud & Volume Trend")
    # Stacked Area Chart (Volume vs Fraud)
    df['Day'] = df['Date'].dt.date
    daily_stats = df.groupby('Day').agg({'Amount': 'count', 'Is Fraud?': 'sum'}).reset_index()
    daily_stats.columns = ['Date', 'Total Volume', 'Fraud Cases']
    
    fig_trend = px.area(daily_stats, x='Date', y=['Total Volume', 'Fraud Cases'],
                        title="Transaction Volume vs. Fraud Frequency",
                        color_discrete_sequence=['#0078D4', '#D13438'],
                        template="plotly_white")
    st.plotly_chart(fig_trend, use_container_width=True)

with row2_right:
    st.subheader("🍩 Risk Distribution")
    # Using 'Use Chip' or 'Merchant State' as a proxy for categorical analysis
    top_states = df['Merchant State'].value_counts().head(5)
    fig_pie = px.pie(values=top_states.values, names=top_states.index, hole=0.6,
                     color_discrete_sequence=px.colors.sequential.RdBu)
    fig_pie.update_layout(showlegend=False)
    st.plotly_chart(fig_pie, use_container_width=True)

# --- Bottom Row: Model Health & Matrix ---
row3_left, row3_right = st.columns(2)

with row3_left:
    st.subheader("📊 Confusion Matrix (Model Accuracy)")
    cm = np.array(metrics['confusion_matrix'])
    fig_cm = px.imshow(cm, text_auto=True, 
                       x=['Predicted Safe', 'Predicted Fraud'],
                       y=['Actual Safe', 'Actual Fraud'],
                       color_continuous_scale='Blues')
    st.plotly_chart(fig_cm, use_container_width=True)

with row3_right:
    st.subheader("📉 Prediction Confidence")
    # Histogram of amounts filtered by fraud status (Stacked Bar effect)
    fig_hist = px.histogram(df, x="Amount", color="Is Fraud?", 
                            nbins=50, barmode="stack",
                            title="Transaction Value Distribution by Class",
                            color_discrete_map={0: "#0078D4", 1: "#D13438"})
    st.plotly_chart(fig_hist, use_container_width=True)

st.success(f"Dashboard Model v1.0 (Threshold: {metrics['best_threshold']:.4f})")