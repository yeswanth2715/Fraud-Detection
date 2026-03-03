import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import joblib
import os

# --- Page Config ---
st.set_page_config(page_title="Fraud Intelligence Command Center", layout="wide")

# --- Custom Styling ---
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 32px; color: #0078D4; font-weight: bold; }
    .main { background-color: #f8f9fa; }
    div.stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_and_preprocess_data():
    # 1. Load Data
    data_path = "data/User0_credit_card_transactions.csv"
    if not os.path.exists(data_path):
        st.error("Dataset not found.")
        st.stop()
    df = pd.read_csv(data_path)
    
    # 2. Fix the Date Error
    # Merge Year, Month, Day columns found in your dataset into a single datetime
    if all(col in df.columns for col in ['Year', 'Month', 'Day']):
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    elif 'Date' not in df.columns:
        # Fallback if names are lowercase
        df['Date'] = pd.to_datetime({'year': df['year'], 'month': df['month'], 'day': df['day']})

    # 3. Clean Amount and Fraud Columns
    df['Amount'] = df['Amount'].replace(r'[\$,]', '', regex=True).astype(float)
    if df['Is Fraud?'].dtype == object:
        df['Is Fraud?'] = df['Is Fraud?'].map({"Yes": 1, "No": 0})
    
    # 4. Load Pre-calculated Metrics
    metrics_path = "metrics/metrics.json"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    return df, metrics

# Initialize
df, metrics = load_and_preprocess_data()

# --- HEADER ---
st.title("🏢 Fraud Intelligence Command Center")
st.markdown("Real-time Operational Performance & Model Health")
st.divider()

# --- ROW 1: KPI SCORECARDS ---
col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.metric("Total Transactions", f"{len(df):,}")
with col2: st.metric("Fraud Detected", f"{int(df['Is Fraud?'].sum()):,}")
with col3: st.metric("Model ROC-AUC", f"{metrics['roc_auc']:.3f}")
with col4: st.metric("Precision", f"{metrics['precision']:.3f}")
with col5: st.metric("F1 Score", f"{metrics['f1_score']:.3f}")

st.divider()

# --- ROW 2: TRENDS AND CATEGORIES ---
row2_left, row2_right = st.columns([2, 1])

with row2_left:
    st.subheader("📈 Fraud & Volume Trend")
    # Grouping by day for the Area Chart
    daily = df.groupby(df['Date'].dt.date).agg({'Amount': 'count', 'Is Fraud?': 'sum'}).reset_index()
    daily.columns = ['Date', 'Total Volume', 'Fraud Cases']
    
    fig_trend = px.area(daily, x='Date', y=['Total Volume', 'Fraud Cases'],
                        title="Daily Activity vs. Fraud Incidents",
                        color_discrete_sequence=['#0078D4', '#D13438'],
                        template="plotly_white")
    st.plotly_chart(fig_trend, use_container_width=True)

with row2_right:
    st.subheader("🍩 Merchant Distribution")
    top_merchants = df['Merchant Name'].value_counts().head(5)
    fig_pie = px.pie(values=top_merchants.values, names=top_merchants.index, hole=0.5,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_pie.update_layout(showlegend=False)
    st.plotly_chart(fig_pie, use_container_width=True)

# --- ROW 3: MODEL HEALTH VISUALS ---
row3_left, row3_right = st.columns(2)

with row3_left:
    st.subheader("📊 Confusion Matrix")
    # Rendering the matrix from metrics.json values
    cm = np.array(metrics['confusion_matrix'])
    fig_cm = px.imshow(cm, text_auto=True,
                       x=['Predicted Safe', 'Predicted Fraud'],
                       y=['Actual Safe', 'Actual Fraud'],
                       color_continuous_scale="Blues")
    st.plotly_chart(fig_cm, use_container_width=True)

with row3_right:
    st.subheader("📉 Transaction Value Analysis")
    fig_hist = px.histogram(df, x='Amount', color='Is Fraud?', 
                            nbins=40, barmode='stack',
                            color_discrete_map={0: "#0078D4", 1: "#D13438"},
                            title="Transaction Amounts by Class")
    st.plotly_chart(fig_hist, use_container_width=True)

st.success(f"Dashboard synchronized with Goldilocks XGBoost (Threshold: {metrics['best_threshold']:.4f})")