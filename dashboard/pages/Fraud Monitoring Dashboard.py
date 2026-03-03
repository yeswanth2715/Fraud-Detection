import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os

# --- Page Config ---
st.set_page_config(page_title="Fraud Intelligence Command Center", layout="wide")

# --- Custom CSS for Power BI Aesthetic ---
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 32px; color: #0078D4; font-weight: bold; }
    .main { background-color: #f8f9fa; }
    div.stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_all_assets():
    # 1. Load Main Data
    data_path = os.path.join("data", "User0_credit_card_transactions.csv")
    if not os.path.exists(data_path):
        st.error("Dataset not found.")
        st.stop()
    df = pd.read_csv(data_path)
    
    # 2. Robust Column Detection (Fixes KeyError)
    d_col = next((c for c in ["Date", "date", "timestamp"] if c in df.columns), None)
    f_col = next((c for c in ["Is Fraud?", "is_fraud", "fraud", "Class"] if c in df.columns), None)
    a_col = next((c for c in ["Amount", "amount", "transaction_amount"] if c in df.columns), None)
    
    if not all([d_col, f_col, a_col]):
        st.error(f"Missing required columns. Found: {list(df.columns)}")
        st.stop()

    # 3. Data Cleaning & Type Conversion
    df[a_col] = df[a_col].replace(r'[\$,]', '', regex=True).astype(float)
    df[d_col] = pd.to_datetime(df[d_col], errors='coerce')
    if df[f_col].dtype == object:
        df[f_col] = df[f_col].map({"Yes": 1, "No": 0, "1": 1, "0": 0})
    df[f_col] = df[f_col].fillna(0).astype(int)

    # 4. Load Pre-calculated Metrics (Fixes Model Probability Error)
    metrics_path = os.path.join("metrics", "metrics.json")
    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)
    
    return df, metrics_data, d_col, f_col, a_col

# Initialize Application Assets
df, metrics, date_col, fraud_col, amount_col = load_all_assets()

# --- HEADER ---
st.title("🏢 Fraud Intelligence Command Center")
st.markdown("Real-time Operational Performance & Model Health")
st.divider()

# --- ROW 1: KPI SCORECARDS ---
col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.metric("Total Transactions", f"{len(df):,}")
with col2: st.metric("Fraud Detected", f"{int(df[fraud_col].sum()):,}")
with col3: st.metric("Model ROC-AUC", f"{metrics['roc_auc']:.3f}")
with col4: st.metric("Precision", f"{metrics['precision']:.3f}")
with col5: st.metric("F1 Score", f"{metrics['f1_score']:.3f}")

st.divider()

# --- ROW 2: TRENDS AND CATEGORICAL DISTRIBUTION ---
row2_left, row2_right = st.columns([2, 1])

with row2_left:
    st.subheader("📈 Fraud & Volume Trend")
    # Grouping by day for the Area Chart
    daily = df.groupby(df[date_col].dt.date).agg({amount_col: 'count', fraud_col: 'sum'}).reset_index()
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
    # Stacked histogram showing amount distribution
    fig_hist = px.histogram(df, x=amount_col, color=fraud_col, 
                            nbins=40, barmode='stack',
                            color_discrete_map={0: "#0078D4", 1: "#D13438"},
                            title="Transaction Amounts by Class")
    st.plotly_chart(fig_hist, use_container_width=True)

st.success(f"Dashboard synchronized with Goldilocks Model (Threshold: {metrics['best_threshold']:.4f})")