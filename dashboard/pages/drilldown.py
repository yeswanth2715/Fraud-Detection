import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="User Drilldown", layout="wide")

st.title("👤 User Drilldown")

# ===============================
# Load Data
# ===============================

@st.cache_data
def load_data():
    return pd.read_csv("data/credit_card_fraud_synthetic.csv")  # adjust if needed

df = load_data()

st.write("Available Columns:", df.columns.tolist())

# ===============================
# Detect User Column Automatically
# ===============================

possible_user_cols = [
    "user_id",
    "User",
    "customer_id",
    "client_id",
    "account_id",
    "User_ID"
]

user_column = None
for col in possible_user_cols:
    if col in df.columns:
        user_column = col
        break

# If not found, fallback to categorical column
if user_column is None:
    candidate_columns = [
        col for col in df.columns
        if df[col].nunique() < 1000 and df[col].nunique() > 1
    ]

    if len(candidate_columns) == 0:
        st.error("No suitable user identifier column found.")
        st.stop()

    user_column = st.selectbox("Select User Identifier Column", candidate_columns)
else:
    st.success(f"Using detected user column: {user_column}")

# ===============================
# User Selection
# ===============================

user_id = st.selectbox("Select User", df[user_column].unique())

user_data = df[df[user_column] == user_id]

# ===============================
# Detect Fraud Column
# ===============================

possible_targets = ["is_fraud", "fraud", "Class", "target", "label"]

fraud_column = None
for col in possible_targets:
    if col in df.columns:
        fraud_column = col
        break

# ===============================
# KPIs
# ===============================

col1, col2, col3 = st.columns(3)

total_tx = len(user_data)

col1.metric("Total Transactions", total_tx)

if fraud_column:
    fraud_count = user_data[fraud_column].sum()
    fraud_rate = (fraud_count / total_tx) * 100 if total_tx > 0 else 0

    col2.metric("Fraud Transactions", int(fraud_count))
    col3.metric("Fraud Rate %", f"{fraud_rate:.2f}%")
else:
    col2.metric("Fraud Transactions", "N/A")
    col3.metric("Fraud Rate %", "N/A")

# ===============================
# Time Series (if timestamp exists)
# ===============================

st.subheader("Transaction Time Series")

possible_time_cols = ["timestamp", "transaction_time", "date", "datetime"]

time_column = None
for col in possible_time_cols:
    if col in df.columns:
        time_column = col
        break

if time_column:
    user_data[time_column] = pd.to_datetime(user_data[time_column])
    user_data_sorted = user_data.sort_values(time_column)

    fig = px.line(
        user_data_sorted,
        x=time_column,
        y=user_data_sorted.index,
        title="Transaction Activity Over Time"
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No time column found for time series visualization.")

# ===============================
# Transaction Table
# ===============================

st.subheader("User Transactions")

st.dataframe(user_data, use_container_width=True)