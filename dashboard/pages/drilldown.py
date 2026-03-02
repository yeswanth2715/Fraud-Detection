import sys
import os
sys.path.append(os.path.abspath("."))

import streamlit as st
import pandas as pd
import plotly.express as px

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "data/User0_credit_card_transactions.csv")

st.set_page_config(layout="wide")
st.title("👤 User Drilldown")

df = pd.read_csv(DATA_PATH)

user_id = st.selectbox("Select User", df["user_id"].unique())

user_df = df[df["user_id"] == user_id]

st.metric("Total Transactions", len(user_df))

fig = px.line(
    user_df,
    x="transaction_date",
    y="amount",
    title="Transaction History"
)

st.plotly_chart(fig, use_container_width=True)

st.dataframe(user_df)