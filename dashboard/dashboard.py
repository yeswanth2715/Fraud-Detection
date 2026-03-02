import streamlit as st

st.set_page_config(
    page_title="Fraud Intelligence Platform",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Fraud Intelligence Platform")
st.markdown("""
Welcome to the enterprise fraud monitoring system.

Use the left sidebar to navigate between:
- 🥇 Model Performance
- 📊 Fraud Monitoring Dashboard
- 👤 User Drilldown
""")