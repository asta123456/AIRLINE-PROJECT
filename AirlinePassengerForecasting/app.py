import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

from src.data_loader import DataLoader
from src.forecast import Forecaster
from src.evaluation import Evaluator

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------

st.set_page_config(
    page_title="Airline Passenger Forecaster",
    page_icon="✈️",
    layout="wide"
)

# ------------------------------------------------
# Custom CSS
# ------------------------------------------------

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    div.stButton > button:first-child {
        background-color: #007bff;
        color: white;
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Sidebar
# ------------------------------------------------

with st.sidebar:

    img_path = os.path.join("assets", "201623.png")

    if os.path.exists(img_path):
        st.image(img_path, width=100)
    else:
        st.warning("Image not found!")

    st.title("Settings")
    future_months = st.slider("Forecast Horizon (Months)", 1, 24, 12)

# ------------------------------------------------
# Data Loading
# ------------------------------------------------

loader = DataLoader()
df = loader.load_data()

# Fix column names if needed
if "total_passengers" in df.columns:
    df.rename(columns={"total_passengers": "Passengers"}, inplace=True)

# Convert date column properly
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)

# ------------------------------------------------
# Title
# ------------------------------------------------

st.title("✈️ Airline Passenger Forecasting Dashboard")

# ------------------------------------------------
# Show Raw Data
# ------------------------------------------------

with st.expander("📄 View Raw Data"):
    st.dataframe(df)

# ------------------------------------------------
# Visualization
# ------------------------------------------------

st.subheader("📊 Historical Passenger Data")

fig = px.line(df, y="Passengers", title="Passenger Trend Over Time")
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# Forecasting
# ------------------------------------------------

st.subheader("🔮 Forecast Future Passengers")

forecaster = Forecaster()
future_df = forecaster.forecast(df, future_months)

# ------------------------------------------------
# Combine Actual + Forecast
# ------------------------------------------------

fig_combined = go.Figure()

fig_combined.add_trace(go.Scatter(
    x=df.index,
    y=df["Passengers"],
    mode='lines',
    name='Actual'
))

fig_combined.add_trace(go.Scatter(
    x=future_df.index,
    y=future_df["Passengers"],
    mode='lines',
    name='Forecast'
))

fig_combined.update_layout(title="Actual vs Forecast")

st.plotly_chart(fig_combined, use_container_width=True)

# ------------------------------------------------
# Evaluation (Optional)
# ------------------------------------------------

st.subheader("📈 Model Evaluation")

evaluator = Evaluator()
metrics = evaluator.evaluate(df)

col1, col2 = st.columns(2)

with col1:
    st.metric("MAE", round(metrics["MAE"], 2))

with col2:
    st.metric("RMSE", round(metrics["RMSE"], 2))
