import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

from src.data_loader import DataLoader
from src.forecast import Forecaster
from src.evaluation import Evaluator

st.set_page_config(page_title="Airline Forecast", layout="wide")

# Sidebar
with st.sidebar:
    st.title("Settings")
    future_months = st.slider("Forecast Months", 1, 24, 12)

# Load data
loader = DataLoader()
df = loader.load_data()

# Fix column names
df.columns = [col.lower() for col in df.columns]

# Handle dataset properly
if "month" in df.columns:
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
    df.set_index("month", inplace=True)

# Rename passengers column safely
if "passengers" in df.columns:
    df.rename(columns={"passengers": "Passengers"}, inplace=True)
elif len(df.columns) == 1:
    df.columns = ["Passengers"]

# Title
st.title("✈️ Airline Passenger Forecasting Dashboard")

# Show data
with st.expander("View Dataset"):
    st.dataframe(df)

# Plot
st.subheader("📊 Historical Data")
fig = px.line(df, y="Passengers")
st.plotly_chart(fig, use_container_width=True)

# Forecast
st.subheader("🔮 Forecast")

try:
    forecaster = Forecaster()
    future_df = forecaster.forecast(df, future_months)

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=df.index,
        y=df["Passengers"],
        name="Actual"
    ))

    fig2.add_trace(go.Scatter(
        x=future_df.index,
        y=future_df["Passengers"],
        name="Forecast"
    ))

    st.plotly_chart(fig2, use_container_width=True)

except Exception as e:
    st.error(f"Forecast error: {e}")

# Evaluation
st.subheader("📈 Evaluation")

try:
    evaluator = Evaluator()
    metrics = evaluator.evaluate(df)

    col1, col2 = st.columns(2)
    col1.metric("MAE", round(metrics["MAE"], 2))
    col2.metric("RMSE", round(metrics["RMSE"], 2))

except:
    st.warning("Evaluation not available")
