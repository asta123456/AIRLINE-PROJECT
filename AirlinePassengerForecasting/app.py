import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

from src.data_loader import DataLoader
from src.forecast import Forecaster
from src.evaluation import Evaluator

# ------------------------------------------------
# Page Config
# ------------------------------------------------
st.set_page_config(
    page_title="Airline Passenger Forecaster",
    page_icon="✈️",
    layout="wide"
)

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
# Load Data
# ------------------------------------------------
loader = DataLoader()
df = loader.load_data()

# Fix passenger column name
if "total_passengers" in df.columns:
    df.rename(columns={"total_passengers": "Passengers"}, inplace=True)

# Handle different date formats safely
if "Month" in df.columns:
    df["Month"] = pd.to_datetime(df["Month"])
    df.set_index("Month", inplace=True)

elif "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

else:
    df.index = pd.to_datetime(df.index)

# Ensure passenger column exists
if "Passengers" not in df.columns:
    df.columns = ["Passengers"]

# ------------------------------------------------
# Title
# ------------------------------------------------
st.title("✈️ Airline Passenger Forecasting Dashboard")

# ------------------------------------------------
# Show Data
# ------------------------------------------------
with st.expander("View Dataset"):
    st.dataframe(df)

# ------------------------------------------------
# Plot Historical Data
# ------------------------------------------------
st.subheader("📊 Historical Passenger Data")

fig = px.line(df, y="Passengers", title="Passenger Trend")
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# Forecast
# ------------------------------------------------
st.subheader("🔮 Forecast Future Passengers")

try:
    forecaster = Forecaster()
    future_df = forecaster.forecast(df, future_months)

    # Combine plots
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=df.index,
        y=df["Passengers"],
        mode="lines",
        name="Actual"
    ))

    fig2.add_trace(go.Scatter(
        x=future_df.index,
        y=future_df["Passengers"],
        mode="lines",
        name="Forecast"
    ))

    st.plotly_chart(fig2, use_container_width=True)

except Exception as e:
    st.error(f"Forecast error: {e}")

# ------------------------------------------------
# Evaluation
# ------------------------------------------------
st.subheader("📈 Model Evaluation")

try:
    evaluator = Evaluator()
    metrics = evaluator.evaluate(df)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("MAE", round(metrics["MAE"], 2))

    with col2:
        st.metric("RMSE", round(metrics["RMSE"], 2))

except Exception as e:
    st.warning("Evaluation not available")
