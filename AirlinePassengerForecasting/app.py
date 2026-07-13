import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

from src.data_loader import DataLoader
from src.forecast import Forecaster
from src.evaluation import Evaluator

# ------------------------------------------------
# Page Configuration & Custom CSS
# ------------------------------------------------

st.set_page_config(
    page_title="Airline Passenger Forecaster",
    page_icon="✈️",
    layout="wide"
)

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
    st.info("Adjust the slider to change the prediction window.")

# ------------------------------------------------
# Data Loading
# ------------------------------------------------

loader = DataLoader("data/airline-passengers.csv")
df = loader.load_data()

# OPTIONAL: rename column for easier use
df.rename(columns={"total_passengers": "Passengers"}, inplace=True)

# Ensure index is datetime
df.index = pd.to_datetime(df.index)

# ------------------------------------------------
# Header
# ------------------------------------------------

st.title("✈️ Airline Passenger Analysis & Forecasting")
st.caption("Predicting global travel trends using RNN")

# ------------------------------------------------
# Tabs
# ------------------------------------------------

tab1, tab2 = st.tabs(["🚀 Model Performance", "🔎 Exploratory Data Analysis"])

with tab1:
    st.subheader("Model Accuracy Metrics")

    mae, mse, rmse = Evaluator().evaluate()

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mae:.2f}")
    m2.metric("MSE", f"{mse:.2f}")
    m3.metric("RMSE", f"{rmse:.2f}")

    st.info("Lower RMSE indicates better model performance.")

with tab2:
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.subheader("Raw Data")
        st.dataframe(df, height=350)

    with col_b:
        st.subheader("Historical Trend")

        fig = px.line(
            df,
            x=df.index,
            y="Passengers",
            template="plotly_white",
            color_discrete_sequence=['#007bff']
        )

        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# Forecast Section
# ------------------------------------------------

st.markdown("---")
st.header("🔮 Generate Future Forecast")

if st.button("Run RNN Model"):
    with st.spinner("Analyzing..."):

        forecaster = Forecaster()
        future = forecaster.forecast(future_months)

        # ensure correct shape
        future = future.reshape(-1)

        last_date = df.index[-1]

        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=future_months,
            freq="MS"
        )

        forecast_df = pd.DataFrame({
            "Month": future_dates,
            "Predicted Passengers": future
        })

    st.success(f"Forecast generated for {future_months} months!")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Forecasted Values")
        st.dataframe(forecast_df, use_container_width=True)

        csv = forecast_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="forecast_results.csv",
            mime="text/csv"
        )

    with col2:
        st.subheader("Combined Projection")

        fig_combined = go.Figure()

        fig_combined.add_trace(go.Scatter(
            x=df.index,
            y=df["Passengers"],
            name="Historical"
        ))

        fig_combined.add_trace(go.Scatter(
            x=forecast_df["Month"],
            y=forecast_df["Predicted Passengers"],
            name="Forecast"
        ))

        fig_combined.update_layout(
            title="Passenger Forecast vs Historical",
            template="plotly_white"
        )

        st.plotly_chart(fig_combined, use_container_width=True)