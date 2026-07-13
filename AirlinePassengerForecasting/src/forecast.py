import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

class Forecaster:

    def forecast(self, df, future_months=12):

        # train model
        model = ARIMA(df["Passengers"], order=(5,1,0))
        model_fit = model.fit()

        # forecast future
        forecast_values = model_fit.forecast(steps=future_months)

        # future dates
        future_dates = pd.date_range(
            start=df.index[-1],
            periods=future_months + 1,
            freq="MS"
        )[1:]

        # dataframe
        future_df = pd.DataFrame({
            "Passengers": forecast_values
        }, index=future_dates)

        return future_df
