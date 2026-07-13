import pandas as pd

class Forecaster:

    def forecast(self, df, future_months=12):

        # last value
        last_value = df["Passengers"].iloc[-1]

        # correct monthly dates
        future_dates = pd.date_range(
            start=df.index[-1],
            periods=future_months + 1,
            freq="MS"   # month start (fixes date issue)
        )[1:]

        # simple forecast
        future_values = [last_value] * future_months

        future_df = pd.DataFrame({
            "Passengers": future_values
        }, index=future_dates)

        return future_df
   
