import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

class Evaluator:

    def evaluate(self, df):

        train = df["Passengers"][:-12]
        test = df["Passengers"][-12:]

        model = ARIMA(train, order=(5,1,0))
        model_fit = model.fit()

        predictions = model_fit.forecast(steps=12)

        mae = mean_absolute_error(test, predictions)
        rmse = np.sqrt(mean_squared_error(test, predictions))

        return {
            "MAE": mae,
            "RMSE": rmse
        }
