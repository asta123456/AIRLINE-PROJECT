from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

class Evaluator:

    def evaluate(self, df):
        y = df["Passengers"]

        # dummy prediction (same as actual)
        y_pred = y.copy()

        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        return {
            "MAE": mae,
            "RMSE": rmse
        }
