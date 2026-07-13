"""
====================================================
Module : evaluation.py
Project: Airline Passenger Forecasting
Purpose: Evaluate LSTM Model Performance
====================================================
"""

print("evaluation.py started")

import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from src.predict import Predictor


class Evaluator:

    """
    Evaluate model performance.
    """

    def __init__(self):
        print("Evaluator Initialized")

    def evaluate(self):

        print("Inside evaluate()")

        # -----------------------------
        # Generate Predictions
        # -----------------------------

        print("Creating Predictor...")
        predictor = Predictor()

        print("Calling predict()...")
        actual, predicted = predictor.predict()

        print("Prediction Completed")

        # -----------------------------
        # Calculate Metrics
        # -----------------------------

        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)

        print("\n========== Model Evaluation ==========")
        print(f"MAE  : {mae:.4f}")
        print(f"MSE  : {mse:.4f}")
        print(f"RMSE : {rmse:.4f}")

        # -----------------------------
        # Save Metrics
        # -----------------------------

        with open("outputs/metrics.txt", "w") as file:
            file.write("Model Evaluation\n")
            file.write("=====================\n")
            file.write(f"MAE  : {mae:.4f}\n")
            file.write(f"MSE  : {mse:.4f}\n")
            file.write(f"RMSE : {rmse:.4f}\n")

        print("\nMetrics Saved Successfully.")

        return mae, mse, rmse


if __name__ == "__main__":

    print("Main block running")

    try:
        evaluator = Evaluator()
        mae, mse, rmse = evaluator.evaluate()

        print("\nEvaluation Completed Successfully.")

    except Exception as e:
        print("\nERROR OCCURRED")
        print(type(e).__name__)
        print(e)