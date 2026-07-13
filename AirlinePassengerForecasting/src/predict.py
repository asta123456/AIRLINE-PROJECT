"""
====================================================
Module : predict.py
Project: Airline Passenger Forecasting
Purpose: Predict Passenger Counts
====================================================
"""

import joblib
import os
import numpy as np


class Predictor:

    def __init__(self):
        # Path to model.pkl
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(BASE_DIR, "..", "model.pkl")

    def predict(self, input_data):
        """
        input_data: numpy array or list
        """

        # Load model
        model = joblib.load(self.model_path)

        # Convert input to numpy array
        input_data = np.array(input_data).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)

        return prediction

