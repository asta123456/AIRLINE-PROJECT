# Scaling & preprocessing
 
"""
====================================================
Module : preprocessing.py
Project: Airline Passenger Forecasting
Purpose: Scale the dataset using MinMaxScaler
====================================================
"""
 
# Import required libraries
import joblib
import os
import sys
import pandas as pd
 
from sklearn.preprocessing import MinMaxScaler

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SCALER_PATH = os.path.join(ROOT_DIR, "models", "scaler.pkl")
 
 
class Preprocessor:
    """
    Preprocess the time series dataset.
    """
 
    def __init__(self):
        """
        Initialize the scaler.
        """
 
        self.scaler = MinMaxScaler(feature_range=(0, 1))
 
    def scale_data(self, df):
        """
        Scale the Passengers column.
 
        Parameters
        ----------
        df : pandas.DataFrame
 
        Returns
        -------
        scaled_df : pandas.DataFrame
        """
 
        print("\nOriginal Data")
        print(df.head())
 
        # Identify the passenger column
        passenger_col = (
            "total_passengers"
            if "total_passengers" in df.columns
            else "Passengers"
            if "Passengers" in df.columns
            else None
        )
 
        if passenger_col is None:
            raise KeyError(
                "Expected a passenger column named 'total_passengers' or 'Passengers'."
            )
 
        # Scale the passenger column
        scaled_values = self.scaler.fit_transform(df[[passenger_col]])
 
        # Convert to DataFrame
        scaled_df = pd.DataFrame(
            scaled_values,
            columns=[passenger_col],
            index=df.index
        )
 
        # print("\nScaled Data")
        # print(scaled_df.head())
 
        # Save the scaler
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        joblib.dump(self.scaler, SCALER_PATH)
 
        print("\nScaler saved successfully.")
 
        return scaled_df
 
if __name__ == "__main__":

    if __package__ is None:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

    from src.data_loader import DataLoader

    DATA_PATH = os.path.join(ROOT_DIR, "data", "airline-passengers.csv")

    # Load data
    loader = DataLoader(DATA_PATH)
    df = loader.load_data()
 
    # Scale data
    preprocessor = Preprocessor()
 
    scaled_df = preprocessor.scale_data(df)
 
    print("\nScaled Dataset")
    print(scaled_df.head())