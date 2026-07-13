import os
import pandas as pd

class DataLoader:

    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.file_path = os.path.join(BASE_DIR, "..", "data", "airline-passengers.csv")

    def load_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset not found at:\n{self.file_path}")

        df = pd.read_csv(self.file_path)
        return df
