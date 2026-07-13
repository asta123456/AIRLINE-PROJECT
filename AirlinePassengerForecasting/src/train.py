from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.sequence_generator import SequenceGenerator
from src.train_test_split import TimeSeriesSplit
from src.model import ModelBuilder

print("train.py started")


class ModelTrainer:

    def __init__(self):
        print("Inside __init__")
        self.data_path = "data/airline-passengers.csv"

    def train(self):

        print("Step 1: Loading Dataset...")

        loader = DataLoader(self.data_path)
        df = loader.load_data()
        print("Dataset Loaded Successfully")

        print("Step 2: Preprocessing...")
        preprocessor = Preprocessor()
        scaled_df = preprocessor.scale_data(df)
        print("Preprocessing Completed")

        print("Step 3: Creating Sequences...")
        generator = SequenceGenerator(sequence_length=12)
        X, y = generator.create_sequences(scaled_df)
        print("Sequences Created")
        print("X Shape:", X.shape)
        print("y Shape:", y.shape)

        print("Step 4: Train-Test Split...")
        splitter = TimeSeriesSplit(train_size=0.80)
        X_train, X_test, y_train, y_test = splitter.split(X, y)

        print("Training Shape:", X_train.shape)
        print("Testing Shape:", X_test.shape)

        print("Step 5: Building Model...")
        builder = ModelBuilder(
            model_type="lstm",
            input_shape=(12, 1)
        )

        model = builder.build_model()
        print("Model Built Successfully")

        print("Step 6: Training Model...")

        history = model.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=8,
            validation_data=(X_test, y_test),
            verbose=1
        )

        print("Training Completed")

        print("Step 7: Saving Model...")
        model.save("models/lstm_model.keras")
        print("Model Saved Successfully")

        return model, history


if __name__ == "__main__":
    print("Main block running")

    try:
        trainer = ModelTrainer()
        model, history = trainer.train()
        print("Program Finished Successfully")

    except Exception as e:
        print("\nERROR OCCURRED:")
        print(type(e).__name__)
        print(e)