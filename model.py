import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Conv1D, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from xgboost import plot_importance
import tensorflow.keras.backend as K
import os
import matplotlib.pyplot as plt

"""
Monthly Spending Prediction Model

This script implements a machine learning model using Long Short-Term Memory (LSTM) networks to predict monthly spending based on historical user transaction data. 
The model takes a sequence of past monthly spending data as input and generates a single output representing the predicted spending for the next month.

Main Components:
1. SpendingPredictionModel Class:
   - __init__: Initializes the model parameters, including sequence length and scalers for feature normalization.
   - load_and_preprocess_data: Loads and preprocesses user transaction data, normalizing continuous features and creating input-output sequences for the model.
   - _create_sequences: Generates sequences of input data and corresponding targets for training and prediction, based on a specified sequence length.
   - build_model: Defines and compiles the LSTM model architecture with two LSTM layers and dropout layers for regularization.
   - train_model: Trains the model on the provided training data, with options to specify epochs and batch size.
   - evaluate_model: Evaluates the model on a test dataset, returning metrics such as loss and mean absolute error (MAE).
   - predict: Makes predictions on new input data and denormalizes the results to the original spending scale.

2. Main Execution:
   - Loads the dataset, initializes the SpendingPredictionModel, and preprocesses the data.
   - Splits the data into training, validation, and test sets.
   - Builds, trains, and evaluates the model, then makes predictions on a sample of test data.

Usage:
- Run the script to train the model and predict monthly spending based on historical data.
- Outputs include training/validation metrics, test performance, and denormalized predictions for interpretability.

Dependencies:
- numpy, pandas, sklearn, tensorflow (keras), os

"""

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

class SpendingPredictionModel:
    
    def __init__(self, seq_length=11):
        """
        Initializes the model with a sequence length for the LSTM.

        Parameters:
        - seq_length: The length of input sequences for the LSTM (e.g. past 3 months of data)
        """
        self.seq_length = seq_length
        self.scaler = MinMaxScaler()
        self.amount_scaler = MinMaxScaler()
        self.model = None
        

    def load_and_preprocess_data(self, data):
        """
        Loads and preprocesses the data for LSTM input.

        Parameters:
        - data: DataFrame containing user transaction data

        Returns:
        - X, y: Processed input features and labels
        """

        # Summarizes spending for each month for each user
        monthly_data = data.groupby(['user_id', 'year', 'month']).agg({
            'amount': 'sum',           # Sum total spending for each category within the month
            'income': 'first',          # Keep income constant for the user
            'family_size': 'first',      # Keep family size constant for the user
            'age': 'first'               # Keep age constant for the user
        }).reset_index()

        # Normalize 'amount' separately using amount_scaler
        monthly_data['amount'] = self.amount_scaler.fit_transform(monthly_data[['amount']])

        # Normalize the other continuous features using the main scaler
        monthly_data[['income', 'family_size', 'age']] = self.scaler.fit_transform(
            monthly_data[['income', 'family_size', 'age']]
        )
        
        # Create sequences and labels
        X, y = self._create_sequences(monthly_data)
        
        return X, y

    def _create_sequences(self, data):
        """
        Creates sequences from the data for LSTM input.

        Parameters:
        - data: Preprocessed DataFrame containing aggregated monthly user data

        Returns:
        - sequences(X), targets(y): Arrays of input sequences and corresponding labels
        """
        sequences = []
        targets = []
        # We create sequences of 3 months length
        for user_id in data['user_id'].unique():
            user_data = data[data['user_id'] == user_id]
            # This loop will allow us to stop when we can no longer create full sequences (3 months + 1 month for target)
            for i in range(len(user_data) - self.seq_length):
                seq = user_data[['amount', 'income', 'family_size', 'age']].iloc[i:i + self.seq_length].values
                target = user_data['amount'].iloc[i + self.seq_length]
                sequences.append(seq)
                targets.append(target)
        return np.array(sequences), np.array(targets)
    

    def build_model(self):
        """
        Builds and compiles the LSTM model.

        """
        num_features = 4
        self.model = Sequential()

        self.model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(self.seq_length, num_features)))
        self.model.add(BatchNormalization())

        self.model.add(LSTM(128, return_sequences=True))
        self.model.add(Dropout(0.1))
        self.model.add(LSTM(64)) 
        self.model.add(Dropout(0.1))  
        self.model.add(Dense(1))  

        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    def train_model(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        """
        Trains the LSTM model.

        Parameters:
        - X_train, y_train: Training data
        - X_val, y_val: Validation data
        - epochs: Number of epochs to train
        - batch_size: Batch size for training

        Returns:
        - history: object containing details about the training process
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() before training.")

        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                 validation_data=(X_val, y_val), verbose=1)
        return history

    def evaluate_model(self, X_test, y_test):
        """
        Evaluates the model on the test data.

        Parameters:
        - X_test, y_test: Test data

        Returns:
        - loss, mae: Test loss and mean absolute error
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() before evaluating.")
        
        loss, mae = self.model.evaluate(X_test, y_test)
        print(f"Test Loss: {loss}, Test MAE: {mae}")
        return loss, mae

    def predict(self, X):
        """
        Makes predictions with the trained model.

        Parameters:
        - X: Input data for prediction

        Returns:
        - Predictions in the original scale
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() before predicting.")
        
        predictions = self.model.predict(X)

        return predictions

if __name__ == "__main__":
   
    data_dir = os.getcwd()
    data_path = os.path.join(data_dir, 'data/synthetic_financial_data_1.5M.csv')
    data = pd.read_csv(data_path)  

    model = SpendingPredictionModel(seq_length=11)

    X, y = model.load_and_preprocess_data(data)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    model.build_model()
    print(model.model.summary())

    history = model.train_model(X_train, y_train, X_val, y_val, epochs=20, batch_size=32)

    loss, mae = model.evaluate_model(X_test, y_test)

    predictions = model.predict(X_test)
 
    predictions_denormalized = model.amount_scaler.inverse_transform(predictions)

    # Denormalize actual test targets
    y_test_denormalized = model.amount_scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate the absolute difference between predicted and actual values
    difference = np.abs(predictions_denormalized - y_test_denormalized)

    # Set the tolerance level (e.g., 10% of the actual value)
    tolerance = 0.1

    # Check if each prediction is within the tolerance range of the actual value
    within_tolerance = difference <= tolerance * np.abs(y_test_denormalized)

    # Calculate accuracy as the percentage of predictions within the tolerance range
    accuracy = np.mean(within_tolerance) * 100

    print(f"Accuracy within ±10% tolerance: {accuracy:.2f}%")

    print('MAPE: ', mean_absolute_percentage_error(y_test_denormalized, predictions_denormalized))

    # Plot predicted vs actual spending
    plt.figure(figsize=(10, 6))
    plt.plot(predictions_denormalized, label="Predicted Spending", linestyle="--", marker="o", color="blue")
    plt.plot(y_test_denormalized, label="Actual Spending", linestyle="-", marker="x", color="orange")
    plt.title("Predicted vs Actual Spending Over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Spending Amount ($)")
    plt.legend()
    plt.grid(True)
    plt.show()