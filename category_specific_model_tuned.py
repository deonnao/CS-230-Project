import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Input, Conv1D, LSTM, Dropout, BatchNormalization, Dense, Attention, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from xgboost import plot_importance
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.layers import Attention
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

def permutation_importance(model, X_test, y_test):
    baseline = mean_absolute_error(y_test, model.predict(X_test))  # Baseline MAE
    importance = {}
    
    for i in range(X_test.shape[2]):  # Iterate over features
        X_temp = X_test.copy()
        np.random.shuffle(X_temp[:, :, i])  # Shuffle one feature
        mae = mean_absolute_error(y_test, model.predict(X_temp))
        importance[f"Feature {i}"] = mae - baseline  # Importance = MAE drop

    return importance

def plot_predictions_vs_actuals(y_test, predictions, category):
    """
    Plots actual vs predicted values for a specific category.
    
    Parameters:
    - y_test: Actual values (denormalized).
    - predictions: Predicted values (denormalized).
    - category: The category being plotted.
    - scaler: The scaler used for normalization (to denormalize y_test and predictions).
    """
   
    # Create a line plot for actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual Spending', color='blue', linestyle='--', marker='o')
    plt.plot(predictions, label='Predicted Spending', color='orange', linestyle='-', marker='x')
    plt.title(f'Predicted vs Actual Spending for {category.capitalize()}')
    plt.xlabel('Time Steps (Test Data)')
    plt.ylabel('Spending Amount ($)')
    plt.legend()
    plt.grid(True)
    plt.show()


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

class SpendingPredictionModel:
    
    def __init__(self, seq_length=3):
        """
        Initializes the model with a sequence length for the LSTM.

        Parameters:
        - seq_length: The length of input sequences for the LSTM (e.g. past 3 months of data)
        """
        self.seq_length = seq_length
        self.scaler = MinMaxScaler()
        self.amount_scaler = MinMaxScaler()
        self.model = None
        

    def load_and_preprocess_data(self, data, category):
        """
        Loads and preprocesses the data for LSTM input.

        Parameters:
        - data: DataFrame containing user transaction data

        Returns:
        - X, y: Processed input features and labels
        """

        # Filter data for the given category
        category_data = data[data['category'] == category].sort_values(['user_id', 'date'])

        print(category_data['amount'].describe())
        print(f"Number of samples: {len(category_data)}")

        # Convert date to datetime format and extract year-month
        category_data['date'] = pd.to_datetime(category_data['date'])
        category_data['year_month'] = category_data['date'].dt.to_period('M')

        # Aggregate monthly spending for the given category
        monthly_data = category_data.groupby(['user_id', 'year_month']).agg({
            'amount': 'sum',  # Aggregate spending for the month
            'income': 'first',  # Assume income is constant within the user
            'age': 'first',  # Age remains constant
            'family_size': 'first'  # Family size remains constant
        }).reset_index()

        # Normalize the spending amounts
        category_data['amount'] = self.amount_scaler.fit_transform(category_data[['amount']])

        # Normalize the features
        additional_features = ['income', 'age', 'family_size']
        monthly_data[['amount'] + additional_features] = self.scaler.fit_transform(
            monthly_data[['amount'] + additional_features]
        )

        # Generate sequences of monthly spending and targets
        sequences, targets = [], []
        for user_id in monthly_data['user_id'].unique():
            user_data = monthly_data[monthly_data['user_id'] == user_id]
            spending = user_data['amount'].values  # Shape: (num_months,)
            features = user_data[['amount'] + additional_features].values  # Shape: (num_months, num_features)
            
            for i in range(len(spending) - self.seq_length):
                sequences.append(features[i:i + self.seq_length])
                targets.append(spending[i + self.seq_length])

        # Convert to numpy arrays
        X = np.array(sequences)  # Shape: (num_samples, seq_length, num_features)
        y = np.array(targets)  # Shape: (num_samples,)

        # Generate sequences and targets
        # sequences, targets = [], []
        # for user_id in category_data['user_id'].unique():
        #     user_data = category_data[category_data['user_id'] == user_id]
        #     user_features = user_data[features].values  # Shape: (num_records, num_features)
        #     spending = user_data['amount'].values  # Shape: (num_records,)
        #     for i in range(len(user_features) - self.seq_length):
        #         sequences.append(user_features[i:i + self.seq_length])
        #         targets.append(spending[i + self.seq_length])

        # # Convert sequences to a 3D array (num_samples, seq_length, num_features)
        # X = np.array(sequences)
        # y = np.array(targets)

        return X, y
        

    def build_model(self, num_features):
        """
        Builds and compiles the LSTM model.

        """
        
        self.model = Sequential()
        self.model.add(
            Conv1D(
                filters=64,
                kernel_size=2,
                activation="relu",
                input_shape=(self.seq_length, num_features),
            )
        )
        self.model.add(BatchNormalization())
        self.model.add(LSTM(128, return_sequences=True))
        self.model.add(Dropout(0.1))
        self.model.add(LSTM(64))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(1, activation="relu"))
        self.model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])



    def train_model(self, X_train, y_train, X_val, y_val, epochs=200, batch_size=32):
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
        return self.model.predict(X)

if __name__ == "__main__":
   
    data_dir = os.getcwd()
    data_path = os.path.join(data_dir, 'data/synthetic_financial_data_1.5M.csv')
    data = pd.read_csv(data_path)

    # Initialize the model
    model = SpendingPredictionModel(seq_length=3)
    # Categories to predict
    categories = data['category'].unique()
    for category in categories:
        category = 'rent'
        print(f"Processing category: {category}")
        
        # Preprocess data for the specific category
        X, y = model.load_and_preprocess_data(data, category='rent')
        print(f"Shape of X: {X.shape}")  # Should be (num_samples, seq_length, 1)
        print(f"Shape of y: {y.shape}")  # Should be (num_samples,)

        num_features = X.shape[2]

        # Split data into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Build and train the model
        model.build_model(num_features)
        print(model.model.summary())

        history = model.train_model(X_train, y_train, X_val, y_val, epochs=200, batch_size=32)

        # print('HISTORY: ', history)

        # Evaluate the model
        loss, mae = model.evaluate_model(X_test, y_test)
        print(f"Category: {category}, Test Loss: {loss}, Test MAE: {mae}")

        # Make predictions
        predictions = model.predict(X_test)

        # Denormalize predictions and actuals
        predictions_denormalized = model.amount_scaler.inverse_transform(predictions)
        y_test_denormalized = model.amount_scaler.inverse_transform(y_test.reshape(-1, 1))
        

        print(f"Category: {category}, Predictions: {predictions_denormalized[:5].flatten()}")
        print(f"Category: {category}, Actuals: {y_test_denormalized[:5].flatten()}")
        
        print('MAPE: ', mean_absolute_percentage_error(y_test_denormalized, predictions_denormalized))

        # Calculate the absolute difference between predicted and actual values
        difference = np.abs(predictions_denormalized - y_test_denormalized)

        # Set the tolerance level (e.g., 10% of the actual value)
        tolerance = 0.1

        # Check if each prediction is within the tolerance range of the actual value
        within_tolerance = difference <= tolerance * np.abs(y_test_denormalized)

        # Calculate accuracy as the percentage of predictions within the tolerance range
        accuracy = np.mean(within_tolerance) * 100

        print(f"Accuracy within ±10% tolerance: {accuracy:.2f}%")


        importance_scores = permutation_importance(model, X_test, y_test)
        print(importance_scores)

        feature_names = ['amount', 'income', 'age', 'family_size']

        # Map feature indices to their names
        importance_scores_named = {feature_names[int(key.split()[1])]: value for key, value in importance_scores.items()}

        # Print feature importance with names
        print("Feature Importance with Names:")
        for feature, importance in importance_scores_named.items():
            print(f"{feature}: {importance}")

        plot_predictions_vs_actuals(y_test_denormalized, predictions_denormalized, category='rent')

    # model = SpendingPredictionModel(seq_length=11, categories=categories)

    # X, y = model.load_and_preprocess_data(data)

    # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # model.build_model()
    # print(model.model.summary())

    # history = model.train_model(X_train, y_train, X_val, y_val, epochs=20, batch_size=32)

    # loss, mae = model.evaluate_model(X_test, y_test)

    # predictions = model.predict(X_test)
 
    # predictions_denormalized = {
    #     cat: model.scalers[cat].inverse_transform(predictions[:, i].reshape(-1, 1))
    #     for i, cat in enumerate(categories)
    # }

    # # Denormalize actual test targets
    # y_test_denormalized = {
    #     cat: model.scalers[cat].inverse_transform(y_test[:, i].reshape(-1, 1))
    #     for i, cat in enumerate(categories)
    # }

    # # Calculate the absolute difference between predicted and actual values
    # difference = np.abs(predictions_denormalized - y_test_denormalized)

    # # Set the tolerance level (e.g., 10% of the actual value)
    # tolerance = 0.1

    # # Check if each prediction is within the tolerance range of the actual value
    # within_tolerance = difference <= tolerance * np.abs(y_test_denormalized)

    # # Calculate accuracy as the percentage of predictions within the tolerance range
    # accuracy = np.mean(within_tolerance) * 100

    # print(f"Accuracy within ±10% tolerance: {accuracy:.2f}%")

    # print('MAPE: ', mean_absolute_percentage_error(y_test_denormalized, predictions_denormalized))

    # # Plot predicted vs actual spending
    # plt.figure(figsize=(10, 6))
    # plt.plot(predictions_denormalized, label="Predicted Spending", linestyle="--", marker="o", color="blue")
    # plt.plot(y_test_denormalized, label="Actual Spending", linestyle="-", marker="x", color="orange")
    # plt.title("Predicted vs Actual Spending Over Time")
    # plt.xlabel("Time Steps")
    # plt.ylabel("Spending Amount ($)")
    # plt.legend()
    # plt.grid(True)
    # plt.show()