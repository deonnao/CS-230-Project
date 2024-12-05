import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from category_specific_model import SpendingPredictionModel

def create_budget(predictions, income, savings_goal, spending_goals):
    """
    Creates a budget combining predictions and user inputs.

    Parameters:
    - predictions: Dictionary of spending predictions per category.
    - income: User's monthly income.
    - savings_goal: Percentage of income to save.
    - spending_goals: Dictionary of user-defined spending goals per category.

    Returns:
    - budget: Dictionary of budget allocations.
    """
    savings = income * savings_goal
    available_income = income - savings
    budget = {}
    total_predicted = sum(predictions.values())
    
    for category, predicted in predictions.items():
        # Prioritize user goals if provided
        goal = spending_goals.get(category, None)
        if goal is not None and goal > 0:
            allocated = min(goal, available_income * (predicted / total_predicted))
        else:
            allocated = available_income * (predicted / total_predicted)

        budget[category] = max(allocated, 0.01)  # Ensures a minimum allocation to each category

    budget['savings'] = savings
    return budget

def calculate_median_spending(user_data):
    """
    Calculates the median spending for each category in the user's data.

    Parameters:
    - user_data: DataFrame containing transaction data for a single user.

    Returns:
    - A dictionary with categories as keys and their median spending as values.
    """
    median_spending = user_data.groupby('category')['amount'].median().to_dict()
    return median_spending

def provide_feedback(predictions, budget, income):
    """
    Provides feedback based on predictions and allocated budget.

    Parameters:
    - predictions: Predicted spending per category.
    - budget: Allocated budget per category.
    - income: User's total income.

    Returns:
    - feedback: List of feedback messages.
    """
    feedback = []
    for category, allocated in budget.items():
        if category == "savings":
            continue
        predicted = predictions.get(category, 0)
        if allocated < predicted:
            feedback.append(f"Consider increasing your {category} budget to match predicted needs of ${predicted:.2f}.")
        elif allocated > 0.5 * income:  
            feedback.append(f"Your {category} budget is quite high; consider reducing it.")

    feedback.append(f"You are saving {budget['savings'] / income * 100:.1f}% of your income.")
    return feedback

def plot_budget(budget):
    """
    Plot a pie chart of the budget.

    Parameters:
    - budget: Dictionary of budget allocations.
    """
    labels = list(budget.keys())
    values = list(budget.values())

    plt.figure(figsize=(8, 6))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Suggested Monthly Budget')
    plt.show()

def main():

    # Load user data
    data_dir = os.getcwd()
    data_path = os.path.join(data_dir, 'data/user0.csv')
    user_data = pd.read_csv(data_path)

    # Step 1: Accept user inputs
    income = float(input("Enter your monthly income: "))
    savings_goal = float(input("Enter your savings goal as a percentage (e.g., 0.2 for 20%): "))
    categories = ['rent', 'education', 'groceries', 'transportation', 'utilities', 'dining_out', 'entertainment', 'shopping', 'healthcare']
    spending_goals = {}

    model = SpendingPredictionModel(seq_length=11)

    for category in categories:
        goal = float(input(f"Enter your spending goal for {category} (or 0 if no goal): "))
        spending_goals[category] = goal if goal > 0 else None

    median_spending = calculate_median_spending(user_data)

    # Step 3: Predict spending
    predictions = {}
    for category in categories:
        X, _ = model.load_and_preprocess_data(user_data, category)
        model.build_model()
        if X.size > 0:
            predictions[category] = model.predict(X)[-1][0]  # Get the latest prediction
        else:
            predictions[category] = 0  # No data, default to 0

    # Fallback to averages where predictions are too low
    for category in predictions.keys():
        if predictions[category] < 10:
            predictions[category] = median_spending.get(category, 0)

    print("Predictions by category:", predictions)

    # Step 4: Create a budget
    budget = create_budget(predictions, income, savings_goal, spending_goals)

    # Step 5: Output the budget
    print("Suggested Budget:")
    for category, amount in budget.items():
        print(f"{category.capitalize()}: ${amount:.2f}")

    feedback = provide_feedback(predictions, budget, income)
    print(feedback)

    # Step 6: Visualize the budget
    plot_budget(budget)

if __name__ == "__main__":
    main()
