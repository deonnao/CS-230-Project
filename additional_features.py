import pandas as pd
import os

# Load your existing dataset 
data_dir = os.getcwd()
data_path = os.path.join(data_dir, 'data/synthetic_financial_data_1.5M copy.csv')
df = pd.read_csv(data_path)  

# Convert 'date' to datetime if needed
df['date'] = pd.to_datetime(df['date'])

# Group by user_id and category to calculate statistics for each user in each category
user_category_stats = df.groupby(['user_id', 'category']).agg(
    avg_spend_category=('amount', 'mean'),        # Average amount spent by user in each category
    median_spend_category=('amount', 'median'),   # Median amount spent by user in each category
    freq_transactions_category=('amount', 'count') # Frequency of transactions in each category
).reset_index()

# Merge the calculated statistics back into the original DataFrame
df = df.merge(user_category_stats, on=['user_id', 'category'], how='left')

# Check the updated DataFrame to ensure new columns were added
print(df.head())

final_path = os.path.join(data_dir, 'data/enhanced_user_specific_data.csv')
df.to_csv(final_path, index=False)
