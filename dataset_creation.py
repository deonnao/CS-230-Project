import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from tqdm import tqdm
import gc

'''AUTHOR: RAMON SILVA'''

class CategoryPattern:
    """Define transaction patterns for different categories"""
    DAILY = 'daily'
    WEEKLY = 'weekly'
    MONTHLY = 'monthly'
    QUARTERLY = 'quarterly'
    SEMESTER = 'semester'
    VARIABLE = 'variable'

def generate_financial_shock(date, severity='medium'):
    """Generate unexpected financial events."""
    shock_types = {
        'low': (100, 500),
        'medium': (500, 2000),
        'high': (2000, 5000)
    }
    shock_categories = ['healthcare', 'car_repair', 'home_repair', 'emergency']
    return {
        'date': date,
        'amount': np.random.uniform(*shock_types[severity]),
        'category': np.random.choice(shock_categories)
    }

def generate_seasonal_factors(date):
    """Generate seasonal spending multipliers."""
    month = date.month
    seasonal_multipliers = {
        12: 1.4,  # December holiday season
        1: 0.8,   # Post-holiday reduction
        11: 1.2,  # Black Friday/Pre-holiday
        7: 1.1,   # Summer activities
        4: 1.1    # Spring shopping
    }
    return seasonal_multipliers.get(month, 1.0)

def generate_user_profile(user_id):
    """Generate realistic user demographic profiles."""
    age = int(np.clip(np.random.normal(35, 12), 18, 80))
    
    # Income distribution varies by age
    if age < 25:
        income_mean = 10.5
        income_std = 0.3
    elif age < 35:
        income_mean = 11.0
        income_std = 0.4
    elif age < 50:
        income_mean = 11.5
        income_std = 0.5
    else:
        income_mean = 11.2
        income_std = 0.6
        
    return {
        'user_id': user_id,
        'age': age,
        'income': np.clip(np.random.lognormal(income_mean, income_std), 25000, 250000),
        'family_size': np.random.choice([1, 2, 3, 4, 5, 6], p=[0.25, 0.3, 0.2, 0.15, 0.07, 0.03]),
        'occupation_risk': np.random.uniform(0, 1),
        'spending_volatility': np.random.uniform(0.5, 1.5)
    }

def determine_user_transaction_count(profile, category):
    """Determine transaction count based on user profile and category."""
    base_counts = {
        'groceries': 52,  # Weekly
        'transportation': 180,  # About every other day
        'dining_out': 104,  # Twice a week
        'entertainment': 24,  # Twice a month
        'shopping': 36,  # Three times a month
        'healthcare': 12,  # Monthly
        'utilities': 12,  # Monthly
    }
    
    base_count = base_counts.get(category, 52)  # Default to weekly if category not found
    
    # Adjust based on profile
    income_factor = np.clip(profile['income'] / 50000, 0.5, 2.0)
    family_factor = np.clip(profile['family_size'] / 2, 0.8, 1.5)
    
    # Add randomness (±20%)
    variation = np.random.uniform(0.8, 1.35)
    
    return int(base_count * income_factor * family_factor * variation)

def generate_variable_amount(params, profile):
    """Generate variable transaction amount based on category parameters and user profile."""
    base_range = params['base_range']
    volatility = params['volatility']
    
    base_amount = np.random.uniform(*base_range)
    income_factor = np.sqrt(profile['income'] / 70000)  # Square root to dampen the effect
    
    amount = base_amount * income_factor
    if volatility > 0:
        amount *= np.random.normal(1, volatility)
    
    return max(0.01, amount)

def generate_education_expenses(user_id, start_date, end_date, base_tuition):
    """Generate education-related expenses with realistic amounts and times."""
    expenses = []
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Determine payment frequency
    payment_type = np.random.choice(['monthly', 'quarterly', 'semester'], p=[0.3, 0.4, 0.3])
    
    if payment_type == 'monthly':
        interval = 30
        tuition_amount = base_tuition / 12  # Monthly tuition
    elif payment_type == 'quarterly':
        interval = 90
        tuition_amount = base_tuition / 4  # Quarterly tuition
    else:  # semester
        interval = 120
        tuition_amount = base_tuition / 3  # Semester tuition
    
    # Generate tuition payments
    current_date = start
    while current_date < end:
        payment_date = current_date + timedelta(days=np.random.randint(0, 5))
        # Add random time
        payment_date = payment_date + timedelta(
            hours=np.random.randint(9, 17),  # Assume tuition payments are made during the day
            minutes=np.random.randint(0, 60),
            seconds=np.random.randint(0, 60)
        )
        if payment_date < end:
            expenses.append({
                'date': payment_date,
                'amount': tuition_amount,
                'subcategory': 'tuition'
            })
        current_date += timedelta(days=interval)
    
    # Generate book expenses (larger amounts at term starts)
    term_starts = []
    if payment_type == 'semester':
        term_starts = [start + timedelta(days=x) for x in [0, 120, 240, 360]]
    else:
        term_starts = [start + timedelta(days=x) for x in [0, 90, 180, 270, 360]]
    
    for term_start in term_starts:
        if term_start < end:
            books_amount = np.random.uniform(200, 800)  # Realistic book costs
            book_date = term_start + timedelta(days=np.random.randint(0, 7))
            # Add random time
            book_date = book_date + timedelta(
                hours=np.random.randint(8, 20),  # Book purchases during store hours
                minutes=np.random.randint(0, 60),
                seconds=np.random.randint(0, 60)
            )
            expenses.append({
                'date': book_date,
                'amount': books_amount,
                'subcategory': 'books'
            })
    
    # Generate printing expenses (small, frequent amounts)
    current_date = start
    while current_date < end:
        num_printing = np.random.randint(2, 4)
        for _ in range(num_printing):
            printing_amount = np.random.uniform(2, 15)
            printing_date = current_date + timedelta(days=np.random.randint(0, 7))
            # Add random time
            printing_date = printing_date + timedelta(
                hours=np.random.randint(8, 22),  # Printing can be done throughout the day
                minutes=np.random.randint(0, 60),
                seconds=np.random.randint(0, 60)
            )
            if printing_date < end:
                expenses.append({
                    'date': printing_date,
                    'amount': printing_amount,
                    'subcategory': 'printing'
                })
        current_date += timedelta(days=7)
    
    return expenses

def generate_fixed_transactions(user_id, start_date, end_date, amount, frequency='monthly'):
    """Generate fixed transactions with consistent amounts and random times."""
    transactions = []
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    intervals = {
        'monthly': 30,
        'quarterly': 90,
        'weekly': 7,
        'daily': 1
    }
    interval = intervals.get(frequency, 30)
    
    current_date = start
    while current_date < end:
        # Add random days within the interval
        payment_date = current_date + timedelta(days=np.random.randint(1, 4))
        # Add random hours, minutes, seconds
        payment_date = payment_date + timedelta(
            hours=np.random.randint(8, 20),
            minutes=np.random.randint(0, 60),
            seconds=np.random.randint(0, 60)
        )
        if payment_date < end:
            transactions.append({
                'date': payment_date,
                'amount': amount
            })
        current_date += timedelta(days=interval)
    
    return transactions

def determine_education_profile(income, age):
    """Determine education type and costs based on user profile."""
    if age < 18:
        return {'type': 'high_school', 'base_cost': np.random.uniform(0, 1000)}
    elif 18 <= age <= 22:
        if income > 100000:
            return {'type': 'private_university', 'base_cost': np.random.uniform(15000, 25000)}
        else:
            return {'type': 'public_university', 'base_cost': np.random.uniform(5000, 15000)}
    elif 23 <= age <= 30:
        return {'type': 'graduate_school', 'base_cost': np.random.uniform(10000, 30000)}
    else:
        return {'type': 'professional_development', 'base_cost': np.random.uniform(500, 5000)}

def generate_transactions_batch(profiles, start_date, end_date, categories, batch_size=1000):
    """Generate transactions with realistic patterns for different categories."""
    all_data = []
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    for profile in tqdm(profiles, desc="Generating transactions", leave=False):
        # Generate fixed rent amount for this user
        monthly_rent = np.random.uniform(1200, 6500)
        rent_transactions = generate_fixed_transactions(
            profile['user_id'], start_date, end_date, monthly_rent, 'monthly')
        
        # Generate education expenses
        edu_profile = determine_education_profile(profile['income'], profile['age'])
        education_transactions = generate_education_expenses(
            profile['user_id'], start_date, end_date, edu_profile['base_cost'])
        
        # Add fixed rent transactions
        for rent in rent_transactions:
            date_obj = pd.to_datetime(rent['date'])
            transaction = {
                'user_id': profile['user_id'],
                'date': date_obj.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                'category': 'rent',
                'subcategory': 'rent_payment',
                'amount': round(rent['amount'], 2),
                'age': profile['age'],
                'income': profile['income'],
                'family_size': profile['family_size'],
                # Add date components
                'day_of_week': date_obj.dayofweek,
                'day_of_month': date_obj.day,
                'month': date_obj.month,
                'year': date_obj.year
            }
            all_data.append(transaction)
        
        # Add education transactions with subcategories
        for edu in education_transactions:
            date_obj = pd.to_datetime(edu['date'])
            transaction = {
                'user_id': profile['user_id'],
                'date': date_obj.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                'category': 'education',
                'subcategory': edu['subcategory'],
                'amount': round(edu['amount'], 2),
                'age': profile['age'],
                'income': profile['income'],
                'family_size': profile['family_size'],
                # Add date components
                'day_of_week': date_obj.dayofweek,
                'day_of_month': date_obj.day,
                'month': date_obj.month,
                'year': date_obj.year
            }
            all_data.append(transaction)
        
        # Generate other category transactions based on their frequency patterns
        for category, params in categories.items():
            if category not in ['rent', 'education']:  # Skip already handled categories
                frequency = params.get('frequency', CategoryPattern.VARIABLE)
                
                if frequency == CategoryPattern.MONTHLY:
                    # Generate fixed monthly transactions (like utilities)
                    monthly_amount = np.random.uniform(*params['base_range'])
                    fixed_transactions = generate_fixed_transactions(
                        profile['user_id'], start_date, end_date, monthly_amount, 'monthly'
                    )
                    for trans in fixed_transactions:
                        date_obj = pd.to_datetime(trans['date'])
                        transaction = {
                            'user_id': profile['user_id'],
                            'date': date_obj.strftime('%Y-%m-%d %H:%M:%S.%f')[:-4],
                            'category': category,
                            'subcategory': f'{category}_payment',
                            'amount': round(trans['amount'], 2),
                            'age': profile['age'],
                            'income': profile['income'],
                            'family_size': profile['family_size'],
                            # Add date components
                            'day_of_week': date_obj.dayofweek,
                            'day_of_month': date_obj.day,
                            'month': date_obj.month,
                            'year': date_obj.year
                        }
                        all_data.append(transaction)
                
                elif frequency == CategoryPattern.WEEKLY:
                    # Generate weekly transactions (like groceries)
                    weekly_transactions = generate_fixed_transactions(
                        profile['user_id'], start_date, end_date,
                        np.random.uniform(*params['base_range']), 'weekly'
                    )
                    for trans in weekly_transactions:
                        amount = trans['amount'] * np.random.normal(1, params['volatility'])
                        date_obj = pd.to_datetime(trans['date'])
                        transaction = {
                            'user_id': profile['user_id'],
                            'date': date_obj.strftime('%Y-%m-%d %H:%M:%S.%f')[:-4],
                            'category': category,
                            'subcategory': f'{category}_expense',
                            'amount': round(max(0.01, amount), 2),
                            'age': profile['age'],
                            'income': profile['income'],
                            'family_size': profile['family_size'],
                            'day_of_week': date_obj.dayofweek,
                            'day_of_month': date_obj.day,
                            'month': date_obj.month,
                            'year': date_obj.year
                        }
                        all_data.append(transaction)
                
                elif frequency == CategoryPattern.DAILY:
                    # Generate daily transactions (like transportation)
                    daily_transactions = generate_fixed_transactions(
                        profile['user_id'], start_date, end_date,
                        np.random.uniform(*params['base_range']), 'daily'
                    )
                    for trans in daily_transactions:
                        amount = trans['amount'] * np.random.normal(1, params['volatility'])
                        date_obj = pd.to_datetime(trans['date'])
                        transaction = {
                            'user_id': profile['user_id'],
                            'date': date_obj.strftime('%Y-%m-%d %H:%M:%S.%f')[:-4],
                            'category': category,
                            'subcategory': f'{category}_expense',
                            'amount': round(max(0.01, amount), 2),
                            'age': profile['age'],
                            'income': profile['income'],
                            'family_size': profile['family_size'],
                            # Add date components
                            'day_of_week': date_obj.dayofweek,
                            'day_of_month': date_obj.day,
                            'month': date_obj.month,
                            'year': date_obj.year
                        }
                        all_data.append(transaction)
                
                else:  # VARIABLE frequency
                    # Generate variable transactions
                    n_transactions = determine_user_transaction_count(profile, category)
                    for _ in range(n_transactions):
                        amount = generate_variable_amount(params, profile)
                        
                        # Determine time range based on category
                        if category == 'dining_out':
                            hour_start, hour_end = 7, 23  # 7 AM to 11 PM
                        elif category == 'entertainment':
                            hour_start, hour_end = 10, 24  # 10 AM to midnight
                        elif category == 'transportation':
                            hour_start, hour_end = 5, 23  # 5 AM to 11 PM
                        elif category == 'online_shopping':
                            hour_start, hour_end = 0, 24  # All hours
                        elif category == 'groceries':
                            hour_start, hour_end = 8, 22  # 8 AM to 10 PM
                        else:
                            hour_start, hour_end = 8, 20  # Default commercial hours

                        date = start + timedelta(
                            days=np.random.randint(0, (end - start).days),
                            hours=np.random.randint(hour_start, hour_end),
                            minutes=np.random.randint(0, 60),
                            seconds=np.random.randint(0, 60)
                        )
                        
                        date_obj = pd.to_datetime(date)
                        
                        transaction = {
                            'user_id': profile['user_id'],
                            'date': date_obj.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                            'category': category,
                            'subcategory': f'{category}_expense',
                            'amount': round(amount, 2),
                            'age': profile['age'],
                            'income': profile['income'],
                            'family_size': profile['family_size'],
                            'day_of_week': date_obj.dayofweek,
                            'day_of_month': date_obj.day,
                            'month': date_obj.month,
                            'year': date_obj.year
                        }
                        all_data.append(transaction)
        
        # Yield batches
        if len(all_data) >= batch_size:
            df = pd.DataFrame(all_data)
            # Ensure correct column order
            df = df[['user_id', 'date', 'category', 'subcategory', 'amount', 'age', 'income', 
                     'family_size', 'day_of_week', 'day_of_month', 'month', 'year']]
            yield df
            all_data = []
            gc.collect()
    
    # Yield any remaining data
    if all_data:
        df = pd.DataFrame(all_data)
        df = df[['user_id', 'date', 'category', 'subcategory', 'amount', 'age', 'income', 
                 'family_size', 'day_of_week', 'day_of_month', 'month', 'year']]
        yield df

# Updated categories dictionary with more detailed patterns
categories = {
    'rent': {
        'frequency': CategoryPattern.MONTHLY,
        'base_range': (1200, 6500),  # Higher rent range
        'volatility': 0,
        'subcategories': ['rent_payment']
    },
    'education': {
        'frequency': CategoryPattern.VARIABLE,
        'base_range': (5000, 25000),  # Higher education costs
        'volatility': 0,
        'subcategories': ['tuition', 'books', 'printing']
    },
    'groceries': {
        'frequency': CategoryPattern.WEEKLY,
        'base_range': (30, 1100),  # Higher grocery range
        'volatility': 0.1,
        'subcategories': ['grocery_shopping']
    },
    'transportation': {
        'frequency': CategoryPattern.DAILY,
        'base_range': (2, 50),
        'volatility': 0.15,
        'subcategories': ['public_transport', 'fuel', 'parking']
    },
    'utilities': {
        'frequency': CategoryPattern.MONTHLY,
        'base_range': (50, 500),  # Higher utilities range
        'volatility': 0.05,
        'subcategories': ['electricity', 'water', 'gas', 'internet']
    },
    'dining_out': {
        'frequency': CategoryPattern.VARIABLE,
        'base_range': (15, 850),  # Higher dining range
        'volatility': 0.2,
        'subcategories': ['restaurants', 'fast_food', 'cafes']
    },
    'entertainment': {
        'frequency': CategoryPattern.VARIABLE,
        'base_range': (20, 2000),  # Higher entertainment range
        'volatility': 0.3,
        'subcategories': ['movies', 'concerts', 'sports_events']
    },
    'shopping': {
        'frequency': CategoryPattern.VARIABLE,
        'base_range': (20, 1500),  # Higher shopping range
        'volatility': 0.25,
        'subcategories': ['clothing', 'electronics', 'home_goods']
    },
    'healthcare': {
        'frequency': CategoryPattern.VARIABLE,
        'base_range': (500, 5000),  # Higher healthcare range
        'volatility': 0.4,
        'subcategories': ['doctor_visits', 'medication', 'insurance']
    }
}

def generate_synthetic_transactions(target_rows=5000000, start_date='2023-01-01', end_date='2024-12-31', 
                                 batch_size=100000, output_file='synthetic_financial_data_5M.csv'):
    """Generate large-scale synthetic financial data with varying transaction counts per user."""
    
    # Estimate number of users needed (adjust multiplier based on transaction patterns)
    estimated_users = target_rows // 2000  # Adjusted for more transactions per user
    print(f"Estimated number of users needed: {estimated_users}")
    
    # Generate user profiles
    profiles = [generate_user_profile(i) for i in range(estimated_users)]
    
    total_rows = 0
    batch_num = 0
    
    # Generate transactions in batches using the generate_transactions_batch function
    for batch_df in generate_transactions_batch(profiles, start_date, end_date, categories, batch_size):
        if total_rows == 0:
            # Write headers only for the first batch
            batch_df.to_csv(output_file, index=False)
        else:
            # Append without headers for subsequent batches
            batch_df.to_csv(output_file, mode='a', header=False, index=False)
        
        total_rows += len(batch_df)
        batch_num += 1
        
        print(f"Processed batch {batch_num}, Total rows: {total_rows:,}", end='\r')
        
        if total_rows >= target_rows:
            break
    
    print(f"\nData generation complete! Total rows: {total_rows:,}")
    return total_rows

def analyze_dataset(file_path):
    """Analyze the generated dataset and print comprehensive statistics."""
    print("\nAnalyzing dataset...")
    
    # Read the full dataset
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate number of months in the dataset
    total_months = (df['date'].max() - df['date'].min()).days / 30.44  # Average days per month
    
    # Basic dataset statistics
    total_users = df['user_id'].nunique()
    total_transactions = len(df)
    
    # Calculate transaction frequency and average values by category per user per month
    print("\n=== Transaction Patterns by Category ===")
    print("\nMonthly Statistics per User by Category:")
    
    # Group by user, category, and month
    df['year_month'] = df['date'].dt.to_period('M')
    category_stats = df.groupby(['user_id', 'category', 'year_month']).agg({
        'amount': ['count', 'mean', 'sum']
    }).reset_index()
    
    # Calculate average monthly frequencies and amounts per category
    monthly_stats = category_stats.groupby('category').agg({
        ('amount', 'count'): ['mean', 'std'],  # Average and std dev of transaction frequency
        ('amount', 'mean'): ['mean', 'std'],   # Average and std dev of transaction amount
        ('amount', 'sum'): ['mean', 'std']     # Average and std dev of monthly spending
    })
    
    # Rename columns for clarity
    monthly_stats.columns = [
        'avg_transactions_per_user_month', 'std_transactions',
        'avg_transaction_amount', 'std_amount',
        'avg_monthly_spending_per_user', 'std_monthly_spending'
    ]
    
    # Add total users per category
    category_user_counts = df.groupby('category')['user_id'].nunique()
    monthly_stats['total_users'] = category_user_counts
    
    # Calculate percentage of users per category
    monthly_stats['user_percentage'] = (monthly_stats['total_users'] / total_users * 100)
    
    # Sort by average monthly spending per user
    monthly_stats = monthly_stats.sort_values('avg_monthly_spending_per_user', ascending=False)
    
    # Print detailed statistics
    print("\nDetailed Monthly Statistics by Category:")
    print("\nCategory-wise Frequency and Amount Analysis:")
    print(f"{'Category':<15} {'Freq/Month':<12} {'Avg Amount':>12} {'Monthly Spend':>15} {'% Users':>10}")
    print("-" * 65)
    
    for category in monthly_stats.index:
        freq = monthly_stats.loc[category, 'avg_transactions_per_user_month']
        avg_amount = monthly_stats.loc[category, 'avg_transaction_amount']
        monthly_spend = monthly_stats.loc[category, 'avg_monthly_spending_per_user']
        user_pct = monthly_stats.loc[category, 'user_percentage']
        
        print(f"{category:<15} {freq:>9.1f}/mo ${avg_amount:>10.2f} ${monthly_spend:>13.2f} {user_pct:>9.1f}%")
    
    print("\nDetailed Statistics for Key Categories:")
    for category in ['rent', 'education', 'groceries', 'transportation']:
        stats = monthly_stats.loc[category]
        print(f"\n{category.upper()}:")
        print(f"Transactions per month: {stats['avg_transactions_per_user_month']:.1f} ± {stats['std_transactions']:.1f}")
        print(f"Average transaction: ${stats['avg_transaction_amount']:.2f} ± ${stats['std_amount']:.2f}")
        print(f"Monthly spending: ${stats['avg_monthly_spending_per_user']:.2f} ± ${stats['std_monthly_spending']:.2f}")
        print(f"Percentage of users: {stats['user_percentage']:.1f}%")
    
    # Transaction timing patterns
    print("\n=== Transaction Timing Patterns ===")
    df['day_of_month'] = df['date'].dt.day
    timing_stats = df.groupby(['category', 'day_of_month']).size().reset_index()
    timing_stats.columns = ['category', 'day_of_month', 'count']
    
    print("\nTypical Transaction Days:")
    for category in ['rent', 'utilities', 'education']:
        category_timing = timing_stats[timing_stats['category'] == category]
        most_common_days = category_timing.nlargest(3, 'count')
        print(f"\n{category.capitalize()}:")
        print(f"Most common days of month: {', '.join(map(str, most_common_days['day_of_month'].values))}")
    
    # Value distribution
    print("\n=== Value Distribution by Category ===")
    value_percentiles = df.groupby('category')['amount'].agg([
        ('min', 'min'),
        ('25th', lambda x: x.quantile(0.25)),
        ('median', 'median'),
        ('75th', lambda x: x.quantile(0.75)),
        ('max', 'max')
    ])
    print("\nAmount Distribution by Category:")
    print(value_percentiles)

    return monthly_stats


if __name__ == "__main__":
    output_dir = os.getcwd()
    output_file = os.path.join(output_dir, 'synthetic_financial_data_5M.csv')
    
    print("Starting large-scale data generation process...")
    print(f"Output will be saved to: {output_file}")
    
    start_time = datetime.now()
    
    total_rows = generate_synthetic_transactions(
        target_rows=1_500_000,
        start_date='2023-01-01',
        end_date='2024-12-31',
        batch_size=100000,
        output_file=output_file
    )
    
    end_time = datetime.now()
    execution_time = end_time - start_time
    
    print("\nGeneration Summary:")
    print(f"Total rows generated: {total_rows:,}")
    print(f"Output file: {output_file}")
    print(f"Execution time: {execution_time}")
    
    # Read and display sample with all columns
    print("\nSample of generated data:")
    sample_df = pd.read_csv(output_file, nrows=5)
    pd.set_option('display.max_columns', None)  # Show all columns
    print(sample_df)

    analyze_dataset(output_file)