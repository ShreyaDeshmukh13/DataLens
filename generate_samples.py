import os
import pandas as pd
import numpy as np

os.makedirs('samples', exist_ok=True)

# Generate Sales Data
np.random.seed(42)
dates = pd.date_range(start='2025-01-01', periods=100)
sales_df = pd.DataFrame({
    'date': dates,
    'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
    'product': np.random.choice(['Laptop', 'Phone', 'Tablet', 'Monitor'], 100),
    'revenue': np.random.normal(5000, 1500, 100).round(2),
    'units_sold': np.random.randint(10, 100, 100),
    'satisfaction_score': np.random.uniform(3.0, 5.0, 100).round(1)
})
# Introduce some missing values
sales_df.loc[5:10, 'revenue'] = np.nan
sales_df.to_csv('samples/sales.csv', index=False)

# Generate HR Data
hr_df = pd.DataFrame({
    'employee_id': range(1001, 1101),
    'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR'], 100, p=[0.5, 0.3, 0.1, 0.1]),
    'salary': np.random.normal(80000, 20000, 100).round(-3),
    'years_at_company': np.random.randint(0, 15, 100),
    'performance_rating': np.random.choice([1, 2, 3, 4, 5], 100, p=[0.05, 0.1, 0.6, 0.2, 0.05]),
    'left_company': np.random.choice(['Yes', 'No'], 100, p=[0.15, 0.85])
})
hr_df.to_csv('samples/hr.csv', index=False)

print("Samples created.")
