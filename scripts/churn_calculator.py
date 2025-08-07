import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_churn_status(df):
    """
    Calculate churn status based on business rules:
    1. No transactions for 3+ months
    2. Average monthly spend drops below 50% of last known average
    3. Mark as Churn = 1
    """
    df = df.copy()
    
    # Assuming we need to create transaction history indicators from existing data
    # Since we don't have transaction dates, we'll use proxy indicators
    
    # Rule 1: Proxy for "no transactions for 3+ months" 
    # Use IsActiveMember = 0 as indicator of recent inactivity
    no_recent_transactions = (df['IsActiveMember'] == 0)
    
    # Rule 2: Proxy for "spending drop"
    # Use low balance relative to salary as indicator of reduced spending
    df['Balance_to_Salary_Ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    spending_dropped = (df['Balance_to_Salary_Ratio'] < 0.5)  # Less than 50% of annual salary in balance
    
    # Rule 3: Additional churn indicators
    # Low engagement: Few products + inactive + low balance
    low_engagement = (
        (df['NumOfProducts'] <= 1) & 
        (df['IsActiveMember'] == 0) & 
        (df['Balance'] < df['Balance'].median())
    )
    
    # Calculate churn score (0-1)
    churn_score = (
        no_recent_transactions.astype(int) * 0.4 +
        spending_dropped.astype(int) * 0.3 +
        low_engagement.astype(int) * 0.3
    )
    
    # Create binary churn label
    # Customer churns if churn_score >= 0.6 (indicating majority of conditions met)
    df['Churn_Calculated'] = (churn_score >= 0.6).astype(int)
    
    # If original Churn column exists, compare and use business logic as override
    if 'Churn' in df.columns:
        # Use calculated churn if it's more restrictive, otherwise keep original
        df['Churn_Final'] = np.maximum(df['Churn'], df['Churn_Calculated'])
    else:
        df['Churn_Final'] = df['Churn_Calculated']
    
    # Clean up temporary columns
    df = df.drop(['Balance_to_Salary_Ratio', 'Churn_Calculated'], axis=1)
    df = df.rename(columns={'Churn_Final': 'Churn'})
    
    return df

def add_churn_indicators(df):
    """Add additional features that help identify churn patterns"""
    df = df.copy()
    
    # Tenure-based risk (customers with very short tenure are higher risk)
    df['Tenure_Risk'] = (df['Tenure'] <= 2).astype(int)
    
    # Age-based risk (middle-aged customers often have higher churn)
    df['Age_Risk'] = ((df['Age'] >= 35) & (df['Age'] <= 55)).astype(int)
    
    # Product engagement risk
    df['Low_Product_Engagement'] = (df['NumOfProducts'] == 1).astype(int)
    
    # Credit card non-adoption risk
    df['No_Credit_Card'] = (df['HasCrCard'] == 0).astype(int)
    
    return df