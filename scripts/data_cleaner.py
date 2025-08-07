from data_loader import load_data
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

#handle age
def handle_age(df):
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

    valid_ages = df[(df['Age'] >= 18) & (df['Age'] <= 100)]['Age']
    age_mode = valid_ages.mode()[0]  # Get the most frequent valid age
    
    # Replace invalid ages with mode
    df.loc[(df['Age'] < 18) | (df['Age'] > 100), 'Age'] = age_mode
    
    # Fill any remaining missing ages with mode
    df['Age'] = df['Age'].fillna(age_mode)
    return df

#data inconsistency
def handle_gender(df):
    # Standardize gender - remove extra spaces and standardize format
    df['Gender'] = df['Gender'].astype(str).str.strip()  # Remove leading/trailing spaces
    df['Gender'] = df['Gender'].str.title()  # Convert to title case
    
    # Additional cleaning for common variations
    df['Gender'] = df['Gender'].replace({
        'M': 'Male',
        'F': 'Female', 
        'MALE': 'Male',
        'FEMALE': 'Female',
        'male': 'Male',
        'female': 'Female'
    })
    
    # Get the mode (most frequent value)
    gender_mode = df['Gender'].mode()[0]
    
    # Fill missing values with mode
    df['Gender'] = df['Gender'].fillna(gender_mode)
    
    # Final validation - ensure only 'Male' or 'Female'
    valid_genders = ['Male', 'Female']
    invalid_mask = ~df['Gender'].isin(valid_genders)
    df.loc[invalid_mask, 'Gender'] = gender_mode
    
    return df
def handle_balance(df):
    # Ensure 'Balance' is numeric, coercing errors to NaN
    df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce')
    
    df.loc[df['Balance'] < 0, 'Balance'] = 0
    
    # Fill missing balance values with median (better than mean for financial data)
    balance_median = df['Balance'].median()
    df['Balance'] = df['Balance'].fillna(balance_median)
    return df


#handle estimated salary
def handle_estimated_salary(df):
    # Ensure 'EstimatedSalary' is numeric, coercing errors to NaN
    df['EstimatedSalary'] = pd.to_numeric(df['EstimatedSalary'], errors='coerce')
    # Replace negative salaries with the median salary
    median_salary = df['EstimatedSalary'].median()
    df.loc[df['EstimatedSalary'] < 0, 'EstimatedSalary'] = median_salary
    # Fill any remaining missing salaries with median
    df['EstimatedSalary'] = df['EstimatedSalary'].fillna(median_salary)
    return df

#handle credit card
def handle_crcard(df):
    
    df['HasCrCard'] = pd.to_numeric(df['HasCrCard'], errors='coerce')

    # Replace any value > 1 or < 0 with the mode
    valid_crcard = df[(df['HasCrCard'] >= 0) & (df['HasCrCard'] <= 1)]['HasCrCard']
    crcard_mode = valid_crcard.mode()[0]  
    
    # Replace invalid values with mode
    df.loc[(df['HasCrCard'] < 0) | (df['HasCrCard'] > 1), 'HasCrCard'] = crcard_mode
    
    # Fill any remaining missing values with mode
    df['HasCrCard'] = df['HasCrCard'].fillna(crcard_mode)
    
    # Ensure values are integers (0 or 1)
    df['HasCrCard'] = df['HasCrCard'].astype(int)
    
    return df

# handle duplicate records
def handle_duplicates(df):
    df = df.drop_duplicates()
    return df


def handle_churn(df):
    # Ensure 'Churn' is numeric (0 or 1)
    df['Churn'] = pd.to_numeric(df['Churn'], errors='coerce')
    
    # Validate values are only 0 or 1
    valid_churn = df[(df['Churn'] >= 0) & (df['Churn'] <= 1)]['Churn']
    churn_mode = valid_churn.mode()[0]
    
    # Replace invalid values with mode
    df.loc[(df['Churn'] < 0) | (df['Churn'] > 1), 'Churn'] = churn_mode
    df['Churn'] = df['Churn'].fillna(churn_mode)
    df['Churn'] = df['Churn'].astype(int)
    return df

# handle isactive member
def handle_isactive_member(df):
    # Ensure 'IsActiveMember' is numeric, coercing errors to NaN
    df['IsActiveMember'] = pd.to_numeric(df['IsActiveMember'], errors='coerce')
    # Replace any value > 1 or < 0 with the mode
    
    valid_active = df[(df['IsActiveMember'] >= 0) & (df['IsActiveMember'] <= 1)]['IsActiveMember']
    active_mode = valid_active.mode()[0]
    # Replace invalid values with mode
    df.loc[(df['IsActiveMember'] < 0) | (df['IsActiveMember'] > 1), 'IsActiveMember'] = active_mode
    # Fill any remaining missing values with mode
    df['IsActiveMember'] = df['IsActiveMember'].fillna(active_mode)
    # Ensure values are integers (0 or 1)
    df['IsActiveMember'] = df['IsActiveMember'].astype(int)
    return df

def clean_data(file_path):
    df = load_data(file_path)
    df = handle_duplicates(df)
    df = handle_gender(df)
    df = handle_age(df)
    df = handle_balance(df)
    df = handle_estimated_salary(df)
    df = handle_crcard(df)
    df = handle_churn(df)
    df = handle_isactive_member(df)
    # df = calculate_churn_status(df)
    # df = add_churn_indicators(df)

    return df

#main
if __name__ == "__main__":
    file_path = '../data/raw/exl_credit_card_churn_data.csv'     
    cleaned_data = clean_data(file_path)
    print("Cleaned Data:================")
    print(cleaned_data.head())
    #save processed data to a new CSV file
    cleaned_data.to_csv('../data/processed/churn_cleaned.csv', index=False)
    print("Data cleaning completed and saved to '../data/processed/churn_cleaned.csv'.")


