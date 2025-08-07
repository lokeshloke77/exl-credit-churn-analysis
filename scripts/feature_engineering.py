import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import joblib
import os
import warnings
warnings.filterwarnings('ignore')
from data_loader import load_data


def create_features(df):
    
    df = df.copy()
    
    # Age groups
    df['Age_Group'] = pd.cut(df['Age'], 
                            bins=[0, 30, 40, 50, 60, 100], 
                            labels=['Young', 'Adult', 'Middle', 'Senior', 'Elderly'])
    
    # Balance categories
    df['Balance_Category'] = pd.cut(df['Balance'],
                                   bins=[-1, 0, 50000, 100000, float('inf')],
                                   labels=['No_Balance', 'Low', 'Medium', 'High'])
    
    # Engagement score (0-1)
    df['Engagement_Score'] = (
        (df['Tenure'] / 10) * 0.3 +
        (df['NumOfProducts'] / 4) * 0.4 +
        df['IsActiveMember'] * 0.2 +
        df['HasCrCard'] * 0.1
    )
    
    # Balance to salary ratio
    df['Balance_Salary_Ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    
    # Products per tenure year
    df['Products_Per_Tenure'] = df['NumOfProducts'] / (df['Tenure'] + 1)
    
    return df

def perform_one_hot_encoding(df):
    """Perform one-hot encoding on categorical features"""
    print("Performing one-hot encoding...")
    
    # Original categorical columns
    categorical_columns = ['Gender']
    
    # New categorical features from feature engineering
    new_categorical = ['Age_Group', 'Balance_Category']
    
    all_categorical = categorical_columns + new_categorical
    
    # One-hot encode all categorical features
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_features = encoder.fit_transform(df[all_categorical])
    encoded_df = pd.DataFrame(encoded_features, 
                             columns=encoder.get_feature_names_out(all_categorical))
    
    # Combine with numerical features 
    numerical_columns = ['Age', 'Tenure', 'Balance', 'NumOfProducts', 
                        'EstimatedSalary', 'HasCrCard', 'IsActiveMember', 'Engagement_Score',
                        'Balance_Salary_Ratio', 'Products_Per_Tenure', 'Churn']
    
    df_processed = pd.concat([df[numerical_columns], encoded_df], axis=1)
    
    print(f"One-hot encoding completed. Shape: {df_processed.shape}")
    return df_processed, encoder

def save_processed_data(df, encoder):
    """Save processed data and encoder"""
    print("Saving processed data and encoder...")
    
    # Create directories
    os.makedirs('../data/processed', exist_ok=True)
    os.makedirs('../feature/encoders', exist_ok=True)
    
    # Save processed data
    df.to_csv('../data/processed/churn_features_encoded.csv', index=False)
    
    # Save encoder
    joblib.dump(encoder, '../feature/encoders/onehot_encoder.pkl')
    
    # Save feature information
    feature_info = {
        'total_features': df.shape[1],
        'feature_names': list(df.columns),
        'categorical_encoded': encoder.get_feature_names_out().tolist(),
        # Update the numerical_features list in feature_info
        'numerical_features': ['Age', 'Tenure', 'Balance', 'NumOfProducts', 
                      'EstimatedSalary', 'HasCrCard', 'IsActiveMember', 'Engagement_Score',
                      'Balance_Salary_Ratio', 'Products_Per_Tenure']
    }
    
    # Save as text file
    with open('../feature/encoders/feature_info.txt', 'w') as f:
        f.write(f"Total Features: {feature_info['total_features']}\n\n")
        f.write("All Features:\n")
        for feature in feature_info['feature_names']:
            f.write(f"- {feature}\n")
        f.write("\nCategorical Encoded Features:\n")
        for feature in feature_info['categorical_encoded']:
            f.write(f"- {feature}\n")
        f.write("\nNumerical Features:\n")
        for feature in feature_info['numerical_features']:
            f.write(f"- {feature}\n")
    
    print("Data and encoder saved successfully!")

def perform_feature_engineering(data="'../data/processed/churn_cleaned.csv"):
    
    print("=== Starting Feature Engineering Pipeline ===")
    
    # 1. Load data
    df_org= load_data(data)
    df_features = create_features(df_org)
    print("Features created successfully.")
    # 2. Perform one-hot encoding
    df_encoded, encoder = perform_one_hot_encoding(df_features)
    print("One-hot encoding completed successfully.")
    # 3. Save processed data and encoder
    save_processed_data(df_encoded, encoder)
    print("Feature engineering pipeline completed successfully.")
    print("=== Feature Engineering Pipeline Finished ===")

if __name__ == "__main__":
    perform_feature_engineering('../data/processed/churn_cleaned.csv')