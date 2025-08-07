import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Import from your existing modules
from feature_engineering import create_features, perform_one_hot_encoding
from model_serializer import load_model_components

def prepare_features(df):
    """Prepare features using feature_engineering functions"""
    df_processed = df.copy()
    
    print("Applying feature engineering...")
    
    # Use create_features from feature_engineering.py
    df_with_features = create_features(df_processed)
    
    # Manual one-hot encoding to match training exactly
    print("Applying one-hot encoding...")
    
    # Gender encoding
    df_with_features['Gender_Male'] = (df_with_features['Gender'] == 'Male').astype(int)
    
    # Age group encoding (drop 'Adult' as reference)
    for age_group in ['Elderly', 'Middle', 'Senior', 'Young']:
        df_with_features[f'Age_Group_{age_group}'] = (df_with_features['Age_Group'] == age_group).astype(int)
    
    # Balance category encoding (drop 'High' as reference)
    for balance_cat in ['Low', 'Medium', 'No_Balance']:
        df_with_features[f'Balance_Category_{balance_cat}'] = (df_with_features['Balance_Category'] == balance_cat).astype(int)
    
    # Select final features in the same order as training
    final_features = [
        'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary',
        'HasCrCard', 'IsActiveMember', 'Engagement_Score', 'Balance_Salary_Ratio',
        'Products_Per_Tenure', 'Gender_Male', 'Age_Group_Elderly',
        'Age_Group_Middle', 'Age_Group_Senior', 'Age_Group_Young',
        'Balance_Category_Low', 'Balance_Category_Medium', 'Balance_Category_No_Balance'
    ]
    
    df_final = df_with_features[final_features]
    
    print(f"Final features shape: {df_final.shape}")
    print(f"Features: {list(df_final.columns)}")
    return df_final

def predict_churn(input_data, model, scaler):
    """Make prediction for customer(s)"""
    try:
        # Convert to DataFrame if dictionary
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Prepare features using feature_engineering functions
        df_features = prepare_features(df)
        
        # Scale numerical features (same as training)
        numerical_columns = ['Age', 'Tenure', 'Balance', 'NumOfProducts', 
                           'EstimatedSalary', 'HasCrCard', 'IsActiveMember', 
                           'Engagement_Score', 'Balance_Salary_Ratio', 'Products_Per_Tenure']
        
        df_scaled = df_features.copy()
        
        # Only scale columns that exist in the dataset
        existing_numerical = [col for col in numerical_columns if col in df_features.columns]
        df_scaled[existing_numerical] = scaler.transform(df_features[existing_numerical])
        
        # Make predictions
        predictions = model.predict(df_scaled)
        probabilities = model.predict_proba(df_scaled)
        
        # Format results
        results = []
        for i in range(len(predictions)):
            result = {
                'prediction': 'Churn' if predictions[i] == 1 else 'No Churn',
                'churn_probability': round(probabilities[i][1], 4),
                'no_churn_probability': round(probabilities[i][0], 4),
                'risk_level': get_risk_level(probabilities[i][1])
            }
            results.append(result)
        
        return results if len(results) > 1 else results[0]
        
    except Exception as e:
        return {'error': f'Prediction failed: {str(e)}'}

def get_risk_level(churn_prob):
    """Categorize risk level based on churn probability"""
    if churn_prob >= 0.7:
        return 'High Risk'
    elif churn_prob >= 0.4:
        return 'Medium Risk'
    else:
        return 'Low Risk'

def main():
    """Run sample predictions"""
    print("=== Churn Prediction System ===")
    
    # Load model and scaler
    try:
        model, scaler = load_model_components('../feature/ml/models')
        if model is None or scaler is None:
            print("Failed to load model components")
            return
        print("Model and scaler loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Sample customer data (matching your original data structure)
    sample_customers = [
        {
            'Gender': 'Male',
            'Age': 42,
            'Tenure': 2,
            'Balance': 0.0,
            'NumOfProducts': 1,
            'HasCrCard': 1,
            'IsActiveMember': 1,
            'EstimatedSalary': 101348.88
        },
        {
            'Gender': 'Female',
            'Age': 41,
            'Tenure': 1,
            'Balance': 83807.86,
            'NumOfProducts': 1,
            'HasCrCard': 0,
            'IsActiveMember': 1,
            'EstimatedSalary': 112542.58
        },
        {
            'Gender': 'Male',
            'Age': 55,
            'Tenure': 8,
            'Balance': 159660.8,
            'NumOfProducts': 4,
            'HasCrCard': 1,
            'IsActiveMember': 0,
            'EstimatedSalary': 45000.00
        }
    ]
    
    print("\n=== Sample Churn Predictions ===")
    print("=" * 60)
    
    for i, customer in enumerate(sample_customers, 1):
        print(f"\nCustomer {i} Profile:")
        print(f"  Age: {customer['Age']}, Gender: {customer['Gender']}")
        print(f"  Balance: ${customer['Balance']:,.2f}, Salary: ${customer['EstimatedSalary']:,.2f}")
        print(f"  Products: {customer['NumOfProducts']}, Tenure: {customer['Tenure']} years")
        print(f"  Active: {'Yes' if customer['IsActiveMember'] else 'No'}, Credit Card: {'Yes' if customer['HasCrCard'] else 'No'}")
        
        result = predict_churn(customer, model, scaler)
        if 'error' in result:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Prediction: {result['prediction']}")
            print(f"  Churn Probability: {result['churn_probability']:.1%}")
            print(f"  Risk Level: {result['risk_level']}")

if __name__ == "__main__":
    main()