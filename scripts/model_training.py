import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')
from data_loader import load_data

# def load_encoded_data():
#         df = load_data('../data/processed/churn_features_encoded.csv')
#         print(f"Encoded data loaded successfully with shape: {df.shape}")
#         print(f"Features: {list(df.columns)}")
#         return df

def prepare_features(df):
    """Prepare features for modeling"""
    print("Preparing features for modeling...")
    
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Define numerical columns for scaling
    numerical_columns = ['Age', 'Tenure', 'Balance', 'NumOfProducts', 
                        'EstimatedSalary', 'HasCrCard', 'IsActiveMember', 
                        'Engagement_Score', 'Balance_Salary_Ratio', 'Products_Per_Tenure']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    return X, y, numerical_columns

def train_random_forest_model(X_train, X_test, y_train, y_test):
    """Train Random Forest model and evaluate performance"""
    print("Training Random Forest model...")
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Feature importance (top 3)
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_3_features = feature_importance.head(3)
    
    # Print results
    print(f"\n=== Model Evaluation Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(cm)
    
    print(f"\nTop 3 Influential Features:")
    for idx, row in top_3_features.iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm,
        'top_3_features': top_3_features,
        'feature_importance': feature_importance
    }
    
    return rf_model, results

def save_model_and_results(model, scaler, results):
    """Save model and results to /feature/ml branch"""
    print("Saving model and results...")
    
    # Create directories
    os.makedirs('../feature/ml/models', exist_ok=True)
    os.makedirs('../feature/ml/results', exist_ok=True)
    os.makedirs('../feature/ml/plots', exist_ok=True)
    
    # Save model components
    joblib.dump(model, '../feature/ml/models/random_forest_model.pkl')
    joblib.dump(scaler, '../feature/ml/models/minmax_scaler.pkl')
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall'],
        'Value': [results['accuracy'], results['precision'], results['recall']]
    })
    results_df.to_csv('../feature/ml/results/model_metrics.csv', index=False)
    
    # Save feature importance
    results['feature_importance'].to_csv('../feature/ml/results/feature_importance.csv', index=False)
    
    # Save confusion matrix
    cm_df = pd.DataFrame(results['confusion_matrix'], 
                        columns=['Predicted_0', 'Predicted_1'],
                        index=['Actual_0', 'Actual_1'])
    cm_df.to_csv('../feature/ml/results/confusion_matrix.csv')
    
    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('../feature/ml/plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_10_features = results['feature_importance'].head(10)
    plt.barh(range(len(top_10_features)), top_10_features['importance'])
    plt.yticks(range(len(top_10_features)), top_10_features['feature'])
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('../feature/ml/plots/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Model and results saved successfully!")

def main():
    """Main ML training pipeline using one-hot encoded data"""
    print("=== Random Forest Model Training with One-Hot Encoded Data ===")
    
    # 1. Load one-hot encoded data
    df = load_data('../data/processed/churn_features_encoded.csv')
    # 2. Prepare features
    X, y, numerical_columns = prepare_features(df)
    
    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    # 4. Normalize numerical features using MinMaxScaler (mandatory)
    print("Normalizing numerical features using MinMaxScaler...")
    scaler = MinMaxScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Only scale the numerical columns
    X_train_scaled[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_test_scaled[numerical_columns] = scaler.transform(X_test[numerical_columns])
    
    # 5. Train Random Forest model
    model, results = train_random_forest_model(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # 6. Save model and results
    save_model_and_results(model, scaler, results)
    
    print("\n=== Model Training Completed Successfully ===")
    print("Check the following directories:")
    print("- Model files: ../feature/ml/models/")
    print("- Results: ../feature/ml/results/")
    print("- Plots: ../feature/ml/plots/")

if __name__ == "__main__":
    main()