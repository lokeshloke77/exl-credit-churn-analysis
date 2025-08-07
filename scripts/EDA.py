import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import data loader
from data_loader import load_data

def basic_data_visualization(data='../data/processed/churn_cleaned.csv'):
    """Minimal ML Pipeline with Feature Engineering and Modeling"""
    print("=== feature engineering Pipeline ===")
    
    # 1. Load data
    df = load_data(data)
    print(f"Loaded data with shape: {df.shape}")

    #feature engineering 
    # age distribution plot 1
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")  # Updated from sns.set

    # Age distribution plot 1
    plt.subplot(2, 2, 1)
    sns.histplot(df['Age'], bins=30, kde=True, color='skyblue')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
   
    # Tenure distribution plot 2
    plt.subplot(2, 2, 2)
    sns.histplot(df['Tenure'], bins=30, kde=True, color='salmon')
    plt.title('Tenure Distribution')
    plt.xlabel('Tenure (Months)')
    plt.ylabel('Frequency')
  
    # HasCrCard distribution plot 3
    plt.subplot(2, 2, 3)
    sns.countplot(x='HasCrCard', data=df, palette='Set2')
    plt.title('Has Credit Card Distribution')
    plt.xlabel('Has Credit Card')
    plt.ylabel('Count')
   
    # Active member distribution plot 4
    plt.subplot(2, 2, 4)
    sns.countplot(x='IsActiveMember', data=df, palette='Set1')
    plt.title('Active Member Distribution')
    plt.xlabel('Is Active Member')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('../feature/eda/basic_data_visualization.png')
    plt.show()
    
def churn_data_visualization(data='../data/processed/churn_cleaned.csv'):
    print("=== Churn Analysis ===")
    df= load_data(data)
    df['Churn'] = df['Churn'].apply(lambda x: 'Churned' if x == 1 else 'Not Churned')
    #show the churn and not churn pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(df['Churn'].value_counts(), labels=df['Churn'].value_counts().index, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff'])
    plt.title('Churn Distribution')
    plt.savefig('../feature/eda/churn_distribution.png')
    plt.show()

def  correlation_matrix(data='../data/processed/churn_cleaned.csv'):
    """Correlation Analysis"""
    df = load_data(data)
    #remove first non numeric column
    df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('Feature Correlation Matrix')
    plt.savefig('../feature/eda/correlation_matrix.png')
    plt.show()







if __name__ == "__main__":
    basic_data_visualization('../data/processed/churn_cleaned.csv')
    churn_data_visualization('../data/processed/churn_cleaned.csv')
    correlation_matrix('../data/processed/churn_cleaned.csv')
