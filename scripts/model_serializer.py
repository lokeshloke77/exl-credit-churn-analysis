import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_model_components(load_dir='../feature/ml/models'):
    """Load saved model and scaler"""
    try:
        model = joblib.load(f'{load_dir}/random_forest_model.pkl')
        scaler = joblib.load(f'{load_dir}/minmax_scaler.pkl')
        print("Model components loaded successfully")
        return model, scaler
    except FileNotFoundError as e:
        print(f"Error loading model components: {e}")
        return None, None
    
if __name__ == "__main__":
    model, scaler = load_model_components()
    if model is None or scaler is None:
        print("Failed to load model components. Exiting.")
    else:
        print("Model and scaler loaded successfully.")
        
    