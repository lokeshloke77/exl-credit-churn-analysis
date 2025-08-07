import pandas as pd

def load_data(file_path):
    """Load data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully with shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
def print_duplicates(df):
    """Print duplicate rows in the DataFrame"""
    duplicates = df[df.duplicated()]
    if not duplicates.empty:
        print(f"Found {len(duplicates)} duplicate rows:")
        print(duplicates)
    else:
        print("No duplicate rows found.")
          

if __name__ == "__main__":
    # Test data loading
    df = load_data('../data/processed/churn_features_encoded.csv')
    # print_duplicates(df)
    if df is not None:
        print(df.head())