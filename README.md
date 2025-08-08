# Credit Card Churn Prediction

## 📊 Project Overview
An end-to-end machine learning solution to predict credit card customer churn using Random Forest classifier with advanced feature engineering and interactive web interface.

## 🎯 Features
- **Data Preprocessing**: Comprehensive data cleaning and validation
- **Feature Engineering**: Advanced feature creation including age groups, balance categories, and engagement scores
- **Machine Learning**: Random Forest classifier with MinMaxScaler normalization
- **Prediction System**: Both single and batch prediction capabilities
- **Web Interface**: Interactive Streamlit application
- **Database Integration**: MySQL data loading and export functionality

## 🛠️ Technologies Used
- **Python 3.8+**
- **Scikit-learn**: Machine learning
- **Pandas & NumPy**: Data manipulation
- **Streamlit**: Web interface
- **MySQL**: Database integration
- **Plotly**: Interactive visualizations
- **Joblib**: Model serialization

## 📁 Project Structure
```
exl-credit-churn-analysis/
├── scripts/
│   ├── data_loader.py          # Data loading utilities
│   ├── data_cleaner.py         # Data preprocessing
│   ├── feature_engineering.py  # Feature creation
│   ├── model_training.py       # ML model training
│   ├── model_predict.py        # Prediction functions
│   ├── model_serializer.py     # Model persistence
│   ├── streamlit_app.py        # Web interface
│   └── csvtoexl.py            # Database operations
├── data/
│   ├── raw/                   # Original data files
│   ├── processed/             # Cleaned data
│   └── exports/               # Exported data
├── feature/ml/
│   ├── models/                # Trained models
│   ├── results/               # Evaluation results
│   └── plots/                 # Visualizations
└── README.md
```

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/lokeshloke77/exl-credit-churn-analysis.git
cd exl-credit-churn-analysis
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn streamlit plotly mysql-connector-python joblib
```

### 3. Run Data Pipeline
```bash
cd scripts
python data_cleaner.py
python feature_engineering.py
python model_training.py
```

### 4. Launch Web Application
```bash
streamlit run streamlit_app.py
```

## 📊 Model Performance
- **Algorithm**: Random Forest Classifier
- **Features**: 18 engineered features
- **Metrics**: Accuracy, Precision, Recall
- **Risk Categories**: High (≥70%), Medium (40-70%), Low (<40%)

## 🔧 Usage Examples

### Single Prediction
```python
from model_predict import predict_churn
from model_serializer import load_model_components

# Load model
model, scaler = load_model_components()

# Customer data
customer = {
    'Gender': 'Male',
    'Age': 42,
    'Tenure': 2,
    'Balance': 0.0,
    'NumOfProducts': 1,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 101348.88
}

# Predict
result = predict_churn(customer, model, scaler)
print(f"Prediction: {result['prediction']}")
print(f"Risk Level: {result['risk_level']}")
```

### Batch Prediction
Upload CSV file through Streamlit interface or use batch processing functions.

## 🗄️ Database Integration
- MySQL database support
- CSV to MySQL loading
- MySQL to CSV export
- Table structure management

## 📈 Business Impact
- **Proactive Retention**: Early identification of at-risk customers
- **Cost Reduction**: Prevent expensive customer acquisition
- **Revenue Protection**: Maintain customer lifetime value
- **Data-Driven Decisions**: Replace intuition with predictive insights

## 🤝 Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author
**Your Name**
- GitHub: [@lokeshloke77](https://github.com/lokeshloke77)


## 🙏 Acknowledgments
- Dataset: EXL Credit Card Churn Data
- Libraries: Scikit-learn, Pandas, Streamlit
- Tools: Python, MySQL, Git#

