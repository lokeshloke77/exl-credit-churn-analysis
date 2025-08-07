import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from model_predict import predict_churn, get_risk_level
from model_serializer import load_model_components
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Credit Card Churn Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
    }
    .high-risk {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
    }
    .medium-risk {
        background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%);
        color: white;
    }
    .low-risk {
        background: linear-gradient(135deg, #48dbfb 0%, #0abde3 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for model loading
if 'model_loaded' not in st.session_state:
    try:
        st.session_state.model, st.session_state.scaler = load_model_components('../feature/ml/models')
        if st.session_state.model is not None and st.session_state.scaler is not None:
            st.session_state.model_loaded = True
        else:
            st.session_state.model_loaded = False
            st.session_state.error_message = "Failed to load model components"
    except Exception as e:
        st.session_state.model_loaded = False
        st.session_state.error_message = str(e)

def main():
    st.markdown('<h1 class="main-header">💳 Credit Card Churn Prediction</h1>', unsafe_allow_html=True)
    
    # Check if model is loaded
    if not st.session_state.model_loaded:
        st.error(f"Failed to load model: {st.session_state.error_message}")
        st.info("Please ensure that the model files are available in '../feature/ml/models/' directory")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["Single Prediction", "Batch Prediction", "Model Info", "Sample Data"])
    
    if page == "Single Prediction":
        single_prediction_page()
    elif page == "Batch Prediction":
        batch_prediction_page()
    elif page == "Model Info":
        model_info_page()
    elif page == "Sample Data":
        sample_data_page()

def single_prediction_page():
    st.header("🎯 Single Customer Prediction")
    
    # Create input form
    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Personal Information")
            age = st.number_input("Age", min_value=18, max_value=100, value=40, step=1)
            gender = st.selectbox("Gender", ["Female", "Male"])
        
        with col2:
            st.subheader("Account Information")
            tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5, step=1)
            balance = st.number_input("Balance ($)", min_value=0.0, value=50000.0, step=100.0)
            num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2, step=1)
            estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=100000.0, step=1000.0)
        
        with col3:
            st.subheader("Engagement")
            has_cr_card = st.selectbox("Has Credit Card", [0, 1], format_func=lambda x: "Yes" if x else "No")
            is_active_member = st.selectbox("Is Active Member", [0, 1], format_func=lambda x: "Yes" if x else "No")
        
        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)
        
        if submitted:
            # Prepare customer data (matching your model_predict.py structure)
            customer_data = {
                'Gender': gender,
                'Age': age,
                'Tenure': tenure,
                'Balance': balance,
                'NumOfProducts': num_products,
                'HasCrCard': has_cr_card,
                'IsActiveMember': is_active_member,
                'EstimatedSalary': estimated_salary
            }
            
            # Make prediction using your model_predict functions
            with st.spinner("Making prediction..."):
                result = predict_churn(customer_data, st.session_state.model, st.session_state.scaler)
            
            # Display results
            if 'error' in result:
                st.error(f"Prediction failed: {result['error']}")
            else:
                display_prediction_result(result, customer_data)

def display_prediction_result(result, customer_data):
    """Display prediction results with styling"""
    
    # Determine risk styling
    risk_level = result['risk_level']
    if risk_level == "High Risk":
        risk_class = "high-risk"
        risk_emoji = "🚨"
    elif risk_level == "Medium Risk":
        risk_class = "medium-risk"
        risk_emoji = "⚠️"
    else:
        risk_class = "low-risk"
        risk_emoji = "✅"
    
    # Main prediction box
    st.markdown(f"""
    <div class="prediction-box {risk_class}">
        <h2>{risk_emoji} {result['prediction']}</h2>
        <h3>Risk Level: {risk_level}</h3>
        <h4>Churn Probability: {result['churn_probability']:.1%}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Churn Probability", f"{result['churn_probability']:.1%}")
    
    with col2:
        st.metric("No Churn Probability", f"{result['no_churn_probability']:.1%}")
    
    with col3:
        st.metric("Risk Level", risk_level)
    
    # Probability visualization
    st.subheader("📊 Probability Breakdown")
    
    fig = go.Figure(data=[
        go.Bar(name='No Churn', x=['Prediction'], y=[result['no_churn_probability']], 
               marker_color='lightgreen'),
        go.Bar(name='Churn', x=['Prediction'], y=[result['churn_probability']], 
               marker_color='lightcoral')
    ])
    
    fig.update_layout(
        barmode='stack',
        title='Churn vs No Churn Probability',
        yaxis_title='Probability',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Customer profile summary
    st.subheader("👤 Customer Profile Summary")
    profile_col1, profile_col2 = st.columns(2)
    
    with profile_col1:
        st.write(f"**Age:** {customer_data['Age']} years")
        st.write(f"**Gender:** {customer_data['Gender']}")
        st.write(f"**Balance:** ${customer_data['Balance']:,.2f}")
        st.write(f"**Tenure:** {customer_data['Tenure']} years")
    
    with profile_col2:
        st.write(f"**Products:** {customer_data['NumOfProducts']}")
        st.write(f"**Active Member:** {'Yes' if customer_data['IsActiveMember'] else 'No'}")
        st.write(f"**Has Credit Card:** {'Yes' if customer_data['HasCrCard'] else 'No'}")
        st.write(f"**Estimated Salary:** ${customer_data['EstimatedSalary']:,.2f}")

def batch_prediction_page():
    st.header("📊 Batch Prediction")
    
    st.write("Upload a CSV file with customer data for batch predictions.")
    st.write("**Required columns:** Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            st.subheader("📋 Data Preview")
            st.dataframe(df.head())
            
            # Check for required columns
            required_cols = ['Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                           'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                return
            
            if st.button("🔮 Predict All", use_container_width=True):
                with st.spinner("Making batch predictions..."):
                    # Make predictions for each row
                    results = []
                    for _, row in df.iterrows():
                        customer_data = row[required_cols].to_dict()
                        result = predict_churn(customer_data, st.session_state.model, st.session_state.scaler)
                        if 'error' not in result:
                            results.append(result)
                        else:
                            # Handle error case
                            results.append({
                                'prediction': 'Error',
                                'churn_probability': 0,
                                'no_churn_probability': 0,
                                'risk_level': 'Error'
                            })
                
                if results:
                    df_results = pd.DataFrame(results)
                    df_combined = pd.concat([df, df_results], axis=1)
                    
                    st.subheader("📈 Prediction Results")
                    st.dataframe(df_combined)
                    
                    # Summary statistics
                    st.subheader("📊 Summary Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    valid_results = [r for r in results if r['prediction'] != 'Error']
                    churn_count = len([r for r in valid_results if r['prediction'] == 'Churn'])
                    total_count = len(valid_results)
                    avg_churn_prob = np.mean([r['churn_probability'] for r in valid_results]) if valid_results else 0
                    
                    with col1:
                        st.metric("Total Customers", total_count)
                    
                    with col2:
                        st.metric("Predicted Churners", churn_count)
                    
                    with col3:
                        st.metric("Average Churn Probability", f"{avg_churn_prob:.1%}")
                    
                    # Risk distribution chart
                    if valid_results:
                        risk_counts = pd.Series([r['risk_level'] for r in valid_results]).value_counts()
                        fig = px.pie(values=risk_counts.values, names=risk_counts.index, 
                                    title="Risk Level Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Download results
                    csv = df_combined.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="churn_predictions.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("No valid predictions were generated")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def model_info_page():
    st.header("🤖 Model Information")
    
    st.subheader("📋 Required Features")
    features = [
        'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
        'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
    ]
    
    features_df = pd.DataFrame({
        'Feature': features,
        'Description': [
            'Customer gender (Male/Female)',
            'Customer age in years',
            'Years as customer',
            'Account balance in dollars',
            'Number of bank products used',
            'Has credit card (0/1)',
            'Is active member (0/1)',
            'Estimated annual salary'
        ]
    })
    st.dataframe(features_df)
    
    st.subheader("ℹ️ Model Details")
    st.write("""
    - **Model Type**: Random Forest Classifier
    - **Features**: 18 engineered features including demographic, financial, and behavioral data
    - **Target**: Binary classification (Churn vs No Churn)
    - **Feature Engineering**: Includes age groups, balance categories, and engagement scores
    - **Risk Levels**: 
        - High Risk: ≥70% churn probability
        - Medium Risk: 40-70% churn probability  
        - Low Risk: <40% churn probability
    """)

def sample_data_page():
    st.header("📝 Sample Data")
    
    st.write("Here are some sample customer profiles you can use for testing:")
    
    # Sample customers (updated to match your model_predict.py structure)
    sample_customers = [
        {
            'Gender': 'Male',
            'Age': 42,
            'Tenure': 2,
            'Balance': 0.0,
            'NumOfProducts': 1,
            'HasCrCard': 1,
            'IsActiveMember': 1,
            'EstimatedSalary': 101348.88,
            'Description': 'Low-risk customer with no balance'
        },
        {
            'Gender': 'Female',
            'Age': 41,
            'Tenure': 1,
            'Balance': 83807.86,
            'NumOfProducts': 1,
            'HasCrCard': 0,
            'IsActiveMember': 1,
            'EstimatedSalary': 112542.58,
            'Description': 'Medium-risk customer with high balance'
        },
        {
            'Gender': 'Male',
            'Age': 55,
            'Tenure': 8,
            'Balance': 159660.8,
            'NumOfProducts': 4,
            'HasCrCard': 1,
            'IsActiveMember': 0,
            'EstimatedSalary': 45000.00,
            'Description': 'High-risk customer - multiple products, inactive'
        }
    ]
    
    for i, customer in enumerate(sample_customers, 1):
        with st.expander(f"Sample Customer {i}: {customer['Description']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Age:** {customer['Age']}")
                st.write(f"**Gender:** {customer['Gender']}")
                st.write(f"**Tenure:** {customer['Tenure']} years")
                st.write(f"**Balance:** ${customer['Balance']:,.2f}")
            
            with col2:
                st.write(f"**Products:** {customer['NumOfProducts']}")
                st.write(f"**Has Credit Card:** {'Yes' if customer['HasCrCard'] else 'No'}")
                st.write(f"**Active Member:** {'Yes' if customer['IsActiveMember'] else 'No'}")
                st.write(f"**Estimated Salary:** ${customer['EstimatedSalary']:,.2f}")
            
            if st.button(f"🔮 Predict Customer {i}", key=f"predict_{i}"):
                customer_data = {k: v for k, v in customer.items() if k != 'Description'}
                with st.spinner("Making prediction..."):
                    result = predict_churn(customer_data, st.session_state.model, st.session_state.scaler)
                
                if 'error' in result:
                    st.error(f"Prediction failed: {result['error']}")
                else:
                    st.success(f"Prediction: {result['prediction']} ({result['risk_level']} - {result['churn_probability']:.1%})")

if __name__ == "__main__":
    main()