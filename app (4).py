
import streamlit as st
import pandas as pd
import joblib

# Load files
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('features.pkl')

st.title("Customer Churn Prediction System")

# Inputs
CreditScore = st.number_input("Credit Score", 300, 900, 600)

Geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

Gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

Age = st.number_input("Age", 18, 100, 30)

Tenure = st.number_input("Tenure", 0, 10, 5)

Balance = st.number_input(
    "Balance",
    0.0,
    300000.0,
    50000.0
)

NumOfProducts = st.number_input(
    "Number Of Products",
    1,
    4,
    1
)

HasCrCard = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

IsActiveMember = st.selectbox(
    "Is Active Member",
    [0, 1]
)

EstimatedSalary = st.number_input(
    "Estimated Salary",
    0.0,
    300000.0,
    50000.0
)

# Encoding
geo_map = {
    'France': 0,
    'Germany': 1,
    'Spain': 2
}

gender_map = {
    'Female': 0,
    'Male': 1
}

# Input dictionary
input_dict = {
    'CreditScore': CreditScore,
    'Geography': geo_map[Geography],
    'Gender': gender_map[Gender],
    'Age': Age,
    'Tenure': Tenure,
    'Balance': Balance,
    'NumOfProducts': NumOfProducts,
    'HasCrCard': HasCrCard,
    'IsActiveMember': IsActiveMember,
    'EstimatedSalary': EstimatedSalary
}

# Convert to DataFrame
input_data = pd.DataFrame([input_dict])

# Arrange columns properly
input_data = input_data[feature_names]

# Scale
input_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_scaled)

# Output
if st.button("Predict"):
    
    if prediction[0] == 1:
        st.error("Customer is likely to churn")
        
    else:
        st.success("Customer is likely to stay")
