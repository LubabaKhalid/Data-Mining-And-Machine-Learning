
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

st.title("💻 Laptop Price Predictor")

st.write("Enter the specifications of the laptop below:")

# Input fields based on commonly used features
brand = st.selectbox("Brand", ['Dell', 'HP', 'Lenovo', 'Asus'])
processor_brand = st.selectbox("Processor Brand", ['Intel', 'AMD'])
num_ratings = st.number_input("Number of Ratings", min_value=0, value=100)
num_reviews = st.number_input("Number of Reviews", min_value=0, value=20)

# Encode inputs to match trained model structure (simplified for now)
input_dict = {
    'Number of Ratings': num_ratings,
    'Number of Reviews': num_reviews,
    'brand_Dell': 1 if brand == 'Dell' else 0,
    'brand_HP': 1 if brand == 'HP' else 0,
    'brand_Lenovo': 1 if brand == 'Lenovo' else 0,
    'brand_Asus': 1 if brand == 'Asus' else 0,
    'processor_brand_Intel': 1 if processor_brand == 'Intel' else 0,
    'processor_brand_AMD': 1 if processor_brand == 'AMD' else 0
}

# Ensure all expected columns are included
expected_columns = [
    'Number of Ratings', 'Number of Reviews',
    'brand_Dell', 'brand_HP', 'brand_Lenovo', 'brand_Asus',
    'processor_brand_Intel', 'processor_brand_AMD'
]
for col in expected_columns:
    if col not in input_dict:
        input_dict[col] = 0

input_df = pd.DataFrame([input_dict])

# Load model (You must have 'laptop_model.pkl' saved)
try:
    model = joblib.load("laptop_model.pkl")
    if st.button("Predict Price"):
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Estimated Laptop Price: ₹{prediction:,.2f}")
except FileNotFoundError:
    st.error("⚠️ Model file 'laptop_model.pkl' not found. Please train and save your model first.")
