import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

@st.cache_data
def load_model():
    
    data = pd.read_csv("laptopPrice.csv")

   
    data = data[['brand', 'processor_brand', 'Number of Ratings', 'Number of Reviews', 'Price']]
    data.dropna(inplace=True)

  
    data = pd.get_dummies(data, columns=['brand', 'processor_brand'], drop_first=True)

    X = data.drop('Price', axis=1)
    y = data['Price']

    model = LinearRegression()
    model.fit(X, y)
    return model, X.columns.tolist()

model, feature_names = load_model()

st.title("💻 Laptop Price Predictor")
brand = st.selectbox("Brand", ['Dell', 'HP', 'Lenovo', 'Asus'])  # Adjust as per your dataset
processor = st.selectbox("Processor Brand", ['Intel', 'AMD'])    # Adjust as needed
num_ratings = st.number_input("Number of Ratings", min_value=0, value=100)
num_reviews = st.number_input("Number of Reviews", min_value=0, value=20)

input_data = {
    'Number of Ratings': num_ratings,
    'Number of Reviews': num_reviews,
}

for feature in feature_names:
    if feature.startswith('brand_'):
        input_data[feature] = 1 if f'brand_{brand}' == feature else 0
    elif feature.startswith('processor_brand_'):
        input_data[feature] = 1 if f'processor_brand_{processor}' == feature else 0

input_df = pd.DataFrame([input_data])

if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Laptop Price: ₹{prediction:,.2f}")
