import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model and preprocessors
with open('linear_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler_X.pkl', 'rb') as scaler_x_file:
    scaler_X = pickle.load(scaler_x_file)

with open('scaler_y.pkl', 'rb') as scaler_y_file:
    scaler_y = pickle.load(scaler_y_file)

with open('label_encoder.pkl', 'rb') as le_file:
    l_encoder = pickle.load(le_file)

# Streamlit app layout
st.title("Housing Price Prediction")
st.write("Enter the details of the house:")

# User input
square_feet = st.number_input("Square Feet", min_value=0)
bedrooms = st.number_input("Bedrooms", min_value=0)
bathrooms = st.number_input("Bathrooms", min_value=0)
neighborhood = st.selectbox("Neighborhood", options=["Rural", "Suburb", "Urban"])
year_built = st.number_input("Year Built", min_value=1900, max_value=2023)

# Predict button
if st.button("Predict"):
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'SquareFeet': [square_feet],
        'Bedrooms': [bedrooms],
        'Bathrooms': [bathrooms],
        'Neighborhood': [l_encoder.transform([neighborhood])[0]],  # Encode neighborhood
        'YearBuilt': [year_built]
    })

    # Scale the input data
    input_data[['SquareFeet', 'Bedrooms', 'Bathrooms', 'YearBuilt']] = scaler_X.transform(input_data[['SquareFeet', 'Bedrooms', 'Bathrooms', 'YearBuilt']])
    
    # Make prediction
    predicted_price_scaled = model.predict(input_data)
    predicted_price = scaler_y.inverse_transform(predicted_price_scaled.reshape(-1, 1))  # Inverse scale

    # Display the predicted price
    st.success(f"Predicted Price: {predicted_price[0][0]:.2f} INR")
