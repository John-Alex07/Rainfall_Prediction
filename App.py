import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
# Load the pre-trained LightGBM model
# model = lgb.Booster(model_file='lightgbm_model.txt')
model = joblib.load("./Code/model_lgb.pkl")

# Function to predict whether it will rain tomorrow
def predict_rain(data):
    # Convert input data to DataFrame
    df = pd.DataFrame(data, index=[0])
    
    # Preprocess the input data (you may need to adjust this based on your preprocessing steps)
    # For example: handle missing values, encode categorical variables, etc.
    
    # Make prediction
    prediction = model.predict(df)[0]
    
    return prediction

# Main function to run the Streamlit web app
def main():
    # Set page title and description
    st.title('Rain Prediction App')
    st.write('This app predicts whether it will rain tomorrow based on meteorological data.')
    
    # Add input fields for meteorological data
    st.header('Input Features')
    min_temp = st.number_input('Minimum Temperature (°C)', value=20.0)
    max_temp = st.number_input('Maximum Temperature (°C)', value=25.0)
    rainfall = st.number_input('Rainfall (mm)', value=0.0)
    sunshine = st.number_input('Sunshine (hours)', value=8.0)
    wind_gust_speed = st.number_input('Wind Gust Speed (km/h)', value=30.0)
    wind_speed_9am = st.number_input('Wind Speed at 9am (km/h)', value=10.0)
    wind_speed_3pm = st.number_input('Wind Speed at 3pm (km/h)', value=15.0)
    humidity_9am = st.number_input('Humidity at 9am (%)', value=70)
    humidity_3pm = st.number_input('Humidity at 3pm (%)', value=50)
    pressure_9am = st.number_input('Pressure at 9am (hPa)', value=1010.0)
    pressure_3pm = st.number_input('Pressure at 3pm (hPa)', value=1015.0)
    temp_9am = st.number_input('Temperature at 9am (°C)', value=22.0)
    temp_3pm = st.number_input('Temperature at 3pm (°C)', value=24.0)
    risk_mm = st.number_input('Risk of Rainfall (mm)', value=0.0)
    
    # Create a dictionary with input data
    input_data = {
        'MinTemp': min_temp,
        'MaxTemp': max_temp,
        'Rainfall': rainfall,
        'Sunshine': sunshine,
        'WindGustSpeed': wind_gust_speed,
        'WindSpeed9am': wind_speed_9am,
        'WindSpeed3pm': wind_speed_3pm,
        'Humidity9am': humidity_9am,
        'Humidity3pm': humidity_3pm,
        'Pressure9am': pressure_9am,
        'Pressure3pm': pressure_3pm,
        'Temp9am': temp_9am,
        'Temp3pm': temp_3pm,
        'RISK_MM': risk_mm
    }
    
    # Predict whether it will rain tomorrow
    if st.button('Predict'):
        prediction = predict_rain(input_data)
        if prediction > 0.5:
            st.error('It will rain tomorrow.')
        else:
            st.success('It will not rain tomorrow.')

# Run the main function
if __name__ == '__main__':
    main()
