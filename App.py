import base64
import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

# Load the pre-trained LightGBM model
model = joblib.load("./Code/model_lgb.pkl")

# Function to get base64 of binary file
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to predict whether it will rain tomorrow
def predict_rain(data):
    df = pd.DataFrame(data, index=[0])
    prediction = model.predict(df)[0]
    return prediction

# Set background image with a semi-transparent black overlay
page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://media.macphun.com/img/uploads/macphun/blog/1359/rain-photography-cover.jpg");
    background-size: cover;
    background-position: center;
}
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5); /* Semi-transparent black overlay */
    z-index: -1;
}
[data-testid="stHeader"] {
    background: rgba(255, 255, 255, 0);
    color: black;
}
.big-font, .header-font, .input-label, .description {
    background: rgba(255, 255, 255, 0.7); /* Semi-transparent white background */
    padding: 10px;
    border-radius: 5px;
    color: black;
}
.big-font {
    font-size: 40px !important;
    font-weight: bold;
    margin-top: 20px;
}
.header-font {
    font-size: 30px !important;
    font-weight: bold;
}
.input-label {
    font-size: 20px !important;
    font-weight: bold;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# Script to increase the visibility of slider value text and add semi-transparent background
slider_css = '''
<style>
div[data-baseweb="slider"] {
    background: rgba(255, 255, 255, 0.5); /* Semi-transparent white background */
    padding: 15px;
    border-radius: 5px;
}
div[data-baseweb="slider"] .css-14ok4jv {
    color: black !important;
    font-size: 15px !important;
    font-weight: 50px;
}
div[data-baseweb="slider"] .css-1n8m44i {
    background: #1f77b4 !important; /* Slider track color */
    font-weight: 50px;
}
div[data-baseweb="slider"] .css-7j44p5 {
    background: #ff7f0e !important; /* Slider thumb color */
    font-weight: 50px;
}
</style>
'''

st.markdown(slider_css, unsafe_allow_html=True)

# Custom CSS for the predict button
button_css = '''
<style>
div.stButton > button {
    color: white !important;
    background-color: #4CAF50 !important; /* Green background */
}
</style>
'''

st.markdown(button_css, unsafe_allow_html=True)

# Set page title and description with increased font size and bold text
st.markdown('<p class="big-font">🌧️ Rain Prediction App</p>', unsafe_allow_html=True)
st.markdown('<p class="description">This app predicts whether it will rain tomorrow based on meteorological data.</p>', unsafe_allow_html=True)

# Add input fields for meteorological data
st.markdown('<p class="header-font">Input Features</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="input-label">Minimum Temperature (°C)</p>', unsafe_allow_html=True)
    min_temp = st.slider('', -30.0, 50.0, 20.0, key='min_temp')
    
    st.markdown('<p class="input-label">Maximum Temperature (°C)</p>', unsafe_allow_html=True)
    max_temp = st.slider('', -30.0, 50.0, 25.0, key='max_temp')
    
    st.markdown('<p class="input-label">Rainfall (mm)</p>', unsafe_allow_html=True)
    rainfall = st.slider('', 0.0, 500.0, 0.0, key='rainfall')
    
    st.markdown('<p class="input-label">Sunshine (hours)</p>', unsafe_allow_html=True)
    sunshine = st.slider('', 0.0, 15.0, 8.0, key='sunshine')
    
    st.markdown('<p class="input-label">Wind Gust Speed (km/h)</p>', unsafe_allow_html=True)
    wind_gust_speed = st.slider('', 0.0, 200.0, 30.0, key='wind_gust_speed')
    
    st.markdown('<p class="input-label">Wind Speed at 9am (km/h)</p>', unsafe_allow_html=True)
    wind_speed_9am = st.slider('', 0.0, 200.0, 10.0, key='wind_speed_9am')
    
    st.markdown('<p class="input-label">Wind Speed at 3pm (km/h)</p>', unsafe_allow_html=True)
    wind_speed_3pm = st.slider('', 0.0, 200.0, 15.0, key='wind_speed_3pm')

with col2:
    st.markdown('<p class="input-label">Humidity at 9am (%)</p>', unsafe_allow_html=True)
    humidity_9am = st.slider('', 0, 100, 70, key='humidity_9am')
    
    st.markdown('<p class="input-label">Humidity at 3pm (%)</p>', unsafe_allow_html=True)
    humidity_3pm = st.slider('', 0, 100, 50, key='humidity_3pm')
    
    st.markdown('<p class="input-label">Pressure at 9am (hPa)</p>', unsafe_allow_html=True)
    pressure_9am = st.slider('', 400.0, 1100.0, 1010.0, key='pressure_9am')
    
    st.markdown('<p class="input-label">Pressure at 3pm (hPa)</p>', unsafe_allow_html=True)
    pressure_3pm = st.slider('', 400.0, 1100.0, 1015.0, key='pressure_3pm')
    
    st.markdown('<p class="input-label">Temperature at 9am (°C)</p>', unsafe_allow_html=True)
    temp_9am = st.slider('', 0.0, 60.0, 22.0, key='temp_9am')
    
    st.markdown('<p class="input-label">Temperature at 3pm (°C)</p>', unsafe_allow_html=True)
    temp_3pm = st.slider('', -30.0, 60.0, 24.0, key='temp_3pm')
    
    st.markdown('<p class="input-label">Risk of Rainfall (mm)</p>', unsafe_allow_html=True)
    risk_mm = st.slider('', 0.0, 5.0, 0.0, key='risk_mm')

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
        st.error('It will rain tomorrow. 🌧️')
    else:
        st.success('It will not rain tomorrow. ☀️')
