import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load saved artifacts
model = joblib.load('logistic_aqi_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')  # This is city encoder
aqi_label_encoder = joblib.load('aqi_label_encoder.pkl')  # AQI encoder
feature_names = joblib.load('feature_names.pkl')
cat_dict = joblib.load('cat_dict.pkl')

# Streamlit UI
st.set_page_config(page_title="Air Quality Health Risk Predictor")
st.title("Air Quality Health Risk Prediction")

def get_user_input():
    # FIXED: Use city names from cat_dict
    city = st.selectbox("City", cat_dict['City'])
    
    pm25 = st.slider("PM2.5 (µg/m³)", 0, 500, 150)
    pm10 = st.slider("PM10 (µg/m³)", 0, 500, 200)
    no = st.slider("NO (µg/m³)", 0, 300, 50)
    no2 = st.slider("NO2 (µg/m³)", 0, 300, 50)
    nox = st.slider("NOx (µg/m³)", 0, 300, 100)
    nh3 = st.slider("NH3 (µg/m³)", 0, 500, 50)
    co = st.slider("CO (mg/m³)", 0, 30, 5)
    so2 = st.slider("SO2 (µg/m³)", 0, 100, 20)
    o3 = st.slider("O3 (µg/m³)", 0, 300, 50)
    benzene = st.slider("Benzene (µg/m³)", 0, 50, 5)
    toluene = st.slider("Toluene (µg/m³)", 0, 50, 5)
    xylene = st.slider("Xylene (µg/m³)", 0, 50, 5)
    aqi = st.slider("AQI", 0, 500, 150)

    # Prepare input in correct order
    data = {name: 0 for name in feature_names}
    
    #Convert city name to encoded value using label_encoder
    data['City'] = label_encoder.transform([city])[0]
    
    data['PM2.5'] = pm25
    data['PM10'] = pm10
    data['NO'] = no
    data['NO2'] = no2
    data['NOx'] = nox
    data['NH3'] = nh3
    data['CO'] = co
    data['SO2'] = so2
    data['O3'] = o3
    data['Benzene'] = benzene
    data['Toluene'] = toluene
    data['Xylene'] = xylene
    data['AQI'] = aqi
    
    return pd.DataFrame([data])

input_df = get_user_input()

if st.button("Predict Health Risk Category"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    
    #Use AQI encoder to get actual category name
    predicted_category = aqi_label_encoder.inverse_transform(prediction)[0]
    
    st.subheader("Prediction Result")
    st.success(f"Predicted Air Quality Risk Category: {predicted_category}")
