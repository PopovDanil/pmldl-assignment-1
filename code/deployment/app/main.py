from datetime import datetime

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://api:8888")

st.set_page_config(page_title="Bike Sharing Prediction", page_icon=":bike:", layout="centered")

st.title(":bike: Bike Sharing Demand Prediction")
st.markdown("Enter the conditions below to predict the number of bike rentals.")

with st.sidebar:
    st.header("API Status")
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        if resp.status_code == 200 and resp.json().get("model_loaded"):
            st.success("API connected, model loaded")
        else:
            st.warning("API reachable but model not loaded")
    except requests.ConnectionError:
        st.error("Cannot reach API")

st.divider()

col1, col2 = st.columns(2)

with col1:
    season = st.selectbox(
        "Season",
        options=[1, 2, 3, 4],
        format_func=lambda x: {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}[x],
    )
    weather = st.selectbox(
        "Weather",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "Clear/Partly Cloudy",
            2: "Mist/Cloudy",
            3: "Light Snow/Rain",
            4: "Heavy Rain/Snow",
        }[x],
    )
    temp = st.slider("Temperature (C)", min_value=-5.0, max_value=45.0, value=20.0, step=0.5)
    atemp = st.slider("Feels Like Temperature (C)", min_value=-10.0, max_value=50.0, value=22.0, step=0.5)

with col2:
    holiday = st.checkbox("Holiday")
    workingday = st.checkbox("Working Day")
    humidity = st.slider("Humidity (%)", min_value=0, max_value=100, value=50)
    windspeed = st.slider("Wind Speed (m/s)", min_value=0.0, max_value=60.0, value=10.0, step=0.5)

date_input = st.date_input("Date", value=datetime.now().date())
time_input = st.time_input("Time", value=datetime.now().time().replace(minute=0, second=0, microsecond=0))

st.divider()

if st.button("Predict Bike Rentals", type="primary", use_container_width=True):
    dt_str = f"{date_input} {time_input}".replace(",", "")
    payload = {
        "season": season,
        "holiday": int(holiday),
        "workingday": int(workingday),
        "weather": weather,
        "temp": temp,
        "atemp": atemp,
        "humidity": humidity,
        "windspeed": windspeed,
        "datetime": dt_str,
    }

    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        if resp.status_code == 200:
            prediction = resp.json()["prediction"]
            st.metric(label="Predicted Bike Rentals", value=f"{prediction:,.0f}")
        else:
            st.error(f"API error: {resp.status_code} - {resp.text}")
    except requests.ConnectionError:
        st.error("Cannot reach the prediction API. Make sure the API service is running.")
