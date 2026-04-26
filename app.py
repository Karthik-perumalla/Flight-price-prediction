# Flight Price Prediction 

import pandas as pd
import numpy as np
import joblib
import streamlit as st
import base64



def set_bg():
    with open("background.jpeg", "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    bg_style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}

    /* Optional: make content slightly transparent for better readability */
    .block-container {{
        background: rgba(255, 255, 255, 0.85);
        padding: 20px;
        border-radius: 10px;
    }}
    </style>
    """
    st.markdown(bg_style, unsafe_allow_html=True)


set_bg()



st.title("✈️ Flight Price Prediction using Linear Models")


@st.cache_resource
def load_model():
    return joblib.load("best_model.joblib")

model = load_model()
st.success("Model loaded successfully!")


airline = st.selectbox("Airline", ["Indigo","Air India","Vistara","SpiceJet","GoAir"])
source_city = st.selectbox("Source City", ["Delhi","Mumbai","Bangalore","Kolkata","Chennai"])
departure_time = st.selectbox("Departure Time", ["Morning","Afternoon","Evening","Night"])
stops = st.selectbox("Stops", ["zero","one","two_or_more"])
arrival_time = st.selectbox("Arrival Time", ["Morning","Afternoon","Evening","Night"])
destination_city = st.selectbox("Destination City", ["Delhi","Mumbai","Bangalore","Kolkata","Chennai"])
travel_class = st.selectbox("Class", ["Economy","Business"])

duration = st.number_input("Duration (hours)", min_value=0.0)
days_left = st.number_input("Days Left", min_value=0)


def preprocess_input():
    input_dict = {
        'airline': airline,
        'source_city': source_city,
        'departure_time': departure_time,
        'stops': stops,
        'arrival_time': arrival_time,
        'destination_city': destination_city,
        'class': travel_class,
        'duration': duration,
        'days_left': days_left
    }

    df = pd.DataFrame([input_dict])
    return df
   

if st.button("Predict Price"):
    try:
        X = preprocess_input()
        prediction = model.predict(X)[0]
        st.success(f"💰 Predicted Price: ₹ {int(prediction)}")

    except Exception as e:
        st.error(f"Error: {e}")
