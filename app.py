# Flight Price Prediction 

import pandas as pd
import numpy as np
import joblib
import streamlit as st

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
