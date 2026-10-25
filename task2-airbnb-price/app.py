import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Airbnb Price Prediction")

# Load the trained model saved by the notebook
model_path = Path(__file__).parent / "airbnb_price_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

st.title("Airbnb Price Prediction System")
st.write("Enter the details of an Airbnb listing in Pakistan and click Predict to get the price of one night in US dollars.")

# Take input from the user
left, right = st.columns(2)

with left:
    city = st.selectbox("City", ["Lahore", "Islamabad", "Karachi", "Rawalpindi"])
    room_type = st.selectbox("Room Type", ["Entire place", "Private room", "Shared room"])
    guests = st.number_input("Number of Guests", min_value=1, max_value=16, value=4)
    has_rating = st.checkbox("The listing has a rating", value=True)
    rating = st.slider("Rating", 1.0, 5.0, 4.8, 0.1, disabled=not has_rating)

with right:
    superhost = st.radio("Superhost", ["No", "Yes"], horizontal=True)
    identity_verified = st.radio("Identity Verified", ["Yes", "No"], horizontal=True)
    response_time = st.selectbox(
        "Host Response Time",
        ["within an hour", "within a few hours", "within a day", "a few days or more", "Unknown"],
    )
    response_rate = st.slider("Host Response Rate (%)", 0, 100, 95, disabled=response_time == "Unknown")

if st.button("Predict"):
    # Convert the input into a feature vector (same columns as the training data)
    # np.nan means missing, the model fills it with the median like in training
    user_input = pd.DataFrame({
        "Guests": [guests],
        "Rating": [rating if has_rating else np.nan],
        "Responserate": [np.nan if response_time == "Unknown" else response_rate / 100],
        "City": [city],
        "Roomtype": [room_type],
        "Superhost": [superhost],
        "Responsetime": [response_time],
        "Identityverified": [identity_verified],
    })

    predicted_price = model.predict(user_input)[0]
    st.success(f"Predicted price of one night: ${predicted_price:.2f}")
    st.caption("The usual error of this model is about $15 (10-fold cross validation RMSE).")
