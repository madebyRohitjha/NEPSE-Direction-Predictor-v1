import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("best_random_forest.pkl")

st.title("📈 NEPSE Direction Predictor")

st.write("Enter today's market data")

open_price = st.number_input("Open")
high = st.number_input("High")
low = st.number_input("Low")
close = st.number_input("Close")
percent_change = st.number_input("Percent Change")
volume = st.number_input("Volume")

daily_range = high - low
open_close_diff = close - open

ma5 = st.number_input("MA5")
ma10 = st.number_input("MA10")
rsi = st.number_input("RSI")

if st.button("Predict"):

    sample = pd.DataFrame([{
        "Open": open_price,
        "High": high,
        "Low": low,
        "Close": close,
        "Percent Change": percent_change,
        "Volume": volume,
        "Daily_Range": daily_range,
        "Open_Close_Diff": open_close_diff,
        "MA5": ma5,
        "MA10": ma10,
        "RSI": rsi
    }])

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0]

    if prediction == 1:
        st.success("📈 NEPSE may go UP")
    else:
        st.error("📉 NEPSE may go DOWN")

    st.write(f"Confidence (DOWN): {probability[0]*100:.2f}%")
    st.write(f"Confidence (UP): {probability[1]*100:.2f}%")