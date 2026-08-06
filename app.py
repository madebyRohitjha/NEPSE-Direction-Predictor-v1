import streamlit as st
import joblib
import pandas as pd

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="NEPSE Predictor",
    page_icon="📈",
    layout="wide"
)

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("best_random_forest.pkl")

# ----------------------------
# Header
# ----------------------------
st.title("📈 NEPSE Direction Predictor")
st.markdown(
    "Predict whether the **NEPSE Index** will move **UP** or **DOWN** on the next trading day."
)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("About")

st.sidebar.write("""
This application uses a **Random Forest Classifier**
trained on historical NEPSE data.

### Features Used
- Open
- High
- Low
- Close
- Percent Change
- Volume
- MA5
- MA10
- RSI

Daily Range and Open-Close Difference are calculated automatically.
""")

# ----------------------------
# Input Section
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    open_price = st.number_input("Open", value=2900.0)
    high = st.number_input("High", value=2925.0)
    low = st.number_input("Low", value=2885.0)
    close = st.number_input("Close", value=2910.0)
    percent_change = st.number_input("Percent Change", value=0.35)

with col2:
    volume = st.number_input("Volume", value=7200000000.0)
    ma5 = st.number_input("MA5", value=2895.0)
    ma10 = st.number_input("MA10", value=2888.0)
    rsi = st.number_input("RSI", value=58.0)

# Automatically calculated features
daily_range = high - low
open_close_diff = close - open_price

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔮 Predict"):

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

    confidence = max(probability)

    st.divider()

    if prediction == 1:
        st.success("📈 Predicted Market Direction: **UP**")
    else:
        st.error("📉 Predicted Market Direction: **DOWN**")

    st.subheader("Prediction Confidence")

    st.progress(float(confidence))

    st.metric(
        label="Confidence",
        value=f"{confidence*100:.2f}%"
    )

    st.write(f"📉 Down Probability: **{probability[0]*100:.2f}%**")
    st.write(f"📈 Up Probability: **{probability[1]*100:.2f}%**")