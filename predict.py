"""
NEPSE Direction Predictor

Author: Rohit Jha
GitHub: https://github.com/madebyrohitjha

© 2026 Rohit Jha. All Rights Reserved.
"""

import joblib
import pandas as pd

# Load trained model
model = joblib.load("best_random_forest.pkl")

# Example input
sample = pd.DataFrame([{
    "Open": 2900,
    "High": 2925,
    "Low": 2885,
    "Close": 2910,
    "Percent Change": 0.35,
    "Volume": 7200000000,
    "Daily_Range": 40,
    "Open_Close_Diff": 10,
    "MA5": 2895,
    "MA10": 2888,
    "RSI": 58
}])

prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0]

print("=" * 50)

if prediction == 1:
    print("📈 Prediction: NEPSE may go UP")
else:
    print("📉 Prediction: NEPSE may go DOWN")

print(f"Confidence (DOWN): {probability[0]*100:.2f}%")
print(f"Confidence (UP): {probability[1]*100:.2f}%")