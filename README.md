# 🇳🇵 NEPSE Direction Predictor v1

A Machine Learning project that predicts whether the NEPSE index will move UP or DOWN on the next trading day using historical market data.

## Project Goal

Predict:

* 1 = Market goes UP tomorrow 📈
* 0 = Market goes DOWN tomorrow 📉

using historical NEPSE trading data.

---

## Dataset

Features used:

* Open
* High
* Low
* Close
* Volume
* Percent Change

Data Period:

* 2021 - 2026

---

## Data Cleaning

Performed:

* Removed unnecessary columns
* Converted Date to datetime format
* Converted Percent Change to numeric values
* Converted Volume to numeric values
* Removed missing or invalid columns

---

## Target Engineering

Created target variable:

```python
df["Target"] = (
    df["Close"].shift(-1)
    > df["Close"]
).astype(int)
```

Meaning:

* 1 → Next day's close is higher
* 0 → Next day's close is lower

---

## Exploratory Analysis

Target Distribution:

* Down Days: 53.91%
* Up Days: 46.09%

Baseline Accuracy:

* 53.91%

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

---

## Current Progress

✅ Dataset Collection

✅ Data Cleaning

✅ Feature Engineering

✅ Target Creation

✅ Target Distribution Analysis

🔄 Model Training (In Progress)

🔄 Performance Evaluation

🔄 Feature Importance Analysis

---

## Author

Rohit Jha

ICT Student | AI & Machine Learning Enthusiast

GitHub: https://github.com/madebyRohitjha
