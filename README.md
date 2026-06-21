# 🇳🇵 NEPSE Direction Predictor v1

A Machine Learning project that predicts whether the NEPSE (Nepal Stock Exchange) index will move **UP 📈** or **DOWN 📉** on the next trading day using historical market data.

---

# 🎯 Project Goal

Build a classification model that predicts:

* **1 → Market goes UP tomorrow**
* **0 → Market goes DOWN tomorrow**

This project aims to explore whether historical market information can be used to predict the next day's market direction.

---

# 📂 Dataset

Dataset: Historical NEPSE Index Data (2021–2026)

### Features Available

* Date
* Open
* High
* Low
* Close
* Percent Change
* Volume

---

# 🧹 Data Cleaning

Performed the following preprocessing steps:

✅ Converted `Date` to datetime format

✅ Removed unnecessary columns

✅ Converted `Percent Change` from string to numeric

✅ Converted `Volume` from string to numeric

✅ Handled invalid and missing values

---

# ⚙️ Feature Engineering

Created additional features to capture market behavior.

### 1. Daily Range

```python
df["Daily_Range"] = df["High"] - df["Low"]
```

Measures market volatility during the day.

---

### 2. Open-Close Difference

```python
df["Open_Close_Diff"] = df["Close"] - df["Open"]
```

Measures the net movement of the market during the trading session.

---

# 🎯 Target Engineering

Created the prediction target:

```python
df["Target"] = (
    df["Close"].shift(-1)
    > df["Close"]
).astype(int)
```

Meaning:

* **1 → Tomorrow's Close > Today's Close**
* **0 → Tomorrow's Close ≤ Today's Close**

---

# 📊 Exploratory Data Analysis

### Target Distribution

* 📉 Down Days: **53.91%**
* 📈 Up Days: **46.09%**

### Baseline Accuracy

A model that always predicts **DOWN** would achieve:

```text
53.91% accuracy
```

Any machine learning model must beat this baseline.

---

# 🤖 Models Implemented

## 1. Logistic Regression

**Accuracy:** 57.83%

Successfully outperformed the baseline.

---

## 2. Decision Tree Classifier

**Accuracy:** 54.78%

Cross Validation Average:

```text
51.91%
```

The model showed signs of overfitting and underperformed compared to Logistic Regression.

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Git
* GitHub

---

# 📈 Current Project Workflow

```text
Raw NEPSE Data
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Target Engineering
        ↓
Exploratory Data Analysis
        ↓
Logistic Regression
        ↓
Decision Tree
        ↓
Model Comparison
        ↓
(Random Forest - Coming Next)
```

---

# 🚀 Upcoming Improvements

* [ ] Random Forest Classifier
* [ ] XGBoost
* [ ] Feature Importance Analysis
* [ ] Hyperparameter Tuning
* [ ] Technical Indicators

  * Moving Averages
  * RSI
  * MACD
* [ ] Interactive Dashboard
* [ ] Model Deployment

---

# 📚 Key Learnings

* Data cleaning and feature engineering are crucial in ML projects.
* Baseline accuracy should always be established before training models.
* Simpler models can outperform complex models on small datasets.
* Stock market prediction is significantly harder than traditional classification problems.

---

# 👨‍💻 Author

**Rohit Jha**

ICT Student | Aspiring AI/ML Engineer

GitHub: https://github.com/madebyRohitjha

---

⭐ If you found this project interesting, feel free to star the repository.
