# 🇳🇵 NEPSE Direction Predictor v1

A Machine Learning project that predicts whether the NEPSE (Nepal Stock Exchange) index will move **UP 📈** or **DOWN 📉** on the next trading day using historical market data.

---

# 🎯 Project Goal

Build a classification model that predicts:

* **1 → Market goes UP tomorrow**
* **0 → Market goes DOWN tomorrow**

using historical NEPSE market information and feature engineering.

---

# 📂 Dataset

**Dataset:** Historical NEPSE Index Data (2021–2026)

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

✅ Converted `Percent Change` to numeric values

✅ Converted `Volume` to numeric values

✅ Removed unnecessary columns

✅ Handled invalid and missing values

---

# ⚙️ Feature Engineering

Created additional features to better represent market behavior.

### Daily Range

```python
df["Daily_Range"] = df["High"] - df["Low"]
```

Measures market volatility during the trading day.

---

### Open-Close Difference

```python
df["Open_Close_Diff"] = df["Close"] - df["Open"]
```

Measures the market's movement from opening to closing.

---

# 🎯 Target Engineering

Created the target variable:

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

Any ML model must beat this baseline.

---

# 🤖 Models Implemented

## 1. Logistic Regression

**Accuracy:** 57.83%

Successfully outperformed the baseline and currently performs the best.

---

## 2. Decision Tree Classifier

**Accuracy:** 54.78%

**Cross Validation Average:** 51.91%

Showed signs of overfitting and underperformed compared to Logistic Regression.

---

## 3. Random Forest Classifier

**Accuracy:** 56.09%

Improved over the baseline and Decision Tree but still slightly underperformed compared to Logistic Regression.

---

# 📈 Model Comparison

| Model                  | Accuracy |
| ---------------------- | -------- |
| Baseline (Always DOWN) | 53.91%   |
| Decision Tree          | 54.78%   |
| Random Forest          | 56.09%   |
| 🥇 Logistic Regression | 57.83%   |

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

# 📂 Project Workflow

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
Random Forest
        ↓
Model Comparison
```

---

# 🚀 Upcoming Improvements

* [ ] Feature Importance Analysis
* [ ] Technical Indicators

  * Moving Average (MA5)
  * Moving Average (MA10)
  * RSI
  * MACD
* [ ] Hyperparameter Tuning
* [ ] XGBoost
* [ ] Interactive Dashboard
* [ ] Model Deployment

---

# 📚 Key Learnings

* Data cleaning and feature engineering are critical in ML projects.
* Establishing a baseline is essential before evaluating models.
* More complex models do not automatically perform better.
* Financial market prediction is significantly harder than traditional classification tasks.

---

# 👨‍💻 Author

**Rohit Jha**

ICT Student | Aspiring AI/ML Engineer

GitHub: https://github.com/madebyRohitjha

---

⭐ If you found this project interesting, feel free to star the repository.
