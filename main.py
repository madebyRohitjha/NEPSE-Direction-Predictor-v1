import pandas as pd

df= pd.read_csv("nepsealpha_export_price_NEPSE_2021-06-28_2026-06-17_unadjusted.csv")

print(df.info())
print(df.head())

print(df["Percent Change"].head())
print(df["Volume"].head())
print(df["Turn Over"].head())

print(df["Turn Over"].value_counts)

df = df.drop(
    columns=["Turn Over"]
)
df["Date"] = pd.to_datetime(
    df["Date"]
)

df["Percent Change"] = (
    df["Percent Change"]
    .str.replace("%","")
    .str.strip()
    .astype(float)
)

df["Volume"] = (
    df["Volume"]
    .str.replace(",","")
    .astype(float)
)

print(df.info())
print(df.isnull().sum())
print(df.head())

df = df.drop(
    columns=["Symbol"]
)

print(df.info())   

df["Target"] = (
    df["Close"].shift(-1)
    >df["Close"]
).astype(int)

print(df["Target"].value_counts())
print(df["Target"].value_counts(normalize=True) * 100)

df["Daily_Range"] = df["High"] - df["Low"]

df["Open_Close_Diff"] = df["Close"] - df["Open"]

print(df.head())


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = df[['Open','High','Low','Close',
        'Daily_Range','Open_Close_Diff']]

y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train,y_train)

preds = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test, preds))

## Logistic Regression Accuracy: 57.83% 

#with python pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

dt_pipeline = Pipeline([
    ("model", DecisionTreeClassifier(
        random_state=42
    ))
])
dt_pipeline.fit(X_train, y_train)

dt_pipeline.fit(X_train, y_train)

y_pred_dt = dt_pipeline.predict(X_test)

from sklearn.metrics import accuracy_score

print(
    "Decision Tree Accuracy:",
    accuracy_score(y_test, y_pred_dt)
)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    dt_pipeline,
    X,
    y,
    cv=5
)

print(scores)
print("Average:", scores.mean())


#random forest model; training

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict

rf_pipeline = Pipeline([
  (  "model",
    RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
  ) 
])

rf_pipeline.fit(X_train,y_train)

y_pred_dt = rf_pipeline.predict(X_test)

print(
    "Random Forest Accuracy:",
    accuracy_score(
        y_test,
        y_pred_dt
    )
)

scores = cross_val_score(
    rf_pipeline,
    X,
    y,
    cv=5
)

print(scores)

print("Average:", scores.mean())

df["MA5"] = df["Close"].rolling(window=5).mean()
df["MA10"] = df["Close"].rolling(window=10).mean()

print(df[["Date", "Close", "MA5", "MA10"]].head(15))

delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
df["RSI"] = 100 - (100 / (1 + gain / loss))

df = df.dropna()

print(df.shape)
print(df.isnull().sum())

X = df[
    [
        "Open",
        "High",
        "Low",
        "Close",
        "Percent Change",
        "Volume",
        "Daily_Range",
        "Open_Close_Diff",
        "MA5",
        "MA10",
        "RSI"
    ]
]

y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(
    "Logistic Regression Accuracy:",
    accuracy_score(y_test, y_pred)
)

rf_pipeline.fit(X_train, y_train)

y_pred_rf = rf_pipeline.predict(X_test)

print(
    "Random Forest Accuracy:",
    accuracy_score(y_test, y_pred_rf)
)

import pandas as pd

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_pipeline.named_steps["model"].feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.xticks(rotation=45)

plt.title("Feature Importance")

plt.show(block=False)
plt.pause(2)
plt.close()

param_grid = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    estimator=rf_pipeline.named_steps["model"],
    param_distributions=param_grid,
    n_iter=20,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print(random_search.best_params_)
print(random_search.best_score_)

best_rf = random_search.best_estimator_

y_pred_best = best_rf.predict(X_test)

print("Tuned Random Forest Test Accuracy:", accuracy_score(y_test, y_pred_best))

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred_best))
print(confusion_matrix(y_test, y_pred_best))