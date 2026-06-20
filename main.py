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

X_train, X_test, Y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train,Y_train)

preds = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test, preds))

## Logistic Regression Accuracy: 57.83%