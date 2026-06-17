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