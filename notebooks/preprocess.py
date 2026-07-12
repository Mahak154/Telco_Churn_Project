import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE, "models"), exist_ok=True)

df = pd.read_csv(os.path.join(BASE, "data", "telco_churn.csv"))

# --- Real-dataset cleanup ---
# Strip whitespace from column names (some Kaggle copies have trailing spaces)
df.columns = df.columns.str.strip()

if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

# TotalCharges is often stored as text in the real dataset, with blank
# strings for ~11 customers who have 0 tenure. Force it to numeric and
# fill those blanks with 0 (since they haven't been billed yet).
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
n_missing = df["TotalCharges"].isna().sum()
if n_missing > 0:
    print(f"Found {n_missing} blank/invalid TotalCharges values — filling with 0 "
          f"(these are typically customers with 0 tenure).")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Some copies of the real dataset use 0/1 for SeniorCitizen already (correct),
# but guard against it being "Yes"/"No" text instead.
if df["SeniorCitizen"].dtype == object:
    df["SeniorCitizen"] = (df["SeniorCitizen"] == "Yes").astype(int)

# Drop exact duplicate rows if any slipped in
before = len(df)
df = df.drop_duplicates()
if len(df) < before:
    print(f"Dropped {before - len(df)} duplicate rows.")

# Feature engineering: tenure buckets
def tenure_bucket(t):
    if t <= 12:
        return "0-1yr"
    elif t <= 24:
        return "1-2yr"
    elif t <= 48:
        return "2-4yr"
    else:
        return "4yr+"

df["TenureBucket"] = df["tenure"].apply(tenure_bucket)

# Feature engineering: average monthly spend so far
df["AvgMonthlySpend"] = (df["TotalCharges"] / df["tenure"].replace(0, 1)).round(2)

# Feature engineering: count of add-on services subscribed
addon_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]
df["NumAddonServices"] = (df[addon_cols] == "Yes").sum(axis=1)

# Target
df["Churn"] = (df["Churn"] == "Yes").astype(int)

# Encode categorical columns
cat_cols = df.select_dtypes(include="object").columns.tolist()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

df.to_csv(os.path.join(BASE, "data", "telco_churn_processed.csv"), index=False)
joblib.dump(encoders, os.path.join(BASE, "models", "label_encoders.joblib"))

print("Processed shape:", df.shape)
print(df.head())
print("\nFinal columns:", df.columns.tolist())
