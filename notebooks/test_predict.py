import joblib
import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE, "models", "churn_model.joblib"))
encoders = joblib.load(os.path.join(BASE, "models", "label_encoders.joblib"))
feature_columns = joblib.load(os.path.join(BASE, "models", "feature_columns.joblib"))

def tenure_bucket(t):
    if t <= 12: return '0-1yr'
    elif t <= 24: return '1-2yr'
    elif t <= 48: return '2-4yr'
    else: return '4yr+'

addon_cols = ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']

def predict_row(row):
    row = dict(row)
    row['TenureBucket'] = tenure_bucket(row['tenure'])
    row['AvgMonthlySpend'] = round(row['TotalCharges'] / max(row['tenure'], 1), 2)
    row['NumAddonServices'] = sum(1 for c in addon_cols if row[c] == 'Yes')
    df = pd.DataFrame([row])
    for col, le in encoders.items():
        if col in df.columns and col != 'Churn':
            df[col] = le.transform(df[col].astype(str))
    df = df[feature_columns]
    proba = float(model.predict_proba(df)[0][1])
    return proba

high_risk = {
    'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes',
    'Dependents': 'No', 'tenure': 5, 'PhoneService': 'Yes',
    'MultipleLines': 'No', 'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No', 'OnlineBackup': 'No',
    'DeviceProtection': 'No', 'TechSupport': 'No',
    'StreamingTV': 'Yes', 'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check', 'MonthlyCharges': 85.5,
    'TotalCharges': 420.75
}

low_risk = dict(high_risk)
low_risk.update({
    'tenure': 60, 'Contract': 'Two year', 'MonthlyCharges': 45.0,
    'TotalCharges': 2700.0, 'OnlineSecurity': 'Yes', 'TechSupport': 'Yes',
    'InternetService': 'DSL', 'PaymentMethod': 'Bank transfer (automatic)'
})

p1 = predict_row(high_risk)
p2 = predict_row(low_risk)

print('High-risk profile churn probability:', round(p1, 4), '-> prediction:', 'Yes' if p1 >= 0.5 else 'No')
print('Low-risk profile churn probability:', round(p2, 4), '-> prediction:', 'Yes' if p2 >= 0.5 else 'No')
