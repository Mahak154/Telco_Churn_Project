"""
generate_data.py
-----------------
Generates a realistic synthetic Telecom Customer Churn dataset.
Modeled on the structure and feature relationships of the well-known
IBM/Kaggle "Telco Customer Churn" dataset (7043 customers, 21 columns),
so it behaves the same way for EDA / modeling / deployment purposes.

Run: python3 generate_data.py
Output: telco_churn.csv (in the same folder)
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 5000  # number of customers

# ---- Demographics ----
gender = np.random.choice(["Male", "Female"], N)
senior_citizen = np.random.choice([0, 1], N, p=[0.84, 0.16])
partner = np.random.choice(["Yes", "No"], N, p=[0.48, 0.52])
dependents = np.random.choice(["Yes", "No"], N, p=[0.30, 0.70])

# ---- Account info ----
tenure = np.random.exponential(scale=24, size=N).astype(int)
tenure = np.clip(tenure, 0, 72)

contract = np.random.choice(
    ["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.21, 0.24]
)
paperless_billing = np.random.choice(["Yes", "No"], N, p=[0.59, 0.41])
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    N, p=[0.34, 0.23, 0.22, 0.21]
)

# ---- Services ----
phone_service = np.random.choice(["Yes", "No"], N, p=[0.90, 0.10])
multiple_lines = np.where(
    phone_service == "No", "No phone service",
    np.random.choice(["Yes", "No"], N, p=[0.42, 0.58])
)
internet_service = np.random.choice(["DSL", "Fiber optic", "No"], N, p=[0.34, 0.44, 0.22])

def dependent_service(internet):
    return np.where(internet == "No", "No internet service",
                     np.random.choice(["Yes", "No"], len(internet), p=[0.39, 0.61]))

online_security   = dependent_service(internet_service)
online_backup     = dependent_service(internet_service)
device_protection = dependent_service(internet_service)
tech_support      = dependent_service(internet_service)
streaming_tv      = dependent_service(internet_service)
streaming_movies  = dependent_service(internet_service)

# ---- Charges ----
base_charge = np.where(internet_service == "Fiber optic", 70,
              np.where(internet_service == "DSL", 45, 20))
addon_cols = [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]
addon_count = sum((col == "Yes").astype(int) for col in addon_cols)
monthly_charges = base_charge + addon_count * 5 + np.random.normal(0, 5, N)
monthly_charges = np.clip(monthly_charges, 18, 120).round(2)

total_charges = (monthly_charges * tenure + np.random.normal(0, 20, N)).round(2)
total_charges = np.clip(total_charges, 0, None)

# ---- Churn (driven by realistic risk factors, mirrors real-world patterns) ----
risk = (
    (contract == "Month-to-month") * 1.8
    + (internet_service == "Fiber optic") * 0.6
    + (payment_method == "Electronic check") * 0.5
    + (tenure < 12) * 1.2
    + (online_security == "No") * 0.4
    + (tech_support == "No") * 0.4
    + (monthly_charges > 80) * 0.5
    + (senior_citizen == 1) * 0.3
    - (partner == "Yes") * 0.3
    - (dependents == "Yes") * 0.3
    - (contract == "Two year") * 1.5
)
prob_churn = 1 / (1 + np.exp(-(risk - 3.1)))
churn = np.where(np.random.rand(N) < prob_churn, "Yes", "No")

customer_id = [f"{np.random.randint(1000,9999)}-{''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5))}" for _ in range(N)]

df = pd.DataFrame({
    "customerID": customer_id,
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Churn": churn,
})

out_path = "telco_churn.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
print(df["Churn"].value_counts(normalize=True))
