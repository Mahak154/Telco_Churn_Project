import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sns.set_theme(style="whitegrid")
OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(BASE, "data", "telco_churn.csv"))
df.columns = df.columns.str.strip()

# TotalCharges can be text with blanks in the real Kaggle dataset —
# convert to numeric here too so charts below don't break.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Basic cleanup used later in preprocessing too, but check here first
print(df.info())
print(df.isnull().sum())
print(df.describe())

# 1. Churn distribution
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="Churn", palette=["#4C72B0", "#DD8452"])
plt.title("Customer Churn Distribution")
plt.tight_layout()
plt.savefig(f"{OUT}/01_churn_distribution.png", dpi=150)
plt.close()

# 2. Churn by contract type
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Contract", hue="Churn", palette=["#4C72B0", "#DD8452"])
plt.title("Churn by Contract Type")
plt.tight_layout()
plt.savefig(f"{OUT}/02_churn_by_contract.png", dpi=150)
plt.close()

# 3. Monthly charges distribution by churn
plt.figure(figsize=(6, 4))
sns.kdeplot(data=df, x="MonthlyCharges", hue="Churn", fill=True, common_norm=False, alpha=0.4)
plt.title("Monthly Charges Distribution by Churn")
plt.tight_layout()
plt.savefig(f"{OUT}/03_monthlycharges_by_churn.png", dpi=150)
plt.close()

# 4. Tenure distribution by churn
plt.figure(figsize=(6, 4))
sns.kdeplot(data=df, x="tenure", hue="Churn", fill=True, common_norm=False, alpha=0.4)
plt.title("Tenure Distribution by Churn")
plt.tight_layout()
plt.savefig(f"{OUT}/04_tenure_by_churn.png", dpi=150)
plt.close()

# 5. Churn by internet service
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="InternetService", hue="Churn", palette=["#4C72B0", "#DD8452"])
plt.title("Churn by Internet Service Type")
plt.tight_layout()
plt.savefig(f"{OUT}/05_churn_by_internet.png", dpi=150)
plt.close()

# 6. Correlation heatmap (numeric + encoded churn)
df_corr = df.copy()
df_corr["Churn_binary"] = (df_corr["Churn"] == "Yes").astype(int)
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn_binary"]
plt.figure(figsize=(6, 5))
sns.heatmap(df_corr[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUT}/06_correlation_heatmap.png", dpi=150)
plt.close()

# 7. Churn by payment method
plt.figure(figsize=(7, 4))
sns.countplot(data=df, y="PaymentMethod", hue="Churn", palette=["#4C72B0", "#DD8452"])
plt.title("Churn by Payment Method")
plt.tight_layout()
plt.savefig(f"{OUT}/07_churn_by_payment.png", dpi=150)
plt.close()

print("\nAll EDA charts saved to", OUT)
