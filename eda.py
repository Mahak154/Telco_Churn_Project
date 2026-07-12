"""
eda.py — Week 2: Exploratory Data Analysis
Generates and saves the charts you'll paste into the internship report.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
df = pd.read_csv("data/telco_churn.csv")

print("Shape:", df.shape)
print(df.isnull().sum().sum(), "missing values")
print(df["Churn"].value_counts())

# 1. Churn distribution
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="Churn", palette=["#2E86AB", "#E63946"])
plt.title("Customer Churn Distribution")
plt.savefig("charts/01_churn_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# 2. Churn by contract type
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Contract", hue="Churn", palette=["#2E86AB", "#E63946"])
plt.title("Churn by Contract Type")
plt.savefig("charts/02_churn_by_contract.png", dpi=150, bbox_inches="tight")
plt.close()

# 3. Tenure distribution by churn
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack", bins=30, palette=["#2E86AB", "#E63946"])
plt.title("Tenure Distribution by Churn")
plt.savefig("charts/03_tenure_by_churn.png", dpi=150, bbox_inches="tight")
plt.close()

# 4. Monthly charges by churn
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", palette=["#2E86AB", "#E63946"])
plt.title("Monthly Charges by Churn")
plt.savefig("charts/04_monthlycharges_by_churn.png", dpi=150, bbox_inches="tight")
plt.close()

# 5. Correlation heatmap (numeric + encoded churn)
df_corr = df.copy()
df_corr["Churn_flag"] = (df_corr["Churn"] == "Yes").astype(int)
num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn_flag"]
plt.figure(figsize=(6, 5))
sns.heatmap(df_corr[num_cols].corr(), annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.savefig("charts/05_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

print("EDA charts saved to charts/")
