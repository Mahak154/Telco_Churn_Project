import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve, confusion_matrix,
                              classification_report)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs")
MODELS = os.path.join(BASE, "models")
os.makedirs(OUT, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

df = pd.read_csv(os.path.join(BASE, "data", "telco_churn_processed.csv"))

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, f"{MODELS}/scaler.joblib")

results = {}

def evaluate(name, model, X_te, y_te, y_pred, y_proba):
    results[name] = {
        "accuracy": accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred),
        "recall": recall_score(y_te, y_pred),
        "f1": f1_score(y_te, y_pred),
        "roc_auc": roc_auc_score(y_te, y_proba),
    }
    print(f"\n--- {name} ---")
    for k, v in results[name].items():
        print(f"{k}: {v:.4f}")
    print(classification_report(y_te, y_pred))

# 1. Logistic Regression (baseline)
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred = log_reg.predict(X_test_scaled)
y_proba = log_reg.predict_proba(X_test_scaled)[:, 1]
evaluate("Logistic Regression", log_reg, X_test_scaled, y_test, y_pred, y_proba)

# 2. Decision Tree (baseline)
dt = DecisionTreeClassifier(max_depth=6, random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
y_proba = dt.predict_proba(X_test)[:, 1]
evaluate("Decision Tree", dt, X_test, y_test, y_pred, y_proba)

# 3. Random Forest
rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]
evaluate("Random Forest", rf, X_test, y_test, y_pred, y_proba)

# 4. Gradient Boosting (advanced model, stands in for XGBoost)
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
y_proba = gb.predict_proba(X_test)[:, 1]
evaluate("Gradient Boosting", gb, X_test, y_test, y_pred, y_proba)

# Cross-validation for the best-looking model (Gradient Boosting)
cv_scores = cross_val_score(gb, X_train, y_train, cv=5, scoring="roc_auc")
print("\nGradient Boosting 5-fold CV ROC-AUC:", cv_scores.mean().round(4), "+/-", cv_scores.std().round(4))

# Hyperparameter tuning on Gradient Boosting
param_grid = {
    "n_estimators": [150, 200, 300],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [2, 3, 4],
}
grid = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid,
                     cv=3, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)
print("\nBest params:", grid.best_params_)
print("Best CV ROC-AUC:", grid.best_score_.round(4))

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]
evaluate("Tuned Gradient Boosting (FINAL)", best_model, X_test, y_test, y_pred, y_proba)

# Save final model
joblib.dump(best_model, f"{MODELS}/churn_model.joblib")
joblib.dump(list(X.columns), f"{MODELS}/feature_columns.joblib")

# --- Charts ---

# Model comparison bar chart
results_df = pd.DataFrame(results).T
results_df.to_csv(f"{OUT}/model_comparison.csv")

plt.figure(figsize=(9, 5))
results_df[["accuracy", "precision", "recall", "f1", "roc_auc"]].plot(kind="bar", ax=plt.gca())
plt.title("Model Comparison")
plt.ylabel("Score")
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT}/08_model_comparison.png", dpi=150)
plt.close()

# ROC curve for final model
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label=f"Tuned GB (AUC={roc_auc_score(y_test, y_proba):.3f})", color="#4C72B0")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Final Model")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/09_roc_curve.png", dpi=150)
plt.close()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title("Confusion Matrix - Final Model")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{OUT}/10_confusion_matrix.png", dpi=150)
plt.close()

# Feature importance
importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(12)
plt.figure(figsize=(7, 5))
sns.barplot(x=importances.values, y=importances.index, color="#4C72B0")
plt.title("Top 12 Feature Importances - Final Model")
plt.tight_layout()
plt.savefig(f"{OUT}/11_feature_importance.png", dpi=150)
plt.close()

print("\nAll model artifacts and charts saved.")
print(results_df)
