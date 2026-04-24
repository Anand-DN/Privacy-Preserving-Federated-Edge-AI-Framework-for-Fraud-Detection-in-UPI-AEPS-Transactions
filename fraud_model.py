# =========================================
# UPI FRAUD DETECTION - FULL PIPELINE
# =========================================

# ========== 1. IMPORTS ==========
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

# ========== 2. LOAD DATA ==========
df = pd.read_csv("upi_synthetic_dataset.csv")

print("Dataset Loaded ✅")
print(df.head())
print("\nClass Distribution:\n", df["is_fraud"].value_counts())

# ========== 3. PREPROCESSING ==========

# Convert categorical → numeric
df = pd.get_dummies(df, columns=["device_type", "upi_app", "location"], drop_first=True)

# Drop unnecessary columns
df = df.drop(["transaction_id", "timestamp", "utr_number"], axis=1)

# Split features & label
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# ========== 4. TRAIN-TEST SPLIT ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain distribution BEFORE SMOTE:")
print(y_train.value_counts())

# ========== 5. HANDLE IMBALANCE ==========
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("\nTrain distribution AFTER SMOTE:")
print(y_train_res.value_counts())

# ========== 6. MODEL 1: LOGISTIC REGRESSION ==========
print("\n🚀 Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_res, y_train_res)

# Predictions
y_pred_lr = lr_model.predict(X_test)
y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

# Evaluation
print("\n===== Logistic Regression Results =====")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))
print("AUC Score:", roc_auc_score(y_test, y_prob_lr))


# ========== 7. MODEL 2: RANDOM FOREST ==========
print("\n🌲 Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_res, y_train_res)

# Predictions
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# Evaluation
print("\n===== Random Forest Results =====")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
print("AUC Score:", roc_auc_score(y_test, y_prob_rf))


# ========== 8. SUMMARY ==========
print("\n✅ Step 2 Completed Successfully!")

print("""
Key Takeaways:
- Logistic Regression = baseline model
- Random Forest = better performance usually
- Focus on Recall (fraud detection)
- AUC > 0.9 = strong model
""")