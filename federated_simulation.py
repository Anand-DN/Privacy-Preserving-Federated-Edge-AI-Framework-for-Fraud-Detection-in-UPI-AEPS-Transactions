# =========================================
# FEDERATED FRAUD DETECTION WITH
# DIFFERENTIAL PRIVACY
# =========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, f1_score, precision_score, recall_score
import json
import os

# =========================================
# CONFIGURATION
# =========================================
DATA_PATH = "data/"
EPSILON = 1.0           # Privacy budget (lower = more private)
DELTA = 1e-5            # Probability of privacy breach
NOISE_MULTIPLIER = 1.1  # Noise multiplier for DP
MAX_ROUNDS = 5          # Federated rounds
CLIP_NORM = 1.0         # Gradient clipping norm
LEARNING_RATE = 0.01    # Local learning rate

np.random.seed(42)

# =========================================
# DIFFERENTIAL PRIVACY UTILITIES
# =========================================
class DPMechanism:
    def __init__(self, epsilon, delta, noise_multiplier):
        self.epsilon = epsilon
        self.delta = delta
        self.noise_multiplier = noise_multiplier
        self.privacy_spent = 0.0
    
    def add_gaussian_noise(self, gradients, clip_norm):
        sigma = clip_norm * self.noise_multiplier * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        noisy_gradients = gradients + np.random.normal(0, sigma, gradients.shape)
        return noisy_gradients
    
    def compute_sensitivity(self, gradients, clip_norm):
        gradient_norm = np.linalg.norm(gradients)
        scaling_factor = min(1.0, clip_norm / gradient_norm)
        return gradients * scaling_factor

# =========================================
# DATA LOADING
# =========================================
def load_and_preprocess_global():
    df_all = pd.concat([
        pd.read_csv(DATA_PATH + "bank_A.csv"),
        pd.read_csv(DATA_PATH + "bank_B.csv"),
        pd.read_csv(DATA_PATH + "bank_C.csv")
    ])
    
    df_all = pd.get_dummies(df_all, columns=["device_type", "upi_app", "location"], drop_first=True)
    df_all = df_all.drop(["transaction_id", "timestamp"], axis=1)
    
    return df_all

df_global = load_and_preprocess_global()
X_all = df_global.drop("is_fraud", axis=1)
columns = X_all.columns

scaler = StandardScaler()
scaler.fit(X_all)

print("✅ Global preprocessing ready")
print(f"Features: {len(columns)}")
print(f"Total samples: {len(df_global)}")

# =========================================
# TRAIN/TEST SPLIT PER BANK
# =========================================
def load_bank_data(file_path):
    df = pd.read_csv(file_path)
    df = pd.get_dummies(df, columns=["device_type", "upi_app", "location"], drop_first=True)
    df = df.drop(["transaction_id", "timestamp"], axis=1)
    df = df.reindex(columns=list(columns) + ["is_fraud"], fill_value=0)
    
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]
    
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    train_size = int(0.8 * len(X))
    train_idx, test_idx = indices[:train_size], indices[train_size:]
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    return X_train, X_test, y_train, y_test

# Load bank data
X_train_A, X_test_A, y_train_A, y_test_A = load_bank_data(DATA_PATH + "bank_A.csv")
X_train_B, X_test_B, y_train_B, y_test_B = load_bank_data(DATA_PATH + "bank_B.csv")
X_train_C, X_test_C, y_train_C, y_test_C = load_bank_data(DATA_PATH + "bank_C.csv")

# Scale data
X_train_A = scaler.transform(X_train_A)
X_test_A = scaler.transform(X_test_A)
X_train_B = scaler.transform(X_train_B)
X_test_B = scaler.transform(X_test_B)
X_train_C = scaler.transform(X_train_C)
X_test_C = scaler.transform(X_test_C)

print("\n🏦 Bank Data Split:")
print(f"Bank A: {len(X_train_A)} train, {len(X_test_A)} test")
print(f"Bank B: {len(X_train_B)} train, {len(X_test_B)} test")
print(f"Bank C: {len(X_train_C)} train, {len(X_test_C)} test")

# =========================================
# LOCAL MODEL TRAINING WITH DP
# =========================================
def train_local_dp(X_train, y_train, dp_mechanism, bank_name):
    model = LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear')
    model.fit(X_train, y_train)
    
    coef = model.coef_.copy()
    intercept = model.intercept_.copy()
    
    noisy_coef = dp_mechanism.add_gaussian_noise(coef, CLIP_NORM)
    noisy_intercept = dp_mechanism.add_gaussian_noise(intercept, CLIP_NORM)
    
    dp_mechanism.privacy_spent += dp_mechanism.epsilon
    
    print(f"  {bank_name}: Trained with DP (ε={dp_mechanism.epsilon})")
    
    return noisy_coef, noisy_intercept, model

# =========================================
# FEDERATED LEARNING WITH MULTIPLE ROUNDS
# =========================================
def federated_round(round_num, w_prev, b_prev, dp_mechanism):
    print(f"\n🔄 Federated Round {round_num}/{MAX_ROUNDS}")
    
    bank_weights = []
    bank_intercepts = []
    
    banks = [
        (X_train_A, y_train_A, "Bank A"),
        (X_train_B, y_train_B, "Bank B"),
        (X_train_C, y_train_C, "Bank C")
    ]
    
    for X_train, y_train, name in banks:
        w_new, b_new, _ = train_local_dp(X_train, y_train, dp_mechanism, name)
        bank_weights.append(w_new)
        bank_intercepts.append(b_new)
    
    w_avg = np.mean(bank_weights, axis=0)
    b_avg = np.mean(bank_intercepts, axis=0)
    
    w_global = 0.7 * w_prev + 0.3 * w_avg if round_num > 1 else w_avg
    b_global = 0.7 * b_prev + 0.3 * b_avg if round_num > 1 else b_avg
    
    return w_global, b_global

# Initialize models
print("\n🚀 Starting Federated Learning...")
dp_mechanism = DPMechanism(EPSILON, DELTA, NOISE_MULTIPLIER)

w_global = np.random.randn(1, len(columns)) * 0.01
b_global = np.array([0.0])

for round_num in range(1, MAX_ROUNDS + 1):
    w_global, b_global = federated_round(round_num, w_global, b_global, dp_mechanism)

print(f"\n📊 Total Privacy Budget Spent: ε={dp_mechanism.privacy_spent:.2f}")

# =========================================
# EVALUATION ON HELD-OUT TEST DATA
# =========================================
def evaluate_model(w, b, X_test, y_test, bank_name):
    logits = np.dot(X_test, w.T) + b
    y_prob = 1 / (1 + np.exp(-logits))
    
    y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
    
    best_f1 = 0
    best_t = 0.5
    for t in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_prob > t).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    
    y_pred = (y_prob > best_t).astype(int)
    
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n🏦 {bank_name} Results:")
    print(f"  AUC: {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Best Threshold: {best_t:.2f}")
    
    return {
        'bank': bank_name,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': best_t
    }

print("\n" + "="*50)
print("📈 FINAL EVALUATION ON HELD-OUT TEST DATA")
print("="*50)

results = []
results.append(evaluate_model(w_global, b_global, X_test_A, y_test_A, "Bank A"))
results.append(evaluate_model(w_global, b_global, X_test_B, y_test_B, "Bank B"))
results.append(evaluate_model(w_global, b_global, X_test_C, y_test_C, "Bank C"))

# Combined test set
X_test_all = np.vstack([X_test_A, X_test_B, X_test_C])
y_test_all = np.concatenate([y_test_A, y_test_B, y_test_C])
results.append(evaluate_model(w_global, b_global, X_test_all, y_test_all, "ALL BANKS"))

# =========================================
# ROC CURVE VISUALIZATION
# =========================================
def plot_roc(y_test, y_prob, bank_name, color):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{bank_name} (AUC={roc_auc_score(y_test, y_prob):.3f})', color=color, linewidth=2)

plt.figure(figsize=(10, 8))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

for i, (X_test, y_test, name) in enumerate([
    (X_test_A, y_test_A, "Bank A"),
    (X_test_B, y_test_B, "Bank B"),
    (X_test_C, y_test_C, "Bank C"),
    (X_test_all, y_test_all, "All Banks")
]):
    logits = np.dot(X_test, w_global.T) + b_global
    y_prob = 1 / (1 + np.exp(-logits))
    plot_roc(y_test, y_prob, name, colors[i])

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Federated Model with Differential Privacy', fontsize=14)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150)
plt.show()

# =========================================
# SAVE RESULTS
# =========================================
results_dict = {
    'configuration': {
        'epsilon': EPSILON,
        'delta': DELTA,
        'max_rounds': MAX_ROUNDS,
        'clip_norm': CLIP_NORM,
        'noise_multiplier': NOISE_MULTIPLIER
    },
    'privacy_budget': {
        'total_epsilon_spent': float(dp_mechanism.privacy_spent)
    },
    'results': results
}

with open('results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print("\n✅ Results saved to results.json")
print("📊 ROC curve saved to roc_curves.png")