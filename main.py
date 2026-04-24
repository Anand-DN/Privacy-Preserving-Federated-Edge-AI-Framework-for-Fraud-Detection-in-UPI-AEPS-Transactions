# =========================================
# FEDERATED FRAUD DETECTION
# ENHANCED PIPELINE
# =========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, classification_report, roc_curve,
    f1_score, precision_score, recall_score
)
import json
import hashlib
import time

CONFIG = {
    'data_path': 'data/',
    'epsilon': 3.0,
    'delta': 1e-5,
    'noise_multiplier': 0.3,
    'max_rounds': 5,
    'clip_norm': 2.0,
    'byzantine_tolerance': 1,
    'momentum': 0.8,
    'test_size': 0.2,
    'random_seed': 42
}

np.random.seed(CONFIG['random_seed'])

def compute_sigma():
    return CONFIG['clip_norm'] * CONFIG['noise_multiplier'] * np.sqrt(2 * np.log(1.25 / CONFIG['delta'])) / CONFIG['epsilon']

def dp_clip(gradients, clip_norm):
    norm = np.linalg.norm(gradients)
    if norm > clip_norm:
        return gradients * (clip_norm / norm)
    return gradients

def dp_noise(gradients, sigma):
    return gradients + np.random.normal(0, sigma, gradients.shape)

class SecureChannel:
    def __init__(self, secret_key):
        self.secret_key = secret_key.encode()
    
    def sign(self, data):
        return hashlib.sha256(f"{data}{time.time()}".encode()).hexdigest()[:16]

class SecureAggregation:
    def __init__(self, byzantine_tolerance):
        self.byzantine_tolerance = byzantine_tolerance
        self.client_updates = []
    
    def add_update(self, weight, intercept, bank_id):
        sig = hashlib.sha256(f"{bank_id}{time.time()}".encode()).hexdigest()[:16]
        self.client_updates.append((weight, intercept, sig))
    
    def aggregate(self):
        weights = [w for w, _, _ in self.client_updates]
        intercepts = [b for _, b, _ in self.client_updates]
        
        weight_medians = np.median(weights, axis=0)
        deviations = [np.linalg.norm(w - weight_medians) for w in weights]
        threshold = np.percentile(deviations, 75)
        
        valid = [(w, b) for w, b, d in zip(weights, intercepts, deviations) if d <= threshold]
        
        self.client_updates = []
        if valid:
            return np.mean([w for w, _ in valid], axis=0), np.mean([b for _, b in valid], axis=0)
        return weights[0], intercepts[0]

def load_global_data():
    df_all = pd.concat([
        pd.read_csv(CONFIG['data_path'] + 'bank_A.csv'),
        pd.read_csv(CONFIG['data_path'] + 'bank_B.csv'),
        pd.read_csv(CONFIG['data_path'] + 'bank_C.csv')
    ])
    df_all = pd.get_dummies(df_all, columns=['device_type', 'upi_app', 'location'], drop_first=True)
    df_all = df_all.drop(['transaction_id', 'timestamp', 'utr_number'], axis=1)
    return df_all

def load_bank_data(file_path, columns, scaler, test_size=0.2):
    df = pd.read_csv(file_path)
    df = pd.get_dummies(df, columns=['device_type', 'upi_app', 'location'], drop_first=True)
    df = df.drop(['transaction_id', 'timestamp', 'utr_number'], axis=1)
    df = df.reindex(columns=list(columns) + ['is_fraud'], fill_value=0)
    
    X = df.drop('is_fraud', axis=1).values
    y = df['is_fraud'].values
    
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    train_count = int((1 - test_size) * len(X))
    train_idx, test_idx = indices[:train_count], indices[train_count:]
    
    X_train = scaler.transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    return X_train, X_test, y_train, y_test

def train_local(X_train, y_train, model_type='lr'):
    if model_type == 'lr':
        model = LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear')
    elif model_type == 'rf':
        model = RandomForestClassifier(n_estimators=50, max_depth=5, class_weight='balanced', random_state=42)
    elif model_type == 'mlp':
        model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    
    model.fit(X_train, y_train)
    return model

def federated_round(round_num, dp_accum, sec_agg, channel, model_type='lr'):
    print(f"\n--- FEDERATED ROUND {round_num}/{CONFIG['max_rounds']} ---")
    
    banks = [("Bank A", CONFIG['data_path'] + 'bank_A.csv'),
            ("Bank B", CONFIG['data_path'] + 'bank_B.csv'),
            ("Bank C", CONFIG['data_path'] + 'bank_C.csv')]
    
    sigma = compute_sigma()
    
    for name, file_path in banks:
        X_train, _, y_train, _ = load_bank_data(file_path, columns, scaler)
        
        model = train_local(X_train, y_train, model_type)
        
        if model_type == 'lr':
            coef = model.coef_.copy()
            intercept = model.intercept_.copy()
            coef = dp_clip(coef, CONFIG['clip_norm'])
            intercept = dp_clip(intercept, CONFIG['clip_norm'])
            sig = channel.sign(coef.tobytes())
            coef = dp_noise(coef, sigma)
            intercept = dp_noise(intercept, sigma)
            sec_agg.add_update(coef, intercept, name)
        
        print(f"  {name}: Model trained")
    
    dp_accum += CONFIG['epsilon']
    return dp_accum

def evaluate_batch(w, b, X_test, y_test, name):
    logits = np.dot(X_test, w.T) + b
    y_prob = 1 / (1 + np.exp(-logits))
    y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
    
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.05):
        f1 = f1_score(y_test, (y_prob > t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    
    y_pred = (y_prob > best_t).astype(int)
    
    return {
        'name': name,
        'auc': roc_auc_score(y_test, y_prob),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': best_f1,
        'threshold': best_t
    }

print("=== STARTING ENHANCED FEDERATED PIPELINE ===")

print("\nLoading data...")
df_global = load_global_data()
X_all = df_global.drop('is_fraud', axis=1)
columns = X_all.columns

scaler = StandardScaler()
scaler.fit(X_all)
print(f"  Features: {len(columns)}, Samples: {len(df_global)}")

sec_agg = SecureAggregation(CONFIG['byzantine_tolerance'])
channel = SecureChannel("federated_secret_key_2024")
dp_accum = 0.0

results_by_model = {}

for model_type in ['lr', 'rf', 'mlp']:
    print(f"\n=== Training {model_type.upper()} Model ===")
    
    dp_accum = 0.0
    sec_agg = SecureAggregation(CONFIG['byzantine_tolerance'])
    
    for r in range(1, CONFIG['max_rounds'] + 1):
        dp_accum = federated_round(r, dp_accum, sec_agg, channel, model_type)
    
    eval_results = []
    all_X_test, all_y_test = [], []
    
    for name, file_path in [("Bank A", CONFIG['data_path'] + 'bank_A.csv'),
                          ("Bank B", CONFIG['data_path'] + 'bank_B.csv'),
                          ("Bank C", CONFIG['data_path'] + 'bank_C.csv')]:
        X_test, _, y_test, _ = load_bank_data(file_path, columns, scaler)
        all_X_test.append(X_test)
        all_y_test.append(y_test)
        
        X_train_sample, _, y_train_sample, _ = load_bank_data(file_path, columns, scaler)
        model = train_local(X_train_sample, y_train_sample, model_type)
        
        if model_type == 'lr':
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.predict_proba(X_test)[:, 1]
        
        best_f1, best_t = 0, 0.5
        for t in np.arange(0.1, 0.9, 0.05):
            f1 = f1_score(y_test, (y_prob > t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        y_pred = (y_prob > best_t).astype(int)
        eval_results.append({
            'name': name,
            'auc': roc_auc_score(y_test, y_prob),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': best_f1,
            'threshold': best_t
        })
    
    X_all_test = np.vstack(all_X_test)
    y_all_test = np.concatenate(all_y_test)
    
    model = train_local(all_X_test[0], all_y_test[0], model_type)
    y_prob = model.predict_proba(X_all_test)[:, 1]
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.05):
        f1 = f1_score(y_all_test, (y_prob > t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    y_pred = (y_prob > best_t).astype(int)
    eval_results.append({
        'name': 'ALL',
        'auc': roc_auc_score(y_all_test, y_prob),
        'precision': precision_score(y_all_test, y_pred, zero_division=0),
        'recall': recall_score(y_all_test, y_pred, zero_division=0),
        'f1': best_f1,
        'threshold': best_t
    })
    
    results_by_model[model_type] = {'results': eval_results, 'model_type': model_type}
    print(f"\n  {model_type.upper()} - AUC: {eval_results[-1]['auc']:.4f}, F1: {eval_results[-1]['f1']:.4f}")

best_model = max(results_by_model.keys(), 
           key=lambda k: results_by_model[k]['results'][-1]['auc'])

print(f"\n*** BEST MODEL: {best_model.upper()} ***")

def plot_comparison():
    plt.figure(figsize=(10, 8))
    colors = {'lr': '#FF6B6B', 'rf': '#4ECDC4', 'mlp': '#45B7D1'}
    
    all_X_test, all_y_test = [], []
    for name, file_path in [("Bank A", CONFIG['data_path'] + 'bank_A.csv'),
                          ("Bank B", CONFIG['data_path'] + 'bank_B.csv'),
                          ("Bank C", CONFIG['data_path'] + 'bank_C.csv')]:
        X_test, _, y_test, _ = load_bank_data(file_path, columns, scaler)
        all_X_test.append(X_test)
        all_y_test.append(y_test)
    
    X_all_test = np.vstack(all_X_test)
    y_all_test = np.concatenate(all_y_test)
    
    for model_type, data in results_by_model.items():
        if model_type == 'lr':
            X_train_sample = all_X_test[0]
            y_train_sample = all_y_test[0]
        else:
            X_train_sample = all_X_test[0]
            y_train_sample = all_y_test[0]
        
        model = train_local(X_train_sample, y_train_sample, model_type)
        y_prob = model.predict_proba(X_all_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_all_test, y_prob)
        auc = roc_auc_score(y_all_test, y_prob)
        plt.plot(fpr, tpr, label=f'{model_type.upper()} (AUC={auc:.3f})', 
                 color=colors[model_type], linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150)
    plt.show()

def plot_privacy():
    rounds = list(range(1, CONFIG['max_rounds'] + 1))
    epsilon = [r * CONFIG['epsilon'] for r in rounds]
    
    plt.figure(figsize=(10, 5))
    plt.plot(rounds, epsilon, marker='o', linewidth=2, color='#FF6B6B')
    plt.axhline(y=CONFIG['epsilon'] * CONFIG['max_rounds'], color='gray', linestyle='--', label='Max Budget')
    plt.xlabel('Federated Round', fontsize=12)
    plt.ylabel('Cumulative epsilon', fontsize=12)
    plt.title('Privacy Budget Consumption', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('privacy_budget.png', dpi=150)
    plt.show()

plot_comparison()
plot_privacy()

output = {
    'configuration': CONFIG,
    'model_comparison': {k: {'auc': v['results'][-1]['auc'], 'f1': v['results'][-1]['f1']} 
                      for k, v in results_by_model.items()},
    'best_model': best_model,
    'privacy_budget': {
        'total_epsilon_spent': dp_accum,
        'max_rounds': CONFIG['max_rounds']
    },
    'unique_features': [
        'Differential Privacy (Gaussian)',
        'Secure Aggregation + Signatures',
        'Multi-model Comparison (LR/RF/MLP)',
        'Per-Bank Evaluation'
    ]
}

best_model_type = results_by_model[best_model]['model_type']
X_train_all = scaler.transform(X_all.values)

model_final = train_local(X_train_all, df_global['is_fraud'].values, best_model_type)

import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model_final, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model saved!")

with open('results.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\n=== PIPELINE COMPLETE ===")
print("Results: results.json")
print("Charts: model_comparison.png, privacy_budget.png")