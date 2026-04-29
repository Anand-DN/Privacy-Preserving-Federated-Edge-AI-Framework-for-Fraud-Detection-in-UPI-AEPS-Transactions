# Privacy-Preserving Federated UPI Fraud Detection

## 1. Introduction

Digital payment systems such as UPI process a large number of transactions every day. With this growth, fraud patterns also become more complex. A fraud detection model must identify suspicious transactions quickly, but financial transaction data is highly sensitive and cannot be freely shared between banks.

This project builds a privacy-preserving UPI fraud detection system using machine learning and federated learning. The system compares traditional machine learning models with federated learning models and shows how banks can collaboratively train a fraud detection system without directly sharing raw customer transaction data.

The project includes:

- Synthetic non-IID UPI transaction data for multiple banks.
- Traditional machine learning baselines: Logistic Regression, Random Forest, and MLP.
- Centralized training baseline.
- Federated Averaging, also called FedAvg.
- Differential privacy and robust aggregation experiments.
- Byzantine client attack simulation.
- Streamlit dashboard for model comparison, fraud risk prediction, and transaction monitoring.

## 2. Problem Statement

Traditional fraud detection systems usually require all transaction data to be collected in one central location. In banking and finance, this is a major privacy and compliance issue because transaction data contains sensitive customer, account, device, and behavioral information.

The main problem addressed in this project is:

> How can multiple banks collaboratively improve UPI fraud detection without sharing raw transaction data?

The project solves this by using federated learning. Each bank trains locally on its own data and only shares model updates. This allows the global model to learn from distributed bank data while reducing direct data exposure.

## 3. Objectives

The objectives of the project are:

1. Build a UPI fraud detection pipeline using transaction-level features.
2. Compare traditional ML models such as LR, RF, and MLP.
3. Compare centralized training with federated training.
4. Implement FedAvg to simulate collaborative multi-bank training.
5. Add differential privacy by clipping updates and adding Gaussian noise.
6. Add robust aggregation to reduce the impact of malicious or abnormal client updates.
7. Evaluate models using AUC, PR-AUC, precision, recall, and F1-score.
8. Build a dashboard for visualization and live transaction risk prediction.

## 4. Reference From Paper.pdf

The provided `Paper.pdf` contains a related-work comparison table. It compares earlier work in federated learning, differential privacy, secure aggregation, fraud detection, explainable AI, non-IID learning, and recent financial FL systems. The table also highlights how this project improves on those works.

The main references and their relevance are summarized below.

| Area | Reference in Paper.pdf | Contribution | Limitation Noted | How This Project Uses or Improves It |
|---|---|---|---|---|
| Federated Learning | McMahan et al., 2017 | Introduced FedAvg and parameter aggregation | No direct real-world application in the paper | Applied FedAvg to a UPI fraud detection setting |
| FL Survey | Kairouz et al., 2021 | Discussed FL challenges such as privacy and communication | Survey, no working implementation | Built a working multi-bank simulation |
| Differential Privacy | Abadi et al., 2016 | Added privacy through noise | Accuracy and privacy tradeoff | Added clipped updates and Gaussian noise in DP-FedAvg |
| Secure Aggregation | Bonawitz et al., 2017 | Secure sharing of model updates | Not domain-specific | Used the idea of protected update sharing in a banking scenario |
| Fraud Detection ML | Carcillo et al., 2021 | Applied ML to fraud detection | Centralized data only | Added decentralized/federated fraud detection |
| Sequence Fraud Detection | Jurgovsky et al., 2018 | Used behavioral transaction patterns | Requires large centralized data | Added real-time prediction dashboard over structured features |
| Non-IID Federated Data | Zhao et al., 2022 | Studied heterogeneous client data | Accuracy can drop with non-IID data | Created non-IID bank data and aligned feature columns |
| FedPAQ | Reisizadeh et al., 2020 | Reduced communication cost | Privacy not the focus | Combined privacy and fraud detection use case |
| Communication-Efficient FL | Kim et al., 2022 | Quantization in FL | Practical deployment gap | Built a simplified deployable pipeline |
| Edge AI Privacy | Wang et al., 2022 | Edge-based private learning | Not fraud-specific | Applied private edge-style learning to financial fraud |
| Membership Attack | Shokri et al., 2017 | Showed privacy risks in ML | No direct mitigation solution | Used FL and DP concepts to reduce raw-data leakage |
| SHAP Explainability | Lundberg et al., 2017 | Explainable ML | Not integrated with FL | Added feature-attribution style explanation in dashboard |
| FL + XAI Fraud | Awosika et al., 2024 | Combined FL and explainable fraud detection | Scalability and deployment challenges | Added end-to-end pipeline and dashboard |
| Recent Financial FL | 2023-2025 works | FL for banking systems | Mostly experimental | Built a complete working prototype |

Based on this paper comparison, the project positions itself as an applied, end-to-end system that combines traditional ML, federated learning, privacy-aware training, robust aggregation, and dashboard deployment.

## 5. Dataset

The project uses a synthetic UPI transaction dataset generated by `gen.py`.

The dataset includes features such as:

- Transaction amount
- Sender ID
- Receiver ID
- Device type
- UPI app
- Location
- Transaction hour
- Night transaction flag
- Weekend flag
- Transaction velocity
- Account age
- Receiver age
- New receiver flag
- Failed attempts in the last 24 hours
- Fraud label

The current dataset contains:

| Property | Value |
|---|---:|
| Total transactions | 12,000 |
| Fraud cases | 678 |
| Fraud rate | 5.65% |
| Banks | Bank A, Bank B, Bank C |

The data is non-IID, meaning each bank has a different distribution of transactions and fraud rates. This is closer to a real multi-bank situation than simply splitting one identical dataset into equal parts.

## 6. System Architecture

The system has four main layers:

1. Data generation and preprocessing
2. Model training and evaluation
3. Federated learning simulation
4. Streamlit dashboard

The high-level workflow is:

```text
Synthetic UPI data
        |
        v
Bank A, Bank B, Bank C datasets
        |
        v
Train traditional ML and federated models
        |
        v
Evaluate using AUC, PR-AUC, F1, precision, recall
        |
        v
Save best prediction model, scaler, plots, and results
        |
        v
Streamlit dashboard for prediction and analysis
```

## 7. Models Used

### 7.1 Logistic Regression

Logistic Regression is a traditional binary classification model. It predicts whether a transaction is fraudulent or genuine by calculating a fraud probability from input features.

In this project, LR is the best overall model by AUC. It is used as the active prediction model in the dashboard.

### 7.2 Random Forest

Random Forest is an ensemble model that uses multiple decision trees. It can capture non-linear patterns and feature interactions. It is included as a traditional ML baseline.

### 7.3 MLP

MLP, or Multi-Layer Perceptron, is a neural network model. It is included to compare deep-learning-style classification against simpler tabular ML models.

### 7.4 Centralized Model

The centralized model trains on combined data from all banks.

```text
Bank A data + Bank B data + Bank C data -> Central training
```

This gives a strong benchmark, but it is not privacy-preserving because raw bank data is collected in one place.

### 7.5 FedAvg Model

FedAvg stands for Federated Averaging.

The process is:

1. Server sends a global model to all banks.
2. Each bank trains the model locally on its own data.
3. Banks send model updates to the server.
4. The server averages the updates.
5. The updated global model is sent back for the next round.

In this project, FedAvg achieves performance very close to centralized training. This is important because it shows that federated learning can preserve much of the performance while avoiding direct raw-data sharing.

### 7.6 DP-FedAvg

DP-FedAvg adds differential privacy to federated learning.

The model updates are:

1. Clipped to limit the influence of any single bank.
2. Noised using Gaussian noise.
3. Aggregated at the server.

This improves privacy but may reduce model accuracy.

### 7.7 Robust DP-FedAvg

Robust DP-FedAvg uses coordinate-wise median aggregation instead of simple averaging. This helps when one bank sends abnormal or malicious updates.

In the project, this model is treated as the recommended privacy-preserving model, not the highest-performing model overall.

## 8. How Fraud Detection Works

When a UPI transaction is entered into the dashboard, the system performs these steps:

1. The transaction details are converted into model features.
2. The features are aligned with the training feature columns.
3. The scaler normalizes the input.
4. The active prediction model calculates a fraud probability.
5. The probability is converted into a risk category.

Risk levels are:

```text
LOW risk     -> probability below 0.30
MEDIUM risk  -> probability between 0.30 and 0.60
HIGH risk    -> probability above 0.60
```

Important fraud signals include:

- High amount
- Night transaction
- High transaction velocity
- New receiver
- Multiple failed attempts
- New account or new receiver account
- Web device usage
- Risky transaction timing and behavior

## 9. Current Results

The dashboard now uses the best overall performing model for active prediction.

Top models by AUC are:

| Rank | Model | AUC | F1 | PR-AUC |
|---:|---|---:|---:|---:|
| 1 | LR | 0.8199 | 0.3163 | 0.2700 |
| 2 | Centralized | 0.8196 | 0.3125 | 0.2696 |
| 3 | FedAvg | 0.8185 | 0.3142 | 0.2680 |

The key observation is that FedAvg performs almost the same as centralized training:

```text
Centralized AUC = 0.8196
FedAvg AUC      = 0.8185
```

This supports the project claim that federated learning can achieve near-centralized fraud detection performance while reducing direct data sharing.

## 10. What Is Unique in This Project

The unique points of this project are:

1. It combines traditional ML and federated learning in one pipeline.
2. It compares LR, RF, MLP, centralized learning, FedAvg, DP-FedAvg, and robust DP-FedAvg.
3. It uses non-IID bank-wise UPI data instead of a simple identical data split.
4. It includes differential privacy using clipped and noised updates.
5. It includes Byzantine attack simulation.
6. It includes robust aggregation using coordinate-wise median.
7. It reports PR-AUC in addition to ROC-AUC, which is important for imbalanced fraud detection.
8. It has a working Streamlit dashboard for live fraud risk prediction.
9. It separates best overall model from best privacy-preserving model.
10. It converts research concepts from the referenced paper into a working prototype.

## 11. Limitations Overcome

The earlier version of the project had some limitations. These were improved in the current version.

| Earlier Limitation | Improvement Made |
|---|---|
| Fraud labels came from a very simple deterministic rule | Replaced with probabilistic, noisy, multi-factor fraud generation |
| Dataset was too easy, causing unrealistically high AUC/F1 | Created harder non-IID data with more realistic fraud distribution |
| Train/test split had evaluation issues | Added proper train, validation, and test splits |
| Scaler was fitted on full data, causing leakage | Scaler is now fitted on training data only |
| Federated learning was mostly simulated but not truly used for final results | Added real iterative FedAvg-style training |
| Dashboard mixed model output heavily with manual rules | Dashboard now loads and uses the saved active model |
| Only AUC was emphasized | Added PR-AUC, precision, recall, and F1 |
| No attack scenario | Added Byzantine client attack simulation |
| No robust aggregation comparison | Added median-based robust aggregation |
| Best model wording was confusing | Separated active best model, best federated model, and recommended private model |

## 12. Current Limitations

Although the project is stronger now, some limitations remain:

1. The dataset is synthetic, not real bank data.
2. The privacy accountant is an approximate implementation, not a production-grade DP accountant.
3. Secure aggregation is represented conceptually through update aggregation, not full cryptographic secure aggregation.
4. The model is designed for research simulation and demonstration, not production deployment.
5. More advanced explainability such as full SHAP integration can be added later.
6. Real-world UPI fraud patterns may change over time, so continuous retraining would be needed.

## 13. Future Scope

Possible future improvements are:

1. Use a real public fraud dataset or anonymized financial transaction dataset.
2. Add a stronger differential privacy accountant such as RDP or moments accountant.
3. Implement real cryptographic secure aggregation.
4. Add SHAP-based explanation for each prediction.
5. Add communication-cost analysis for federated rounds.
6. Add client dropout simulation.
7. Add more banks and more realistic non-IID distributions.
8. Deploy the dashboard using Docker or Streamlit Cloud.
9. Add database support for transaction logs.
10. Add real-time alerting for high-risk UPI transactions.

## 14. Conclusion

This project demonstrates a privacy-preserving UPI fraud detection system that combines traditional machine learning and federated learning. The LR model currently gives the best overall AUC and is used for active prediction. The centralized and FedAvg models achieve almost the same AUC, showing that federated learning can approach centralized performance without direct raw-data sharing.

The project also includes differential privacy, robust aggregation, and attack simulation, making it stronger than a basic fraud detection project. Based on the related work in `Paper.pdf`, the project contributes by converting multiple research ideas into a practical end-to-end prototype with model training, evaluation, saved artifacts, and a working dashboard.

## 15. Important Project Files

| File | Purpose |
|---|---|
| `gen.py` | Generates synthetic non-IID UPI fraud data |
| `main.py` | Runs ML and federated experiments |
| `app.py` | Streamlit dashboard |
| `results.json` | Stores metrics, configuration, and model selection |
| `model.pkl` | Saved active prediction model |
| `scaler.pkl` | Saved scaler used during prediction |
| `model_comparison.png` | Model comparison chart |
| `privacy_budget.png` | Privacy budget chart |
| `per_bank_f1.png` | Per-bank performance chart |
| `Paper.pdf` | Related work comparison used to frame the project |
