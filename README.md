# 💳 Explainable AI – Financial Fraud Detection System

## 📌 Project Overview

This project presents a robust, **explainable AI-driven fraud detection system** designed to detect fraudulent financial transactions with high precision. Built on real-world data and advanced ML algorithms, the system leverages explainability tools like **SHAP** and **LIME** to deliver transparency and trust alongside predictive performance.

---

## 🧠 Objectives

1. Build a high-performance classifier to detect fraudulent transactions.
2. Evaluate multiple models and optimize performance.
3. Apply explainable AI to enhance transparency of predictions.
4. Deliver a web application to make real-time predictions with interpretability.

---

## 📊 Dataset

- Source: [IEEE-CIS + Vesta Corporation](https://www.kaggle.com/competitions/ieee-fraud-detection)
- Files:
  - `transaction.csv`: 590,540 transactions, 394 features
  - `identity.csv`: 144,233 identity records, 41 features
- Merged on: `TransactionID`

### Class Distribution
- Legitimate: 96.5%
- Fraudulent: 3.5% (high class imbalance)

---

## 🔍 Exploratory Data Analysis (EDA)

- **Fraud is more frequent on weekends and early month dates.**
- **Credit cards, mobile devices, and mail.com domains show higher fraud rates.**
- **Fraudulent transactions average higher amounts than legitimate ones.**
- Temporal, product, and device usage patterns are strong fraud indicators.

---

## ⚙️ Data Preprocessing

- **Memory optimization** (~50% reduction)
- **Missing value handling** (flags, KNN imputation, mean filling)
- **Feature engineering** (transaction ratios, log transforms, PCA)
- **Class imbalance**: SMOTE oversampling + cost-sensitive learning

---

## 🤖 Modeling

### Algorithms Compared
- Random Forest
- Logistic Regression
- Decision Tree
- XGBoost
- LightGBM ✅ (Best)

### LightGBM Performance
- Accuracy: 97%
- AUC-ROC: 0.96
- Precision: 88%
- Recall: 81% (→ 90% after optimization)
- F1-Score: 0.84

### Optimization Techniques
- SMOTE oversampling
- Feature interactions
- Threshold tuning
- Ensemble modeling

---

## 🧠 Explainable AI (XAI)

### SHAP – Shapley Additive Explanations
- Global and local interpretability
- Top feature contributions for each prediction
- Interactive plots for feature importance

### Other Tools
- LIME for local explanation
- Feature importance heatmaps

---

## 🖥️ Web Application (Streamlit)

A user-friendly dashboard for fraud detection and interpretability:

### Key Features
- Real-time transaction scoring
- Interactive SHAP & LIME visualizations
- Feature importance charts
- User input forms with prediction explanations

---


