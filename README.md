# 📌 Customer Churn Prediction System

## Overview

This project implements an **end-to-end, production-style machine learning system** to predict customer churn in a telecom setting and expose predictions through a REST API.

The objective is not only to predict churn, but to **support business decision-making** by identifying high-risk customers early and enabling targeted retention strategies.

---

## 🎯 Problem Statement

Customer churn directly impacts revenue, customer lifetime value, and acquisition costs.  
This system predicts whether a customer is likely to churn based on demographic attributes, service usage patterns, and account information.

Key questions addressed:
- Which customers are at highest risk of churn?
- What factors are driving churn behavior?
- How can predictions be operationalized for retention actions?

---

## 🧠 Business Framing

This is a **binary classification problem** where prediction errors have asymmetric business costs:

- **False Negatives** (missed churners): lost revenue and customer lifetime value  
- **False Positives** (incorrect churn alerts): unnecessary retention incentives and operational cost  

As a result, model evaluation and threshold selection are guided by **business trade-offs**, not raw accuracy.

---

## 📊 Data & Features

The dataset represents a telecom subscription environment and includes:

- Customer demographics  
- Account tenure and contract details  
- Service usage and billing behavior  

Feature engineering considerations:
- Explicit leakage prevention
- Normalization of usage-related features
- Derived indicators such as tenure buckets and customer value flags

---

## ⚙️ Modeling Approach

### Baseline Model
- **Logistic Regression**  
  Used as an interpretable benchmark for churn prediction.

### Primary Model
- **RandomForestClassifier**  
  Chosen to capture non-linear relationships and feature interactions.

### Pipeline Design
- End-to-end `scikit-learn` Pipeline
- `ColumnTransformer` for consistent preprocessing of numerical and categorical features
- Reproducible training and inference workflow

---

## 📈 Evaluation Strategy

Models are evaluated using metrics appropriate for imbalanced classification:

- Precision
- Recall
- ROC–AUC
- Confusion Matrix analysis

Decision thresholds are tuned to balance **retention cost vs churn risk**, rather than optimizing a single metric in isolation.

---

## 🚀 Deployment & Serving

The trained model is deployed via a **Flask REST API**, enabling:

- Real-time churn prediction for individual customers
- Clear separation of training and inference logic
- Model persistence and versioning using `joblib`

The API design mirrors how predictive models are typically exposed in production systems.

---

## 🛠️ Tech Stack

- Python, Pandas, NumPy  
- Scikit-learn (Pipeline, ColumnTransformer)  
- RandomForestClassifier  
- Flask (REST API)  
- Joblib  

---

## ⚠️ Limitations

- Evaluation focuses on classification performance rather than long-term retention impact  
- Business cost assumptions are illustrative and not company-specific  
- No explicit temporal modeling of churn behavior is included  

---

## 🔮 Possible Extensions

- Cost-sensitive learning
- Customer Lifetime Value (CLV) integration
- SHAP-based explainability for decision support
- Batch and streaming inference workflows

---

## ✅ Key Takeaways

- Churn prediction is fundamentally a **business decision problem**, not just a modeling task  
- Robust preprocessing pipelines are critical for deployment readiness  
- Interpretable baselines and non-linear models both play complementary roles  

