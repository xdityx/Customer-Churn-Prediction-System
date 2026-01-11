# 📌 Customer Churn Prediction System

This project implements a production-style machine learning pipeline to predict customer churn in a telecom setting and expose predictions through a REST API.

The goal is to identify customers at high churn risk early so the business can take targeted retention actions.

---

## 🔍 Problem Statement

Customer churn leads to revenue loss and higher acquisition costs.  
This system predicts whether a customer is likely to churn based on demographic, behavioral, and usage data.

---

## 🛠️ Tech Stack

- Python, Pandas, NumPy  
- Scikit-learn (Pipeline, ColumnTransformer)  
- RandomForestClassifier  
- Flask (REST API)  
- Joblib  

---

## ⚙️ System Architecture

1. Data cleaning & validation  
2. Feature engineering (tenure, normalized usage, customer value flags)  
3. End-to-end ML pipeline  
4. Threshold tuning  
5. REST API deployment  
