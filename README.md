# 🏦 Bank Customer Churn Prediction

## 📘 Overview
Customer churn is a major challenge for financial institutions as retaining existing customers is often more cost-effective than acquiring new ones.  
This project focuses on **predicting customer churn in a bank** using machine learning techniques. By analyzing customer demographics, account information, and banking activity, the model identifies which customers are most likely to leave the bank, enabling data-driven retention strategies.

---

## 📊 Dataset Information
**Source:** [Kaggle - Bank Churn Modeling](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers)  

**Dataset Details:**
- **Rows:** ~10,000 customer records  
- **Features:** 14 attributes including demographics, account balance, product usage, and activity  
- **Target Variable:** `Exited` (1 = Customer Churned, 0 = Customer Retained)

**Key Features Used:**
- `CreditScore`, `Age`, `Gender`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`

**Preprocessing Steps:**
1. Handled missing values (if any) and removed duplicates  
2. Label encoded categorical variables (`Gender`, `Geography`)  
3. Standardized numerical features using `StandardScaler`  
4. Split dataset into **80% training** and **20% testing**  

---

## ⚙️ Methods and Approach

### 🔹 Problem Formulation
The task is a **binary classification** problem — predicting whether a customer will churn (`1`) or not (`0`).  

### 🔹 Workflow
```mermaid
graph TD;
    A[Data Collection] --> B[Data Cleaning & Preprocessing];
    B --> C[Exploratory Data Analysis];
    C --> D[Feature Engineering & Selection];
    D --> E[Model Training];
    E --> F[Model Evaluation & Comparison];
    F --> G[Insights & Prescriptive Analysis];

| Model               | Description                                            | Accuracy |
| ------------------- | ------------------------------------------------------ | -------- |
| Logistic Regression | Baseline model for binary classification               | 81.2%    |
| Random Forest       | Ensemble model capturing non-linear relationships      | 85.9%    |
| XGBoost             | Gradient boosting algorithm providing best performance | 87.3%    |
| Decision Tree       | Simple interpretable model                             | 82.1%    |
| KNN                 | Instance-based learning algorithm                      | 80.7%    |

