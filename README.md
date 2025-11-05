# 📊 Customer Churn Prediction using Machine Learning

## 📘 Overview
This project focuses on predicting customer churn using machine learning techniques. Customer churn occurs when customers discontinue using a company’s service. By analyzing behavioral and demographic data, this model helps businesses understand the drivers of churn and improve customer retention.

---

## 🧠 Project Objective
The primary goal is to build and evaluate machine learning models that:
- Identify customers likely to churn.
- Discover the most influential features affecting churn.
- Improve business decision-making through predictive analytics.

---

## 📂 Dataset
**File Used:** `churn.csv`  

The dataset contains various customer-related features including:
- **Demographics:** Gender, SeniorCitizen, Partner, Dependents  
- **Account Information:** Tenure, Contract type, Payment method  
- **Services Subscribed:** PhoneService, InternetService, Streaming options  
- **Billing Details:** MonthlyCharges, TotalCharges  
- **Target Variable:** `Churn` (Yes/No)

---

## ⚙️ Workflow
1. **Data Loading & Cleaning**
   - Loaded dataset using `pandas`
   - Checked for missing values, corrected data types, and removed inconsistencies

2. **Exploratory Data Analysis (EDA)**
   - Visualized key relationships using `matplotlib`, `seaborn`, and `plotly`
   - Identified churn trends across gender, tenure, and service type

3. **Data Preprocessing**
   - Encoded categorical variables (Label Encoding & One-Hot Encoding)
   - Scaled numerical features
   - Split data into training and test sets (80:20 ratio)

4. **Model Training**
   - Implemented and compared multiple models:
     - Logistic Regression  
     - Decision Tree  
     - Random Forest  
     - XGBoost  

5. **Model Evaluation**
   - Evaluated performance using:
     - Accuracy
     - Precision, Recall, F1-score
     - ROC-AUC Score
     - Confusion Matrix

6. **Model Selection**
   - Compared metrics and selected **XGBoost** as the final model based on superior results.

---

## 📈 Evaluation Metrics

### 🔹 Logistic Regression
| Metric | Score |
|--------|--------|
| Accuracy | 0.8645 |
| Precision (Class 0 / 1) | 0.88 / 0.75 |
| Recall (Class 0 / 1) | 0.96 / 0.47 |
| F1-Score (Class 0 / 1) | 0.92 / 0.58 |
| Weighted Avg F1 | 0.85 |

---

### 🔹 XGBoost (Best Performing Model)
| Metric | Score |
|--------|--------|
| **Accuracy** | **0.870** |
| **ROC-AUC** | **0.868** |
| Precision (Class 0 / 1) | 0.89 / 0.74 |
| Recall (Class 0 / 1) | 0.96 / 0.51 |
| F1-Score (Class 0 / 1) | 0.92 / 0.61 |
| Weighted Avg F1 | 0.86 |

**Key Observations:**
1. XGBoost and Random Forest outperform linear models.
2. Logistic Regression provides solid baseline accuracy.
3. XGBoost achieves the highest recall and overall balanced performance.

---

## 📊 Insights
- Customers with **month-to-month contracts** and **higher monthly charges** tend to churn more.
- **Senior citizens** and **single customers** are more likely to leave.
- Longer-tenure customers are more loyal.
- Lack of **tech support** or **online security** increases churn probability.

---

## 🧩 Technologies Used
- **Python**
- **NumPy**, **Pandas**
- **Matplotlib**, **Seaborn**, **Plotly**
- **Scikit-learn**
- **XGBoost**
- **Jupyter Notebook**

---

## 🚀 How to Run the Project
1. Clone this repository:
   ```bash
   [git clone https://github.com/yourusername/Customer-Churn-Prediction.git](https://github.com/Sujal-Acharya/Machine-Learning-.git)
