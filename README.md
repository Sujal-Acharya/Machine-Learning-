# 📊 Customer Churn Prediction using Machine Learning

## 📘 Overview
This project aims to **predict customer churn** using Machine Learning techniques. Customer churn refers to when customers stop doing business with a company. By analyzing behavioral and demographic data, this model identifies key factors influencing churn and helps businesses retain valuable customers.

---

## 🧠 Project Objective
The main goal of this project is to build a predictive model that:
- Analyzes customer data to detect patterns related to churn.
- Predicts whether a customer will churn or stay.
- Provides data-driven insights to improve customer retention.

---

## 📂 Dataset
**File:** `churn.csv`  
The dataset contains various customer attributes such as:
- `customerID`: Unique customer identifier  
- `gender`: Male/Female  
- `SeniorCitizen`: Whether the customer is a senior citizen  
- `Partner`, `Dependents`: Family-related attributes  
- `tenure`: Duration with the company  
- `PhoneService`, `InternetService`: Service-related attributes  
- `Contract`, `PaymentMethod`: Billing-related details  
- `MonthlyCharges`, `TotalCharges`: Payment information  
- `Churn`: Target variable (Yes/No)

---

## ⚙️ Workflow
1. **Data Loading and Exploration**
   - Imported necessary libraries like `pandas`, `numpy`, `matplotlib`, `seaborn`, and `plotly`.
   - Loaded the dataset and explored its structure, missing values, and data types.

2. **Exploratory Data Analysis (EDA)**
   - Visualized distributions and relationships using:
     - Histograms
     - Correlation heatmaps
     - Box plots and count plots
   - Identified patterns and important features contributing to churn.

3. **Data Preprocessing**
   - Handled missing values and outliers.
   - Encoded categorical variables using Label Encoding/One-Hot Encoding.
   - Scaled numerical features for better model performance.
   - Split the data into training and testing sets.

4. **Model Building**
   - Trained multiple models including:
     - Logistic Regression  
     - Decision Tree Classifier  
     - Random Forest Classifier  
     - Support Vector Machine (SVM)
   - Compared their performance metrics (accuracy, precision, recall, F1-score).

5. **Model Evaluation**
   - Evaluated models using confusion matrix and classification report.
   - Visualized model performance through ROC curves and feature importance charts.

6. **Result & Insights**
   - Identified the most significant predictors of churn.
   - The best-performing model achieved high accuracy in predicting customer churn.
   - Insights can be used to design targeted retention campaigns.

---

## 📈 Key Insights
- Customers with **month-to-month contracts** are more likely to churn.
- **Higher monthly charges** correlate with increased churn risk.
- Customers without **online security or tech support** show higher churn tendencies.
- Long-term customers tend to stay loyal.

---

## 🧩 Technologies Used
- **Python**
- **Pandas, NumPy**
- **Matplotlib, Seaborn, Plotly**
- **Scikit-learn**
- **Jupyter Notebook**

---

## 🚀 How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Customer-Churn-Prediction.git
