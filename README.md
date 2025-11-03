# Machine-Learning-
# 🏦 Bank Customer Churn Prediction  

### 📘 Overview  
Customer retention is one of the most pressing challenges for banks today — acquiring new customers can cost **5x more** than retaining existing ones.  
This project focuses on predicting **which customers are likely to leave (churn)** using machine learning models. By identifying at-risk customers early, banks can design proactive retention strategies, improve customer satisfaction, and reduce revenue loss.  

The project uses a dataset of **10,000 customers** to analyze demographic, financial, and behavioral attributes. Multiple models were compared — including Logistic Regression, Random Forest, and XGBoost — to build an effective classification system.  

---

## 📊 Dataset Information  

**Source:** [Kaggle – Bank Churn Modeling Dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)  
**Records:** 10,000  
**Features:** 14 attributes  

**Key Columns:**
- `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `EstimatedSalary`
- `Geography`, `Gender`, `IsActiveMember`
- `Exited` – Target variable (1 = Churned, 0 = Retained)

### 🔧 Preprocessing Steps  
- Removed redundant fields: `RowNumber`, `CustomerId`, `Surname`  
- Encoded categorical variables (`Gender`, `Geography`) using Label Encoding  
- Standardized numerical columns using `StandardScaler`  
- Checked and treated outliers where necessary  
- Split dataset into **80% training** and **20% testing**

---

## ⚙️ Methods and Approach  

### 💡 Workflow Overview  
```mermaid
flowchart TD
A[Data Collection] --> B[Data Preprocessing]
B --> C[Exploratory Data Analysis]
C --> D[Feature Encoding & Scaling]
D --> E[Model Training & Evaluation]
E --> F[Result Interpretation & Prescriptive Analysis]
