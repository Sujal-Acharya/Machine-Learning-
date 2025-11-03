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

---
```
## 📈 Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to uncover patterns and relationships between variables that influence churn behavior. The visualizations below helped in understanding customer demographics, product usage, and financial profiles.

### 🔍 Key Findings:
- **Age vs Churn:** Customers aged **40–60** exhibit the highest churn tendency.
- **Gender vs Churn:** Female customers show a slightly higher churn rate than males.
- **Geography vs Churn:** Customers from **Germany** have a notably higher churn rate compared to Spain or France.
- **Balance vs Churn:** Mid-range balances correlate with higher churn likelihood.
- **Product Usage:** Customers holding only **one product** are more likely to leave than multi-product customers.

### 🧮 Visualization Samples (using Plotly Express)
```python
import plotly.express as px

px.histogram(df, x='Age', color='Exited', barmode='group', title='Age Distribution vs Churn')
px.box(df, x='Gender', y='Age', color='Exited', title='Gender-wise Churn by Age')
px.bar(df.groupby('Geography')['Exited'].mean().reset_index(), x='Geography', y='Exited', color='Geography', title='Churn Rate by Country')
px.scatter(df, x='Balance', y='EstimatedSalary', color='Exited', title='Balance vs Salary and Churn Relationship')
```
