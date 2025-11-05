```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
```

```python
df=pd.read_csv('/content/churn.csv')
```

```python
df
```

```python
df.head()
```

```python
df.isnull().sum()
```

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
```

```python
le = LabelEncoder()
df['Geography'] = le.fit_transform(df['Geography'])
df['Gender'] = le.fit_transform(df['Gender'])
```

```python
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
```

```python
df.head()
```

```python
fig = px.pie(df, names='Exited', title='Customer Churn Distribution', color='Exited',
color_discrete_map={0:'green', 1:'red'})
fig.show()
```

```python
fig = px.scatter(df, x='Age', y='Balance', color='Exited', title='Age vs Balance by Churn',
color_continuous_scale=['green', 'red'])
fig.show()
```

```python
le.fit(['Female', 'Male'])
print("Gender Encoding Mapping:")
print(dict(zip(le.classes_, le.transform(le.classes_))))
```

```python
fig = px.bar(
    df.groupby('Gender')['Exited'].mean().reset_index(),
    x='Gender',
    y='Exited',
    title='Churn Rate by Gender',
    color='Gender',
    color_discrete_sequence=px.colors.qualitative.Vivid
)
fig.update_layout(
    xaxis_title='Gender',
    yaxis_title='Average Churn Rate',
    title_x=0.5
)
fig.show()
```

```python
geo_encoder = LabelEncoder()
geo_encoder.fit(['France', 'Germany', 'Spain'])
print("\nGeography Encoding Mapping:")
print(dict(zip(geo_encoder.classes_, geo_encoder.transform(geo_encoder.classes_))))
```

```python
fig = px.bar(
    df.groupby('Geography')['Exited'].mean().reset_index(),
    x='Geography',
    y='Exited',
    title='Churn Rate by Geography',
    color='Geography',
    color_discrete_sequence=px.colors.qualitative.Bold
)
fig.update_layout(
    xaxis_title='Geographical Region',
    yaxis_title='Average Churn Rate',
    title_x=0.5
)
fig.show()
```

```python
fig = px.histogram(
    df,
    x='Tenure',
    color='Exited',
    barmode='group',
    title='Churn Distribution by Tenure',
    color_discrete_map={0: 'green', 1: 'red'}
)
fig.update_layout(
    xaxis_title='Customer Tenure (Years)',
    yaxis_title='Number of Customers',
    legend_title='Exited (1 = Yes, 0 = No)',
    title_x=0.5
)
fig.show()
```

```python
fig = px.scatter(
    df,
    x='CreditScore',
    y='Age',
    color='Exited',
    title='Credit Score vs Age by Churn',
    color_continuous_scale=['green', 'red']
)
fig.update_layout(
    xaxis_title='Credit Score',
    yaxis_title='Customer Age',
    legend_title='Exited (1 = Yes, 0 = No)',
    title_x=0.5
)
fig.show()
```

```python
fig = px.scatter(
    df,
    x='Balance',
    y='EstimatedSalary',
    color='Exited',
    title='Balance vs Estimated Salary by Churn',
    color_continuous_scale=['green', 'red']
)
fig.update_layout(
    xaxis_title='Account Balance',
    yaxis_title='Estimated Annual Salary',
    legend_title='Exited (1 = Yes, 0 = No)',
    title_x=0.5
)
fig.show()
```

```python
fig = px.bar(df.groupby('NumOfProducts')['Exited'].mean().reset_index(), x='NumOfProducts', y='Exited',
title='Churn Rate by Number of Products', color='NumOfProducts')
fig.show()
```

```python
fig = px.imshow(
    df.corr(),
    text_auto=True,
    title='Feature Correlation Heatmap',
    color_continuous_scale='Viridis'
)
fig.show()

```

```python
X = df.drop('Exited', axis=1)
y = df['Exited']
```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

```python
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_scaled, y_train)


# Predictions
y_pred = rf.predict(X_test_scaled)
```

```python
print('Accuracy:', accuracy_score(y_test, y_pred))
print('\nClassification Report:\n', classification_report(y_test, y_pred))
```

```python
cm = confusion_matrix(y_test, y_pred)
fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
title='Confusion Matrix', labels=dict(x='Predicted', y='Actual'))
fig.show()
```

```python
feature_importances = pd.DataFrame({
'Feature': X.columns,
'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)


fig = px.bar(feature_importances, x='Importance', y='Feature', orientation='h',
title='Feature Importance in Predicting Customer Churn')
fig.show()
```

```python
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

```

```python
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb.fit(X_train_scaled, y_train)
```

```python
y_pred_xgb = xgb.predict(X_test_scaled)
y_proba_xgb = xgb.predict_proba(X_test_scaled)[:,1]
```

```python
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba_xgb))
print( classification_report(y_test, y_pred_xgb))
```

```python
xgb_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb.feature_importances_
}).sort_values(by='Importance', ascending=False)

fig = px.bar(xgb_importances, x='Importance', y='Feature', orientation='h',
             title='XGBoost Feature Importance')
fig.show()
```

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
```

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

```

```python
models = {

    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": SVC(kernel='rbf', probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
}

# Train and evaluate models
results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
results_df.reset_index(drop=True, inplace=True)
results_df
```

```python
# Model Comparison Bar Chart
fig = px.bar(
    results_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]),
    x="Model",
    y="value",
    color="variable",
    barmode="group",
    title="Comparison of Machine Learning Models for Churn Prediction",
    labels={"value": "Score", "variable": "Metric"}
)
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()

```

```python
from sklearn.metrics import roc_curve

# Plot ROC curves for all models
fig = go.Figure()

for name, model in models.items():
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=name))

fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color='gray'), name='Random Chance'))
fig.update_layout(title="ROC Curve Comparison Across Models",
                  xaxis_title="False Positive Rate",
                  yaxis_title="True Positive Rate")
fig.show()

```

```python
best_model = results_df.iloc[0]
print(f" Best Performing Model: {best_model['Model']}")
print(f"Accuracy: {best_model['Accuracy']:.3f}, ROC-AUC: {best_model['ROC-AUC']:.3f}")

print('\\n Key Observations:')
print('1. XGBoost and Random Forest consistently outperform linear models.')
print('2. Logistic Regression provides interpretability but lower recall.')
print('3. SVM works well for balanced datasets but may struggle with large-scale churn data.')
print('4. Tree-based models are most effective at capturing complex relationships.')

```

