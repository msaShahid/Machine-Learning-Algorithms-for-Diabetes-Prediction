import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_csv("dataset.csv")

# Replace 0s in important columns with median
columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in columns_to_replace:
    df[col] = df[col].replace(0, df[col].median())

# Feature and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier()
}

# Store metrics
results = []

# Evaluation
for name, model in models.items():
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]  # Needed for ROC AUC

    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_proba)

    results.append({
        "Model": name,
        "Training Accuracy": train_acc,
        "Testing Accuracy": test_acc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    })

# Create DataFrame
results_df = pd.DataFrame(results)
print("\nModel Evaluation Summary:")
print(results_df.sort_values(by="ROC AUC", ascending=False))

# Optional: Save to CSV
results_df.to_csv("model_comparison_results.csv", index=False)

# Set seaborn style
sns.set(style="whitegrid")

# Plot: Training vs Testing Accuracy
plt.figure(figsize=(10, 6))
accuracy_plot = results_df.set_index('Model')[['Training Accuracy', 'Testing Accuracy']]
accuracy_plot.plot(kind='bar', figsize=(10, 6), colormap='Set2')
plt.title("Training vs Testing Accuracy")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()

# Plot: Precision, Recall, F1 Score
plt.figure(figsize=(10, 6))
metrics_plot = results_df.set_index('Model')[['Precision', 'Recall', 'F1 Score']]
metrics_plot.plot(kind='bar', figsize=(10, 6), colormap='Set1')
plt.title("Precision, Recall, and F1 Score by Model")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()

# Plot: ROC AUC
plt.figure(figsize=(8, 5))
sns.barplot(x="ROC AUC", y="Model", data=results_df.sort_values("ROC AUC", ascending=False), palette="viridis")
plt.title("ROC AUC Score by Model")
plt.xlabel("ROC AUC")
plt.ylabel("Model")
plt.tight_layout()
plt.grid(True, axis='x')
plt.show()

