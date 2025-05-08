import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_csv("dataset.csv")

# Replace 0s in specific columns with median
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

results = []

print("\nEvaluating models...\n")

# Model evaluation loop
for name, model in models.items():
    print(f"Training and evaluating: {name}")
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Get probabilities for ROC AUC
    try:
        y_test_proba = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_test_proba = None
        print(f"Warning: {name} does not support predict_proba. ROC AUC will be set to NaN.")

    # Evaluation metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_test_proba) if y_test_proba is not None else float('nan')

    results.append({
        "Model": name,
        "Training Accuracy": round(train_acc * 100, 2),
        "Testing Accuracy": round(test_acc * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1 Score": round(f1 * 100, 2),
        "ROC AUC": round(roc_auc * 100, 2) if y_test_proba is not None else np.nan
    })

# Create DataFrame and sort
results_df = pd.DataFrame(results)
results_df_sorted = results_df.sort_values(by="ROC AUC", ascending=False)

print("\nModel Evaluation Summary (in % and sorted by ROC AUC):")
print(results_df_sorted.to_string(index=False))

# Save results to CSV
results_df_sorted.to_csv("model_comparison_results.csv", index=False)

# Seaborn style
sns.set(style="whitegrid")

# Plot: Training vs Testing Accuracy
accuracy_plot = results_df.set_index('Model')[['Training Accuracy', 'Testing Accuracy']]
accuracy_plot.plot(kind='bar', figsize=(10, 6), colormap='Set2')
plt.title("Training vs Testing Accuracy (%)")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()

# Plot: Precision, Recall, F1 Score
metrics_plot = results_df.set_index('Model')[['Precision', 'Recall', 'F1 Score']]
metrics_plot.plot(kind='bar', figsize=(10, 6), colormap='Set1')
plt.title("Precision, Recall, and F1 Score by Model (%)")
plt.ylabel("Score (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()

# Plot: ROC AUC
plt.figure(figsize=(8, 5))
sns.barplot( x="ROC AUC", y="Model", hue="Model", data=results_df_sorted, palette="viridis", legend=False )
plt.title("ROC AUC Score by Model (%)")
plt.xlabel("ROC AUC (%)")
plt.ylabel("Model")
plt.tight_layout()
plt.grid(True, axis='x')
plt.show()

# Confusion Matrix
for i, (name, model) in enumerate(models.items()):
    y_test_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Diabetic", "Diabetic"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", values_format='d')
    ax.set_title(f"Confusion Matrix - {name}")
    plt.grid(False)
    plt.show()

# ROC Curve for All Models
plt.figure(figsize=(10, 7))
for name, model in models.items():
    try:
        y_test_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        roc_auc = roc_auc_score(y_test, y_test_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
    except AttributeError:
        print(f"Skipping ROC Curve for {name}: no predict_proba")

plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.title("ROC Curve Comparison of All Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
