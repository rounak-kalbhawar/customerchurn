import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load predictions
file_path = os.path.join("data", "predicted_churn_output.csv")

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"ERROR: File not found at {file_path}")
    exit(1)

if 'Churn_Predicted' not in df.columns:
    print("ERROR: 'Churn_Predicted' column not found in CSV.")
    exit(1)

# -------------------------
# Count Plot
# -------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn_Predicted', data=df, hue='Churn_Predicted', palette='coolwarm', legend=False)
plt.title("Churn Prediction Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Churn Predicted (0 = No, 1 = Yes)", fontsize=12)
plt.ylabel("Number of Customers", fontsize=12)
plt.xticks([0, 1], ['No', 'Yes'], fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# -------------------------
# Box Plot: Monthly Charges vs Churn
# -------------------------
if 'MonthlyCharges' in df.columns:
    plt.figure(figsize=(7, 5))
    sns.boxplot(x='Churn_Predicted', y='MonthlyCharges', data=df, hue='Churn_Predicted', palette='coolwarm', legend=False)
    plt.title("Monthly Charges vs Churn Prediction", fontsize=14, fontweight='bold')
    plt.xlabel("Churn Predicted (0 = No, 1 = Yes)", fontsize=12)
    plt.ylabel("Monthly Charges", fontsize=12)
    plt.xticks([0, 1], ['No', 'Yes'], fontsize=10)
    plt.tight_layout()
    plt.show()
else:
    print("MonthlyCharges column not found. Skipping boxplot.")

# -------------------------
# Pie Chart
# -------------------------
plt.figure(figsize=(5, 5))
churn_counts = df['Churn_Predicted'].value_counts()
labels = ['No Churn', 'Churn']
colors = ['#66c2a5', '#fc8d62']
plt.pie(churn_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
plt.title("Churn Prediction Proportion", fontsize=13, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.show()
