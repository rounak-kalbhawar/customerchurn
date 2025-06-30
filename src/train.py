# --------------------------
# Import Modules
# --------------------------

import pandas as pd
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# --------------------------
# Load & Prepare Data
# --------------------------
def load_data(path):
    df = pd.read_csv(path)
    df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

    # Feature engineering
    df['ChargesRatio'] = df['TotalCharges'] / df['MonthlyCharges']
    df['tenure_group'] = pd.cut(df['tenure'],
                                bins=[0, 12, 24, 48, 72],
                                labels=['0-12', '13-24', '25-48', '49-72'])

    df = pd.get_dummies(df, drop_first=True)
    return df

# --------------------------
# Confusion Matrix Plot
# --------------------------
def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    labels = ['No Churn', 'Churn']

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar=False, linewidths=0.5, linecolor='gray')

    plt.title("Confusion Matrix - Churn Prediction", fontsize=13, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

# --------------------------
# Professional Feature Importance Plot
# --------------------------
def plot_feature_importance(model, top_n=10):
    booster = model.get_booster()
    importance = booster.get_score(importance_type='gain')
    importance_df = pd.DataFrame(importance.items(), columns=["Feature", "Gain"])
    importance_df = importance_df.sort_values(by="Gain", ascending=False).head(top_n)
    importance_df = importance_df[::-1]  # For horizontal bar chart (smallest on bottom)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(importance_df["Feature"], importance_df["Gain"], color='steelblue')

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 3, bar.get_y() + bar.get_height() / 2,
                 f"{width:.2f}", va='center', fontsize=10)

    plt.title("Top 10 Most Important Features for Churn Prediction", fontsize=14, fontweight='bold')
    plt.xlabel("Gain-Based Importance Score", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    max_score = importance_df["Gain"].max()
    plt.xlim(0, max_score * 1.15)

    plt.tight_layout()
    plt.show()

# --------------------------
# Train + Tune Model
# --------------------------
def run_grid_search(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ratio = round((y_train == 0).sum() / (y_train == 1).sum(), 2)

    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=ratio,
        random_state=42
    )

    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1]
    }

    grid = GridSearchCV(estimator=xgb_model,
                        param_grid=param_grid,
                        scoring='roc_auc',
                        cv=3,
                        verbose=2,
                        n_jobs=-1)

    grid.fit(X_train, y_train)
    print("Best Parameters:", grid.best_params_)

    best_model = grid.best_estimator_

    # Evaluate
    y_pred = best_model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    print(f"\nAUC Score: {auc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Visualize
    plot_confusion(y_test, y_pred)
    plot_feature_importance(best_model)

    return best_model

# --------------------------
# Main Script
# --------------------------
if __name__ == "__main__":
    df = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    model = run_grid_search(df)
    pickle.dump(model, open("models/churn_model.pkl", "wb"))
