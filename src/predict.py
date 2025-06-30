import pandas as pd
import pickle
import os

# ---------- Load model ----------
model_path = os.path.join("models", "churn_model.pkl")
model = pickle.load(open(model_path, "rb"))

# ---------- Load new data ----------
input_path = os.path.join("data", "new_customers.csv")
df = pd.read_csv(input_path)

# ---------- Drop unused columns ----------
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# ---------- Handle TotalCharges as numeric ----------
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# ---------- Feature Engineering ----------
df['ChargesRatio'] = df['TotalCharges'] / df['MonthlyCharges']
df['tenure_group'] = pd.cut(df['tenure'],
                            bins=[0, 12, 24, 48, 72],
                            labels=['0-12', '13-24', '25-48', '49-72'])

# ---------- One-Hot Encoding ----------
df_encoded = pd.get_dummies(df, drop_first=True)

# ---------- Align columns with model's training data ----------
# Load training columns used for model
train_data = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
train_data.drop(['customerID'], axis=1, inplace=True)
train_data['TotalCharges'] = pd.to_numeric(train_data['TotalCharges'], errors='coerce')
train_data.dropna(inplace=True)
train_data['ChargesRatio'] = train_data['TotalCharges'] / train_data['MonthlyCharges']
train_data['tenure_group'] = pd.cut(train_data['tenure'],
                                    bins=[0, 12, 24, 48, 72],
                                    labels=['0-12', '13-24', '25-48', '49-72'])
train_encoded = pd.get_dummies(train_data.drop('Churn', axis=1), drop_first=True)

# Align new data with training columns
df_encoded = df_encoded.reindex(columns=train_encoded.columns, fill_value=0)

# ---------- Predict ----------
predictions = model.predict(df_encoded)
df['Churn_Predicted'] = predictions

# ---------- Save results ----------
output_path = os.path.join("data", "predicted_churn_output.csv")
df.to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")
