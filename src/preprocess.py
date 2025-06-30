import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def preprocess_data(df: pd.DataFrame) -> tuple:
    df = df.copy()
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.fillna(df.median(numeric_only= True), inplace=True)

    cat_cols = df.select_dtypes(include='object').columns.tolist()
    encoders={}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders

