import pandas as pd
from skilearn.preprocessing import LabelEncoder

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def preprocess_data(df: DataFrame) -> tuple:
    df = df.copy()
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    df['TotalChanges'] = pd.to_numeric(df['TotalChanges'], errors='coerce')
    df.fillna.median(numeric_only= True)

    cat_cols = df.select_dtypes(include='object').column.tolist()
    encoders={}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders

