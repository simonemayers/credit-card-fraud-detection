# utils.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import DATA_PATH

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=['id'])
    return df

def scale_amount(df):
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    return df
