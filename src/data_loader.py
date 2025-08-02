# src/data_loader.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_csv_dataset(csv_path):
    df = pd.read_csv(csv_path)
    labels = df.pop("label") if "label" in df.columns else None
    return df, labels