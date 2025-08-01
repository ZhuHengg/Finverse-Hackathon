# src/data_loader.py
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_csv_dataset(csv_path):
    df = pd.read_csv(csv_path)
    categorical_cols = ['device_os', 'nav_path', 'ip_country']
    df = pd.get_dummies(df, columns=categorical_cols)

    if 'user_id' in df.columns:
        df = df.drop(columns=['user_id'])

    numerical_cols = ['login_hour', 'typing_speed_cpm']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    labels = None
    if 'label' in df.columns:
        labels = df['label']
        df = df.drop(columns=['label'])

    return df, labels

# Example usage to link to your CSV file
df, labels = load_csv_dataset(r"C:\Users\zhuheng\Documents\VSC\Machine Learning\Finverse\Data\ghostpattern_sessions.csv")
print("Loaded DataFrame:")
print(df)
print("Labels:")
print(labels)