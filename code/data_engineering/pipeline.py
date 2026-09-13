import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data() -> pd.DataFrame:
    return pd.read_csv('/opt/airflow/data/raw/train.csv', parse_dates=["datetime"])


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["temp_bin"] = pd.cut(df["temp"], bins=[-np.inf, 10, 20, 30, np.inf], labels=[0, 1, 2, 3]).astype(int)
    df["humidity_bin"] = pd.cut(df["humidity"], bins=[-np.inf, 40, 70, 100], labels=[0, 1, 2]).astype(int)
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    q_low = df["count"].quantile(0.01)
    q_high = df["count"].quantile(0.99)
    df = df[(df["count"] >= q_low) & (df["count"] <= q_high)].copy()
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    drop_cols = ["datetime", "casual", "registered"]
    features = [c for c in df.columns if c not in drop_cols + ["count"]]
    X = df[features]
    y = df["count"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def run() -> None:
    os.makedirs('/opt/airflow/data/processed', exist_ok=True)

    df = load_data()
    df = impute_missing(df)
    df = engineer_features(df)
    df = remove_outliers(df)

    X_train, X_test, y_train, y_test = split_data(df)

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    train.to_csv("/opt/airflow/data/processed/train.csv", index=False)
    test.to_csv("/opt/airflow/data/processed/test.csv", index=False)

if __name__ == "__main__":
    run()
