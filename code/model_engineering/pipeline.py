import os

import catboost as cb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

import mlflow

DATA_DIR = "/opt/airflow/data/processed"
MODEL_DIR = "/opt/airflow/models"

TARGET = "count"
CATEGORICAL_COLS = ["season", "holiday", "workingday", "weather"]


def load_train_test_split():
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def prepare_features(df):
    df = df.copy()

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(int)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return X, y


def train_model(X_train, y_train, X_test, y_test, n_trials=50):
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 200, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
            "verbose": 0,
            "random_seed": 42,
            "loss_function": "RMSE",
            "cat_features": [c for c in CATEGORICAL_COLS if c in X_train.columns],
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rmse_scores = []

        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = cb.CatBoostRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50)
            preds = model.predict(X_val)
            rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))

        return np.mean(rmse_scores)

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("bike-sharing-catboost")

    with mlflow.start_run(run_name="optuna-hp-search"):
        study = optuna.create_study(direction="minimize", study_name="catboost-hp-tuning")
        study.optimize(objective, n_trials=n_trials)

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_cv_rmse", study.best_value)

        final_params = study.best_params
        final_params.update({
            "verbose": 0,
            "random_seed": 42,
            "loss_function": "RMSE",
            "cat_features": [c for c in CATEGORICAL_COLS if c in X_train.columns],
        })

        model = cb.CatBoostRegressor(**final_params)
        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

        test_preds = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_mae = mean_absolute_error(y_test, test_preds)
        test_r2 = r2_score(y_test, test_preds)

        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_artifact("code/model_engineering/pipeline.py")

    return model


def save_model(model, filename="catboost_model.cbm"):
    os.makedirs(MODEL_DIR, exist_ok=True)
    filepath = os.path.join(MODEL_DIR, filename)
    model.save_model(filepath)
    return filepath


def run(n_trials=10):
    train_df, test_df = load_train_test_split()

    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)

    model = train_model(X_train, y_train, X_test, y_test, n_trials=n_trials)
    path = save_model(model)

    print(f"Model saved to {path}")

    return model, path
