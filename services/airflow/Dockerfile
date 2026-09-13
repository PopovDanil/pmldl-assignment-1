FROM apache/airflow:3.3.1

USER airflow

RUN pip install --no-cache-dir \
    mlflow \
    scikit-learn \
    xgboost \
    catboost \
    optuna \
    skl2onnx \
    joblib \
    pandas \
    numpy \
    docker
