# Bike Sharing Demand Prediction Pipeline

Automated MLOps pipeline for predicting bike sharing demand using CatBoost, with Airflow orchestration, MLflow tracking, and Docker deployment.

## Project Structure

```
├── code/
│   ├── data_engineering/     # Data loading, cleaning, splitting
│   ├── model_engineering/    # Feature engineering, model training, evaluation
│   └── deployment/          # API (FastAPI) and App (Streamlit) with Dockerfiles
├── data/
│   ├── raw/                 # Raw dataset files
│   └── processed/           # Processed train/test splits
├── models/                  # Trained model files
├── services/airflow/        # Airflow configuration and DAGs
│   ├── dags/                # Pipeline DAG definitions
│   ├── docker-compose.yml   # Airflow services setup
│   └── Dockerfile           # Custom Airflow image
└── mlflow/                  # MLflow artifacts
```

## Prerequisites

- Docker and Docker Compose
- 4GB+ RAM, 2+ CPUs recommended
- Raw data in `data/raw/train.csv`

## Starting the Pipeline

1. Navigate to the Airflow directory:
   ```bash
   cd services/airflow
   ```

2. Start all Airflow services (including MLflow):
   ```bash
   docker compose up -d
   ```

3. Wait for services to initialize (~2-3 minutes), then access:
   - **Airflow UI**: http://localhost:8080 (login: `airflow` / password: `airflow`)
   - **MLflow UI**: http://localhost:5000

4. In Airflow UI, enable the `ML_pipeline` DAG to start automatic execution every 5 minutes.

## Pipeline Stages

The pipeline runs three sequential stages automatically:

1. **Data Engineering**: Loads raw data, imputes missing values, removes outliers, splits into train/test
2. **Model Engineering**: Trains CatBoost with Optuna hyperparameter tuning, logs metrics to MLflow
3. **Deployment**: Builds and runs Docker containers for API and web app

## Accessing the Deployed Application

After the pipeline completes the deployment stage:

- **Web App**: http://localhost:8501
- **API Endpoint**: http://localhost:8888
- **API Health Check**: http://localhost:8888/health

## Testing the Pipeline

### Test API directly
```bash
curl -X POST http://localhost:8888/predict \
  -H "Content-Type: application/json" \
  -d '{
    "season": 3,
    "holiday": 0,
    "workingday": 1,
    "weather": 1,
    "temp": 25.0,
    "atemp": 27.0,
    "humidity": 50,
    "windspeed": 10.0,
    "datetime": "2012-06-15 14:00:00"
  }'
```

### Test via Web App
1. Open http://localhost:8501
2. Select input parameters (season, weather, temperature, etc.)
3. Click "Predict Bike Rentals"
4. View the predicted demand

## Manual Pipeline Execution

To run a single pipeline execution:
1. Go to Airflow UI → DAGs → `ML_pipeline`
2. Click the play button (▶) → "Trigger DAG"

## Stopping the Pipeline

```bash
cd services/airflow
docker compose down
```