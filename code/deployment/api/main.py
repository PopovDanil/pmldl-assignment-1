import os
from contextlib import asynccontextmanager

import catboost as cb
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODELS_DIR = os.environ.get("MODELS_DIR", "/opt/airflow/models")
MODEL_PATHS = [
    os.path.join(MODELS_DIR, "catboost_model.cbm"),
    "./catboost_model.cbm",
]
model: cb.CatBoostRegressor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    for path in MODEL_PATHS:
        if path and os.path.isfile(path):
            model = cb.CatBoostRegressor()
            model.load_model(path)
            break
    yield
    model = None


app = FastAPI(title="Bike Sharing Prediction API", lifespan=lifespan)


class BikeSharingInput(BaseModel):
    season: int = Field(..., description="Season (1=spring, 2=summer, 3=fall, 4=winter)")
    holiday: int = Field(..., description="Whether the day is a holiday")
    workingday: int = Field(..., description="Whether the day is a working day")
    weather: int = Field(..., description="Weather condition")
    temp: float = Field(..., description="Temperature in Celsius")
    atemp: float = Field(..., description="Feeling temperature in Celsius")
    humidity: int = Field(..., description="Humidity")
    windspeed: float = Field(..., description="Wind speed")
    datetime: str = Field(..., description="Datetime string (e.g. 2012-12-31 23:00:00)")


class PredictionResponse(BaseModel):
    prediction: float


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: BikeSharingInput):
    if model is None:
        raise ValueError("Model not loaded")
    dt = pd.to_datetime(input_data.datetime)

    data = {
        "season": input_data.season,
        "holiday": input_data.holiday,
        "workingday": input_data.workingday,
        "weather": input_data.weather,
        "temp": input_data.temp,
        "atemp": input_data.atemp,
        "humidity": input_data.humidity,
        "windspeed": input_data.windspeed,
        "year": dt.year,
        "month": dt.month,
        "day": dt.day,
        "hour": dt.hour,
        "dayofweek": dt.dayofweek,
        "is_weekend": int(dt.dayofweek in [5, 6]),
        "temp_bin": _bin_temp(input_data.temp),
        "humidity_bin": _bin_humidity(input_data.humidity),
    }

    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return PredictionResponse(prediction=round(float(prediction), 2))


def _bin_temp(temp: float) -> int:
    if temp <= 10:
        return 0
    elif temp <= 20:
        return 1
    elif temp <= 30:
        return 2
    else:
        return 3


def _bin_humidity(humidity: int) -> int:
    if humidity <= 40:
        return 0
    elif humidity <= 70:
        return 1
    else:
        return 2
