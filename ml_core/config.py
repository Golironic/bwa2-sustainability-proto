"""Shared paths and feature schema for everything in ml_core/.

Single source of truth: model_trainer, model_caller, model_tester and
local_model_tester all import from here, so they can't drift apart.
"""
from pathlib import Path

# This points to the ml_core/ folder itself
ML_DIR = Path(__file__).resolve().parent

MODEL_PATH = ML_DIR / "aqi_xgboost_model.json"
CATEGORIES_PATH = ML_DIR / "region_categories.json"
CSV_PATH = ML_DIR / "aqi_weather_historical.csv"

# Model inputs, in training order (19 features).
# aqi_lag_0h = the current hour's AQI; the model predicts the NEXT hour.
FEATURE_COLS = [
    'hour', 'dayofweek', 'month', 'region',
    'temperature_aqi', 'humidity_aqi', 'wind_speed_aqi', 'wind_direction_aqi',
    'aqi_lag_0h', 'aqi_lag_1h', 'aqi_lag_2h', 'aqi_lag_3h', 'aqi_lag_24h',
    'temp_lag_1h', 'wind_lag_1h',
    'aqi_roll_mean_3h', 'aqi_roll_mean_6h', 'aqi_roll_mean_24h', 'aqi_roll_std_24h',
]

TARGET_COL = 'target_aqi_next_hour'