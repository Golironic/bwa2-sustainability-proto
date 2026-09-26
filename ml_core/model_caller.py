import json
import pandas as pd
import xgboost as xgb
from pathlib import Path
import numpy as np

# Absolute path relative to model_caller.py
ML_DIR = Path(__file__).resolve().parent
MODEL_PATH = ML_DIR / "aqi_xgboost_model.json"
CATEGORIES_PATH = ML_DIR / "region_categories.json"

class AQICaller:
    def __init__(self, model_path=MODEL_PATH, categories_path=CATEGORIES_PATH):
        """
        Loads the XGBoost model weights and region category mappings into memory.
        Instantiate this ONCE when the backend server boots up.
        """
        self.model_path = Path(model_path)
        self.categories_path = Path(categories_path)

        if not self.model_path.exists() or not self.categories_path.exists():
            raise FileNotFoundError(
                f"Missing ML assets in {ML_DIR}.\n"
                f"Expected:\n  - {self.model_path}\n  - {self.categories_path}"
            )

        # 1. Load Model
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(self.model_path))

        # 2. Load Regional Metadata
        with open(self.categories_path, "r") as f:
            self.region_categories = json.load(f)

        # 3. Exact 18 features in training order
        self.feature_cols = [
            'hour', 'dayofweek', 'month', 'region',
            'temperature_aqi', 'humidity_aqi', 'wind_speed_aqi', 'wind_direction_aqi',
            'aqi_lag_1h', 'aqi_lag_2h', 'aqi_lag_3h', 'aqi_lag_24h',
            'temp_lag_1h', 'wind_lag_1h',
            'aqi_roll_mean_3h', 'aqi_roll_mean_6h', 'aqi_roll_mean_24h', 'aqi_roll_std_24h'
        ]

        self.required_input_cols = {
            'timestamp', 'region', 'aqi_value', 
            'temperature_aqi', 'humidity_aqi', 'wind_speed_aqi', 'wind_direction_aqi'
        }

    def predict(self, recent_history_df: pd.DataFrame) -> float:
        """
        Takes raw historical data for a region, constructs the 18 lag/rolling features,
        and returns the next-hour AQI forecast.

        Parameters:
            recent_history_df (pd.DataFrame): DataFrame with AT LEAST 25 continuous 
            hourly rows for a single region.

        Returns:
            float: Predicted next-hour AQI (rounded to 2 decimal places).
        """
        # --- Validation Checks ---
        missing_cols = self.required_input_cols - set(recent_history_df.columns)
        if missing_cols:
            raise ValueError(f"Input DataFrame missing required columns: {missing_cols}")

        if len(recent_history_df) < 25:
            raise ValueError(
                f"Received {len(recent_history_df)} rows. At least 25 hourly rows are "
                f"required to compute 24-hour lag and rolling features."
            )

        # --- Data Prep ---
        df = recent_history_df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp').reset_index(drop=True)

        # --- 18 Feature Engineering ---
        # AQI Lags
        df['aqi_lag_1h'] = df['aqi_value'].shift(1)
        df['aqi_lag_2h'] = df['aqi_value'].shift(2)
        df['aqi_lag_3h'] = df['aqi_value'].shift(3)
        df['aqi_lag_24h'] = df['aqi_value'].shift(24)

        # Weather Lags
        df['temp_lag_1h'] = df['temperature_aqi'].shift(1)
        df['wind_lag_1h'] = df['wind_speed_aqi'].shift(1)

        # Rolling Statistics
        df['aqi_roll_mean_3h'] = df['aqi_lag_1h'].rolling(3, min_periods=1).mean()
        df['aqi_roll_mean_6h'] = df['aqi_lag_1h'].rolling(6, min_periods=1).mean()
        df['aqi_roll_mean_24h'] = df['aqi_lag_1h'].rolling(24, min_periods=1).mean()
        df['aqi_roll_std_24h'] = df['aqi_lag_1h'].rolling(24, min_periods=1).std()

        # Temporal Features
        df['hour'] = df['timestamp'].dt.hour
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month

        # Categorical Region Mapping
        df['region'] = pd.Categorical(df['region'], categories=self.region_categories)

        # --- Extract Latest Row & Predict ---
        # The last row contains features calculated using all prior history
        latest_row = df.iloc[[-1]][self.feature_cols]

        prediction = self.model.predict(latest_row)[0]
        return float(np.round(prediction, 2))

# --- Run Fast Inference ---
# if __name__ == "__main__":
#     # 1. Load trained resources once
#     model, categories = load_inference_pipeline()
    
#     # 2. Define real-time input features (from user upload / live API)
#     new_sample = {
#         'hour': 18,
#         'dayofweek': 24,
#         'month': 8,
#         'temperature_aqi': 31.0,
#         'humidity_aqi': 57.0,
#         'region': "Gandhinagar",
#         'wind_speed_aqi': 11,
#         'wind_direction_aqi': 225.0,
#     }
    
#     # 3. Predict immediately
#     predicted_val = predict_aqi(model, categories, new_sample)
#     print(f"🔥 Predicted AQI: {predicted_val:.2f}")

