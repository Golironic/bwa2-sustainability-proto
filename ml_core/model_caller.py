import json
import pandas as pd
import xgboost as xgb
from pathlib import Path
import numpy as np

# Paths and feature list come from config.py
try:
    from ml_core.config import ML_DIR, MODEL_PATH, CATEGORIES_PATH, CSV_PATH, FEATURE_COLS
except ImportError:  # run directly from inside ml_core/
    from config import ML_DIR, MODEL_PATH, CATEGORIES_PATH, CSV_PATH, FEATURE_COLS

class AQICaller:
    # Mapping known external aliases from Person 1/Person 2 to your trained categories
    REGION_ALIASES = {
        "delhi": "Delhi",
        "delhi ncr": "Delhi",
        "new delhi": "Delhi",
        "gandhinagar": "Gandhinagar",
        "mumbai": "Mumbai",
        "gurgaon": "Gurugram",
        "greater noida": "Noida",
        "ludhiana station": "Ludhiana",
        # Add any specific sub-grid tags Person 1 used for API limits
    }
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

        # 3. Exact 19 features in training order
        self.feature_cols = list(FEATURE_COLS)

        self.required_input_cols = {
            'timestamp', 'region', 'aqi_value', 
            'temperature_aqi', 'humidity_aqi', 'wind_speed_aqi', 'wind_direction_aqi'
        }

    def get_supported_regions(self) -> list:
        """Returns the list of valid region categories the model was trained on."""
        return self.region_categories

    def normalize_input_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps standard Open-Meteo or raw BigQuery column aliases to required names.
        """
        df = df.copy()

        # Unpack 'other_pollutants' if raw JSON string/dict exists from BigQuery
        if 'other_pollutants' in df.columns:
            def parse_other_pollutants(val):
                if isinstance(val, str):
                    try:
                        return json.loads(val)
                    except json.JSONDecodeError:
                        return {}
                elif isinstance(val, dict):
                    return val
                return {}

            parsed_json = df['other_pollutants'].apply(parse_other_pollutants)
            
            if 'temperature_aqi' not in df.columns:
                df['temperature_aqi'] = parsed_json.apply(lambda x: x.get('temperature'))
            if 'humidity_aqi' not in df.columns:
                df['humidity_aqi'] = parsed_json.apply(lambda x: x.get('relativehumidity'))
            if 'wind_speed_aqi' not in df.columns:
                df['wind_speed_aqi'] = parsed_json.apply(lambda x: x.get('wind_speed'))
            if 'wind_direction_aqi' not in df.columns:
                df['wind_direction_aqi'] = parsed_json.apply(lambda x: x.get('wind_direction'))

        # Map common aliases
        rename_dict = {
            'us_aqi': 'aqi_value',
            'temperature': 'temperature_aqi',
            'relativehumidity': 'humidity_aqi',
            'humidity': 'humidity_aqi',
            'wind_speed': 'wind_speed_aqi',
            'wind_direction': 'wind_direction_aqi'
        }
        return df.rename(columns=rename_dict)

    def predict(self, recent_history_df: pd.DataFrame) -> float:
        """
        Takes raw historical data for a region, constructs the 19 lag/rolling features,
        and returns the next-hour AQI forecast.

        Parameters:
            recent_history_df (pd.DataFrame): DataFrame with AT LEAST 25 continuous 
            hourly rows for a single region.

        Returns:
            float: Predicted next-hour AQI (rounded to 2 decimal places).
        """

        df = self.normalize_input_schema(recent_history_df)
        
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

        # Sanitize all region entries in the DataFrame
        df['region'] = df['region'].apply(self.sanitize_region)
        df['region'] = pd.Categorical(df['region'], categories=self.region_categories)

        # --- 19 Feature Engineering ---
        # Current-hour AQI (the newest observation; the target is the NEXT hour)
        df['aqi_lag_0h'] = df['aqi_value']

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

        # --- Extract Latest Row & Predict ---
        # The last row contains features calculated using all prior history
        latest_row = df.iloc[[-1]][self.feature_cols]

        prediction = self.model.predict(latest_row)[0]
        return float(np.round(prediction, 2))

    def sanitize_region(self, input_region: str) -> str:
        """Sanitizes incoming region strings to match region_categories.json."""
        if not isinstance(input_region, str):
            raise TypeError(f"Expected string for region, got {type(input_region).__name__}")

        raw_region = input_region.strip()

        # 1. Direct match
        if raw_region in self.region_categories:
            return raw_region

        # 2. Alias lookup (case-insensitive)
        if raw_region.lower() in self.REGION_ALIASES:
            return self.REGION_ALIASES[raw_region.lower()]

        # 3. Case-insensitive search across valid categories
        for category in self.region_categories:
            if category.lower() == raw_region.lower():
                return category

        # 4. If no match, raise explicit error with allowed categories
        raise ValueError(
            f"Region '{input_region}' is not supported by the ML model. "
            f"Supported regions are: {self.region_categories}"
        )

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