import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

# Paths, feature list and shared feature engineering
try:
    from ml_core.config import ML_DIR, MODEL_PATH, CATEGORIES_PATH, FEATURE_COLS
    from ml_core.features import add_features, NUMERIC_FEATURES
except ImportError:  # run directly from inside ml_core/
    from config import ML_DIR, MODEL_PATH, CATEGORIES_PATH, FEATURE_COLS
    from features import add_features, NUMERIC_FEATURES


class AQICaller:
    # Mapping known external aliases to the trained categories. Every value MUST be a
    # category the model was trained on (enforced in sanitize_region).
    REGION_ALIASES = {
        "delhi": "Delhi",
        "delhi-ncr": "Delhi",
        "delhi ncr": "Delhi",
        "new delhi": "Delhi",
        "gandhinagar": "Gandhinagar",
        "mumbai": "Mumbai",
        "gurgaon": "Gurugram",
        "greater noida": "Noida",
        "ludhiana": "Punjab",          # 'Punjab' was trained on Ludhiana's coordinates
        "ludhiana station": "Punjab",
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

        # 3. The 19 features in training order (from config.py)
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

        # Map common aliases (never create duplicate columns if the target name exists)
        rename_dict = {
            'us_aqi': 'aqi_value',
            'temperature': 'temperature_aqi',
            'relativehumidity': 'humidity_aqi',
            'humidity': 'humidity_aqi',
            'wind_speed': 'wind_speed_aqi',
            'wind_direction': 'wind_direction_aqi'
        }
        rename_dict = {k: v for k, v in rename_dict.items()
                       if k in df.columns and v not in df.columns}
        return df.rename(columns=rename_dict)

    def predict(self, recent_history_df: pd.DataFrame) -> float:
        """
        Takes raw historical data for ONE region, builds the 19 features and
        returns the next-hour AQI forecast.

        Parameters:
            recent_history_df (pd.DataFrame): AT LEAST 25 consecutive hourly rows
            for a single region. Gaps within the last 24 hours raise a ValueError
            (the lag/rolling features would otherwise be wrong).

        Returns:
            float: Predicted next-hour AQI (rounded to 2 decimal places).
        """
        # Use the NORMALIZED frame from here on
        df = self.normalize_input_schema(recent_history_df)

        # --- Validation Checks ---
        missing_cols = self.required_input_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Input DataFrame missing required columns: {missing_cols}")

        if len(df) < 25:
            raise ValueError(
                f"Received {len(df)} rows. At least 25 hourly rows are "
                f"required to compute 24-hour lag and rolling features."
            )

        # Sanitize region entries; must be a single region
        df['region'] = df['region'].apply(self.sanitize_region)
        if df['region'].nunique() != 1:
            raise ValueError(
                f"predict() expects a single region, got {sorted(df['region'].unique())}"
            )

        # --- Same feature code as training (strict hourly grid) ---
        feats = add_features(df, training=False)
        latest = feats.iloc[[-1]].copy()

        bad = [c for c in NUMERIC_FEATURES if pd.isna(latest.iloc[0][c])]
        if bad:
            raise ValueError(
                f"Cannot build features {bad}: the history has gaps or missing values "
                f"within the last 24 hours (need 25 consecutive hourly rows)."
            )

        latest['region'] = pd.Categorical(latest['region'], categories=self.region_categories)
        prediction = self.model.predict(latest[self.feature_cols])[0]
        return float(np.round(prediction, 2))

    def sanitize_region(self, input_region: str) -> str:
        """Sanitizes incoming region strings to match region_categories.json."""
        if not isinstance(input_region, str):
            raise TypeError(f"Expected string for region, got {type(input_region).__name__}")

        raw_region = input_region.strip()

        # 1. Direct match
        if raw_region in self.region_categories:
            return raw_region

        # 2. Alias lookup (case-insensitive) - the target must be a trained category
        alias = self.REGION_ALIASES.get(raw_region.lower())
        if alias is not None:
            if alias in self.region_categories:
                return alias
            raise ValueError(
                f"Alias '{input_region}' maps to '{alias}', which the model was not trained on. "
                f"Supported regions are: {self.region_categories}"
            )

        # 3. Case-insensitive search across valid categories
        for category in self.region_categories:
            if category.lower() == raw_region.lower():
                return category

        # 4. If no match, raise explicit error with allowed categories
        raise ValueError(
            f"Region '{input_region}' is not supported by the ML model. "
            f"Supported regions are: {self.region_categories}"
        )