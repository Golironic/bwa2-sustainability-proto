import json
import os
from pathlib import Path
import pandas as pd
import xgboost as xgb
from google.cloud import bigquery
from dotenv import load_dotenv
from google.oauth2 import service_account

# --- Paths, feature list and shared feature engineering ---
try:
    from ml_core.config import MODEL_PATH, CATEGORIES_PATH, CSV_PATH, FEATURE_COLS, TARGET_COL
    from ml_core.features import add_features, assign_split
except ImportError:  # run directly from inside ml_core/
    from config import MODEL_PATH, CATEGORIES_PATH, CSV_PATH, FEATURE_COLS, TARGET_COL
    from features import add_features, assign_split

USE_BIGQUERY = False  # Set to True if pulling directly from BigQuery


def load_raw_data() -> pd.DataFrame:
    if not USE_BIGQUERY:
        df = pd.read_csv(CSV_PATH)
        print(f"Loaded {len(df)} rows from local CSV.")
        return df
load_dotenv()
# --- Setup Absolute Paths relative to THIS script ---
ML_DIR = Path(__file__).resolve().parent
MODEL_PATH = ML_DIR / "aqi_xgboost_model.json"
CATEGORIES_PATH = ML_DIR / "region_categories.json"
CSV_PATH = ML_DIR / "aqi_weather_historical.csv"

USE_BIGQUERY = False;

if USE_BIGQUERY:
    project_id = os.environ.get("GCP_PROJECT_ID")
    credentials_path = os.environ.get("FIREBASE_CREDENTIALS_PATH")
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    bq_client = bigquery.Client(credentials=credentials, project=project_id)

    DATASET_NAME = "air_quality" 
    TABLE_NAME_ONE = "aqi_readings"
    TABLE_NAME_TWO = "weather_readings"
    TABLE_NAME_THREE = "fire_hotspots"

    sql_query = f"""
        SELECT 
            aqi.timestamp,
            aqi.aqi_value,
            aqi.region,
            SAFE_CAST(JSON_VALUE(aqi.other_pollutants, '$.temperature') AS FLOAT64) AS temperature_aqi,
            SAFE_CAST(JSON_VALUE(aqi.other_pollutants, '$.wind_speed') AS FLOAT64) AS wind_speed_aqi,
            SAFE_CAST(JSON_VALUE(aqi.other_pollutants, '$.relativehumidity') AS FLOAT64) AS humidity_aqi,
            SAFE_CAST(JSON_VALUE(aqi.other_pollutants, '$.wind_direction') AS FLOAT64) AS wind_direction_aqi
        FROM `{project_id}.{DATASET_NAME}.{TABLE_NAME_ONE}` AS aqi
        WHERE aqi.timestamp IS NOT NULL
        AND aqi_value IS NOT NULL
        ORDER BY aqi.timestamp ASC
    """
    
    # JOIN `{project_id}.{DATASET_NAME}.{TABLE_NAME_THREE}` AS fire
    # ON aqi.region = fire.region
    # AND TIMESTAMP_TRUNC(aqi.timestamp, HOUR) = TIMESTAMP_TRUNC(fire.timestamp, HOUR)

    df = bq_client.query(sql_query).to_dataframe()
    print(f"Successfully fetched {len(df)} rows from BigQuery.")
else:
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from local CSV.")

model = None  # Initialize model variable

def add_time_series_features(df):
    # Ensure proper ordering
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['region', 'timestamp']).reset_index(drop=True)

    # 1. Simple Lags
    df['aqi_lag_1h'] = df.groupby('region')['aqi_value'].shift(1)
    df['aqi_lag_2h'] = df.groupby('region')['aqi_value'].shift(2)
    df['aqi_lag_3h'] = df.groupby('region')['aqi_value'].shift(3)
    df['aqi_lag_24h'] = df.groupby('region')['aqi_value'].shift(24)

    # 2. Weather Lags (Optional: Temperature & Wind Trends)
    df['temp_lag_1h'] = df.groupby('region')['temperature_aqi'].shift(1)
    df['wind_lag_1h'] = df.groupby('region')['wind_speed_aqi'].shift(1)

    # 3. Rolling Aggregates (applied on aqi_lag_1h to avoid target leakage)
    df['aqi_roll_mean_3h'] = df.groupby('region')['aqi_lag_1h'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df['aqi_roll_mean_6h'] = df.groupby('region')['aqi_lag_1h'].transform(
        lambda x: x.rolling(6, min_periods=1).mean()
    )
    df['aqi_roll_mean_24h'] = df.groupby('region')['aqi_lag_1h'].transform(
        lambda x: x.rolling(24, min_periods=1).mean()
    )
    df['aqi_roll_std_24h'] = df.groupby('region')['aqi_lag_1h'].transform(
        lambda x: x.rolling(24, min_periods=1).std()
    )

    # Drop initial NaN rows created by 24-hour shifting
    df = df.dropna().reset_index(drop=True)
    return df

try:
    print("Engineering temporal, lag, and rolling features...")
    df = add_time_series_features(df);

    # 2. Feature Engineering (Prepare temporal features locally)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['region'] = df['region'].astype('category')

    # Features (X) vs Target (y)
    X = df[[
        'hour', 'dayofweek', 'month', 'region',
        'temperature_aqi', 'humidity_aqi', 'wind_speed_aqi', 'wind_direction_aqi',
        # --- New Engineered Features ---
        'aqi_lag_1h', 'aqi_lag_2h', 'aqi_lag_3h', 'aqi_lag_24h',
        'temp_lag_1h', 'wind_lag_1h',
        'aqi_roll_mean_3h', 'aqi_roll_mean_6h', 'aqi_roll_mean_24h', 'aqi_roll_std_24h'
    ]]

# 1. Create the future target: shift AQI backward by 1 row per region
    df['target_aqi_next_hour'] = df.groupby('region')['aqi_value'].shift(-1)

    # 2. Shifting creates a NaN in the very last row of each region (since there is no "next hour" available). Drop these.
    df = df.dropna(subset=['target_aqi_next_hour']).reset_index(drop=True)

    # 3. Set y to the new future target
    y = df['target_aqi_next_hour']

    # 80/20 Chronological split
    split_idx = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Dataset split: {len(X_train)} training rows | {len(X_val)} validation rows.")

    print("Starting XGBoost training...")

    model = xgb.XGBRegressor(
        n_estimators=5000,
        learning_rate=0.015,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",          # Fast histogram binning for 300k+ rows
        enable_categorical=True,
        random_state=42,
        early_stopping_rounds=50,    # Stop if validation RMSE doesn't improve for 50 trees
        eval_metric="rmse"
    )

# Fit model with evaluation feedback
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100  # Logs training progress every 100 trees
    )

    print(f"\n✅ Training complete! Best tree iteration: {model.best_iteration}")

# 5. Save the model to disk after training
    model.save_model(str(MODEL_PATH))
    print("✅ Model saved to disk as 'aqi_xgboost_model.json'")

    categories = df['region'].cat.categories.tolist()
    with open(str(CATEGORIES_PATH), "w") as f:
        json.dump(categories, f)
    print(f"💾 Saved 'aqi_xgboost_model.json' and 'region_categories.json' successfully to {ML_DIR}.")
except Exception as e:
    print(f" Error fetching or training data: {e}")
