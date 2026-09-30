import json
import os

import pandas as pd
import xgboost as xgb

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

    # Google / dotenv packages are only needed on this path
    from dotenv import load_dotenv
    from google.cloud import bigquery
    from google.oauth2 import service_account
    load_dotenv()

    project_id = os.environ.get("GCP_PROJECT_ID")
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    bq_client = bigquery.Client(credentials=credentials, project=project_id)

    DATASET_NAME = "air_quality"
    TABLE_NAME_ONE = "aqi_readings"
    TABLE_NAME_THREE = "fire_hotspots"

    sql_query = f"""
        SELECT a.ts AS timestamp, a.aqi_value, a.region,
               w.temperature_c AS temperature_aqi, w.humidity_pct AS humidity_aqi, w.wind_speed_kmh AS wind_speed_aqi, w.wind_direction_deg AS wind_direction_aqi
        FROM (
          SELECT TIMESTAMP_TRUNC(timestamp, HOUR) AS ts, region, AVG(aqi_value) AS aqi_value
          FROM `{project_id}.{DATASET_NAME}.{TABLE_NAME_ONE}`
          WHERE aqi_value IS NOT NULL
          GROUP BY ts, region
        ) a
        JOIN (
          SELECT TIMESTAMP_TRUNC(timestamp, HOUR) AS ts, region, AVG(temperature_c) AS temperature_c,
                 AVG(humidity_pct) AS humidity_pct, AVG(wind_speed_kmh) AS wind_speed_kmh,
                 AVG(wind_direction_deg) AS wind_direction_deg
          FROM `{project_id}.{DATASET_NAME}.weather_readings`
          GROUP BY ts, region
        ) w USING (ts, region)
        ORDER BY ts
    """
    # JOIN `{project_id}.{DATASET_NAME}.{TABLE_NAME_THREE}` AS fire
    # ON aqi.region = fire.region
    # AND TIMESTAMP_TRUNC(aqi.timestamp, HOUR) = TIMESTAMP_TRUNC(fire.timestamp, HOUR)

    df = bq_client.query(sql_query).to_dataframe()
    print(f"Successfully fetched {len(df)} rows from BigQuery.")
    return df


def main():
    df = load_raw_data()

    print("Engineering temporal, lag, and rolling features (strict hourly grid)...")
    df = add_features(df, training=True)

    # Per-region chronological split: 70% train | 15% val (early stopping) | 15% test
    # (test is only used by model_tester.py). Done while 'region' is still a string.
    split = assign_split(df)

    categories = sorted(df['region'].unique().tolist())
    df['region'] = pd.Categorical(df['region'], categories=categories)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    tr, va = split == 'train', split == 'val'
    print(f"Dataset split: {tr.sum()} train | {va.sum()} val | {(split == 'test').sum()} test rows.")

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
    model.fit(
        X[tr], y[tr],
        eval_set=[(X[va], y[va])],
        verbose=100  # Logs training progress every 100 trees
    )
    print(f"\n✅ Training complete! Best tree iteration: {model.best_iteration}")

    model.save_model(str(MODEL_PATH))
    with open(CATEGORIES_PATH, "w") as f:
        json.dump(categories, f)
    print(f"💾 Saved '{MODEL_PATH.name}' and '{CATEGORIES_PATH.name}' to {MODEL_PATH.parent}.")


if __name__ == "__main__":
    # No try/except on purpose: a failed training run must fail loudly.
    main()