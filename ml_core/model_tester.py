import json
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from google.cloud import bigquery
from google.oauth2 import service_account
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# --- 1. Configurations & Model Loading ---
try:
    from ml_core.config import ML_DIR, MODEL_PATH, CATEGORIES_PATH, CSV_PATH, FEATURE_COLS
except ImportError:  # run directly from inside ml_core/
    from config import ML_DIR, MODEL_PATH, CATEGORIES_PATH, CSV_PATH, FEATURE_COLS

USE_BIGQUERY = False  # Set to True if pulling directly from BigQuery

if not os.path.exists(MODEL_PATH) or not os.path.exists(CATEGORIES_PATH):
    raise FileNotFoundError(
        "Model or categories file missing. Ensure 'aqi_xgboost_model.json' "
        "and 'region_categories.json' are in the working directory."
    )

print("📥 Loading saved model and regional metadata...")
model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)

with open(CATEGORIES_PATH, "r") as f:
    region_categories = json.load(f)


# --- 2. Load Evaluation Data ---
if USE_BIGQUERY:
    project_id = os.environ.get("GCP_PROJECT_ID")
    credentials_path = os.environ.get("FIREBASE_CREDENTIALS_PATH")
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    bq_client = bigquery.Client(credentials=credentials, project=project_id)

    sql_query = f"""
        SELECT 
            timestamp, aqi_value, region,
            SAFE_CAST(JSON_VALUE(other_pollutants, '$.temperature') AS FLOAT64) AS temperature_aqi,
            SAFE_CAST(JSON_VALUE(other_pollutants, '$.wind_speed') AS FLOAT64) AS wind_speed_aqi,
            SAFE_CAST(JSON_VALUE(other_pollutants, '$.relativehumidity') AS FLOAT64) AS humidity_aqi,
            SAFE_CAST(JSON_VALUE(other_pollutants, '$.wind_direction') AS FLOAT64) AS wind_direction_aqi
        FROM `{project_id}.air_quality.aqi_readings`
        WHERE timestamp IS NOT NULL AND aqi_value IS NOT NULL
        ORDER BY timestamp ASC
    """
    df = bq_client.query(sql_query).to_dataframe()
else:
    df = pd.read_csv(CSV_PATH)


# --- 3. Reconstruct 19 Features (Matching Training Schema) ---
def add_time_series_features(data):
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values(by=['region', 'timestamp']).reset_index(drop=True)

    # Current-hour AQI
    data['aqi_lag_0h'] = data['aqi_value']

    # AQI Lags
    data['aqi_lag_1h'] = data.groupby('region')['aqi_value'].shift(1)
    data['aqi_lag_2h'] = data.groupby('region')['aqi_value'].shift(2)
    data['aqi_lag_3h'] = data.groupby('region')['aqi_value'].shift(3)
    data['aqi_lag_24h'] = data.groupby('region')['aqi_value'].shift(24)

    # Weather Lags
    data['temp_lag_1h'] = data.groupby('region')['temperature_aqi'].shift(1)
    data['wind_lag_1h'] = data.groupby('region')['wind_speed_aqi'].shift(1)

    # Rolling Statistics
    data['aqi_roll_mean_3h'] = data.groupby('region')['aqi_lag_1h'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    data['aqi_roll_mean_6h'] = data.groupby('region')['aqi_lag_1h'].transform(
        lambda x: x.rolling(6, min_periods=1).mean()
    )
    data['aqi_roll_mean_24h'] = data.groupby('region')['aqi_lag_1h'].transform(
        lambda x: x.rolling(24, min_periods=1).mean()
    )
    data['aqi_roll_std_24h'] = data.groupby('region')['aqi_lag_1h'].transform(
        lambda x: x.rolling(24, min_periods=1).std()
    )

    return data.dropna().reset_index(drop=True)

print("🛠️ Processing lag features and preparing validation split...")
df_featured = add_time_series_features(df)

# Same target and per-region split as model_trainer.py (must stay identical)
df_featured['target_aqi_next_hour'] = df_featured.groupby('region')['aqi_value'].shift(-1)
df_featured = df_featured.dropna(subset=['target_aqi_next_hour']).reset_index(drop=True)
rank_in_region = df_featured.groupby('region').cumcount()
region_size = df_featured.groupby('region')['region'].transform('size')
is_val = rank_in_region >= (region_size * 0.8)

# Temporal & Categorical Features
df_featured['hour'] = df_featured['timestamp'].dt.hour
df_featured['dayofweek'] = df_featured['timestamp'].dt.dayofweek
df_featured['month'] = df_featured['timestamp'].dt.month
df_featured['region'] = pd.Categorical(df_featured['region'], categories=region_categories)

feature_cols = FEATURE_COLS

X = df_featured[feature_cols]
y = df_featured['target_aqi_next_hour']

# Re-create identical per-region chronological validation slice (last 20% of each region)
X_val = X[is_val]
y_val = y[is_val]
val_df = df_featured[is_val]

print(f"🔬 Evaluating on {len(X_val)} validation rows...")


# --- 4. Predictions & Global Accuracy Metrics ---
y_pred = model.predict(X_val)

mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred)
mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100

print("\n" + "=" * 50)
print("📊 GLOBAL MODEL ACCURACY METRICS")
print("=" * 50)
print(f"R² Score (Variance Explained) : {r2:.4f} ({r2 * 100:.2f}%)")
print(f"MAE  (Mean Absolute Error)     : {mae:.2f} AQI points")
print(f"RMSE (Root Mean Squared Error) : {rmse:.2f} AQI points")
print(f"MAPE (Mean Absolute % Error)  : {mape:.2f}%")
print("=" * 50)


# --- 5. Regional Accuracy Breakdown ---
val_df_copy = val_df.copy()
val_df_copy['predicted_aqi'] = y_pred
val_df_copy['error'] = np.abs(val_df_copy['target_aqi_next_hour'] - val_df_copy['predicted_aqi'])

regional_metrics = []
for region_name, group in val_df_copy.groupby('region', observed=False):
    if len(group) == 0:
        continue
    r_mae = mean_absolute_error(group['target_aqi_next_hour'], group['predicted_aqi'])
    r_rmse = np.sqrt(mean_squared_error(group['target_aqi_next_hour'], group['predicted_aqi']))
    r_r2 = r2_score(group['target_aqi_next_hour'], group['predicted_aqi'])
    regional_metrics.append({
        'Region': region_name,
        'Samples': len(group),
        'MAE': round(r_mae, 2),
        'RMSE': round(r_rmse, 2),
        'R²': round(r_r2, 4)
    })

reg_summary_df = pd.DataFrame(regional_metrics).sort_values('MAE')
print("\n📍 REGIONAL ACCURACY BREAKDOWN")
print(reg_summary_df.to_string(index=False))


# --- 6. Feature Importance Ranking ---
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance_Score': model.feature_importances_
}).sort_values('Importance_Score', ascending=False)

print("\n" + "=" * 50)
print("🏆 TOP 10 MOST IMPORTANT FEATURES")
print("=" * 50)
print(importance_df.head(10).to_string(index=False))


# --- 7. Random Prediction Comparison Sample ---
sample_comparison = val_df_copy[['timestamp', 'region', 'target_aqi_next_hour', 'predicted_aqi', 'error']].sample(10, random_state=42)
sample_comparison = sample_comparison.rename(columns={'target_aqi_next_hour': 'actual_aqi'}).round(2)

print("\n" + "=" * 50)
print("👀 RANDOM 10 PREDICTION COMPARISON SAMPLES")
print("=" * 50)
print(sample_comparison.to_string(index=False))