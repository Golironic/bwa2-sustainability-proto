import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from ml_core.config import MODEL_PATH, CATEGORIES_PATH, CSV_PATH, FEATURE_COLS, TARGET_COL
    from ml_core.features import add_features, assign_split
except ImportError:  # run directly from inside ml_core/
    from config import MODEL_PATH, CATEGORIES_PATH, CSV_PATH, FEATURE_COLS, TARGET_COL
    from features import add_features, assign_split

USE_BIGQUERY = False  # Set to True if pulling directly from BigQuery

# --- 1. Model Loading ---
if not os.path.exists(MODEL_PATH) or not os.path.exists(CATEGORIES_PATH):
    raise FileNotFoundError(
        "Model or categories file missing. Run model_trainer.py first "
        "(expects aqi_xgboost_model.json and region_categories.json in ml_core/)."
    )

print("📥 Loading saved model and regional metadata...")
model = xgb.XGBRegressor()
model.load_model(str(MODEL_PATH))

with open(CATEGORIES_PATH, "r") as f:
    region_categories = json.load(f)


# --- 2. Load Evaluation Data ---
if USE_BIGQUERY:
    from dotenv import load_dotenv
    from google.cloud import bigquery
    from google.oauth2 import service_account
    load_dotenv()

    project_id = os.environ.get("GCP_PROJECT_ID")
    credentials_path = os.environ.get("FIREBASE_CREDENTIALS_PATH")
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    bq_client = bigquery.Client(credentials=credentials, project=project_id)

    sql_query = f"""
        SELECT a.ts AS timestamp, a.aqi_value, a.region,
               w.temperature_c AS temperature_aqi, w.humidity_pct AS humidity_aqi, w.wind_speed_kmh AS wind_speed_aqi, w.wind_direction_deg AS wind_direction_aqi
        FROM (
          SELECT TIMESTAMP_TRUNC(timestamp, HOUR) AS ts, region, AVG(aqi_value) AS aqi_value
          FROM `{project_id}.air_quality.aqi_readings`
          WHERE aqi_value IS NOT NULL
          GROUP BY ts, region
        ) a
        JOIN (
          SELECT TIMESTAMP_TRUNC(timestamp, HOUR) AS ts, region, AVG(temperature_c) AS temperature_c,
                 AVG(humidity_pct) AS humidity_pct, AVG(wind_speed_kmh) AS wind_speed_kmh,
                 AVG(wind_direction_deg) AS wind_direction_deg
          FROM `{project_id}.air_quality.weather_readings`
          GROUP BY ts, region
        ) w USING (ts, region)
        ORDER BY ts
    """
    df = bq_client.query(sql_query).to_dataframe()
else:
    df = pd.read_csv(CSV_PATH)


# --- 3. Same features, target and split as model_trainer.py ---
print("🛠️ Processing lag features and preparing the held-out test split...")
df_featured = add_features(df, training=True)
split = assign_split(df_featured)

test_df = df_featured[split == 'test'].copy()
unknown = set(test_df['region']) - set(region_categories)
if unknown:
    raise ValueError(f"Regions in the data but not in region_categories.json: {unknown}. Retrain the model.")
test_df['region'] = pd.Categorical(test_df['region'], categories=region_categories)

X_test = test_df[FEATURE_COLS]
y_test = test_df[TARGET_COL]

print(f"🔬 Evaluating on {len(X_test)} test rows (never used for training or early stopping)...")


# --- 4. Predictions & Global Accuracy Metrics ---
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
nonzero = y_test > 0  # MAPE is undefined where the actual AQI is 0
mape = np.mean(np.abs((y_test[nonzero] - y_pred[nonzero]) / y_test[nonzero])) * 100

print("\n" + "=" * 50)
print("📊 GLOBAL MODEL ACCURACY METRICS (next-hour AQI)")
print("=" * 50)
print(f"R² Score (Variance Explained) : {r2:.4f} ({r2 * 100:.2f}%)")
print(f"MAE  (Mean Absolute Error)     : {mae:.2f} AQI points")
print(f"RMSE (Root Mean Squared Error) : {rmse:.2f} AQI points")
print(f"MAPE (Mean Absolute % Error)  : {mape:.2f}%  (rows with actual AQI = 0 excluded)")
print("=" * 50)


# --- 5. Regional Accuracy Breakdown ---
res = test_df.copy()
res['predicted_aqi'] = y_pred
res['error'] = np.abs(res[TARGET_COL] - res['predicted_aqi'])

regional_metrics = []
for region_name, group in res.groupby('region', observed=True):
    regional_metrics.append({
        'Region': region_name,
        'Samples': len(group),
        'MAE': round(mean_absolute_error(group[TARGET_COL], group['predicted_aqi']), 2),
        'RMSE': round(np.sqrt(mean_squared_error(group[TARGET_COL], group['predicted_aqi'])), 2),
        'R²': round(r2_score(group[TARGET_COL], group['predicted_aqi']), 4)
    })

reg_summary_df = pd.DataFrame(regional_metrics).sort_values('MAE')
print("\n📍 REGIONAL ACCURACY BREAKDOWN")
print(reg_summary_df.to_string(index=False))


# --- 6. Feature Importance Ranking ---
importance_df = pd.DataFrame({
    'Feature': FEATURE_COLS,
    'Importance_Score': model.feature_importances_
}).sort_values('Importance_Score', ascending=False)

print("\n" + "=" * 50)
print("🏆 TOP 10 MOST IMPORTANT FEATURES")
print("=" * 50)
print(importance_df.head(10).to_string(index=False))


# --- 7. Random Prediction Comparison Sample ---
sample_comparison = res[['timestamp', 'region', TARGET_COL, 'predicted_aqi', 'error']].sample(10, random_state=42)
sample_comparison = sample_comparison.rename(columns={TARGET_COL: 'actual_next_hour_aqi'}).round(2)

print("\n" + "=" * 50)
print("👀 RANDOM 10 PREDICTION COMPARISON SAMPLES")
print("=" * 50)
print(sample_comparison.to_string(index=False))