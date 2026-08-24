import json
import os
import pandas as pd
import xgboost as xgb
from google.cloud import bigquery, firestore
from datetime import datetime, timezone
from dotenv import load_dotenv
from google.oauth2 import service_account

load_dotenv()  # Load environment variables from .env file

project_id = os.environ.get("GCP_PROJECT_ID")
credentials_path = os.environ.get("FIREBASE_CREDENTIALS_PATH")
credentials = service_account.Credentials.from_service_account_file(credentials_path)

# 1. Fetch BigQuery Data to Local Memory (Cost: Free/Pennies)
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
model = None  # Initialize model variable

try:

    df = bq_client.query(sql_query).to_dataframe()
    print(f"Successfully fetched {len(df)} rows from BigQuery.")

# df = pd.read_csv("aqi_weather_historical.csv")
# 2. Feature Engineering (Prepare temporal features locally)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month

# Features (X) vs Target (y)
    df['region'] = df['region'].astype('category')
    X = df[['hour', 'dayofweek', 'month', 'temperature_aqi', 'humidity_aqi', 'region', 'wind_speed_aqi', 'wind_direction_aqi']]
    y = df['aqi_value']

# Split a small 20% validation slice to monitor training live
    X_train, X_val = X[:int(len(X)*0.8)], X[int(len(X)*0.8):]
    y_train, y_val = y[:int(len(y)*0.8)], y[int(len(y)*0.8):]

    model = xgb.XGBRegressor(n_estimators=50, 
                        max_depth=3, 
                        learning_rate=0.1,
                        random_state=42,
                        enable_categorical=True,
                        )

# Fit model with evaluation feedback
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=10  # Logs training progress every 10 trees
    )

    print("✅ Local training complete!")

# 5. Save the model to disk after training
    model.save_model("aqi_xgboost_model.json")
    print("✅ Model saved to disk as 'aqi_xgboost_model.json'")

    categories = df['region'].cat.categories.tolist()
    with open("region_categories.json", "w") as f:
        json.dump(categories, f)
        print("✅ Model and categorical metadata saved successfully!")

except Exception as e:
    print(f" Error fetching or training data: {e}")

# 6. Predict Future AQI (e.g., Next Hour Prediction)
# Example feature input: Hour=14, Day=Monday(0), Temp=32°C, Humidity=65%
# try:
#     if model is None:
#         raise ValueError("Model is not trained. Cannot make predictions.")
    
#     sample_input = pd.DataFrame([[14, 0, 1, 32.0, 65.0, 1, 10.0, 180.0]], columns=['hour', 'dayofweek', 'month', 'temperature_aqi', 'humidity_aqi', 'region', 'wind_speed_aqi', 'wind_direction_aqi'])
#     predicted_aqi = float(model.predict(sample_input)[0])

#     print(f"Predicted AQI: {predicted_aqi:.2f}")
# except Exception as e:
#     print(f" Error predicting AQI: {e}")
