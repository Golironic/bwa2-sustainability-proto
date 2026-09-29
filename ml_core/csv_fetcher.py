import os
import requests
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

ML_DIR = Path(__file__).resolve().parent
CSV_PATH = ML_DIR / "aqi_weather_historical.csv"

REGIONS = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Gandhinagar": {"lat": 23.2156, "lon": 72.6369},
    "Punjab": {"lat": 30.9010, "lon": 75.8573},
    "Gurugram": {"lat": 28.4595, "lon": 77.0266},
    "Noida": {"lat": 28.5355, "lon": 77.3910},
    "Faridabad": {"lat": 28.4089, "lon": 77.3178},
    "Ghaziabad": {"lat": 28.6692, "lon": 77.4538},
    "Ludhiana": {"lat": 30.9010, "lon": 75.8573},
    "Amritsar": {"lat": 31.6340, "lon": 74.8723},
    "Patiala": {"lat": 30.3398, "lon": 76.3869},
    "Bathinda": {"lat": 30.2110, "lon": 74.9455},
    "Jalandhar": {"lat": 31.3260, "lon": 75.5762}
}

START_DATE = "2024-01-01"
END_DATE = "2026-08-31"

# Identical math from Person 1's fetch_aqi_weather.py
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), 
    (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)
]

def pm25_to_aqi(pm25):
    if pd.isna(pm25) or pm25 < 0:
        return None
    c = int(pm25 * 10) / 10.0
    for c_lo, c_hi, aqi_lo, aqi_hi in PM25_BREAKPOINTS:
        if c_lo <= c <= c_hi:
            return round(((aqi_hi - aqi_lo) / (c_hi - c_lo)) * (c - c_lo) + aqi_lo, 1)
    return 500.0

def fetch_region_data(region_name, lat, lon, start_date, end_date):
    print(f"Fetching data for {region_name}...")
    
    # 1. Weather
    w_res = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": lat, "longitude": lon, "start_date": start_date, "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
    }).json().get("hourly", {})
    
    df_w = pd.DataFrame({
        "timestamp": w_res["time"],
        "temperature_aqi": w_res["temperature_2m"],
        "humidity_aqi": w_res["relative_humidity_2m"],
        "wind_speed_aqi": w_res["wind_speed_10m"],
        "wind_direction_aqi": w_res["wind_direction_10m"]
    })

    # 2. AQI (Fetch PM2.5 and convert manually to match backend contract)
    aq_res = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params={
        "latitude": lat, "longitude": lon, "start_date": start_date, "end_date": end_date,
        "hourly": "pm2_5"
    }).json().get("hourly", {})
    
    df_aq = pd.DataFrame({"timestamp": aq_res["time"], "pm2_5": aq_res["pm2_5"]})
    df_aq["aqi_value"] = df_aq["pm2_5"].apply(pm25_to_aqi)

# Merge weather and AQI
    df = pd.merge(df_w, df_aq, on="timestamp", how="inner").drop(columns=["pm2_5"])
    
    # Forward fill up to 3 consecutive missing hours per region before dropping
    df["aqi_value"] = df["aqi_value"].ffill(limit=3)
    df["region"] = region_name
    return df

all_frames = [fetch_region_data(r, c["lat"], c["lon"], START_DATE, END_DATE) for r, c in REGIONS.items()]
full_df = pd.concat(all_frames, ignore_index=True).dropna()

full_df.to_csv(str(CSV_PATH), index=False)
print(f"Saved aligned PM2.5 dataset to '{CSV_PATH}'")

# --- OPTION A: Save as Local CSV (For quick offline XGBoost training) ---
# csv_df = full_df.rename(columns={
#     "us_aqi": "aqi_value",
#     "temperature": "temperature_aqi",
#     "relativehumidity": "humidity_aqi",
#     "wind_speed": "wind_speed_aqi",
#     "wind_direction": "wind_direction_aqi"
# })
# csv_df.to_csv(str(CSV_PATH), index=False)
# print(f"Saved local dataset to '{CSV_PATH}' for XGBoost training.")


# --- OPTION B: Load to GCP BigQuery (Matches your SQL schema) ---
# project_id = os.environ.get("GCP_PROJECT_ID")
# credentials_path = os.environ.get("FIREBASE_CREDENTIALS_PATH")

# if project_id and credentials_path and os.path.exists(credentials_path):
#     print("Uploading dataset to BigQuery...")
#     credentials = service_account.Credentials.from_service_account_file(credentials_path)
#     bq_client = bigquery.Client(credentials=credentials, project=project_id)

#     DATASET_NAME = "air_quality"
#     TABLE_NAME = "aqi_readings"
    
#     bq_records = []
#     for _, row in full_df.iterrows():
#         other_pollutants = json.dumps({
#             "temperature": float(row["temperature"]),
#             "wind_speed": float(row["wind_speed"]),
#             "relativehumidity": float(row["relativehumidity"]),
#             "wind_direction": float(row["wind_direction"])
#         })
#         bq_records.append({
#             "timestamp": pd.to_datetime(row["timestamp"]).strftime("%Y-%m-%d %H:%M:%S UTC"),
#             "aqi_value": float(row["us_aqi"]),
#             "region": row["region"],
#             "other_pollutants": other_pollutants
#         })

#     df_bq = pd.DataFrame(bq_records)
#     table_ref = f"{project_id}.{DATASET_NAME}.{TABLE_NAME}"

#     job_config = bigquery.LoadJobConfig(
#         schema=[
#             bigquery.SchemaField("timestamp", "TIMESTAMP"),
#             bigquery.SchemaField("aqi_value", "FLOAT64"),
#             bigquery.SchemaField("region", "STRING"),
#             bigquery.SchemaField("other_pollutants", "JSON"),
#         ],
#         write_disposition="WRITE_TRUNCATE"  # Replaces existing baseline sample data
#     )

#     job = bq_client.load_table_from_dataframe(df_bq, table_ref, job_config=job_config)
#     job.result()
#     print(f"Successfully loaded {len(df_bq)} rows into BigQuery table `{table_ref}`!")
# else:
#     print("GCP credentials not configured; skipped BigQuery upload.")