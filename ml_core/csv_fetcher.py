import requests
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# This points to the ml_core/ folder itself
ML_DIR = Path(__file__).resolve().parent
CSV_PATH = ML_DIR / "aqi_weather_historical.csv"

# Setup target regions and coordinates
REGIONS = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Gandhinagar": {"lat": 23.2156, "lon": 72.6369},
    "Punjab": {"lat": 30.9010, "lon": 75.8573},  # Ludhiana station

    "Delhi": {"lat": 28.6139, "lon": 77.2090},
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

def fetch_region_data(region_name, lat, lon, start_date, end_date):
    print(f"Fetching real historical data for {region_name}...")
    
    # 1. Fetch Open-Meteo Historical Weather
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
    }
    res_w = requests.get(weather_url, params=weather_params).json()
    hourly_w = res_w.get("hourly", {})
    
    df_w = pd.DataFrame({
        "timestamp": hourly_w["time"],
        "temperature": hourly_w["temperature_2m"],
        "relativehumidity": hourly_w["relative_humidity_2m"],
        "wind_speed": hourly_w["wind_speed_10m"],
        "wind_direction": hourly_w["wind_direction_10m"]
    })

    # 2. Fetch Open-Meteo Air Quality Reanalysis Data
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "us_aqi,pm2_5"
    }
    res_aq = requests.get(aq_url, params=aq_params).json()
    hourly_aq = res_aq.get("hourly", {})
    
    df_aq = pd.DataFrame({
        "timestamp": hourly_aq["time"],
        "us_aqi": hourly_aq["us_aqi"]
    })

    # Merge weather and AQI on timestamp
    df = pd.merge(df_w, df_aq, on="timestamp", how="inner")
    df["region"] = region_name
    return df

# Download for all regions
all_frames = []
for region, coords in REGIONS.items():
    df_region = fetch_region_data(region, coords["lat"], coords["lon"], START_DATE, END_DATE)
    all_frames.append(df_region)

full_df = pd.concat(all_frames, ignore_index=True)

# Drop rows with null values to clean training dataset
full_df = full_df.dropna(subset=["us_aqi", "temperature", "relativehumidity", "wind_speed", "wind_direction"])

print(f"\nFetched total {len(full_df)} hourly records across all regions.")

# --- OPTION A: Save as Local CSV (For quick offline XGBoost training) ---
csv_df = full_df.rename(columns={
    "us_aqi": "aqi_value",
    "temperature": "temperature_aqi",
    "relativehumidity": "humidity_aqi",
    "wind_speed": "wind_speed_aqi",
    "wind_direction": "wind_direction_aqi"
})
csv_df.to_csv(str(CSV_PATH), index=False)
print(f"Saved local dataset to '{CSV_PATH}' for XGBoost training.")


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