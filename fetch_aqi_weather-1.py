"""
Pulls current AQI data (OpenAQ) + weather data (Open-Meteo) for 4 target
regions and loads them into BigQuery.
 
NOTE ON SCHEMA: aqi_readings is in "wide" format to match the team's
shared data contract - one row per station, with pm25/pm10/aqi_value as
columns (not one row per station+parameter). aqi_value is a simple US
EPA AQI calculation derived from PM2.5 (the most commonly available and
most health-relevant pollutant); if a station only reports other
pollutants (e.g. only NO2, no PM2.5), aqi_value will be null - real US
AQI is a max across pollutant sub-indices, this is a reasonable proxy
for a 3-day build, not the full EPA methodology.
 
Regions: Delhi-NCR, Punjab, Gandhinagar, Mumbai
 
Run:
    python fetch_aqi_weather.py
 
Requires:
    pip install google-cloud-bigquery requests python-dotenv
    A GCP project with BigQuery API enabled, and
    GOOGLE_APPLICATION_CREDENTIALS env var pointing to firebase-key.json
    (the same service account key works for BigQuery if it has the right role,
    otherwise generate a separate key from IAM & Admin > Service Accounts)
"""
 
import os
import json
import requests
from datetime import datetime, timezone
from google.cloud import bigquery
 
# ---- CONFIG ----
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "firebase-key.json")
 
PROJECT_ID = "carc-f5b14"   # <-- replace with your actual project id
DATASET_ID = "air_quality"
TABLE_ID = "aqi_readings"
WEATHER_TABLE_ID = "weather_readings"
OPENAQ_API_KEY = "f8a04510cf2ecb447233f89cc674759d3211d9162e94f1fbc003fbc60c5f100d"   # <-- get a free key at explore.openaq.org
 
REGIONS = {
    "Delhi-NCR":    {"lat": 28.6139, "lon": 77.2090, "radius_km": 60},
    "Punjab":       {"lat": 30.9010, "lon": 75.8573, "radius_km": 100},
    "Gandhinagar":  {"lat": 23.2156, "lon": 72.6369, "radius_km": 40},
    "Mumbai":       {"lat": 19.0760, "lon": 72.8777, "radius_km": 40},
}
 
# ---- BIGQUERY SETUP ----
client = bigquery.Client(project=PROJECT_ID)
dataset_ref = f"{PROJECT_ID}.{DATASET_ID}"
 
def ensure_dataset():
    try:
        client.get_dataset(dataset_ref)
    except Exception:
        ds = bigquery.Dataset(dataset_ref)
        ds.location = "asia-south1"
        client.create_dataset(ds)
        print(f"Created dataset {dataset_ref}")
 
def ensure_table(table_id, schema):
    table_ref = f"{dataset_ref}.{table_id}"
    try:
        client.get_table(table_ref)
    except Exception:
        table = bigquery.Table(table_ref, schema=schema)
        client.create_table(table)
        print(f"Created table {table_ref}")
    return table_ref
 
AQI_SCHEMA = [
    # --- fields from the shared team schema (aqi_readings) ---
    bigquery.SchemaField("station_id", "STRING"),
    bigquery.SchemaField("location", "RECORD", fields=[
        bigquery.SchemaField("lat", "FLOAT"),
        bigquery.SchemaField("lng", "FLOAT"),
    ]),
    bigquery.SchemaField("timestamp", "TIMESTAMP"),
    bigquery.SchemaField("pm25", "FLOAT"),
    bigquery.SchemaField("pm10", "FLOAT"),
    bigquery.SchemaField("aqi_value", "FLOAT"),  # derived US EPA AQI from pm25, see module docstring
    bigquery.SchemaField("source", "STRING"),
    # --- extra fields kept for context, not required by the contract ---
    bigquery.SchemaField("region", "STRING"),
    bigquery.SchemaField("station_name", "STRING"),
    bigquery.SchemaField("other_pollutants", "STRING"),  # JSON string of any non-pm25/pm10 params seen
]
 
WEATHER_SCHEMA = [
    bigquery.SchemaField("region", "STRING"),
    bigquery.SchemaField("lat", "FLOAT"),
    bigquery.SchemaField("lon", "FLOAT"),
    bigquery.SchemaField("temperature_c", "FLOAT"),
    bigquery.SchemaField("humidity_pct", "FLOAT"),
    bigquery.SchemaField("wind_speed_kmh", "FLOAT"),
    bigquery.SchemaField("wind_direction_deg", "FLOAT"),
    bigquery.SchemaField("timestamp", "TIMESTAMP"),
    bigquery.SchemaField("source", "STRING"),
]
 
# ---- AQI CALCULATION ----
# Simplified US EPA AQI breakpoints for PM2.5 (µg/m³ -> 0-500 AQI scale).
# Real EPA AQI uses 24-hr averaged PM2.5; here we apply the same breakpoint
# formula to the latest instantaneous reading, which is a reasonable proxy
# for a fast build but not official methodology.
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]
 
def pm25_to_aqi(pm25):
    """Convert a PM2.5 concentration (µg/m³) to a US EPA AQI value."""
    if pm25 is None:
        return None
    for c_lo, c_hi, aqi_lo, aqi_hi in PM25_BREAKPOINTS:
        if c_lo <= pm25 <= c_hi:
            return round(((aqi_hi - aqi_lo) / (c_hi - c_lo)) * (pm25 - c_lo) + aqi_lo, 1)
    return 500.0 if pm25 > 500.4 else None
 
# ---- FETCH FUNCTIONS ----
def fetch_openaq(region_name, lat, lon, radius_km):
    """Pull latest AQI measurements near a region from OpenAQ v3 API.
 
    NOTE: /v3/locations only returns station + sensor *metadata* (id, name,
    parameter) - it does NOT include a "latest" reading inline. Actual
    readings live at /v3/locations/{id}/latest and have to be fetched
    separately per location, then matched back to sensor metadata via
    sensorsId.
 
    Returns one row PER STATION (not per parameter) - pm25/pm10 pivoted
    into columns, any other pollutants bundled into other_pollutants as a
    JSON string, and aqi_value derived from pm25.
    """
    headers = {"X-API-Key": OPENAQ_API_KEY}
 
    loc_resp = requests.get(
        "https://api.openaq.org/v3/locations",
        params={
            "coordinates": f"{lat},{lon}",
            "radius": min(radius_km * 1000, 25000),  # OpenAQ max radius is 25km
            "limit": 20,
        },
        headers=headers,
        timeout=20,
    )
    loc_resp.raise_for_status()
    locations = loc_resp.json().get("results", [])
 
    rows = []
    for loc in locations:
        station_id = loc.get("id")
        station_name = loc.get("name", "unknown")
        coords = loc.get("coordinates", {})
 
        # Build a lookup of sensor metadata (parameter name/units) by sensorsId
        sensor_meta = {}
        for sensor in loc.get("sensors", []):
            sensor_meta[sensor.get("id")] = {
                "parameter": sensor.get("parameter", {}).get("name", "unknown"),
                "unit": sensor.get("parameter", {}).get("units", ""),
            }
 
        # Fetch actual latest readings for this location
        latest_resp = requests.get(
            f"https://api.openaq.org/v3/locations/{station_id}/latest",
            headers=headers,
            timeout=20,
        )
        if latest_resp.status_code != 200:
            continue
        latest_results = latest_resp.json().get("results", [])
        if not latest_results:
            continue
 
        # Pivot: collect all parameter readings for this station into one dict
        pm25 = None
        pm10 = None
        other = {}
        latest_ts = None
        station_lat, station_lon = coords.get("latitude"), coords.get("longitude")
 
        for reading in latest_results:
            sensors_id = reading.get("sensorsId")
            meta = sensor_meta.get(sensors_id, {})
            param = (meta.get("parameter") or "").lower()
            value = reading.get("value")
            ts = (reading.get("datetime") or {}).get("utc")
            if ts and (latest_ts is None or ts > latest_ts):
                latest_ts = ts
            reading_coords = reading.get("coordinates")
            if reading_coords:
                station_lat = reading_coords.get("latitude", station_lat)
                station_lon = reading_coords.get("longitude", station_lon)
 
            if param in ("pm25", "pm2.5"):
                pm25 = value
            elif param == "pm10":
                pm10 = value
            elif param and value is not None:
                other[param] = value
 
        rows.append({
            "station_id": str(station_id),
            "location": {"lat": station_lat, "lng": station_lon},
            "timestamp": latest_ts or datetime.now(timezone.utc).isoformat(),
            "pm25": pm25,
            "pm10": pm10,
            "aqi_value": pm25_to_aqi(pm25),
            "source": "OpenAQ",
            "region": region_name,
            "station_name": station_name,
            "other_pollutants": json.dumps(other) if other else None,
        })
    return rows
 
def fetch_weather(region_name, lat, lon):
    """Pull current weather from Open-Meteo (no API key needed)."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
        "timezone": "Asia/Kolkata",
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json().get("current", {})
    return {
        "region": region_name,
        "lat": lat,
        "lon": lon,
        "temperature_c": data.get("temperature_2m"),
        "humidity_pct": data.get("relative_humidity_2m"),
        "wind_speed_kmh": data.get("wind_speed_10m"),
        "wind_direction_deg": data.get("wind_direction_10m"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "Open-Meteo",
    }
 
# ---- MAIN ----
def main():
    ensure_dataset()
    aqi_table = ensure_table(TABLE_ID, AQI_SCHEMA)
    weather_table = ensure_table(WEATHER_TABLE_ID, WEATHER_SCHEMA)
 
    all_aqi_rows = []
    all_weather_rows = []
 
    for region_name, info in REGIONS.items():
        print(f"Fetching {region_name}...")
        try:
            aqi_rows = fetch_openaq(region_name, info["lat"], info["lon"], info["radius_km"])
            print(f"  {len(aqi_rows)} AQI readings")
            all_aqi_rows.extend(aqi_rows)
        except Exception as e:
            print(f"  AQI fetch failed for {region_name}: {e}")
 
        try:
            weather_row = fetch_weather(region_name, info["lat"], info["lon"])
            all_weather_rows.append(weather_row)
            print(f"  weather OK")
        except Exception as e:
            print(f"  Weather fetch failed for {region_name}: {e}")
 
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
    )
 
    if all_aqi_rows:
        job = client.load_table_from_json(all_aqi_rows, aqi_table, job_config=job_config)
        job.result()  # wait for the load job to finish
        if job.errors:
            print("AQI insert errors:", job.errors)
        else:
            print(f"Inserted {len(all_aqi_rows)} AQI rows into {aqi_table}")
    else:
        print("No AQI rows to insert.")
 
    if all_weather_rows:
        job = client.load_table_from_json(all_weather_rows, weather_table, job_config=job_config)
        job.result()
        if job.errors:
            print("Weather insert errors:", job.errors)
        else:
            print(f"Inserted {len(all_weather_rows)} weather rows into {weather_table}")
    else:
        print("No weather rows to insert.")
 
if __name__ == "__main__":
    main()