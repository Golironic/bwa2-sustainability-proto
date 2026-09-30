"""
Pulls current AQI data (OpenAQ) + weather data (Open-Meteo) for 4 target
regions and loads them into BigQuery.

CHANGES IN THIS VERSION (vs. the original)
  1. Stale stations are skipped. OpenAQ returns a station's last-ever
     reading as "latest" even if it died years ago; we now drop anything
     older than MAX_STATION_AGE_HOURS (first via the cheap `datetimeLast`
     field on /locations, then again on the actual reading timestamp).
  2. No more duplicate rows. Before loading, we look up which
     (station_id, timestamp) pairs already exist in BigQuery and skip them.
  3. Delhi-NCR and Punjab are queried from several city centres (OpenAQ
     caps radius at 25 km, so one point can't cover a 100 km region).
     Stations are de-duplicated by id within a run.
  4. /locations is paginated, so Mumbai is no longer cut off at 20 stations.
  5. If a region has NO fresh station data, we fall back to Open-Meteo's
     air-quality API (CAMS model output, NOT station data). Those rows are
     labelled source="Open-Meteo CAMS (modelled)" so nobody mistakes them
     for measurements.
  6. pm25_to_aqi no longer returns None for values that fall between
     breakpoints (e.g. 12.05) and handles negative sensor values.
  7. Weather timestamp is the observation time from the API, not fetch time.
  8. 429 (rate limit) responses are retried with backoff.

NOTE ON SCHEMA: aqi_readings is "wide" (one row per station, pm25/pm10/
aqi_value as columns) to match the team's shared data contract. aqi_value
is a US EPA AQI derived from PM2.5 only - a proxy, not the full EPA method.

Run:
    python fetch_aqi_weather.py

Requires:
    pip install google-cloud-bigquery requests python-dotenv
    .env with OPENAQ_API_KEY=...
    GOOGLE_APPLICATION_CREDENTIALS pointing to firebase-key.json
"""
import os
import json
import time
import requests
from datetime import datetime, timedelta, timezone
from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()

# ---- CONFIG ----
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "firebase-key.json")

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "carc-f5b14")
DATASET_ID = "air_quality"
TABLE_ID = "aqi_readings"
WEATHER_TABLE_ID = "weather_readings"
OPENAQ_API_KEY = os.environ.get("OPENAQ_API_KEY")

MAX_STATION_AGE_HOURS = 24      # drop stations whose latest reading is older than this
OPENAQ_MAX_RADIUS_M = 25000     # OpenAQ hard limit
LOCATIONS_PAGE_SIZE = 100
MAX_PAGES_PER_POINT = 3
MIN_FRESH_STATIONS = 1          # region below this -> add modelled fallback rows
DEDUPE_LOOKBACK_HOURS = 48      # how far back to look for already-loaded rows

# "center" is used for weather. "points" are the OpenAQ search centres
# (name, lat, lon); each is searched within 25 km.
REGIONS = {
    "Delhi-NCR": {
        "center": (28.6139, 77.2090),
        "points": [
            ("Delhi", 28.6139, 77.2090),
            ("Gurugram", 28.4595, 77.0266),
            ("Noida", 28.5355, 77.3910),
            ("Faridabad", 28.4089, 77.3178),
            ("Ghaziabad", 28.6692, 77.4538),
        ],
    },
    "Punjab": {
        "center": (30.9010, 75.8573),
        "points": [
            ("Ludhiana", 30.9010, 75.8573),
            ("Amritsar", 31.6340, 74.8723),
            ("Patiala", 30.3398, 76.3869),
            ("Bathinda", 30.2110, 74.9455),
            ("Jalandhar", 31.3260, 75.5762),
        ],
    },
    "Gandhinagar": {
        "center": (23.2156, 72.6369),
        "points": [("Gandhinagar", 23.2156, 72.6369)],
    },
    "Mumbai": {
        "center": (19.0760, 72.8777),
        "points": [("Mumbai", 19.0760, 72.8777)],
    },
}

# ---- BIGQUERY SETUP ----
_client = None

def get_client():
    """Return a shared BigQuery client, created lazily on first call.

    Keeping construction here (rather than at module level) means that
    importing this module does NOT open a GCP connection, so local testing
    and `--help` flags work without credentials.
    """
    global _client
    if _client is None:
        _client = bigquery.Client(project=PROJECT_ID)
    return _client

dataset_ref = f"{PROJECT_ID}.{DATASET_ID}"


def ensure_dataset():
    try:
        get_client().get_dataset(dataset_ref)
    except Exception:
        ds = bigquery.Dataset(dataset_ref)
        ds.location = "asia-south1"
        get_client().create_dataset(ds)
        print(f"Created dataset {dataset_ref}")


def ensure_table(table_id, schema):
    table_ref = f"{dataset_ref}.{table_id}"
    try:
        get_client().get_table(table_ref)
    except Exception:
        table = bigquery.Table(table_ref, schema=schema)
        get_client().create_table(table)
        print(f"Created table {table_ref}")
    return table_ref


AQI_SCHEMA = [
    bigquery.SchemaField("station_id", "STRING"),
    bigquery.SchemaField("location", "RECORD", fields=[
        bigquery.SchemaField("lat", "FLOAT"),
        bigquery.SchemaField("lng", "FLOAT"),
    ]),
    bigquery.SchemaField("timestamp", "TIMESTAMP"),
    bigquery.SchemaField("pm25", "FLOAT"),
    bigquery.SchemaField("pm10", "FLOAT"),
    bigquery.SchemaField("aqi_value", "FLOAT"),
    bigquery.SchemaField("source", "STRING"),
    bigquery.SchemaField("region", "STRING"),
    bigquery.SchemaField("station_name", "STRING"),
    bigquery.SchemaField("other_pollutants", "STRING"),
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

# ---- HELPERS ----
def parse_ts(ts):
    """Parse an ISO timestamp string (or datetime) into an aware UTC datetime."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))


def get_with_retry(url, retries=4, **kwargs):
    """GET with exponential backoff (1s, 2s, 4s, ...) on HTTP 429, 5xx and
    network errors. Returns the last response if it never succeeds, or
    re-raises the last network error if there was never a response."""
    resp, last_exc = None, None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30, **kwargs)
            last_exc = None
            if resp.status_code != 429 and resp.status_code < 500:
                return resp
        except requests.RequestException as e:
            resp, last_exc = None, e
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    if resp is None and last_exc is not None:
        raise last_exc
    return resp


def existing_keys(table_id, key_fields, hours=DEDUPE_LOOKBACK_HOURS):
    """Keys already in BigQuery (recent rows only), so we don't re-insert them."""
    query = f"""
        SELECT {', '.join(key_fields)}
        FROM `{dataset_ref}.{table_id}`
        WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
    """
    try:
        return {tuple(row[f] for f in key_fields) for row in get_client().query(query).result()}
    except Exception as e:
        print(f"  Warning: could not read existing keys from {table_id} ({e}); skipping dedupe")
        return set()


# ---- AQI CALCULATION ----
# US EPA PM2.5 breakpoints (pre-2024 table). Input is truncated to 0.1 ug/m3
# (as EPA specifies) so there are no gaps between brackets.
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
    """Convert a PM2.5 concentration (ug/m3) to a US EPA AQI value."""
    if pm25 is None or pm25 < 0:  # negative = sensor glitch
        return None
    c = int(pm25 * 10) / 10.0     # truncate to 1 decimal
    for c_lo, c_hi, aqi_lo, aqi_hi in PM25_BREAKPOINTS:
        if c_lo <= c <= c_hi:
            return round(((aqi_hi - aqi_lo) / (c_hi - c_lo)) * (c - c_lo) + aqi_lo, 1)
    return 500.0  # above the top of the scale


# ---- FETCH: OPENAQ ----
def build_station_row(loc, region_name, headers, cutoff):
    """Turn one OpenAQ location into a wide row, or None if stale/no data."""
    # Cheap pre-filter: locations carry datetimeLast, so skip dead stations
    # without spending an API call on /latest.
    last_seen = (loc.get("datetimeLast") or {}).get("utc")
    if last_seen and parse_ts(last_seen) < cutoff:
        return "stale"

    station_id = loc.get("id")
    coords = loc.get("coordinates") or {}

    sensor_meta = {}
    for sensor in loc.get("sensors", []):
        sensor_meta[sensor.get("id")] = (sensor.get("parameter", {}).get("name") or "").lower()

    latest_resp = get_with_retry(
        f"https://api.openaq.org/v3/locations/{station_id}/latest", headers=headers
    )
    if latest_resp is None or latest_resp.status_code != 200:
        return None
    latest_results = latest_resp.json().get("results", [])
    if not latest_results:
        return None

    pm25 = pm10 = None
    other = {}
    latest_ts = None
    station_lat, station_lon = coords.get("latitude"), coords.get("longitude")

    for reading in latest_results:
        ts = (reading.get("datetime") or {}).get("utc")
        if ts and parse_ts(ts) < cutoff:
            continue  # ignore this specific sensor reading if it's stale

        param = sensor_meta.get(reading.get("sensorsId"), "")
        value = reading.get("value")
        if ts and (latest_ts is None or parse_ts(ts) > parse_ts(latest_ts)):
            latest_ts = ts
        rc = reading.get("coordinates")
        if rc:
            station_lat = rc.get("latitude", station_lat)
            station_lon = rc.get("longitude", station_lon)

        if param in ("pm25", "pm2.5"):
            pm25 = value
        elif param == "pm10":
            pm10 = value
        elif param and value is not None:
            other[param] = value

    # Second freshness check on the actual reading time.
    if latest_ts is None or parse_ts(latest_ts) < cutoff:
        return "stale"

    return {
        "station_id": str(station_id),
        "location": {"lat": station_lat, "lng": station_lon},
        "timestamp": latest_ts,
        "pm25": pm25,
        "pm10": pm10,
        "aqi_value": pm25_to_aqi(pm25),
        "source": "OpenAQ",
        "region": region_name,
        "station_name": loc.get("name", "unknown"),
        "other_pollutants": json.dumps(other) if other else None,
    }


def fetch_openaq(region_name, points):
    """Fresh stations across all search points of a region.
    Returns (rows, stale_count)."""
    headers = {"X-API-Key": OPENAQ_API_KEY}
    cutoff = datetime.now(timezone.utc) - timedelta(hours=MAX_STATION_AGE_HOURS)
    seen_ids = set()
    rows, stale = [], 0

    for name, lat, lon in points:
        for page in range(1, MAX_PAGES_PER_POINT + 1):
            resp = get_with_retry(
                "https://api.openaq.org/v3/locations",
                params={
                    "coordinates": f"{lat},{lon}",
                    "radius": OPENAQ_MAX_RADIUS_M,
                    "limit": LOCATIONS_PAGE_SIZE,
                    "page": page,
                },
                headers=headers,
            )
            resp.raise_for_status()
            locations = resp.json().get("results", [])

            for loc in locations:
                loc_id = loc.get("id")
                if loc_id in seen_ids:
                    continue
                seen_ids.add(loc_id)
                result = build_station_row(loc, region_name, headers, cutoff)
                if result == "stale":
                    stale += 1
                elif result:
                    rows.append(result)

            if len(locations) < LOCATIONS_PAGE_SIZE:
                break
    return rows, stale


# ---- FETCH: MODELLED FALLBACK ----
def fetch_openmeteo_aq(region_name, name, lat, lon):
    """Modelled (CAMS) PM2.5/PM10 for a point. Used ONLY when a region has no
    fresh station data. AQI is derived from PM2.5 with the same function as
    real stations so values are comparable."""
    resp = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "pm10,pm2_5",
            "timezone": "UTC",
        },
        timeout=20,
    )
    resp.raise_for_status()
    cur = resp.json().get("current", {})
    t = cur.get("time")  # e.g. "2026-09-20T09:00" (UTC because timezone=UTC)
    ts = f"{t}:00Z" if t and len(t) == 16 else datetime.now(timezone.utc).isoformat()
    pm25 = cur.get("pm2_5")
    return {
        "station_id": f"openmeteo-{lat:.2f},{lon:.2f}",
        "location": {"lat": lat, "lng": lon},
        "timestamp": ts,
        "pm25": pm25,
        "pm10": cur.get("pm10"),
        "aqi_value": pm25_to_aqi(pm25),
        "source": "Open-Meteo CAMS (modelled)",
        "region": region_name,
        "station_name": f"{name} (modelled grid cell)",
        "other_pollutants": None,
    }


# ---- FETCH: WEATHER ----
def fetch_weather(region_name, lat, lon):
    """Current weather from Open-Meteo (no API key)."""
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
            "timezone": "UTC",
        },
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json().get("current", {})
    t = data.get("time")  # observation time, UTC because timezone=UTC
    ts = f"{t}:00Z" if t and len(t) == 16 else datetime.now(timezone.utc).isoformat()
    return {
        "region": region_name,
        "lat": lat,
        "lon": lon,
        "temperature_c": data.get("temperature_2m"),
        "humidity_pct": data.get("relative_humidity_2m"),
        "wind_speed_kmh": data.get("wind_speed_10m"),
        "wind_direction_deg": data.get("wind_direction_10m"),
        "timestamp": ts,
        "source": "Open-Meteo",
    }


# ---- LOAD ----
def load_rows(rows, table_ref, label):
    if not rows:
        print(f"No new {label} rows to insert.")
        return
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
    )
    job = get_client().load_table_from_json(rows, table_ref, job_config=job_config)
    job.result()
    if job.errors:
        print(f"{label} insert errors:", job.errors)
    else:
        print(f"Inserted {len(rows)} {label} rows into {table_ref}")


# ---- MAIN ----
def main():
    if not OPENAQ_API_KEY:
        raise RuntimeError("OPENAQ_API_KEY not set - add it to .env")

    ensure_dataset()
    aqi_table = ensure_table(TABLE_ID, AQI_SCHEMA)
    weather_table = ensure_table(WEATHER_TABLE_ID, WEATHER_SCHEMA)

    seen_aqi = existing_keys(TABLE_ID, ["station_id", "timestamp"])
    seen_weather = existing_keys(WEATHER_TABLE_ID, ["region", "timestamp"])

    new_aqi_rows, new_weather_rows = [], []

    for region_name, info in REGIONS.items():
        print(f"Fetching {region_name}...")

        # --- AQI (real stations) ---
        region_rows = []
        try:
            region_rows, stale = fetch_openaq(region_name, info["points"])
            usable = [r for r in region_rows if r["aqi_value"] is not None]
            print(f"  {len(region_rows)} fresh stations ({len(usable)} with PM2.5), "
                  f"{stale} stale skipped")
        except Exception as e:
            print(f"  AQI fetch failed for {region_name}: {e}")
            usable = []

        # --- AQI fallback (modelled) when no usable station data ---
        if len(usable) < MIN_FRESH_STATIONS:
            print(f"  No fresh station data -> using modelled Open-Meteo fallback")
            for name, lat, lon in info["points"]:
                try:
                    region_rows.append(fetch_openmeteo_aq(region_name, name, lat, lon))
                except Exception as e:
                    print(f"  Fallback failed for {name}: {e}")

        for row in region_rows:
            key = (row["station_id"], parse_ts(row["timestamp"]))
            if key in seen_aqi:
                continue
            seen_aqi.add(key)
            new_aqi_rows.append(row)

        # --- Weather ---
        try:
            lat, lon = info["center"]
            w = fetch_weather(region_name, lat, lon)
            key = (w["region"], parse_ts(w["timestamp"]))
            if key not in seen_weather:
                seen_weather.add(key)
                new_weather_rows.append(w)
            print("  weather OK")
        except Exception as e:
            print(f"  Weather fetch failed for {region_name}: {e}")

    load_rows(new_aqi_rows, aqi_table, "AQI")
    load_rows(new_weather_rows, weather_table, "weather")


if __name__ == "__main__":
    main()