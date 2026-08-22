"""
Pulls active fire detections (NASA FIRMS) for 4 target regions and loads
them into BigQuery. Used as a proxy for agricultural/stubble burning and
other localized fire-driven pollution events.

NOTE ON SCHEMA: table name and field names (location, timestamp,
confidence, brightness, source) are aligned to the team's shared data
contract, which specced Google Earth Engine as the fire data source into
a `fire_hotspots` table. We're using NASA FIRMS directly instead (free,
same underlying VIIRS/MODIS satellite data, no Earth Engine account
needed) - flagged to the team, not a silent swap. Extra fields (region,
frp, satellite, daynight, fetched_at) are kept beyond the contract for
richer hotspot scoring but don't break anything Person 3's dashboard
expects.

Regions: Delhi-NCR, Punjab, Gandhinagar, Mumbai

Run:
    python fetch_firms.py

Requires:
    pip install google-cloud-bigquery requests
    A free FIRMS MAP_KEY - register at:
    https://firms.modaps.eosdis.nasa.gov/api/map_key/
    (takes seconds, just needs an email - keep this key secret, don't
    hardcode/commit it, same as the OpenAQ key)

    A GCP project with BigQuery API enabled, and
    GOOGLE_APPLICATION_CREDENTIALS env var pointing to firebase-key.json
"""

import os
import csv
import io
import requests
from datetime import datetime, timezone
from google.cloud import bigquery

# ---- CONFIG ----
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "firebase-key.json")

PROJECT_ID = "carc-f5b14"
DATASET_ID = "air_quality"
FIRMS_TABLE_ID = "fire_hotspots"  # matches shared team schema

# Get your own free key: https://firms.modaps.eosdis.nasa.gov/api/map_key/
FIRMS_MAP_KEY = os.environ.get("FIRMS_MAP_KEY", "")

# VIIRS NOAA-20 NRT: ~375m resolution, near-real-time, good default.
# Alternatives: VIIRS_SNPP_NRT, MODIS_NRT (coarser, 1km)
FIRMS_SOURCE = "VIIRS_NOAA20_NRT"
DAY_RANGE = 5  # 1 = most recent day only (FIRMS allows up to 10)

# Bounding boxes as (west, south, east, north) - wider than the AQI/weather
# point+radius since FIRMS area queries need a box, not a center point.
REGIONS = {
    "Delhi-NCR":   {"bbox": (76.5, 28.0, 77.9, 29.2)},
    "Punjab":      {"bbox": (73.9, 29.5, 76.9, 32.5)},
    "Gandhinagar": {"bbox": (72.0, 22.7, 73.2, 23.7)},
    "Mumbai":      {"bbox": (72.6, 18.7, 73.2, 19.4)},
}

# ---- BIGQUERY SETUP ----
client = bigquery.Client(project=PROJECT_ID)
dataset_ref = f"{PROJECT_ID}.{DATASET_ID}"

def ensure_table(table_id, schema):
    table_ref = f"{dataset_ref}.{table_id}"
    try:
        client.get_table(table_ref)
    except Exception:
        table = bigquery.Table(table_ref, schema=schema)
        client.create_table(table)
        print(f"Created table {table_ref}")
    return table_ref

FIRMS_SCHEMA = [
    # --- fields from the shared team schema (fire_hotspots) ---
    bigquery.SchemaField("location", "RECORD", fields=[
        bigquery.SchemaField("lat", "FLOAT"),
        bigquery.SchemaField("lng", "FLOAT"),
    ]),
    bigquery.SchemaField("timestamp", "TIMESTAMP"),   # actual satellite detection time (acq_date + acq_time)
    bigquery.SchemaField("confidence", "STRING"),      # low/nominal/high (VIIRS) or 0-100 (MODIS)
    bigquery.SchemaField("brightness", "FLOAT"),       # fire radiative brightness (K)
    bigquery.SchemaField("source", "STRING"),          # e.g. "VIIRS_NOAA20_NRT" (via NASA FIRMS, not Earth Engine)
    # --- extra fields kept for richer scoring/debugging, not required by the contract ---
    bigquery.SchemaField("region", "STRING"),
    bigquery.SchemaField("frp", "FLOAT"),              # fire radiative power (MW) - proxy for intensity
    bigquery.SchemaField("satellite", "STRING"),
    bigquery.SchemaField("daynight", "STRING"),
    bigquery.SchemaField("fetched_at", "TIMESTAMP"),   # when our script pulled this row
]

# ---- FETCH FUNCTION ----
def fetch_firms(region_name, bbox, day_range=DAY_RANGE):
    """Pull active fire detections for a bounding box from NASA FIRMS."""
    if not FIRMS_MAP_KEY:
        raise RuntimeError("FIRMS_MAP_KEY not set - export it or set it in the script")

    west, south, east, north = bbox
    area = f"{west},{south},{east},{north}"
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_MAP_KEY}/{FIRMS_SOURCE}/{area}/{day_range}"

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    # FIRMS returns an error message as plain text (not CSV) if the key or
    # params are bad - guard against silently parsing that as "0 rows".
    text = resp.text.strip()
    if text.lower().startswith("invalid") or "error" in text[:200].lower():
        raise RuntimeError(f"FIRMS API error for {region_name}: {text[:200]}")

    reader = csv.DictReader(io.StringIO(text))
    rows = []
    fetched_at = datetime.now(timezone.utc).isoformat()

    for r in reader:
        lat = float(r["latitude"]) if r.get("latitude") else None
        lon = float(r["longitude"]) if r.get("longitude") else None
        brightness = float(r["bright_ti4"]) if r.get("bright_ti4") else (
            float(r["brightness"]) if r.get("brightness") else None
        )

        # Build the actual satellite detection timestamp from acq_date + acq_time
        # (FIRMS gives these as separate fields, e.g. "2026-08-22" + "0512" = 05:12 UTC)
        acq_date = r.get("acq_date", "")
        acq_time_raw = r.get("acq_time", "")
        detection_ts = None
        if acq_date and acq_time_raw:
            try:
                hh = acq_time_raw.zfill(4)[:2]
                mm = acq_time_raw.zfill(4)[2:]
                detection_ts = f"{acq_date}T{hh}:{mm}:00Z"
            except Exception:
                detection_ts = fetched_at
        else:
            detection_ts = fetched_at

        rows.append({
            "location": {"lat": lat, "lng": lon},
            "timestamp": detection_ts,
            "confidence": r.get("confidence", ""),
            "brightness": brightness,
            "source": FIRMS_SOURCE,
            "region": region_name,
            "frp": float(r["frp"]) if r.get("frp") else None,
            "satellite": r.get("satellite", ""),
            "daynight": r.get("daynight", ""),
            "fetched_at": fetched_at,
        })
    return rows

# ---- MAIN ----
def main():
    firms_table = ensure_table(FIRMS_TABLE_ID, FIRMS_SCHEMA)

    all_rows = []
    for region_name, info in REGIONS.items():
        print(f"Fetching {region_name}...")
        try:
            rows = fetch_firms(region_name, info["bbox"])
            print(f"  {len(rows)} fire detections")
            all_rows.extend(rows)
        except Exception as e:
            print(f"  FIRMS fetch failed for {region_name}: {e}")

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
    )

    if all_rows:
        job = client.load_table_from_json(all_rows, firms_table, job_config=job_config)
        job.result()
        if job.errors:
            print("FIRMS insert errors:", job.errors)
        else:
            print(f"Inserted {len(all_rows)} fire rows into {firms_table}")
    else:
        print("No fire rows to insert.")

if __name__ == "__main__":
    main()