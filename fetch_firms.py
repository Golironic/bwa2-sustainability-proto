"""
Pulls active fire detections (NASA FIRMS) for 4 target regions and loads
them into BigQuery (table: fire_hotspots). Used as a proxy for agricultural
/ stubble burning.

CHANGES IN THIS VERSION (vs. the original)
  1. No more duplicate rows. FIRMS returns the last DAY_RANGE days on every
     call, so re-running re-inserted the same detections. We now skip
     detections already in BigQuery (matched on lat, lng, timestamp).
  2. The FIRMS key can no longer leak into logs. requests puts the URL
     (which contains the key) in exception messages; we catch those and
     re-raise a message without the URL.
  3. load_dotenv() is called, so FIRMS_MAP_KEY in .env is picked up.

NOTE ON SCHEMA: table/field names follow the team's shared contract
(location, timestamp, confidence, brightness, source). We use NASA FIRMS
instead of Earth Engine (same VIIRS/MODIS data, no EE account needed).
Extra fields (region, frp, satellite, daynight, fetched_at) are additions.

Run:
    python fetch_firms.py

Requires:
    pip install google-cloud-bigquery requests python-dotenv
    .env with FIRMS_MAP_KEY=...
"""

import os
import csv
import io
import requests
from datetime import datetime, timezone
from google.cloud import bigquery
from dotenv import load_dotenv
from google.api_core.exceptions import NotFound

load_dotenv()

# ---- CONFIG ----
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "firebase-key.json")

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "carc-f5b14")
DATASET_ID = "air_quality"
FIRMS_TABLE_ID = "fire_hotspots"

FIRMS_MAP_KEY = os.environ.get("FIRMS_MAP_KEY")

FIRMS_SOURCE = "VIIRS_NOAA20_NRT"
DAY_RANGE = 5  # FIRMS allows 1-10

# (west, south, east, north)
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
    except NotFound:
        table = bigquery.Table(table_ref, schema=schema)
        client.create_table(table)
        print(f"Created table {table_ref}")
    return table_ref


FIRMS_SCHEMA = [
    bigquery.SchemaField("location", "RECORD", fields=[
        bigquery.SchemaField("lat", "FLOAT"),
        bigquery.SchemaField("lng", "FLOAT"),
    ]),
    bigquery.SchemaField("timestamp", "TIMESTAMP"),
    bigquery.SchemaField("confidence", "STRING"),
    bigquery.SchemaField("brightness", "FLOAT"),
    bigquery.SchemaField("source", "STRING"),
    bigquery.SchemaField("region", "STRING"),
    bigquery.SchemaField("frp", "FLOAT"),
    bigquery.SchemaField("satellite", "STRING"),
    bigquery.SchemaField("daynight", "STRING"),
    bigquery.SchemaField("fetched_at", "TIMESTAMP"),
]


# ---- HELPERS ----
def parse_ts(ts):
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))


def fire_key(lat, lng, ts):
    """Identity of a detection. Rounded so float round-trips through BigQuery match."""
    return (round(lat, 5), round(lng, 5), parse_ts(ts))


def existing_fire_keys(hours=None):
    """Detections already loaded within the lookback window (default: FIRMS window)."""
    hours = hours or (DAY_RANGE + 1) * 24
    query = f"""
        SELECT location.lat AS lat, location.lng AS lng, timestamp
        FROM `{dataset_ref}.{FIRMS_TABLE_ID}`
        WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
    """
    try:
        return {fire_key(r["lat"], r["lng"], r["timestamp"])
                for r in client.query(query).result()
                if r["lat"] is not None and r["lng"] is not None}
    except Exception as e:
        print(f"  Warning: could not read existing fire rows ({e}); skipping dedupe")
        return set()


# ---- FETCH ----
def fetch_firms(region_name, bbox, day_range=DAY_RANGE, start_date=None, source=None):
    """Pull active fire detections for a bounding box from NASA FIRMS.

    start_date ("YYYY-MM-DD") makes FIRMS return day_range days starting at
    that date (used by backfill_history.py). source overrides FIRMS_SOURCE."""
    source = source or FIRMS_SOURCE
    if not FIRMS_MAP_KEY:
        raise RuntimeError("FIRMS_MAP_KEY not set - add it to .env")

    west, south, east, north = bbox
    area = f"{west},{south},{east},{north}"
    url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
           f"{FIRMS_MAP_KEY}/{source}/{area}/{day_range}")
    if start_date:
        url += f"/{start_date}"

    # Never let the URL (it contains the key) reach an error message.
    try:
        resp = requests.get(url, timeout=30)
    except requests.RequestException as e:
        raise RuntimeError(f"FIRMS request failed for {region_name}: {type(e).__name__}") from None
    if resp.status_code != 200:
        detail = resp.text.strip()[:120].replace(FIRMS_MAP_KEY, "***")
        raise RuntimeError(f"FIRMS HTTP {resp.status_code} for {region_name}: {detail}")

    text = resp.text.strip()
    if text.lower().startswith("invalid") or "error" in text[:200].lower():
        raise RuntimeError(f"FIRMS API error for {region_name}: {text[:200]}")

    reader = csv.DictReader(io.StringIO(text))
    rows = []
    fetched_at = datetime.now(timezone.utc).isoformat()

    for r in reader:
        lat = float(r["latitude"]) if r.get("latitude") else None
        lon = float(r["longitude"]) if r.get("longitude") else None
        if lat is None or lon is None:
            continue
        brightness = float(r["bright_ti4"]) if r.get("bright_ti4") else (
            float(r["brightness"]) if r.get("brightness") else None
        )

        acq_date = r.get("acq_date", "")
        acq_time_raw = r.get("acq_time", "")
        if acq_date and acq_time_raw:
            t = acq_time_raw.zfill(4)
            detection_ts = f"{acq_date}T{t[:2]}:{t[2:]}:00Z"
        else:
            detection_ts = fetched_at

        rows.append({
            "location": {"lat": lat, "lng": lon},
            "timestamp": detection_ts,
            "confidence": r.get("confidence", ""),
            "brightness": brightness,
            "source": source,
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
    seen = existing_fire_keys()
    print(f"{len(seen)} detections already in BigQuery (last {DAY_RANGE + 1} days)")

    new_rows = []
    for region_name, info in REGIONS.items():
        print(f"Fetching {region_name}...")
        try:
            rows = fetch_firms(region_name, info["bbox"])
        except Exception as e:
            print(f"  FIRMS fetch failed for {region_name}: {e}")
            continue

        fresh = 0
        for row in rows:
            key = fire_key(row["location"]["lat"], row["location"]["lng"], row["timestamp"])
            if key in seen:
                continue
            seen.add(key)
            new_rows.append(row)
            fresh += 1
        print(f"  {len(rows)} detections from FIRMS, {fresh} new")

    if not new_rows:
        print("No new fire rows to insert.")
        return

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
    )
    job = client.load_table_from_json(new_rows, firms_table, job_config=job_config)
    job.result()
    if job.errors:
        print("FIRMS insert errors:", job.errors)
    else:
        print(f"Inserted {len(new_rows)} fire rows into {firms_table}")


if __name__ == "__main__":
    main()