"""
Wires Person 2's real XGBoost model (ml_core/) into the live pipeline: reads
AQI + weather history from BigQuery, maps our 4 zones onto the model's
trained region categories, recursively forecasts 24 hours ahead using their
single-step AQICaller, and writes the result to the `forecast` table.

IMPORTANT — read before trusting the output:

  1. REGION MAPPING. Our 4 zone names don't match the model's 13 trained
     category strings 1:1. "Punjab", "Gandhinagar" and "Mumbai" match
     exactly (same centre coordinates the model saw in training).
     "Delhi-NCR" has no exact match; we map it to "Delhi" below (same
     centre point) - a stand-in for the whole NCR region, not a perfect one.

  2. AQI DEFINITION MISMATCH. Our aqi_value in BigQuery is a PM2.5-only
     proxy (see fetch_aqi_weather.py). The model was trained on Open-Meteo's
     us_aqi, which blends multiple pollutants - same column name, different
     numbers for the same moment. This script currently feeds the model our
     PM2.5-only number as-is (the "ship as-is, flag it" option), so
     predictions will be somewhat biased, more so on days another pollutant
     (e.g. NO2) is the real driver. See the message this script shipped
     with for two other fix options if this matters for your demo.

  3. CONTINUITY. The model needs an unbroken hourly sequence to compute its
     24-hour lag and rolling features correctly - a gap in the data
     silently shifts what "24 hours ago" means rather than raising an
     error. This script resamples to a strict hourly grid and REFUSES to
     forecast a zone with a gap bigger than MAX_GAP_HOURS, rather than
     silently guessing.

  4. WEATHER HOLD. Weather is held constant at its last observed value for
     the whole 24-hour horizon - there's no live weather forecast wired in.

  5. Import ml_core.model_caller directly (as this script does), not
     `from ai_pipeline import predict_next_hour_aqi`. Importing ai_pipeline
     also initialises the Gemini client and Firebase admin SDK as a side
     effect of the import itself, which you don't need just to forecast,
     and which will error ("app already exists") if it's ever imported
     twice in the same process.

Run:
    python run_forecast.py
    python run_forecast.py --dry-run

Requires (in addition to the other scripts' requirements):
    pip install xgboost
    ml_core/aqi_xgboost_model.json  (the trained model file - not in this
        repo yet; drop it in ml_core/ as Person 2 instructed)
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "firebase-key.json")

# Import the inference class directly - see caveat 5 above.
from ml_core.model_caller import AQICaller

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "carc-f5b14")
DATASET_ID = "air_quality"
FORECAST_TABLE = f"{PROJECT_ID}.{DATASET_ID}.forecast"

HORIZON_HOURS = 24
HISTORY_HOURS = 72       # pulled with margin so resampling/continuity checks have room
MAX_GAP_HOURS = 50        # refuse to forecast a zone with any gap in its history bigger than this
MAX_STALENESS_HOURS = 3

# Our zone -> the model's trained region category. See caveat 1 above.
ZONE_TO_MODEL_REGION = {
    "Delhi-NCR": "Delhi",       # not an exact match - same centre coordinates only
    "Punjab": "Punjab",
    "Gandhinagar": "Gandhinagar",
    "Mumbai": "Mumbai",
}

FORECAST_SCHEMA = [
    bigquery.SchemaField("zone_id", "STRING"),
    bigquery.SchemaField("forecast_timestamp", "TIMESTAMP"),
    bigquery.SchemaField("predicted_aqi", "FLOAT"),
    bigquery.SchemaField("confidence_interval", "RECORD", fields=[
        bigquery.SchemaField("lower", "FLOAT"),
        bigquery.SchemaField("upper", "FLOAT"),
    ]),
    bigquery.SchemaField("generated_at", "TIMESTAMP"),
]

MODEL_INPUT_COLS = ["timestamp", "region", "aqi_value",
                     "temperature_aqi", "humidity_aqi", "wind_speed_aqi", "wind_direction_aqi"]


def load_zone_history(client, zone, hours):
    """AQI + weather, hour-joined, for one zone. Raw rows - not yet checked for gaps."""
    query = f"""
        SELECT a.ts AS timestamp, a.aqi_value,
               w.temperature_c, w.humidity_pct, w.wind_speed_kmh, w.wind_direction_deg
        FROM (
          SELECT TIMESTAMP_TRUNC(timestamp, HOUR) AS ts, AVG(aqi_value) AS aqi_value
          FROM `{PROJECT_ID}.{DATASET_ID}.aqi_readings`
          WHERE region = '{zone}' AND aqi_value IS NOT NULL
            AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
          GROUP BY ts
        ) a
        JOIN (
          SELECT TIMESTAMP_TRUNC(timestamp, HOUR) AS ts, AVG(temperature_c) AS temperature_c,
                 AVG(humidity_pct) AS humidity_pct, AVG(wind_speed_kmh) AS wind_speed_kmh,
                 AVG(wind_direction_deg) AS wind_direction_deg
          FROM `{PROJECT_ID}.{DATASET_ID}.weather_readings`
          WHERE region = '{zone}'
            AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
          GROUP BY ts
        ) w USING (ts)
        ORDER BY ts
    """
    return client.query(query).to_dataframe()


def prepare_hourly_series(df, max_gap_hours=MAX_GAP_HOURS):
    """Resample to a strict hourly grid and refuse silently-wrong lag features.

    Returns (clean_df, None) on success, or (None, reason) if the data can't
    be trusted. A gap bigger than max_gap_hours is a hard failure, not a fill -
    AQICaller's .shift()/.rolling() work by ROW POSITION, so a silently
    filled long gap would make "24 hours ago" actually mean something else.
    """
    if df is None or df.empty or len(df) < 2:
        return None, "no data"
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")

    deltas_h = df["timestamp"].diff().dt.total_seconds().dropna() / 3600.0
    if (deltas_h > max_gap_hours).any():
        return None, f"gap of {deltas_h.max():.1f}h in the history (max allowed {max_gap_hours}h)"

    full_idx = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    hourly = df.set_index("timestamp").reindex(full_idx)
    hourly = hourly.ffill(limit=max_gap_hours)  # smooths only the small, already-approved gaps
    if hourly.isna().any().any():
        return None, "gaps remained after fill"
    hourly = hourly.reset_index().rename(columns={"index": "timestamp"})
    return hourly, None


def forecast_zone(caller, zone, model_region, raw_df, now):
    """24 recursive single-step predictions for one zone, or None to skip."""
    hourly, err = prepare_hourly_series(raw_df)
    if err:
        print(f"  {zone}: SKIPPED ({err})")
        return None

    latest_ts = hourly["timestamp"].max()
    if now - latest_ts > timedelta(hours=MAX_STALENESS_HOURS):
        print(f"  {zone}: SKIPPED (latest reading is {latest_ts}, too old)")
        return None
    if len(hourly) < 25:
        print(f"  {zone}: SKIPPED (need 25h of continuous history, have {len(hourly)})")
        return None

    working = hourly.rename(columns={
        "temperature_c": "temperature_aqi", "humidity_pct": "humidity_aqi",
        "wind_speed_kmh": "wind_speed_aqi", "wind_direction_deg": "wind_direction_aqi",
    })[["timestamp", "aqi_value", "temperature_aqi", "humidity_aqi", "wind_speed_aqi", "wind_direction_aqi"]].copy()
    working["region"] = model_region
    working = working[MODEL_INPUT_COLS]

    # Weather held constant for the whole horizon - see caveat 4 above.
    last_weather = working.iloc[-1][["temperature_aqi", "humidity_aqi", "wind_speed_aqi", "wind_direction_aqi"]].to_dict()

    rows = []
    for h in range(1, HORIZON_HOURS + 1):
        try:
            pred = caller.predict(working)
        except Exception as e:
            print(f"  {zone}: SKIPPED mid-forecast at h+{h} ({e})")
            return None
        pred = min(max(float(pred), 0.0), 500.0)
        ts = latest_ts + timedelta(hours=h)
        new_row = {"timestamp": ts, "region": model_region, "aqi_value": pred, **last_weather}
        working = pd.concat([working, pd.DataFrame([new_row])[MODEL_INPUT_COLS]], ignore_index=True)
        rows.append({
            "zone_id": zone,
            "forecast_timestamp": ts.isoformat(),
            "predicted_aqi": round(pred, 1),
            # No confidence interval yet - that needs a real residual-std number
            # from Person 2's model_tester.py (its printed RMSE), not a guess.
            "confidence_interval": {"lower": None, "upper": None},
            "generated_at": now.isoformat(),
        })
    return rows


def run(dry_run=False):
    client = bigquery.Client(project=PROJECT_ID)
    caller = AQICaller()  # loads ml_core/aqi_xgboost_model.json + region_categories.json once

    now = datetime.now(timezone.utc)
    all_rows = []
    for zone, model_region in ZONE_TO_MODEL_REGION.items():
        raw = load_zone_history(client, zone, HISTORY_HOURS)
        rows = forecast_zone(caller, zone, model_region, raw, now)
        if rows:
            print(f"  {zone}: {len(rows)} hourly forecasts, "
                  f"AQI {rows[0]['predicted_aqi']} -> {rows[-1]['predicted_aqi']} "
                  f"(as '{model_region}')")
            all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No zone produced a forecast; existing forecast table left untouched.")

    if dry_run:
        print(json.dumps(all_rows[:2], indent=2))
        print(f"(dry run: {len(all_rows)} rows NOT written)")
        return 0

    job = client.load_table_from_json(
        all_rows, FORECAST_TABLE,
        job_config=bigquery.LoadJobConfig(
            schema=FORECAST_SCHEMA,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        ),
    )
    job.result()
    print(f"Wrote {len(all_rows)} rows to {FORECAST_TABLE}")
    return len(all_rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    try:
        run(dry_run=args.dry_run)
    except RuntimeError as e:
        print(e)
        sys.exit(1)
