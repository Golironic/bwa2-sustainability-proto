"""
One-off historical backfill so Person 2's Vertex AI forecast has training data.

What it loads (per region, using each region's centre point):
  weather_readings : hourly temperature / humidity / wind  (Open-Meteo)
  aqi_readings     : hourly PM2.5 / PM10 -> AQI            (Open-Meteo air-quality
                     API = CAMS MODEL output, NOT station measurements; every
                     row is labelled source="Open-Meteo CAMS (modelled)")
  fire_hotspots    : NASA FIRMS detections, in 5-day chunks

Why modelled AQI: OpenAQ history needs one call per sensor per window and most
of your stations are dead, so it can't give a continuous series. The modelled
series gives Person 2 one gap-free hourly series per zone (= region).
CAVEAT for the team: the forecast is trained on modelled history but live
station readings will differ (CAMS tends to under-read Delhi), so treat
forecast numbers as indicative, and say so in the pitch.

Safe to re-run: rows already in BigQuery are skipped.

Usage:
    python backfill_history.py                 # 60 days, everything
    python backfill_history.py --days 90       # max ~90 for weather/AQI
    python backfill_history.py --skip-fires    # weather + AQI only
    python backfill_history.py --fire-source VIIRS_NOAA20_SP   # archive fire product

Requires the same .env / credentials as the other scripts (FIRMS_MAP_KEY for fires).
"""
import argparse
from datetime import datetime, timedelta, timezone

import fetch_aqi_weather as A   # reuses clients, schemas, pm25_to_aqi, REGIONS
import fetch_firms as F

MAX_PAST_DAYS = 90              # Open-Meteo allows up to 92
FIRE_CHUNK_DAYS = 5             # FIRMS area API rejects day ranges above 5 (HTTP 400)
BACKFILL_SOURCE_AQI = "Open-Meteo CAMS (modelled)"
BACKFILL_SOURCE_WX = "Open-Meteo hourly (backfill)"


def hourly_series(payload, now):
    """Yield (aware_datetime, index, hourly_dict) for each past-or-present hour."""
    h = payload.get("hourly") or {}
    for i, t in enumerate(h.get("time", [])):
        ts = A.parse_ts(f"{t}:00Z" if len(t) == 16 else t)
        if ts <= now:                      # forecast_days=1 also returns later hours today
            yield ts, i, h


def backfill_weather(days, now):
    seen = A.existing_keys(A.WEATHER_TABLE_ID, ["region", "timestamp"], hours=days * 24 + 48)
    table = f"{A.dataset_ref}.{A.WEATHER_TABLE_ID}"
    failed = []
    for region, info in A.REGIONS.items():
        lat, lon = info["center"]
        try:
            resp = A.get_with_retry(
                "https://api.open-meteo.com/v1/forecast",
                retries=6,
                params={
                    "latitude": lat, "longitude": lon, "timezone": "UTC",
                    "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
                    "past_days": days, "forecast_days": 1,
                },
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:
            print(f"  weather {region}: FAILED ({e}) - re-run the script to retry")
            failed.append(f"weather:{region}")
            continue

        rows = []
        for ts, i, h in hourly_series(payload, now):
            if (region, ts) in seen:
                continue
            row = {
                "region": region, "lat": lat, "lon": lon,
                "temperature_c": h["temperature_2m"][i],
                "humidity_pct": h["relative_humidity_2m"][i],
                "wind_speed_kmh": h["wind_speed_10m"][i],
                "wind_direction_deg": h["wind_direction_10m"][i],
                "timestamp": ts.isoformat(),
                "source": BACKFILL_SOURCE_WX,
            }
            if all(row[k] is None for k in ("temperature_c", "humidity_pct", "wind_speed_kmh")):
                continue
            seen.add((region, ts))
            rows.append(row)
        print(f"  weather {region}: {len(rows)} new hourly rows")
        A.load_rows(rows, table, f"weather backfill ({region})")  # saved immediately
    return failed


def backfill_aqi(days, now):
    seen = A.existing_keys(A.TABLE_ID, ["station_id", "timestamp"], hours=days * 24 + 48)
    table = f"{A.dataset_ref}.{A.TABLE_ID}"
    failed = []
    for region, info in A.REGIONS.items():
        lat, lon = info["center"]
        try:
            resp = A.get_with_retry(
                "https://air-quality-api.open-meteo.com/v1/air-quality",
                retries=6,
                params={
                    "latitude": lat, "longitude": lon, "timezone": "UTC",
                    "hourly": "pm10,pm2_5",
                    "past_days": days, "forecast_days": 1,
                },
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:
            print(f"  aqi {region}: FAILED ({e}) - re-run the script to retry")
            failed.append(f"aqi:{region}")
            continue

        # Same station_id format as the live fallback in fetch_aqi_weather.py,
        # so history and live rows form ONE continuous series per zone.
        station_id = f"openmeteo-{lat:.2f},{lon:.2f}"
        rows = []
        for ts, i, h in hourly_series(payload, now):
            pm25, pm10 = h["pm2_5"][i], h["pm10"][i]
            if pm25 is None and pm10 is None:
                continue
            if (station_id, ts) in seen:
                continue
            seen.add((station_id, ts))
            rows.append({
                "station_id": station_id,
                "location": {"lat": lat, "lng": lon},
                "timestamp": ts.isoformat(),
                "pm25": pm25, "pm10": pm10,
                "aqi_value": A.pm25_to_aqi(pm25),
                "source": BACKFILL_SOURCE_AQI,
                "region": region,
                "station_name": f"{region} centre (modelled grid cell)",
                "other_pollutants": None,
            })
        print(f"  aqi {region}: {len(rows)} new hourly rows")
        A.load_rows(rows, table, f"AQI backfill ({region})")  # saved immediately
    return failed


def backfill_fires(days, now, source):
    firms_table = F.ensure_table(F.FIRMS_TABLE_ID, F.FIRMS_SCHEMA)
    seen = F.existing_fire_keys(hours=days * 24 + 48)
    rows = []
    failed = []
    start = (now - timedelta(days=days)).date()
    end = now.date()
    for region, info in F.REGIONS.items():
        d, region_new = start, 0
        while d <= end:
            chunk = min(FIRE_CHUNK_DAYS, (end - d).days + 1)
            try:
                fetched = F.fetch_firms(region, info["bbox"], day_range=chunk,
                                        start_date=d.isoformat(), source=source)
            except Exception as e:
                print(f"  fires {region} from {d}: FAILED ({e})")
                failed.append(f"fires:{region}@{d}")
                fetched = []
            for row in fetched:
                key = F.fire_key(row["location"]["lat"], row["location"]["lng"], row["timestamp"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
                region_new += 1
            d += timedelta(days=chunk)
        print(f"  fires {region}: {region_new} new detections")

    if not rows:
        print("No new fire rows to insert.")
        return failed
    job_config = F.bigquery.LoadJobConfig(
        write_disposition=F.bigquery.WriteDisposition.WRITE_APPEND,
        source_format=F.bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
    )
    job = F.client.load_table_from_json(rows, firms_table, job_config=job_config)
    job.result()
    print(f"Inserted {len(rows)} fire rows into {firms_table}" if not job.errors
          else f"Fire insert errors: {job.errors}")
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--skip-weather", action="store_true")
    ap.add_argument("--skip-aqi", action="store_true")
    ap.add_argument("--skip-fires", action="store_true")
    ap.add_argument("--fire-source", default=F.FIRMS_SOURCE,
                    help="FIRMS product; use VIIRS_NOAA20_SP for older archive data")
    args = ap.parse_args()

    days = max(1, min(args.days, MAX_PAST_DAYS))
    now = datetime.now(timezone.utc)
    print(f"Backfilling last {days} days...")

    A.ensure_dataset()
    A.ensure_table(A.TABLE_ID, A.AQI_SCHEMA)
    A.ensure_table(A.WEATHER_TABLE_ID, A.WEATHER_SCHEMA)

    failed = []
    if not args.skip_weather:
        print("Weather:")
        failed += backfill_weather(days, now)
    if not args.skip_aqi:
        print("AQI (modelled):")
        failed += backfill_aqi(days, now)
    if not args.skip_fires:
        print("Fires:")
        failed += backfill_fires(days, now, args.fire_source)

    if failed:
        print(f"\nFinished with {len(failed)} failed request(s): {', '.join(failed[:8])}"
              + (" ..." if len(failed) > 8 else ""))
        print("Everything else was saved. Re-run the same command to fill the gaps "
              "(already-loaded rows are skipped).")
    else:
        print("Done.")


if __name__ == "__main__":
    main()