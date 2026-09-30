"""
One-off backfill so run_forecast.py has enough continuous hourly history.

Pulls the last PAST_DAYS days of HOURLY modelled air quality (Open-Meteo CAMS)
and weather (Open-Meteo) for each region centre and appends them to the same
BigQuery tables fetch_aqi_weather.py writes to. Rows already present are
skipped, so it is safe to run more than once.

Rows use the SAME station_id / source labels as the modelled fallback in
fetch_aqi_weather.py, so:
  - dedupe against fallback rows works (same key), and
  - hotspot_scoring.py treats them as modelled data, never as real stations.

Run:
    python backfill_aqi_weather.py

Needs fetch_aqi_weather.py in the same folder (its helpers are reused).
"""
from datetime import datetime, timezone

import requests

from fetch_aqi_weather import (
    REGIONS, TABLE_ID, WEATHER_TABLE_ID, AQI_SCHEMA, WEATHER_SCHEMA,
    ensure_dataset, ensure_table, existing_keys, get_with_retry,
    load_rows, parse_ts, pm25_to_aqi,
)

PAST_DAYS = 3  # 72h; Open-Meteo allows more, but the forecast only needs ~25h


def hourly_rows(url, params):
    """Fetch an Open-Meteo hourly block and return it as a list of dicts,
    dropping any hours that are still in the future."""
    resp = get_with_retry(url, params=params)
    resp.raise_for_status()
    h = resp.json().get("hourly", {})
    times = h.get("time", [])
    now = datetime.now(timezone.utc)
    out = []
    for i, t in enumerate(times):
        ts = f"{t}:00Z"
        if parse_ts(ts) > now:
            continue
        row = {k: (v[i] if i < len(v) else None) for k, v in h.items() if k != "time"}
        row["ts"] = ts
        out.append(row)
    return out


def main():
    ensure_dataset()
    aqi_table = ensure_table(TABLE_ID, AQI_SCHEMA)
    weather_table = ensure_table(WEATHER_TABLE_ID, WEATHER_SCHEMA)

    lookback = (PAST_DAYS + 1) * 24
    seen_aqi = existing_keys(TABLE_ID, ["station_id", "timestamp"], hours=lookback)
    seen_weather = existing_keys(WEATHER_TABLE_ID, ["region", "timestamp"], hours=lookback)

    new_aqi, new_weather = [], []

    for region, info in REGIONS.items():
        lat, lon = info["center"]
        print(f"Backfilling {region}...")

        # --- modelled air quality ---
        try:
            rows = hourly_rows(
                "https://air-quality-api.open-meteo.com/v1/air-quality",
                {"latitude": lat, "longitude": lon, "hourly": "pm10,pm2_5",
                 "past_days": PAST_DAYS, "forecast_days": 1, "timezone": "UTC"},
            )
            fresh = 0
            for r in rows:
                pm25 = r.get("pm2_5")
                if pm25 is None:
                    continue
                key = (f"openmeteo-{lat:.2f},{lon:.2f}", parse_ts(r["ts"]))
                if key in seen_aqi:
                    continue
                seen_aqi.add(key)
                new_aqi.append({
                    "station_id": key[0],
                    "location": {"lat": lat, "lng": lon},
                    "timestamp": r["ts"],
                    "pm25": pm25,
                    "pm10": r.get("pm10"),
                    "aqi_value": pm25_to_aqi(pm25),
                    "source": "Open-Meteo CAMS (modelled)",
                    "region": region,
                    "station_name": f"{region} (modelled grid cell, backfill)",
                    "other_pollutants": None,
                })
                fresh += 1
            print(f"  AQI: {len(rows)} hours from API, {fresh} new")
        except requests.RequestException as e:
            print(f"  AQI backfill failed for {region}: {type(e).__name__}")

        # --- weather ---
        try:
            rows = hourly_rows(
                "https://api.open-meteo.com/v1/forecast",
                {"latitude": lat, "longitude": lon,
                 "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
                 "past_days": PAST_DAYS, "forecast_days": 1, "timezone": "UTC"},
            )
            fresh = 0
            for r in rows:
                key = (region, parse_ts(r["ts"]))
                if key in seen_weather:
                    continue
                seen_weather.add(key)
                new_weather.append({
                    "region": region, "lat": lat, "lon": lon,
                    "temperature_c": r.get("temperature_2m"),
                    "humidity_pct": r.get("relative_humidity_2m"),
                    "wind_speed_kmh": r.get("wind_speed_10m"),
                    "wind_direction_deg": r.get("wind_direction_10m"),
                    "timestamp": r["ts"],
                    "source": "Open-Meteo",
                })
                fresh += 1
            print(f"  weather: {len(rows)} hours from API, {fresh} new")
        except requests.RequestException as e:
            print(f"  Weather backfill failed for {region}: {type(e).__name__}")

    load_rows(new_aqi, aqi_table, "AQI")
    load_rows(new_weather, weather_table, "weather")


if __name__ == "__main__":
    main()