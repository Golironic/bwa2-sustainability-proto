"""
Hotspot scoring logic (Person 1's task from the shared team plan).

For every citizen report in Firestore, computes a hotspot_score (0-100)
by combining three signals near that report's location:
  1. Report density  - how many OTHER citizen reports were filed nearby
     recently (more reports near each other = more likely something real
     is happening, not a one-off).
  2. Fire activity    - any active NASA FIRMS fire detections nearby
     (proxy for stubble/open burning).
  3. AQI deviation    - how far the nearest AQI station's reading is
     above that region's own baseline (mean) AQI - a station reading that's
     way above its own region's normal is more meaningful than an absolute
     threshold, since baseline pollution varies a lot by city.

If Person 2's Gemini severity score (ai_analysis.severity, 1-5) is present
on a report, it's folded in as a weight - a cluster of high-severity
reports scores higher than the same number of low-severity ones. If it's
not present yet (Person 2 hasn't run), severity defaults to a neutral
weight so scoring still works standalone.

Writes the result back to Firestore: citizen_reports/{doc_id}.hotspot_score

Run:
    python hotspot_scoring.py

Requires:
    pip install google-cloud-firestore google-cloud-bigquery
    GOOGLE_APPLICATION_CREDENTIALS env var pointing to firebase-key.json
"""

import os
import math
from datetime import datetime, timedelta, timezone
from google.cloud import firestore
from google.cloud import bigquery

# ---- CONFIG ----
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "firebase-key.json")

PROJECT_ID = "carc-f5b14"
DATASET_ID = "air_quality"

REPORT_RADIUS_KM = 5       # "nearby" radius for report density + fire check
REPORT_LOOKBACK_HOURS = 48  # only count reports/fires from the last N hours
FIRE_LOOKBACK_HOURS = 120   # fires linger as a signal longer than reports

# Score weights - tune these based on what your team wants to prioritize.
# Must sum to 1.0 across the three components.
WEIGHT_REPORT_DENSITY = 0.35
WEIGHT_FIRE_ACTIVITY = 0.30
WEIGHT_AQI_DEVIATION = 0.35

db = firestore.Client(project=PROJECT_ID)
bq = bigquery.Client(project=PROJECT_ID)


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points, in km."""
    if None in (lat1, lon1, lat2, lon2):
        return float("inf")
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def load_recent_fires():
    """Pull recent fire detections from BigQuery once, reused for all reports."""
    query = f"""
        SELECT location.lat AS lat, location.lng AS lng, timestamp
        FROM `{PROJECT_ID}.{DATASET_ID}.fire_hotspots`
        WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {FIRE_LOOKBACK_HOURS} HOUR)
    """
    try:
        return [dict(row) for row in bq.query(query).result()]
    except Exception as e:
        print(f"Warning: could not load fire_hotspots ({e}) - fire component will be 0")
        return []


def load_aqi_with_baseline():
    """Pull current AQI readings plus each region's baseline (mean) AQI."""
    query = f"""
        WITH latest AS (
          SELECT region, station_id, location.lat AS lat, location.lng AS lng,
                 aqi_value, timestamp
          FROM `{PROJECT_ID}.{DATASET_ID}.aqi_readings`
          WHERE aqi_value IS NOT NULL
        ),
        baseline AS (
          SELECT region, AVG(aqi_value) AS region_baseline_aqi
          FROM latest
          GROUP BY region
        )
        SELECT latest.*, baseline.region_baseline_aqi
        FROM latest
        JOIN baseline USING (region)
    """
    try:
        return [dict(row) for row in bq.query(query).result()]
    except Exception as e:
        print(f"Warning: could not load aqi_readings ({e}) - AQI component will be 0")
        return []


def score_report_density(report, all_reports):
    """Count other reports within radius + lookback window. Weighted by
    their Gemini severity if available (defaults to neutral weight 3/5)."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=REPORT_LOOKBACK_HOURS)
    lat, lon = report["lat"], report["lon"]
    weighted_count = 0.0
    for other in all_reports:
        if other["id"] == report["id"]:
            continue
        if other["created_at"] and other["created_at"] < cutoff:
            continue
        dist = haversine_km(lat, lon, other["lat"], other["lon"])
        if dist <= REPORT_RADIUS_KM:
            severity = other.get("severity") or 3  # neutral default if Gemini hasn't scored it
            weighted_count += severity / 5.0
    # Normalize: 5+ weighted nearby reports = max density score
    return min(weighted_count / 5.0, 1.0) * 100


def score_fire_activity(report, fires):
    lat, lon = report["lat"], report["lon"]
    nearby = [f for f in fires if haversine_km(lat, lon, f["lat"], f["lng"]) <= REPORT_RADIUS_KM]
    if not nearby:
        return 0.0
    # 3+ nearby fire detections = max fire score
    return min(len(nearby) / 3.0, 1.0) * 100


def score_aqi_deviation(report, aqi_rows):
    lat, lon = report["lat"], report["lon"]
    nearest, nearest_dist = None, float("inf")
    for row in aqi_rows:
        dist = haversine_km(lat, lon, row["lat"], row["lng"])
        if dist < nearest_dist:
            nearest, nearest_dist = row, dist
    if nearest is None or nearest_dist > 100:  # no station within 100km, skip
        return 0.0
    baseline = nearest["region_baseline_aqi"] or 0
    current = nearest["aqi_value"] or 0
    if baseline <= 0:
        return 0.0
    deviation_ratio = (current - baseline) / baseline
    # 50%+ above baseline = max deviation score
    return max(0.0, min(deviation_ratio / 0.5, 1.0)) * 100


def compute_hotspot_score(report, all_reports, fires, aqi_rows):
    density = score_report_density(report, all_reports)
    fire = score_fire_activity(report, fires)
    aqi = score_aqi_deviation(report, aqi_rows)
    total = (
        density * WEIGHT_REPORT_DENSITY
        + fire * WEIGHT_FIRE_ACTIVITY
        + aqi * WEIGHT_AQI_DEVIATION
    )
    return round(total, 1), {"density": round(density, 1), "fire": round(fire, 1), "aqi": round(aqi, 1)}


def main():
    print("Loading citizen reports from Firestore...")
    reports_ref = db.collection("citizen_reports")
    docs = list(reports_ref.stream())

    all_reports = []
    for doc in docs:
        d = doc.to_dict()
        loc = d.get("location")
        # Handle both a Firestore GeoPoint and a plain {lat, lon} map,
        # since the schema allows either.
        if hasattr(loc, "latitude"):
            lat, lon = loc.latitude, loc.longitude
        elif isinstance(loc, dict):
            lat, lon = loc.get("lat"), loc.get("lng") or loc.get("lon")
        else:
            lat, lon = d.get("lat"), d.get("lon")  # fallback for older docs

        created_at = d.get("created_at") or d.get("timestamp")
        ai = d.get("ai_analysis") or {}

        all_reports.append({
            "id": doc.id,
            "lat": lat,
            "lon": lon,
            "created_at": created_at,
            "severity": ai.get("severity"),
        })

    print(f"  {len(all_reports)} reports loaded")

    print("Loading recent fire detections from BigQuery...")
    fires = load_recent_fires()
    print(f"  {len(fires)} recent fire detections")

    print("Loading AQI readings + regional baselines from BigQuery...")
    aqi_rows = load_aqi_with_baseline()
    print(f"  {len(aqi_rows)} AQI readings across regions")

    print("Scoring reports...")
    updated = 0
    for report in all_reports:
        if report["lat"] is None or report["lon"] is None:
            continue
        score, breakdown = compute_hotspot_score(report, all_reports, fires, aqi_rows)
        reports_ref.document(report["id"]).update({
            "hotspot_score": score,
            "hotspot_score_breakdown": breakdown,  # extra, for demo/debug - not in contract but harmless
        })
        updated += 1
        print(f"  {report['id']}: score={score} (density={breakdown['density']}, "
              f"fire={breakdown['fire']}, aqi={breakdown['aqi']})")

    print(f"Done. Updated hotspot_score on {updated} reports.")


if __name__ == "__main__":
    main()
