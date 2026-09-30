"""
Hotspot scoring logic (Person 1's task from the shared team plan).

For every citizen report in Firestore, computes a hotspot_score (0-100)
by combining three signals near that report's location:
  1. Report density  - other citizen reports nearby, recently, weighted by
     Gemini severity (ai_analysis.severity, 1-5; neutral 3 if missing).
  2. Fire activity   - distinct, non-low-confidence NASA FIRMS detections nearby.
  3. AQI             - nearest FRESH station's latest reading, scored as a blend of
     (a) absolute level (0 at AQI 100 -> 100 at AQI 300) and (b) deviation above
     its region's recent baseline (mean over BASELINE_DAYS). Deviation alone
     gave chronically polluted cities (e.g. Delhi at ~160) a score of 0.

CHANGES IN THIS VERSION (vs. the original)
  - AQI query takes ONE row per station (the latest) and ignores stations
    older than AQI_MAX_AGE_HOURS. Previously it saw ~9 copies per station
    plus readings from 2016-2022 and used an arbitrary one.
  - AQI baseline is the region's mean over the last BASELINE_DAYS days
    (deduplicated), so stale/dead-station rows no longer drag it around and
    a region with a single station still gets a meaningful baseline.
  - Fire query uses SELECT DISTINCT and drops low-confidence detections, so
    re-loaded/duplicate rows can't inflate the fire score.
  - Density normalization now matches its comment: a neutral (severity 3)
    report counts as 1.0, so 5 nearby neutral reports = max score.
  - Naive datetimes are treated as UTC (avoids a TypeError on comparison).
  - Firestore writes are batched.
  - Nearest-station cutoff is a config value (AQI_MAX_STATION_KM).

Run:
    python hotspot_scoring.py

Requires:
    pip install google-cloud-firestore google-cloud-bigquery python-dotenv
"""

import os
import math
from datetime import datetime, timedelta, timezone
from google.cloud import firestore
from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()

# ---- CONFIG ----
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "firebase-key.json")

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "carc-f5b14")
DATASET_ID = "air_quality"

REPORT_RADIUS_KM = 5         # "nearby" radius for report density + fire check
REPORT_LOOKBACK_HOURS = 48   # only count reports from the last N hours
FIRE_LOOKBACK_HOURS = 120    # fires linger as a signal longer than reports
AQI_MAX_AGE_HOURS = 24       # ignore stations whose latest reading is older
BASELINE_DAYS = 7            # window for the regional baseline AQI
AQI_MAX_STATION_KM = 50      # nearest station must be within this distance

# AQI component = blend of two signals:
#   absolute  : how bad is the air right now (0 at AQI_FLOOR, 100 at AQI_CEILING)
#   deviation : how far above this region's own recent baseline (50%+ = 100)
# AQI_ABS_WEIGHT is the share given to "absolute"; 0.0 restores the old
# deviation-only behaviour, 1.0 uses absolute only.
AQI_ABS_WEIGHT = 0.5
AQI_FLOOR = 100              # US AQI 100 = top of "Moderate": no credit below this
AQI_CEILING = 300            # US AQI 300 = start of "Hazardous": full credit

# Weights must sum to 1.0
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


def as_utc(dt):
    """Make datetimes comparable: naive -> assume UTC; non-datetimes -> None."""
    if not isinstance(dt, datetime):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def is_low_confidence(conf):
    """VIIRS: 'l'/'n'/'h'. MODIS: 0-100. Treat VIIRS-low and MODIS<30 as noise."""
    if conf is None:
        return False
    c = str(conf).strip().lower()
    if c in ("l", "low"):
        return True
    try:
        return float(c) < 30
    except ValueError:
        return False


def load_recent_fires():
    """Distinct recent fire detections, loaded once and reused for all reports."""
    query = f"""
        SELECT DISTINCT location.lat AS lat, location.lng AS lng, timestamp, confidence
        FROM `{PROJECT_ID}.{DATASET_ID}.fire_hotspots`
        WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {FIRE_LOOKBACK_HOURS} HOUR)
    """
    try:
        rows = [dict(row) for row in bq.query(query).result()]
    except Exception as e:
        print(f"Warning: could not load fire_hotspots ({e}) - fire component will be 0")
        return []
    kept = [r for r in rows if not is_low_confidence(r.get("confidence"))]
    if len(kept) < len(rows):
        print(f"  ignored {len(rows) - len(kept)} low-confidence fire detections")
    return kept


def load_aqi_with_baseline():
    """Latest fresh reading per station + each region's recent baseline AQI."""
    query = f"""
        WITH recent AS (
          -- distinct readings in the baseline window (removes duplicate loads,
          -- and excludes years-old readings from dead stations)
          SELECT DISTINCT region, station_id, source, location.lat AS lat, location.lng AS lng,
                 aqi_value, timestamp
          FROM `{PROJECT_ID}.{DATASET_ID}.aqi_readings`
          WHERE aqi_value IS NOT NULL
            AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {BASELINE_DAYS * 24} HOUR)
        ),
        baseline AS (
          -- one baseline per (region, source): real stations are compared with
          -- real history, modelled points with modelled history - never mixed
          SELECT region, source, AVG(aqi_value) AS region_baseline_aqi
          FROM recent
          GROUP BY region, source
        ),
        fresh AS (
          SELECT * FROM recent
          WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {AQI_MAX_AGE_HOURS} HOUR)
        ),
        real_regions AS (
          SELECT DISTINCT region FROM fresh WHERE source = 'OpenAQ'
        ),
        latest AS (
          -- one row per station (its most recent); modelled rows are used only
          -- for regions that have no fresh real station
          SELECT * FROM fresh
          WHERE source = 'OpenAQ'
             OR region NOT IN (SELECT region FROM real_regions)
          QUALIFY ROW_NUMBER() OVER (PARTITION BY station_id ORDER BY timestamp DESC) = 1
        )
        SELECT latest.*, baseline.region_baseline_aqi
        FROM latest
        JOIN baseline USING (region, source)
    """
    try:
        return [dict(row) for row in bq.query(query).result()]
    except Exception as e:
        print(f"Warning: could not load aqi_readings ({e}) - AQI component will be 0")
        return []


def score_report_density(report, all_reports):
    """Severity-weighted count of other recent reports within radius.
    A neutral report (severity 3) counts as 1.0; 5 of them = max score."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=REPORT_LOOKBACK_HOURS)
    lat, lon = report["lat"], report["lon"]
    weighted_count = 0.0
    for other in all_reports:
        if other["id"] == report["id"]:
            continue
        created = as_utc(other["created_at"])
        if created is None or created < cutoff:
            continue
        if haversine_km(lat, lon, other["lat"], other["lon"]) <= REPORT_RADIUS_KM:
            severity = other.get("severity") or 3
            weighted_count += severity / 3.0
    return min(weighted_count / 5.0, 1.0) * 100


def score_fire_activity(report, fires):
    lat, lon = report["lat"], report["lon"]
    nearby = [f for f in fires if haversine_km(lat, lon, f["lat"], f["lng"]) <= REPORT_RADIUS_KM]
    if not nearby:
        return 0.0
    return min(len(nearby) / 3.0, 1.0) * 100  # 3+ detections = max


def nearest_station(report, aqi_rows):
    """Nearest fresh station within AQI_MAX_STATION_KM, or None."""
    nearest, nearest_dist = None, float("inf")
    for row in aqi_rows:
        dist = haversine_km(report["lat"], report["lon"], row["lat"], row["lng"])
        if dist < nearest_dist:
            nearest, nearest_dist = row, dist
    if nearest is None or nearest_dist > AQI_MAX_STATION_KM:
        return None
    return nearest


def score_aqi_absolute(aqi_value):
    """0 at AQI_FLOOR, 100 at AQI_CEILING, linear between."""
    if aqi_value is None:
        return 0.0
    frac = (aqi_value - AQI_FLOOR) / float(AQI_CEILING - AQI_FLOOR)
    return max(0.0, min(frac, 1.0)) * 100


def score_aqi_deviation(nearest):
    """How far the station is above its region's baseline (50%+ = max)."""
    if nearest is None:
        return 0.0
    baseline = nearest["region_baseline_aqi"] or 0
    current = nearest["aqi_value"] or 0
    if baseline <= 0:
        return 0.0
    deviation_ratio = (current - baseline) / baseline
    return max(0.0, min(deviation_ratio / 0.5, 1.0)) * 100


def score_aqi(report, aqi_rows):
    nearest = nearest_station(report, aqi_rows)
    if nearest is None:
        return {"combined": 0.0, "absolute": 0.0, "deviation": 0.0}
    absolute = score_aqi_absolute(nearest["aqi_value"])
    deviation = score_aqi_deviation(nearest)
    combined = AQI_ABS_WEIGHT * absolute + (1 - AQI_ABS_WEIGHT) * deviation
    return {"combined": combined, "absolute": absolute, "deviation": deviation}

def compute_hotspot_score(report, all_reports, fires, aqi_rows):
    density = score_report_density(report, all_reports)
    fire = score_fire_activity(report, fires)
    aqi_parts = score_aqi(report, aqi_rows)
    aqi = aqi_parts["combined"]
    total = (
        density * WEIGHT_REPORT_DENSITY
        + fire * WEIGHT_FIRE_ACTIVITY
        + aqi * WEIGHT_AQI_DEVIATION
    )
    return round(total, 1), {
        "density": round(density, 1),
        "fire": round(fire, 1),
        "aqi": round(aqi, 1),                       # blended (what feeds the total)
        "aqi_absolute": round(aqi_parts["absolute"], 1),   # extra, for debugging
        "aqi_deviation": round(aqi_parts["deviation"], 1), # extra, for debugging
    }


def main():
    print("Loading citizen reports from Firestore...")
    reports_ref = db.collection("citizen_reports")
    docs = list(reports_ref.stream())

    all_reports = []
    for doc in docs:
        d = doc.to_dict()
        loc = d.get("location")
        if hasattr(loc, "latitude"):
            lat, lon = loc.latitude, loc.longitude
        elif isinstance(loc, dict):
            lat, lon = loc.get("lat"), loc.get("lng") or loc.get("lon")
        else:
            lat, lon = d.get("lat"), d.get("lon")

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
    print(f"  {len(fires)} usable fire detections")

    print("Loading fresh AQI readings + regional baselines from BigQuery...")
    aqi_rows = load_aqi_with_baseline()
    print(f"  {len(aqi_rows)} fresh stations (one latest reading each)")

    print("Scoring reports...")
    batch = db.batch()
    pending = 0
    updated = 0
    for report in all_reports:
        if report["lat"] is None or report["lon"] is None:
            continue
        score, breakdown = compute_hotspot_score(report, all_reports, fires, aqi_rows)
        batch.update(reports_ref.document(report["id"]), {
            "hotspot_score": score,
            "hotspot_score_breakdown": breakdown,
        })
        pending += 1
        updated += 1
        print(f"  {report['id']}: score={score} (density={breakdown['density']}, "
              f"fire={breakdown['fire']}, aqi={breakdown['aqi']})")
        if pending >= 400:  # Firestore batch limit is 500
            batch.commit()
            batch = db.batch()
            pending = 0
    if pending:
        batch.commit()

    print(f"Done. Updated hotspot_score on {updated} reports.")


if __name__ == "__main__":
    main()