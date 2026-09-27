"""
Read-only diagnostic for hotspot_scoring.py. Writes NOTHING.
Shows, for every citizen report, exactly why each score component is what it is.

Run (same folder as hotspot_scoring.py):
    python debug_scoring.py
"""
from datetime import datetime, timezone
import hotspot_scoring as H


def km(d):
    return "none" if d == float("inf") else f"{d:.1f} km"


def load_reports():
    out = []
    for doc in H.db.collection("citizen_reports").stream():
        d = doc.to_dict()
        loc = d.get("location")
        if hasattr(loc, "latitude"):
            lat, lon = loc.latitude, loc.longitude
        elif isinstance(loc, dict):
            lat, lon = loc.get("lat"), loc.get("lng") or loc.get("lon")
        else:
            lat, lon = d.get("lat"), d.get("lon")
        out.append({
            "id": doc.id, "lat": lat, "lon": lon,
            "created_at": d.get("created_at") or d.get("timestamp"),
            "severity": (d.get("ai_analysis") or {}).get("severity"),
            "raw_location": loc,
        })
    return out


def main():
    reports = load_reports()
    fires = H.load_recent_fires()
    aqi = H.load_aqi_with_baseline()
    now = datetime.now(timezone.utc)

    print("\n=== FRESH AQI STATIONS USED FOR SCORING ===")
    if not aqi:
        print("  (none)")
    for s in aqi:
        print(f"  {s['region']:<12} {s['station_id']:<24} ({s['lat']:.3f}, {s['lng']:.3f})  "
              f"aqi={s['aqi_value']}  baseline={s['region_baseline_aqi']:.1f}  at {s['timestamp']}")

    print("\n=== PER REPORT ===")
    for r in reports:
        print(f"\nReport {r['id']}")
        print(f"  location field : {r['raw_location']}")
        print(f"  parsed lat/lon : {r['lat']}, {r['lon']}")
        created = H.as_utc(r["created_at"])
        age = f"{(now - created).total_seconds() / 3600:.1f} h ago" if created else "MISSING/unparseable"
        print(f"  created_at     : {r['created_at']}  ({age})   severity={r['severity']}")
        if r["lat"] is None or r["lon"] is None:
            print("  -> no coordinates, cannot be scored")
            continue

        # density
        print(f"  DENSITY: other reports within {H.REPORT_RADIUS_KM} km (counted only if <{H.REPORT_LOOKBACK_HOURS}h old):")
        for o in reports:
            if o["id"] == r["id"]:
                continue
            oc = H.as_utc(o["created_at"])
            oage = f"{(now - oc).total_seconds() / 3600:.0f}h old" if oc else "no date"
            print(f"      {o['id']}: {km(H.haversine_km(r['lat'], r['lon'], o['lat'], o['lon']))} away, {oage}")

        # fire
        dists = sorted(H.haversine_km(r["lat"], r["lon"], f["lat"], f["lng"]) for f in fires)
        within = sum(1 for d in dists if d <= H.REPORT_RADIUS_KM)
        print(f"  FIRE   : nearest detection {km(dists[0]) if dists else 'none'}; "
              f"{within} within {H.REPORT_RADIUS_KM} km (need 3 for max)")

        # aqi
        best, bd = None, float("inf")
        for s in aqi:
            d = H.haversine_km(r["lat"], r["lon"], s["lat"], s["lng"])
            if d < bd:
                best, bd = s, d
        if best is None:
            print("  AQI    : no fresh stations at all")
        else:
            dev = (best["aqi_value"] - best["region_baseline_aqi"]) / best["region_baseline_aqi"] * 100
            note = "TOO FAR (limit %s km)" % H.AQI_MAX_STATION_KM if bd > H.AQI_MAX_STATION_KM else "in range"
            print(f"  AQI    : nearest station {best['station_id']} ({best['region']}) is {km(bd)} away [{note}]; "
                  f"aqi={best['aqi_value']} vs baseline {best['region_baseline_aqi']:.1f} => {dev:+.0f}% (need +50% for max)")
            parts = H.score_aqi(r, aqi)
            print(f"           absolute score {parts['absolute']:.1f} (0 at AQI {H.AQI_FLOOR}, 100 at {H.AQI_CEILING}), "
                  f"deviation score {parts['deviation']:.1f} -> blended {parts['combined']:.1f} "
                  f"(absolute weight {H.AQI_ABS_WEIGHT})")


if __name__ == "__main__":
    main()