"""
Seeds TEST citizen reports into Firestore so you can verify hotspot scoring
end to end.

What it creates
  - 5 reports clustered within ~1 km of the densest recent fire in BigQuery
    (mixed severities 3-5, created "now")  -> should score high on density + fire
  - 1 control report in Gandhinagar with no fire nearby -> should score ~0
All test docs have IDs starting with "test_" and is_test=True, and follow the
shared contract (location GeoPoint, category, text, language, timestamp,
ai_analysis{severity, description, detected_issue}).

Usage:
    python seed_test_reports.py            # create test reports
    python hotspot_scoring.py              # score them
    python debug_scoring.py                # see the breakdown
    python seed_test_reports.py --delete   # remove ALL test_* docs when done

NOTE: Firestore is shared with the team, so test reports will show on
Person 3's map until you run --delete. They're labelled "TEST" in the text.
"""
import sys
import math
import random
from google.cloud import firestore
import hotspot_scoring as H

COLLECTION = "citizen_reports"
CONTROL = (23.2156, 72.6369)  # Gandhinagar centre, no fires nearby

CLUSTER = [  # (severity, description)
    (4, "Thick smoke rising from a burning field next to the road"),
    (5, "Large area of crop residue on fire, smoke drifting over houses"),
    (3, "Smell of burning and haze near the village"),
    (4, "Fire visible in the field, residents coughing"),
    (5, "Heavy smoke reducing visibility on the highway"),
]


def densest_fire(fires):
    """The fire detection with the most other detections within REPORT_RADIUS_KM."""
    best, best_n = None, -1
    for f in fires:
        n = sum(1 for g in fires
                if H.haversine_km(f["lat"], f["lng"], g["lat"], g["lng"]) <= H.REPORT_RADIUS_KM)
        if n > best_n:
            best, best_n = f, n
    return best, best_n


def make_doc(lat, lng, severity, description):
    return {
        "location": firestore.GeoPoint(lat, lng),
        "category": "open_burning",
        "text": f"TEST: {description}",
        "language": "en",
        "timestamp": firestore.SERVER_TIMESTAMP,
        "ai_analysis": {
            "severity": severity,
            "description": description,
            "detected_issue": "biomass_burning",
        },
        "is_test": True,
    }


def seed():
    fires = H.load_recent_fires()
    if not fires:
        raise SystemExit("No usable fire detections in BigQuery - run fetch_firms.py first.")

    centre, n = densest_fire(fires)
    print(f"Densest fire area: ({centre['lat']:.4f}, {centre['lng']:.4f}) "
          f"with {n} detections within {H.REPORT_RADIUS_KM} km")

    rng = random.Random(42)  # reproducible offsets
    col = H.db.collection(COLLECTION)
    for i, (sev, desc) in enumerate(CLUSTER, start=1):
        dlat = rng.uniform(-0.008, 0.008)  # ~0.9 km
        dlng = rng.uniform(-0.008, 0.008) / max(math.cos(math.radians(centre["lat"])), 0.1)
        col.document(f"test_cluster_{i}").set(
            make_doc(centre["lat"] + dlat, centre["lng"] + dlng, sev, desc))
        print(f"  created test_cluster_{i} (severity {sev})")

    col.document("test_control").set(
        make_doc(CONTROL[0], CONTROL[1], 3, "Mild haze, nothing unusual"))
    print("  created test_control (Gandhinagar, no fire nearby)")
    print("\nNext: python hotspot_scoring.py  (then python debug_scoring.py)")


def delete():
    col = H.db.collection(COLLECTION)
    removed = 0
    for doc in col.stream():
        if doc.id.startswith("test_"):
            doc.reference.delete()
            removed += 1
            print(f"  deleted {doc.id}")
    print(f"Deleted {removed} test reports.")


if __name__ == "__main__":
    if "--delete" in sys.argv:
        delete()
    else:
        seed()