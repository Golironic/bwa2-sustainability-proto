"""
Mock data generators for the Air Quality Platform dashboard.

Field names match the shared schema in the hackathon plan exactly, so
swapping mock data for real data later (Day 2 evening) is a one-line
change in data_sources.py — nothing in app.py needs to change.
"""
import random
from datetime import datetime, timedelta

import pandas as pd

# Center the mock points around Delhi NCR (CPCB/IMD territory) — change
# this if your city/region differs.
CENTER_LAT, CENTER_LON = 28.6139, 77.2090


def _jitter(base, spread=0.15):
    return base + random.uniform(-spread, spread)


def get_mock_aqi_readings(n=25):
    rows = []
    for i in range(n):
        rows.append({
            "station_id": f"STN{i:03d}",
            "lat": _jitter(CENTER_LAT),
            "lon": _jitter(CENTER_LON),
            "timestamp": datetime.now() - timedelta(minutes=random.randint(0, 60)),
            "pm25": round(random.uniform(20, 300), 1),
            "pm10": round(random.uniform(30, 400), 1),
            "aqi_value": random.randint(50, 450),
            "source": random.choice(["OpenAQ", "CPCB"]),
        })
    return pd.DataFrame(rows)


def get_mock_fire_hotspots(n=10):
    rows = []
    for i in range(n):
        rows.append({
            "lat": _jitter(CENTER_LAT, 0.3),
            "lon": _jitter(CENTER_LON, 0.3),
            "timestamp": datetime.now() - timedelta(hours=random.randint(0, 12)),
            "confidence": random.randint(50, 100),
            "brightness": round(random.uniform(300, 400), 1),
            "source": random.choice(["MODIS", "VIIRS"]),
        })
    return pd.DataFrame(rows)


def get_mock_citizen_reports(n=15):
    categories = ["Burning", "Vehicle Smoke", "Industrial", "Dust", "Construction", "Other"]
    issues = [
        "Crop residue burning", "Diesel generator smoke", "Garbage burning",
        "Road dust", "Factory emissions",
    ]
    rows = []
    for i in range(n):
        severity = random.randint(1, 5)
        rows.append({
            "id": f"report_{i}",
            "lat": _jitter(CENTER_LAT, 0.2),
            "lon": _jitter(CENTER_LON, 0.2),
            "category": random.choice(categories),
            "text": "Mock citizen report text describing the issue.",
            "language": random.choice(["en", "hi"]),
            "photo_url": None,
            "voice_url": None,
            "timestamp": datetime.now() - timedelta(minutes=random.randint(0, 500)),
            "severity": severity,
            "description": random.choice(issues),
            "detected_issue": random.choice(issues),
            "hotspot_score": round(random.uniform(0, 1), 2),
        })
    return pd.DataFrame(rows)


def get_mock_forecast(hours=24):
    rows = []
    base = random.randint(100, 250)
    for h in range(hours):
        val = max(20, base + random.randint(-40, 40))
        rows.append({
            "zone_id": "ZONE_1",
            "forecast_timestamp": datetime.now() + timedelta(hours=h),
            "predicted_aqi": val,
            "confidence_low": max(0, val - 30),
            "confidence_high": val + 30,
        })
    return pd.DataFrame(rows)
