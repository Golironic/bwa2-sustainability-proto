"""
Single place every part of the app calls to get data. This is what
makes "swap mock for real" a one-line change: as long as Person 1's
BigQuery tables and Person 2's Firestore fields match the shared
schema, nothing in app.py has to change — only USE_MOCK below.
"""
import streamlit as st
import pandas as pd

import mock_data

# --- Flip this to False on Day 2 evening once real data is flowing ---
USE_MOCK = True

# Fill these in when USE_MOCK = False
FIREBASE_KEY_PATH = "firebase-key.json"   # Person 1 shares this file with you
BQ_PROJECT_ID = "your-gcp-project-id"
BQ_DATASET = "air_quality"


@st.cache_resource
def _get_firestore_client():
    import firebase_admin
    from firebase_admin import credentials, firestore, storage
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_KEY_PATH)
        firebase_admin.initialize_app(cred, {"storageBucket": f"{BQ_PROJECT_ID}.appspot.com"})
    return firestore.client()


@st.cache_resource
def _get_bq_client():
    from google.cloud import bigquery
    return bigquery.Client(project=BQ_PROJECT_ID)


@st.cache_data(ttl=60)
def get_aqi_readings():
    if USE_MOCK:
        return mock_data.get_mock_aqi_readings()
    client = _get_bq_client()
    query = f"""
        SELECT station_id, location.lat AS lat, location.lon AS lon,
               timestamp, pm25, pm10, aqi_value, source
        FROM `{BQ_PROJECT_ID}.{BQ_DATASET}.aqi_readings`
        WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 6 HOUR)
    """
    return client.query(query).to_dataframe()


@st.cache_data(ttl=60)
def get_fire_hotspots():
    if USE_MOCK:
        return mock_data.get_mock_fire_hotspots()
    client = _get_bq_client()
    query = f"""
        SELECT location.lat AS lat, location.lon AS lon,
               timestamp, confidence, brightness, source
        FROM `{BQ_PROJECT_ID}.{BQ_DATASET}.fire_hotspots`
        WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
    """
    return client.query(query).to_dataframe()


@st.cache_data(ttl=30)
def get_citizen_reports():
    if USE_MOCK:
        return mock_data.get_mock_citizen_reports()
    db = _get_firestore_client()
    docs = (
        db.collection("citizen_reports")
        .order_by("timestamp", direction="DESCENDING")
        .limit(200)
        .stream()
    )
    rows = []
    for doc in docs:
        d = doc.to_dict()
        loc = d.get("location", {}) or {}
        ai = d.get("ai_analysis", {}) or {}
        lat = loc.get("lat") if isinstance(loc, dict) else getattr(loc, "latitude", None)
        lon = loc.get("lon") if isinstance(loc, dict) else getattr(loc, "longitude", None)
        rows.append({
            "id": doc.id,
            "lat": lat,
            "lon": lon,
            "category": d.get("category"),
            "text": d.get("text"),
            "language": d.get("language"),
            "photo_url": d.get("photo_url"),
            "voice_url": d.get("voice_url"),
            "timestamp": d.get("timestamp"),
            "severity": ai.get("severity"),
            "description": ai.get("description"),
            "detected_issue": ai.get("detected_issue"),
            "hotspot_score": d.get("hotspot_score"),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=60)
def get_forecast():
    if USE_MOCK:
        return mock_data.get_mock_forecast()
    client = _get_bq_client()
    query = f"""
        SELECT zone_id, forecast_timestamp, predicted_aqi, confidence_interval
        FROM `{BQ_PROJECT_ID}.{BQ_DATASET}.forecast`
        ORDER BY forecast_timestamp
    """
    return client.query(query).to_dataframe()


def submit_citizen_report(report: dict, photo_file=None, voice_file=None):
    """Writes a new citizen report. In mock mode this just stores it in
    session state so the form is fully clickable from Day 1, without
    needing Firestore set up yet."""
    if USE_MOCK:
        st.session_state.setdefault("mock_reports", []).append(report)
        return True

    db = _get_firestore_client()
    if photo_file is not None:
        report["photo_url"] = _upload_file(photo_file, "photos")
    if voice_file is not None:
        report["voice_url"] = _upload_file(voice_file, "voice")
    db.collection("citizen_reports").add(report)
    return True


def _upload_file(file, folder):
    from firebase_admin import storage
    bucket = storage.bucket()
    blob = bucket.blob(f"{folder}/{file.name}")
    blob.upload_from_file(file, content_type=file.type)
    blob.make_public()
    return blob.public_url
