"""
Single place every part of the app calls to get data. This is what
makes "swap mock for real" a one-line change: as long as Person 1's
BigQuery tables and Person 2's Firestore fields match the shared
schema, nothing in app.py has to change — only USE_MOCK below.
"""
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Load variables from .env file into environment
load_dotenv()

import mock_data

# --- Flip this to False on Day 2 evening once real data is flowing ---
USE_MOCK = False

# Fill these in when USE_MOCK = False
FIREBASE_KEY_PATH = os.getenv("FIREBASE_KEY_PATH", "firebase-key.json")
BQ_PROJECT_ID = os.getenv("BQ_PROJECT_ID", "carc-f5b14")
BQ_DATASET = os.getenv("BQ_DATASET", "air_quality")

# Cloudinary handles photo/voice uploads (Firestore/BigQuery stay on Google —
# Cloudinary only replaces Firebase Storage, not the databases). Get these
# three values from your Cloudinary dashboard (cloudinary.com/console).
# Keep CLOUDINARY_API_SECRET out of git the same way you keep
# firebase-key.json out — see the .gitignore note in the README.
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")


@st.cache_resource
def _get_firestore_client():
    import firebase_admin
    from firebase_admin import credentials, firestore
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_KEY_PATH)
        firebase_admin.initialize_app(cred)
    return firestore.client()


@st.cache_resource
def _get_bq_client():
    from google.cloud import bigquery
    from google.oauth2 import service_account
    credentials = service_account.Credentials.from_service_account_file(FIREBASE_KEY_PATH)
    return bigquery.Client(project=BQ_PROJECT_ID, credentials=credentials)


def _safe_bq_query(client, query, table_label, warn_if_empty=True):
    """Runs a BigQuery query, returning an empty DataFrame (with a
    lightweight on-screen note) instead of crashing the whole app if the
    target table doesn't exist yet. Expected mid-hackathon when a
    teammate's pipeline hasn't populated its table yet (e.g. `forecast`
    depends on Person 2's Vertex AI job landing on Day 2). Any other
    BigQuery error (bad field name, permissions, etc.) still raises
    normally, since those need fixing rather than hiding.

    Also flags the case where the query runs fine but matches zero rows
    (table exists, but nothing recent) — this used to fail silently and
    just show up as "0" everywhere with no explanation.
    """
    from google.api_core.exceptions import NotFound
    try:
        df = client.query(query).to_dataframe()
    except NotFound:
        st.warning(f"BigQuery table for `{table_label}` wasn't found yet — showing empty data until it's built.", icon="⏳")
        return pd.DataFrame()
    if warn_if_empty and df.empty:
        st.info(
            f"`{table_label}` query ran fine but matched 0 rows. The table exists, "
            "but nothing in it falls inside the query's time window — check that the "
            "pipeline is actually writing recent rows, and that timestamps are stored "
            "in UTC (a common bug is writing local time and treating it as UTC).",
            icon="ℹ️",
        )
    return df


@st.cache_data(ttl=60)
def get_aqi_readings():
    if USE_MOCK:
        return mock_data.get_mock_aqi_readings()
    client = _get_bq_client()
    query = f"""
        SELECT station_id, location.lat AS lat, location.lng AS lon,
               timestamp, pm25, pm10, aqi_value, source
        FROM `{BQ_PROJECT_ID}.{BQ_DATASET}.aqi_readings`
        WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 6 HOUR)
    """
    return _safe_bq_query(client, query, "aqi_readings")


@st.cache_data(ttl=60)
def get_fire_hotspots():
    if USE_MOCK:
        return mock_data.get_mock_fire_hotspots()
    client = _get_bq_client()
    query = f"""
        SELECT location.lat AS lat, location.lng AS lon,
               timestamp, confidence, brightness, source
        FROM `{BQ_PROJECT_ID}.{BQ_DATASET}.fire_hotspots`
        WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
    """
    return _safe_bq_query(client, query, "fire_hotspots")


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
        lon = loc.get("lng") if isinstance(loc, dict) else getattr(loc, "longitude", None)
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
    return _safe_bq_query(client, query, "forecast", warn_if_empty=False)


def submit_citizen_report(report: dict, photo_file=None, voice_file=None):
    """Writes a new citizen report. In mock mode this just stores it in
    session state so the form is fully clickable from Day 1, without
    needing Firestore set up yet."""
    if USE_MOCK:
        st.session_state.setdefault("mock_reports", []).append(report)
        return True

    db = _get_firestore_client()
    if photo_file is not None:
        report["photo_url"] = _upload_file(photo_file, "photos", resource_type="image")
    if voice_file is not None:
        report["voice_url"] = _upload_file(voice_file, "voice", resource_type="video")
    db.collection("citizen_reports").add(report)
    return True


@st.cache_resource
def _configure_cloudinary():
    import cloudinary
    cloudinary.config(
        cloud_name=CLOUDINARY_CLOUD_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET,
        secure=True,
    )
    return True


def _upload_file(file, folder, resource_type):
    """Uploads a Streamlit UploadedFile straight to Cloudinary — no local
    save or byte conversion needed, since Cloudinary's SDK accepts any
    file-like object with .read()/.name, which is exactly what
    st.file_uploader() returns.

    resource_type matters: Cloudinary defaults to "image", which fails
    for audio. Voice notes need resource_type="video" (Cloudinary files
    audio under its "video" category — there's no separate "audio" type).
    """
    import cloudinary.uploader
    _configure_cloudinary()
    result = cloudinary.uploader.upload(
        file,
        folder=folder,
        resource_type=resource_type,
        use_filename=True,
        unique_filename=True,
        overwrite=False,
    )
    return result["secure_url"]
