# Air Quality Platform

A 3-day hackathon project that pulls together live AQI, weather, and fire/burning data with AI-powered citizen reporting to surface real-time air quality hotspots and short-term forecasts.

## Overview

The platform combines three data sources — government air quality monitors, weather data, and satellite fire detection — with crowdsourced citizen reports (photo, voice, and text) analyzed by Gemini. Everything is unified through a shared Firestore/BigQuery schema so the backend, AI, and frontend teams can build in parallel without blocking each other.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Data Pipeline    │     │  AI / ML          │     │  Frontend / Dashboard │
│  (BigQuery)       │     │  (Gemini, Speech,  │     │  (Streamlit)          │
│                   │     │   Translate,       │     │                       │
│  - AQI (OpenAQ/    │     │   Vertex AI)       │     │  - Map view           │
│    CPCB)           │     │                    │     │  - Forecast chart      │
│  - Weather (IMD/    │◄───┤  Reads citizen      │────►│  - Alerts panel        │
│    OpenWeatherMap)  │     │  reports, writes    │     │  - Citizen report form │
│  - Fire hotspots    │     │  ai_analysis to     │     │                       │
│    (Earth Engine)   │     │  Firestore          │     │                       │
│  - Hotspot scoring  │     │                    │     │                       │
└─────────┬─────────┘     └──────────────────┘     └───────────┬───────────┘
          │                                                    │
          └───────────────────► Firestore / BigQuery ◄─────────┘
                              (shared data contract)
```

## Team & Ownership

| Track | Owner | Focus |
|---|---|---|
| **Data Pipeline & Backend** | Person 1 | Firestore schema, BigQuery ingestion (AQI, weather, fire), hotspot scoring |
| **AI / ML Integration** | Person 2 | Gemini multimodal analysis, Speech-to-Text, Translation, Vertex AI forecasting |
| **Frontend / Dashboard & Demo** | Person 3 | Streamlit dashboard, citizen report form, map layers, demo/pitch |

Each track can be built independently against the shared schema below — no work is blocked waiting on another track.

## Shared Data Contract

All integration happens through these exact field names. No additional coordination is needed beyond matching this schema.

### Firestore: `citizen_reports`

| Field | Type | Written by |
|---|---|---|
| `location` | geopoint or lat/lng | Person 3 (citizen report form) |
| `category` | string | Person 3 |
| `text` | string | Person 3 |
| `language` | string | Person 3 |
| `photo_url` | string, optional | Person 3 |
| `voice_url` | string, optional | Person 3 |
| `timestamp` | server timestamp | Person 3 |
| `ai_analysis` | map: `severity` (1–5), `description`, `detected_issue` | **Person 2** |
| `hotspot_score` | number, optional | **Person 1** |

### BigQuery: `aqi_readings`
`station_id`, `location` (lat/lng), `timestamp`, `pm25`, `pm10`, `aqi_value`, `source`

### BigQuery: `fire_hotspots` (from Earth Engine)
`location` (lat/lng), `timestamp`, `confidence`, `brightness`, `source` (MODIS/VIIRS)

### BigQuery / Firestore: `forecast` (from Vertex AI)
`zone_id`, `forecast_timestamp`, `predicted_aqi`, `confidence_interval`

## How Integration Works

- **Shared Firebase project** — no one needs their own; everyone connects using a shared `firebase-key.json` (distributed securely, not via public channels).
- **AI → Dashboard** — Person 2 writes results into the `ai_analysis` field on the same Firestore document; Person 3 reads it directly. No separate integration step.
- **Backend → Dashboard** — Person 3's Streamlit app queries BigQuery directly via the `google-cloud-bigquery` Python client, using the same credentials.
- **Unblocked development** — if a piece isn't ready yet, build against mock data matching the schema. Swapping mock for real data later is a one-line change as long as field names match.

## Getting Started

1. Clone the repo and review the shared schema above — confirm field names before writing any integration code.
2. Get a Gemini API key from [aistudio.google.com](https://aistudio.google.com).
3. Request the shared `firebase-key.json` from Person 1 (backend owner) — do not share it via public channels.
4. Set up Google Cloud credentials for BigQuery access (`google-cloud-bigquery` client).
5. Each track can start building immediately:
   - **Backend**: connect to Firestore/BigQuery and begin ingestion.
   - **AI/ML**: start with dummy/sample data; no need to wait for the live pipeline.
   - **Frontend**: start with mock AQI points and mock hotspots; swap in live data later.

## Tech Stack

- **Data**: BigQuery, Firestore, Google Earth Engine
- **AI/ML**: Gemini (multimodal), Cloud Speech-to-Text, Translation API, Vertex AI AutoML
- **Frontend**: Streamlit
- **Data sources**: OpenAQ / CPCB (AQI), IMD / OpenWeatherMap (weather), MODIS/VIIRS (fire)

## Timeline / Sync Points

- **Kickoff**: everyone reads the schema, confirms field names, gets API keys
- **End of Day 1**: `firebase-key.json` + BigQuery access shared; mock data confirmed working across all tracks
- **Day 2 evening**: real pipeline data wired into AI calls and the dashboard
- **Day 3**: full integration test + demo rehearsal
