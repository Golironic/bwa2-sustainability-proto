# 🌫️️ Atmosphere Console — Real-Time AQI & Risk Intelligence Platform

Atmosphere Console is an end-to-end environmental monitoring dashboard, satellite tracking system, and predictive analytics platform. It ingests live ground-station air quality metrics (OpenAQ/CPCB), satellite thermal hotspot observations (NASA FIRMS), meteorological data (Open-Meteo), and multimodal citizen reports, running them through an XGBoost forecasting engine and contextual risk scoring algorithms.

---

## 👥 Team & Attribution

This platform is a collaborative effort, split across three core domains:

*   **Shikhar (ML & AI Pipeline):** Architected the machine learning and AI orchestration.
    *   *Files:* `csv_fetcher`, `features`, `local_mode_tester`, `model_caller`, `model_tester`, `model_trainer`, `ai_pipeline`
*   **Parth (Data Ingestion & Backend):** Handled live API ingestions, database operations, and proximity risk scoring.
    *   *Files:* `data_sources`, `fetch_aqi_weather`, `fetch_firms`, `hotspot_scoring`, `run_forecast`
*   **Krish (Frontend & UI Design):** Built the Streamlit dashboard, interactive map layers, and the custom global AQI glassmorphism design system.
    *   *Files:* `app.py`, `config.toml`, `theme.py`

---

## 🏛️ System Architecture

~~~text
                                  ┌───────────────────────────┐
                                  │   OpenAQ / CPCB Sensors   │
                                  └─────────────┬─────────────┘
                                                │
┌───────────────────────────┐                   ▼                  ┌───────────────────────────┐
│   NASA FIRMS Satellites   │ ────────► [Ingestion Pipeline] ◄─────│    Open-Meteo Weather     │
└───────────────────────────┘                   │                  └───────────────────────────┘
                                                ▼
┌───────────────────────────┐       ┌───────────────────────┐      ┌───────────────────────────┐
│ Citizen Incident Reports  │ ────► │ Google BigQuery Data  │ ◄─── │ Gemini Severity Engine &  │
│  (Cloudinary / Firestore) │       │   Warehouse / Store   │      │ Composite Hotspot Scoring │
└───────────────────────────┘       └───────────┬───────────┘      └───────────────────────────┘
                                                │
                                                ▼
                                    ┌───────────────────────┐
                                    │ XGBoost Time-Series   │
                                    │    Forecast Engine    │
                                    └───────────┬───────────┘
                                                │
                                                ▼
                                    ┌───────────────────────┐
                                    │  Atmosphere Console   │
                                    │  (Streamlit Dashboard)│
                                    └───────────────────────┘
~~~

---

## ✨ Features

- **Live Geospatial Console**: Map layers (PyDeck & Folium) displaying live AQI stations, active NASA thermal fire hotspots, and geolocated citizen submissions.
- **Predictive AQI Horizon**: 24-hour recursive time-series forecasting powered by XGBoost with custom confidence interval bands.
- **Contextual Hotspot Scoring**: Composite 0–100 risk score evaluations factoring in report density, satellite fire proximity, and regional baseline AQI deviations.
- **Multimodal Citizen Submissions**: Geolocation-tagged incident reports supporting text, photo, and voice uploads integrated via Cloudinary and Firestore.
- **Glassmorphism UI System**: Dark-mode theme built around global EPA AQI severity color standards.

---

## 📁 Repository Structure

~~~text
.
├── Frontend & UI (Krish)
│   ├── app.py                  # Streamlit dashboard application & interactive tabs
│   ├── theme.py                # CSS design tokens, AQI color scales, UI components
│   └── .streamlit/
│       └── config.toml         # Streamlit server and theme configuration
│
├── Data Ingestion & Backend (Parth)
│   ├── data_sources.py         # Unified data abstraction layer (Live Cloud vs. Mock Mode)
│   ├── fetch_aqi_weather.py    # Ingests OpenAQ & Open-Meteo with station deduplication
│   ├── fetch_firms.py          # NASA FIRMS satellite thermal anomaly ingestion pipeline
│   ├── hotspot_scoring.py      # Composite 0–100 risk scoring engine
│   └── run_forecast.py         # Executes 24-hour time-series forecasting job
│
├── ML & AI Pipeline (Shikhar)
│   ├── ai_pipeline.py          # Orchestrates the end-to-end AI/ML workflow
│   ├── csv_fetcher.py          # Historical data extraction and parsing
│   ├── features.py             # Feature engineering for the ML models
│   ├── local_mode_tester.py    # Local environment testing for the ML pipeline
│   ├── model_caller.py         # Inference scripts for XGBoost and LLM scoring
│   ├── model_tester.py         # Model evaluation and metrics generation
│   └── model_trainer.py        # XGBoost model training and validation pipeline
│
├── .env.example                # Environment credential template
└── requirements.txt            # Package dependencies
~~~

---

## 🚀 Quickstart Guide

### 1. Requirements & Setup
- Python 3.10 or higher
- Google Cloud Platform account with BigQuery & Firestore enabled
- GCP Service Account JSON key with `BigQuery Data Editor` and `Cloud Datastore User` roles

~~~bash
# Clone the repository
git clone https://github.com/your-org/atmosphere-console.git
cd atmosphere-console

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
~~~

### 2. Environment Configuration
Copy `.env.example` to `.env` and fill in your credential keys:
~~~bash
cp .env.example .env
~~~

Key environment variables:
~~~env
GCP_PROJECT_ID=your-gcp-project-id
GOOGLE_APPLICATION_CREDENTIALS=path/to/service_account.json
BIGQUERY_DATASET=atmosphere_data
GEMINI_API_KEY=your-gemini-api-key
OPENAQ_API_KEY=your-openaq-api-key
FIRMS_MAP_KEY=your-nasa-firms-map-key
CLOUDINARY_CLOUD_NAME=your-cloud-name
CLOUDINARY_API_KEY=your-api-key
CLOUDINARY_API_SECRET=your-api-secret
USE_MOCK=False
~~~

---

## 🔄 Running Data Pipelines

Execute data collection and inference scripts individually or schedule them via Cron / Cloud Scheduler:

1. **Ingest Sensor & Weather Readings**:
   ~~~bash
   python fetch_aqi_weather.py
   ~~~
2. **Fetch Satellite Thermal Anomalies**:
   ~~~bash
   python fetch_firms.py
   ~~~
3. **Run Hotspot Risk Scoring Engine**:
   ~~~bash
   python hotspot_scoring.py
   ~~~
4. **Train/Test AI Pipeline**:
   ~~~bash
   python ai_pipeline.py
   ~~~
5. **Generate 24-Hour Forecast Predictions**:
   ~~~bash
   python run_forecast.py
   ~~~

---

## 💻 Running the Dashboard

Start the Streamlit application:
~~~bash
streamlit run app.py
~~~

*Note*: To test the UI offline without cloud connections, set `USE_MOCK = True` inside `data_sources.py`.