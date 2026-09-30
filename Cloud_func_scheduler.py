"""
Cloud Function entry point that dispatches to fetch_aqi_weather and fetch_firms.

Previously this file was two scripts concatenated together, causing:
  - duplicate main() definitions (the second silently overwrote the first)
  - module-level code from the second script running at import time
  - the REGIONS dict being overwritten, breaking KeyError: 'points' lookups

Now it is a thin dispatcher. All real logic lives in the two dedicated modules.

Deploy:
    gcloud functions deploy run_my_fetcher \
        --runtime python312 \
        --trigger-http \
        --entry-point run_my_fetcher \
        --set-env-vars GCP_PROJECT_ID=...,OPENAQ_API_KEY=...,FIRMS_MAP_KEY=...

Run locally (both fetchers):
    python Cloud_func_scheduler.py
"""

import fetch_aqi_weather
import fetch_firms


def run_my_fetcher(request):
    """HTTP Cloud Function entry point. Runs both data fetchers in sequence."""
    errors = []

    try:
        print("=== fetch_aqi_weather ===")
        fetch_aqi_weather.main()
    except Exception as e:
        print(f"fetch_aqi_weather failed: {e}")
        errors.append(f"aqi_weather: {e}")

    try:
        print("=== fetch_firms ===")
        fetch_firms.main()
    except Exception as e:
        print(f"fetch_firms failed: {e}")
        errors.append(f"firms: {e}")

    if errors:
        return f"Completed with errors: {'; '.join(errors)}", 500
    return "Success", 200


if __name__ == "__main__":
    run_my_fetcher(request=None)