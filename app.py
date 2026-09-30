from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
import folium
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation
from dotenv import load_dotenv

load_dotenv()

import data_sources as ds
import theme

st.set_page_config(page_title="Atmosphere Console", page_icon="🌫️", layout="wide")

import os
mb_key = os.environ.get("MAPBOX_API_KEY", "")
if mb_key.startswith('"') and mb_key.endswith('"'): mb_key = mb_key[1:-1]

st.markdown(theme.CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Sidebar — system status
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="eyebrow">System Status</div>', unsafe_allow_html=True)
    dot_color = theme.ACCENT if ds.USE_MOCK else theme.GOOD
    mode_label = "MOCK DATA" if ds.USE_MOCK else "LIVE"
    st.markdown(
        f'<div style="margin:10px 0 18px 0; font-family:\'Space Grotesk\',sans-serif; '
        f'font-size:1.1rem; font-weight:600;">'
        f'<span class="status-dot" style="background:{dot_color};"></span>{mode_label}</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Last refreshed {datetime.now().strftime('%H:%M:%S')}")
    if st.button("↻ Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Data Sources</div>', unsafe_allow_html=True)
    st.caption("AQI stations · OpenAQ / CPCB")
    st.caption("Fire hotspots · MODIS / VIIRS")
    st.caption("Citizen reports · Gemini-scored")
    if ds.USE_MOCK:
        st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)
        st.caption("Flip `USE_MOCK = False` in data_sources.py once Firestore + BigQuery are live.")

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.markdown(
    f'{theme.live_pulse("AIR QUALITY PLATFORM &middot; LIVE MONITORING")}'
    '<h1 class="hero-title">Atmosphere Console</h1>',
    unsafe_allow_html=True,
)
if ds.USE_MOCK:
    st.caption("Running on mock data — every field name matches the shared schema, so this is demo-ready right now.")

aqi_df = ds.get_aqi_readings()
fire_df = ds.get_fire_hotspots()
reports_df = ds.get_citizen_reports()

mean_val = aqi_df["aqi_value"].mean() if not aqi_df.empty else 0.0
avg_aqi = float(mean_val) if not pd.isna(mean_val) else 0.0
st.markdown(theme.render_sky_strip(avg_aqi), unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Stat row
# ---------------------------------------------------------------------
high_severity = reports_df[reports_df["severity"].fillna(0) >= 4] if not reports_df.empty else pd.DataFrame()
high_aqi = aqi_df[aqi_df["aqi_value"] >= 200] if not aqi_df.empty else pd.DataFrame()
active_alerts = len(high_severity) + len(high_aqi)

cards = [
    theme.stat_card("Avg AQI", f"{avg_aqi:.0f}", theme.get_aqi_category(avg_aqi)[0],
                     color=theme.get_aqi_category(avg_aqi)[1], icon="gauge"),
    theme.stat_card("Fire Hotspots", str(len(fire_df)), "active detections", color=theme.EMBER, icon="flame"),
    theme.stat_card("Citizen Reports", str(len(reports_df)), "submitted", color=theme.ACCENT, icon="report"),
    theme.stat_card("Active Alerts", str(active_alerts), "need attention",
                     color=theme.UNHEALTHY if active_alerts else theme.GOOD, icon="bell"),
]
st.markdown(theme.stat_row(cards), unsafe_allow_html=True)

tab_map, tab_forecast, tab_alerts, tab_report = st.tabs(
    ["🗺️  Map", "📈  Forecast", "🚨  Alerts", "📝  Report an Issue"]
)

forecast_df = ds.get_forecast()

# ---------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------
with tab_map:
    st.markdown(theme.panel_heading("Live Map"), unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    show_aqi = col1.checkbox("AQI stations & Forecast", value=True)
    show_fire = col2.checkbox("Fire hotspots", value=True)
    show_reports = col3.checkbox("Citizen reports", value=True)
    
    # Custom City Navigation & Map Layout
    map_col, nav_col = st.columns([5, 1])
    
    with nav_col:
        st.markdown("<div class='eyebrow'>Focus Cities</div>", unsafe_allow_html=True)
        # Use session state to control map center
        if "map_center" not in st.session_state:
            st.session_state.map_center = [22.0, 79.0]
            st.session_state.map_zoom = 5
        
        def jump_to(lat, lon, zoom=10):
            st.session_state.map_center = [lat, lon]
            st.session_state.map_zoom = zoom

        if st.button("Delhi-NCR"): jump_to(28.6139, 77.2090)
        if st.button("Punjab"): jump_to(31.1471, 75.3412)
        if st.button("Gandhinagar"): jump_to(23.2156, 72.6369)
        if st.button("Mumbai"): jump_to(19.0760, 72.8777)
        if st.button("All India"): jump_to(22.0, 79.0, 5)

    with map_col:
        import folium
        import streamlit.components.v1 as components
        import os
        import json
        
        # Satellite map without base political borders to avoid disputes
        m = folium.Map(
            location=st.session_state.map_center, 
            zoom_start=st.session_state.map_zoom, 
            tiles=f"https://api.mapbox.com/styles/v1/mapbox/dark-v11/tiles/{{z}}/{{x}}/{{y}}?access_token={mb_key}",
            attr="Mapbox",
            zoom_control=False
        )

        # Overlay official India boundaries
        if os.path.exists("india_boundary.geojson"):
            with open("india_boundary.geojson", "r") as f:
                geojson_data = json.load(f)
            folium.GeoJson(
                geojson_data,
                name="India Border",
                style_function=lambda x: {"fillColor": "transparent", "color": "#E8A23D", "weight": 2, "opacity": 0.8}
            ).add_to(m)

        # Add data layers
        if show_aqi and not aqi_df.empty:
            next_hour = {}
            if not forecast_df.empty:
                for zone, df_zone in forecast_df.groupby("zone_id"):
                    fut = df_zone[df_zone["forecast_timestamp"] >= pd.Timestamp.utcnow()]
                    next_hour[zone] = fut.iloc[0]["predicted_aqi"] if not fut.empty else df_zone.iloc[-1]["predicted_aqi"]
            
            ZONES_CENTERS = {
                "Delhi-NCR": (28.6139, 77.2090),
                "Punjab": (31.1471, 75.3412),
                "Gandhinagar": (23.2156, 72.6369),
                "Mumbai": (19.0760, 72.8777),
            }
            
            def get_nearest_zone(lat, lon):
                best_zone, best_dist = None, 99999
                for z, (zlat, zlon) in ZONES_CENTERS.items():
                    dist = (lat - zlat)**2 + (lon - zlon)**2
                    if dist < best_dist:
                        best_dist = dist
                        best_zone = z
                return best_zone if best_dist < 3.0 else ""

            for _, r in aqi_df.dropna(subset=["lat", "lon"]).iterrows():
                val = r["aqi_value"]
                cat, hex_color = theme.get_aqi_category(val)
                
                nearest_zone = get_nearest_zone(r["lat"], r["lon"])
                pred_val = next_hour.get(nearest_zone, "--")
                pred_str = f"{pred_val:.0f}" if isinstance(pred_val, float) else pred_val
                
                html = f"""
                <div class="map-pin" style="background-color: {hex_color}ee; border: 2px solid white; border-radius: 8px; 
                            padding: 4px; color: #fff; font-family: sans-serif; font-size: 12px;
                            font-weight: bold; text-align: center; width: 80px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); animation: slideUpFade 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards, gentlePulse 2s infinite ease-out;">
                    <div>AQI: {val:.0f}</div>
                    <div style="font-size: 10px; font-weight: normal; opacity: 0.9;">Pred: {pred_str}</div>
                </div>
                """
                folium.Marker(
                    location=[r["lat"], r["lon"]],
                    icon=folium.DivIcon(html=html, icon_size=(80, 36), icon_anchor=(40, 18)),
                    tooltip=f"Station: {r.get('station_id', 'Unknown')}"
                ).add_to(m)

        if show_fire and not fire_df.empty:
            for _, r in fire_df.dropna(subset=["lat", "lon"]).iterrows():
                folium.CircleMarker(
                    location=[r["lat"], r["lon"]],
                    radius=4, color=theme.EMBER, fill=True, fill_opacity=0.8, weight=1
                ).add_to(m)

        if show_reports and not reports_df.empty:
            for _, r in reports_df.dropna(subset=["lat", "lon"]).iterrows():
                sev_color = theme.get_severity_color(r.get("severity", 1))
                folium.CircleMarker(
                    location=[r["lat"], r["lon"]],
                    radius=6, color=sev_color, fill=True, fill_opacity=1, weight=2,
                    tooltip=f"Report: Severity {r.get('severity', 1)}"
                ).add_to(m)

        # Inject CSS into the Folium iframe for animations
        map_css = """
        <style>
        @keyframes slideUpFade {
            0% { opacity: 0; transform: translateY(20px) scale(0.98); }
            100% { opacity: 1; transform: translateY(0) scale(1); }
        }
        @keyframes gentlePulse {
            0% { box-shadow: 0 0 0 0 rgba(255,255,255,0.4); }
            70% { box-shadow: 0 0 0 10px rgba(255,255,255,0); }
            100% { box-shadow: 0 0 0 0 rgba(255,255,255,0); }
        }
        </style>
        """
        m.get_root().html.add_child(folium.Element(map_css))

        # Render static HTML for extremely fast performance
        components.html(m.get_root().render(), height=500)

# ---------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------
with tab_forecast:
    with st.container(border=True):
        st.markdown(theme.panel_heading("AQI Forecast"), unsafe_allow_html=True)
        if forecast_df.empty:
            st.markdown(
                theme.empty_state("No forecast data yet — this fills in once Person 2's Vertex AI model runs (Day 2)."),
                unsafe_allow_html=True,
            )
        else:
            fig = go.Figure()
            colors = [theme.ACCENT, theme.GOOD, theme.SENSITIVE, theme.UNHEALTHY, theme.HAZARDOUS]
            
            for i, (zone, df_zone) in enumerate(forecast_df.groupby("zone_id")):
                df_zone = df_zone.sort_values("forecast_timestamp")
                color = colors[i % len(colors)]
                
                if {"confidence_low", "confidence_high"}.issubset(df_zone.columns):
                    r, g, b, _ = theme.hex_to_rgba(color)
                    fig.add_trace(go.Scatter(
                        x=pd.concat([df_zone["forecast_timestamp"], df_zone["forecast_timestamp"][::-1]]),
                        y=pd.concat([df_zone["confidence_high"], df_zone["confidence_low"][::-1]]),
                        fill="toself", fillcolor=f"rgba({r},{g},{b},0.12)",
                        line=dict(width=0), hoverinfo="skip", showlegend=False,
                    ))
                fig.add_trace(go.Scatter(
                    x=df_zone["forecast_timestamp"], y=df_zone["predicted_aqi"],
                    mode="lines", line=dict(color=color, width=3, shape="spline"), name=zone,
                    connectgaps=True
                ))
                
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="IBM Plex Sans", color=theme.TEXT_SECONDARY),
                margin=dict(l=10, r=10, t=10, b=10), height=360, showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(gridcolor=theme.BORDER, showline=False, title=None),
                yaxis=dict(gridcolor=theme.BORDER, title="AQI", zeroline=False),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            with st.expander("Raw forecast data"):
                st.dataframe(forecast_df, use_container_width=True)

# ---------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------
with tab_alerts:
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown(theme.panel_heading("High-Severity Citizen Reports"), unsafe_allow_html=True)
            if high_severity.empty:
                st.markdown(theme.empty_state("No high-severity reports right now."), unsafe_allow_html=True)
            else:
                rows = []
                for _, r in high_severity.sort_values("severity", ascending=False).iterrows():
                    color = theme.get_severity_color(r["severity"])
                    title = f"{r.get('category', 'Report')} — {r.get('description') or r.get('detected_issue') or ''}"
                    ts = r["timestamp"]
                    ts_str = ts.strftime("%H:%M") if hasattr(ts, "strftime") else str(ts)
                    meta = f"Severity {int(r['severity'])} &middot; {ts_str}"
                    rows.append(theme.alert_row(title, meta, color))
                st.markdown("".join(rows), unsafe_allow_html=True)

    with col2:
        with st.container(border=True):
            st.markdown(theme.panel_heading("Stations Above AQI 200 (Unhealthy)"), unsafe_allow_html=True)
            if high_aqi.empty:
                st.markdown(theme.empty_state("No stations currently above 200."), unsafe_allow_html=True)
            else:
                rows = []
                for _, r in high_aqi.sort_values("aqi_value", ascending=False).iterrows():
                    _, color = theme.get_aqi_category(r["aqi_value"])
                    title = f"Station {r.get('station_id', '—')} &middot; AQI {r['aqi_value']:.0f}"
                    meta = f"{r.get('source', '')}"
                    rows.append(theme.alert_row(title, meta, color))
                st.markdown("".join(rows), unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Citizen report form
# ---------------------------------------------------------------------
with tab_report:
    st.markdown('<div class="eyebrow">New Submission</div>', unsafe_allow_html=True)
    st.markdown('<h3 style="margin:4px 0 16px 0;">Report an air quality issue</h3>', unsafe_allow_html=True)

    # Default fallback (Delhi NCR), matching the rest of the app.
    DEFAULT_LAT, DEFAULT_LON = 28.6139, 77.2090
    if "report_lat" not in st.session_state:
        st.session_state.report_lat = DEFAULT_LAT
        st.session_state.report_lon = DEFAULT_LON
        st.session_state.loc_version = 0

    def _set_report_location(lat, lon):
        """Updates the picked location and bumps loc_version — but only on
        an actual change. loc_version feeds into the number_input keys
        below so a real pick refreshes their displayed value, while a
        manual edit in between picks doesn't get silently overwritten
        (widgets inside st.form don't rerun until submit, so the location
        picker has to live outside the form and hand off through
        session_state instead)."""
        lat, lon = round(lat, 6), round(lon, 6)
        if (lat, lon) != (st.session_state.report_lat, st.session_state.report_lon):
            st.session_state.report_lat = lat
            st.session_state.report_lon = lon
            st.session_state.loc_version += 1
            st.rerun()

    with st.container(border=True):
        st.markdown(theme.panel_heading("Set Location"), unsafe_allow_html=True)
        loc_col1, loc_col2 = st.columns([1, 2])

        with loc_col1:
            st.caption("Use your device location:")
            geo = streamlit_geolocation()
            if geo and geo.get("latitude") is not None and geo.get("longitude") is not None:
                _set_report_location(geo["latitude"], geo["longitude"])
            st.caption(f"Selected: {st.session_state.report_lat:.5f}, {st.session_state.report_lon:.5f}")

        with loc_col2:
            st.caption("Or click the map to drop a pin:")
            m = folium.Map(
                location=[st.session_state.report_lat, st.session_state.report_lon],
                zoom_start=12,
                tiles=f"https://api.mapbox.com/styles/v1/mapbox/dark-v11/tiles/{{z}}/{{x}}/{{y}}?access_token={mb_key}",
                attr="Mapbox",
            )
            folium.Marker(
                [st.session_state.report_lat, st.session_state.report_lon],
                tooltip="Selected location",
            ).add_to(m)
            map_data = st_folium(m, height=280, use_container_width=True, key="report_location_map")
            if map_data and map_data.get("last_clicked"):
                _set_report_location(map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"])

    with st.form("citizen_report_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        # Keyed on loc_version so a geolocation/map pick refreshes these
        # fields, while a manual edit between picks is left alone.
        lat = col1.number_input(
            "Latitude", value=st.session_state.report_lat, format="%.6f",
            key=f"lat_input_{st.session_state.loc_version}",
        )
        lon = col2.number_input(
            "Longitude", value=st.session_state.report_lon, format="%.6f",
            key=f"lon_input_{st.session_state.loc_version}",
        )

        category = st.selectbox(
            "Category", ["Burning", "Vehicle Smoke", "Industrial", "Dust", "Construction", "Other"]
        )
        text = st.text_area("Describe what you're seeing")
        language = st.selectbox("Language", ["en", "hi", "other"])
        photo = st.file_uploader("Photo (optional)", type=["jpg", "jpeg", "png"])
        voice = st.file_uploader("Voice note (optional)", type=["mp3", "wav", "m4a"])
        submitted = st.form_submit_button("Submit report")

        if submitted:
            report = {
                "location": {"lat": lat, "lng": lon},
                "category": category,
                "text": text,
                "language": language,
                "timestamp": datetime.now(timezone.utc),
            }
            ds.submit_citizen_report(report, photo_file=photo, voice_file=voice)
            st.success("Report submitted.")
