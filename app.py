from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
import folium
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation

import data_sources as ds
import theme

st.set_page_config(page_title="Atmosphere Console", page_icon="🌫️", layout="wide")
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

avg_aqi = float(aqi_df["aqi_value"].mean()) if not aqi_df.empty else 0.0
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

# ---------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------
with tab_map:
    with st.container(border=True):
        st.markdown(theme.panel_heading("Live Map"), unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        show_aqi = col1.checkbox("AQI stations", value=True)
        show_fire = col2.checkbox("Fire hotspots", value=True)
        show_reports = col3.checkbox("Citizen reports", value=True)

        layers = []
        all_points = []  # (lon, lat) pairs across every visible, non-empty layer — used to auto-fit the view
        if show_aqi and not aqi_df.empty:
            aqi_plot = aqi_df.dropna(subset=["lat", "lon"]).copy()
            if not aqi_plot.empty:
                aqi_plot["color"] = aqi_plot["aqi_value"].apply(
                    lambda v: theme.hex_to_rgba(theme.get_aqi_category(v)[1], 190)
                )
                layers.append(pdk.Layer(
                    "ScatterplotLayer", data=aqi_plot, get_position="[lon, lat]",
                    get_radius=380, get_fill_color="color", stroked=True,
                    get_line_color=[18, 21, 26, 200], line_width_min_pixels=1, pickable=True,
                ))
                all_points.extend(zip(aqi_plot["lon"], aqi_plot["lat"]))
        if show_fire and not fire_df.empty:
            fire_plot = fire_df.dropna(subset=["lat", "lon"]).copy()
            if not fire_plot.empty:
                fire_plot["color"] = [theme.hex_to_rgba(theme.EMBER, 210) for _ in range(len(fire_plot))]
                layers.append(pdk.Layer(
                    "ScatterplotLayer", data=fire_plot, get_position="[lon, lat]",
                    get_radius=340, get_fill_color="color", stroked=True,
                    get_line_color=[18, 21, 26, 200], line_width_min_pixels=1, pickable=True,
                ))
                all_points.extend(zip(fire_plot["lon"], fire_plot["lat"]))
        if show_reports and not reports_df.empty:
            reports_plot = reports_df.dropna(subset=["lat", "lon"]).copy()
            if not reports_plot.empty:
                reports_plot["color"] = reports_plot["severity"].fillna(1).apply(
                    lambda s: theme.hex_to_rgba(theme.get_severity_color(s), 210)
                )
                layers.append(pdk.Layer(
                    "ScatterplotLayer", data=reports_plot, get_position="[lon, lat]",
                    get_radius=300, get_fill_color="color", stroked=True,
                    get_line_color=[18, 21, 26, 200], line_width_min_pixels=1, pickable=True,
                ))
                all_points.extend(zip(reports_plot["lon"], reports_plot["lat"]))

        if all_points:
            # Auto-fit the view to wherever the real data actually is, instead
            # of a fixed fallback point that only made sense for aqi_df.
            view_state = pdk.data_utils.compute_view(list(all_points))
            view_state.zoom = min(view_state.zoom, 12)  # don't over-zoom for a single point/tight cluster
        else:
            view_state = pdk.ViewState(latitude=28.6139, longitude=77.2090, zoom=10)

        st.pydeck_chart(pdk.Deck(
            map_style="dark",
            initial_view_state=view_state,
            layers=layers,
            tooltip={"text": "AQI/Severity data point"},
        ))
        st.markdown(theme.map_legend(), unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------
with tab_forecast:
    with st.container(border=True):
        st.markdown(theme.panel_heading("AQI Forecast"), unsafe_allow_html=True)
        forecast_df = ds.get_forecast()
        if forecast_df.empty:
            st.markdown(
                theme.empty_state("No forecast data yet — this fills in once Person 2's Vertex AI model runs (Day 2)."),
                unsafe_allow_html=True,
            )
        else:
            fig = go.Figure()
            if {"confidence_low", "confidence_high"}.issubset(forecast_df.columns):
                fig.add_trace(go.Scatter(
                    x=pd.concat([forecast_df["forecast_timestamp"], forecast_df["forecast_timestamp"][::-1]]),
                    y=pd.concat([forecast_df["confidence_high"], forecast_df["confidence_low"][::-1]]),
                    fill="toself", fillcolor="rgba(232,162,61,0.12)",
                    line=dict(width=0), hoverinfo="skip", showlegend=False,
                ))
            fig.add_trace(go.Scatter(
                x=forecast_df["forecast_timestamp"], y=forecast_df["predicted_aqi"],
                mode="lines", line=dict(color=theme.ACCENT, width=3, shape="spline"), name="Predicted AQI",
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="IBM Plex Sans", color=theme.TEXT_SECONDARY),
                margin=dict(l=10, r=10, t=10, b=10), height=360, showlegend=False,
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
                tiles="CartoDB dark_matter",
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
                "timestamp": datetime.now(),
            }
            ds.submit_citizen_report(report, photo_file=photo, voice_file=voice)
            st.success("Report submitted. Person 2's Gemini pipeline will pick it up on the next run.")
