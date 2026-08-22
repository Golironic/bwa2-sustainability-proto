"""
Design tokens and reusable UI components for the Air Quality Platform.

Concept: an "atmosphere console" — the dashboard reads like a live
instrument panel for the sky itself. The AQI color scale (the real
global standard for communicating air quality) is used functionally
everywhere severity appears, instead of an arbitrary brand accent.
Panels are treated as frosted glass catching ambient light from that
scale, echoing how haze itself scatters and softens light — that's the
justification for the blur/glow treatment below, not decoration for its
own sake. Everything else stays quiet so the scale carries the meaning.
"""

# ---------------------------------------------------------------------
# Color tokens
# ---------------------------------------------------------------------
BG_DEEP = "#101318"        # app background (top of gradient)
BG_DEEP_2 = "#171B22"      # app background (bottom of gradient)
BG_SURFACE = "#1B1F27"     # card / panel base (used under glass blur)
BG_RAISED = "#242A35"      # hover / active state
BORDER = "#2E3440"         # hairline dividers
BORDER_SOFT = "rgba(255,255,255,0.06)"   # glass panel top-edge highlight
TEXT_PRIMARY = "#ECE9E2"   # haze-diffused off-white
TEXT_SECONDARY = "#8E96A6"  # muted slate for labels/captions
ACCENT = "#E8A23D"         # sodium-streetlight amber — the one interactive accent
EMBER = "#FF6B4A"          # fire hotspots — distinct from the AQI scale on purpose

# The real AQI scale (EPA-style breakpoints), refined/desaturated to sit
# together as a cohesive palette rather than stock traffic-light colors.
GOOD = "#5FBF73"
MODERATE = "#E3C63D"
SENSITIVE = "#F0A94E"       # Unhealthy for Sensitive Groups
UNHEALTHY = "#E2634B"
VERY_UNHEALTHY = "#9B5DE0"
HAZARDOUS = "#7A3040"

BREAKPOINTS = [
    (50, "Good", GOOD),
    (100, "Moderate", MODERATE),
    (150, "Unhealthy (Sensitive)", SENSITIVE),
    (200, "Unhealthy", UNHEALTHY),
    (300, "Very Unhealthy", VERY_UNHEALTHY),
    (500, "Hazardous", HAZARDOUS),
]

SEVERITY_COLORS = {1: GOOD, 2: MODERATE, 3: SENSITIVE, 4: UNHEALTHY, 5: HAZARDOUS}

# Small hand-drawn line icons (24x24, stroke=currentColor) — kept generic
# and geometric so they read clearly at small sizes without relying on an
# external icon font or network request.
ICONS = {
    "gauge": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M4 15a8 8 0 1 1 16 0"/><path d="M12 15l4.2-5.2"/><circle cx="12" cy="15" r="1.4" fill="currentColor" stroke="none"/></svg>',
    "flame": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3c1 3-2.5 4-2.5 7.5a2.5 2.5 0 0 0 5 0c0-1.2-.6-2-1-2.8 2.4 1 4 3.4 4 6.1a5.5 5.5 0 0 1-11 0C6.5 9.8 9 7.3 12 3z"/></svg>',
    "report": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M4 5.5A1.5 1.5 0 0 1 5.5 4h13A1.5 1.5 0 0 1 20 5.5v9a1.5 1.5 0 0 1-1.5 1.5H9l-4 4v-4H5.5A1.5 1.5 0 0 1 4 14.5v-9z"/></svg>',
    "bell": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M6 10a6 6 0 1 1 12 0c0 3.2 1 4.6 2 5.5H4c1-.9 2-2.3 2-5.5z"/><path d="M10 19a2 2 0 0 0 4 0"/></svg>',
}


def get_aqi_category(value):
    """Returns (label, color) for a raw 0-500 AQI value."""
    if value is None:
        return "Unknown", TEXT_SECONDARY
    for limit, label, color in BREAKPOINTS:
        if value <= limit:
            return label, color
    return BREAKPOINTS[-1][1], BREAKPOINTS[-1][2]


def get_severity_color(severity):
    try:
        return SEVERITY_COLORS.get(int(severity), TEXT_SECONDARY)
    except (TypeError, ValueError):
        return TEXT_SECONDARY


def hex_to_rgba(hex_color, alpha=200):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return [r, g, b, alpha]


def _compact(html):
    """Collapses a multi-line HTML template into one unbroken line.

    Streamlit's markdown renderer treats indented text (or a blank/
    whitespace-only line) as a Markdown code block rather than raw HTML.
    Our templates are written multi-line and indented for readability in
    the source, and when several are concatenated (e.g. one card after
    another) a trailing whitespace-only line can slip in between them and
    get read as a blank-line separator. Stripping every line and joining
    with a single space removes both problems (blank lines and stray
    indentation) regardless of how the pieces get combined later, and a
    single space between block-level tags never affects layout.
    """
    return " ".join(line.strip() for line in html.strip().splitlines() if line.strip())


# ---------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------
CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

.stApp {{
  background:
    radial-gradient(ellipse 900px 500px at 15% -10%, rgba(232,162,61,0.10), transparent 60%),
    radial-gradient(ellipse 700px 500px at 100% 0%, rgba(155,93,224,0.08), transparent 55%),
    linear-gradient(180deg, {BG_DEEP} 0%, {BG_DEEP_2} 100%);
  color: {TEXT_PRIMARY};
  font-family: 'IBM Plex Sans', sans-serif;
}}

[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}

h1, h2, h3 {{
  font-family: 'Space Grotesk', sans-serif !important;
  letter-spacing: -0.01em;
  color: {TEXT_PRIMARY};
}}

[data-testid="stCaptionContainer"], .stCaption {{
  font-family: 'IBM Plex Mono', monospace !important;
  color: {TEXT_SECONDARY} !important;
  font-size: 0.8rem !important;
}}

/* Custom scrollbar */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 8px; }}
::-webkit-scrollbar-thumb:hover {{ background: {ACCENT}66; }}

/* Keyboard focus, kept visible per accessibility floor */
a:focus-visible, button:focus-visible, [tabindex]:focus-visible {{
  outline: 2px solid {ACCENT} !important;
  outline-offset: 2px;
}}

[data-testid="stSidebar"] {{
  background: {BG_SURFACE};
  border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] * {{ color: {TEXT_PRIMARY}; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
  gap: 4px;
  background: {BG_SURFACE}cc;
  backdrop-filter: blur(12px);
  padding: 6px;
  border-radius: 12px;
  border: 1px solid {BORDER};
}}
.stTabs [data-baseweb="tab"] {{
  height: 42px;
  border-radius: 8px;
  color: {TEXT_SECONDARY};
  font-weight: 500;
  background: transparent;
  transition: color 0.15s ease, background 0.15s ease;
}}
.stTabs [data-baseweb="tab"]:hover {{ color: {TEXT_PRIMARY}; }}
.stTabs [aria-selected="true"] {{
  background: {BG_RAISED} !important;
  color: {ACCENT} !important;
  box-shadow: inset 0 0 0 1px {ACCENT}33;
}}
.stTabs [data-baseweb="tab-highlight"] {{ background: transparent !important; }}
.stTabs [data-baseweb="tab-border"] {{ display: none; }}

/* Buttons */
.stButton > button, [data-testid="stFormSubmitButton"] button {{
  background: {ACCENT};
  color: #1A1200;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  padding: 0.5rem 1.25rem;
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
.stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {{
  transform: translateY(-1px);
  box-shadow: 0 6px 18px {ACCENT}55;
  color: #1A1200;
}}
.stButton > button:active, [data-testid="stFormSubmitButton"] button:active {{
  transform: translateY(0);
}}

/* Inputs */
.stTextArea textarea, .stNumberInput input,
.stSelectbox > div > div, [data-testid="stFileUploaderDropzone"] {{
  background: {BG_SURFACE} !important;
  border: 1px solid {BORDER} !important;
  color: {TEXT_PRIMARY} !important;
  border-radius: 8px !important;
  transition: border-color 0.15s ease;
}}
.stTextArea textarea:focus, .stNumberInput input:focus {{
  border-color: {ACCENT}88 !important;
}}
label, [data-testid="stWidgetLabel"] p {{ color: {TEXT_SECONDARY} !important; }}

/* Glass panel treatment — shared by forms, bordered containers, dataframes */
[data-testid="stForm"],
[data-testid="stVerticalBlockBorderWrapper"] {{
  background: {BG_SURFACE}b3;
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
  border: 1px solid {BORDER_SOFT};
  border-radius: 16px;
  box-shadow: 0 12px 34px rgba(0,0,0,0.32), inset 0 1px 0 {BORDER_SOFT};
}}
[data-testid="stForm"] {{ padding: 1.75rem; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ padding: 0.25rem; }}

/* Dataframes */
[data-testid="stDataFrame"] {{
  border: 1px solid {BORDER};
  border-radius: 12px;
  overflow: hidden;
}}

/* Checkboxes */
[data-testid="stCheckbox"] label p {{ color: {TEXT_PRIMARY} !important; font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; }}

/* --- Custom components --- */
.eyebrow {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: {TEXT_SECONDARY};
}}

.hero-title {{
  background: linear-gradient(90deg, {TEXT_PRIMARY} 30%, {ACCENT}dd 100%);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 6px 0 4px 0;
  font-size: 2.3rem;
}}

.live-pulse {{ display: inline-flex; align-items: center; gap: 7px; }}
.live-pulse-dot {{ position: relative; width: 7px; height: 7px; border-radius: 50%; background: {GOOD}; }}
.live-pulse-dot::after {{
  content: ''; position: absolute; inset: -5px; border-radius: 50%;
  border: 1.5px solid {GOOD}; animation: ping 2.2s cubic-bezier(0,0,0.2,1) infinite;
}}
@keyframes ping {{
  0% {{ transform: scale(0.6); opacity: 0.9; }}
  75%, 100% {{ transform: scale(1.9); opacity: 0; }}
}}
@media (prefers-reduced-motion: reduce) {{
  .live-pulse-dot::after {{ animation: none; }}
}}

.stat-row {{ display: flex; gap: 14px; flex-wrap: wrap; margin: 0.75rem 0 1.75rem 0; }}
.stat-card {{
  flex: 1 1 170px;
  position: relative;
  overflow: hidden;
  background: {BG_SURFACE}b3;
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
  border: 1px solid {BORDER_SOFT};
  border-radius: 14px;
  padding: 16px 18px;
  box-shadow: 0 10px 28px rgba(0,0,0,0.28), inset 0 1px 0 {BORDER_SOFT};
  transition: transform 0.18s ease, box-shadow 0.18s ease;
}}
.stat-card:hover {{
  transform: translateY(-3px);
  box-shadow: 0 16px 36px rgba(0,0,0,0.36), inset 0 1px 0 {BORDER_SOFT};
}}
.stat-card-accent {{
  position: absolute; top: 0; left: 0; right: 0; height: 3px;
}}
.stat-card-top {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }}
.stat-card-icon {{ display: flex; align-items: center; justify-content: center; opacity: 0.85; }}
.stat-card-label {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.68rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: {TEXT_SECONDARY};
}}
.stat-card-value {{
  font-family: 'Space Grotesk', sans-serif;
  font-size: 2.1rem;
  font-weight: 600;
  color: {TEXT_PRIMARY};
  line-height: 1.1;
}}
.stat-card-sub {{ font-size: 0.78rem; color: {TEXT_SECONDARY}; margin-top: 5px; }}

.sky-strip-wrap {{
  background: {BG_SURFACE}b3;
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
  border: 1px solid {BORDER_SOFT};
  border-radius: 16px;
  padding: 20px 24px 16px 24px;
  margin: 0.5rem 0 1.75rem 0;
  box-shadow: 0 12px 34px rgba(0,0,0,0.32), inset 0 1px 0 {BORDER_SOFT};
}}
.sky-strip-header {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 16px; flex-wrap: wrap; gap: 8px; }}
.sky-strip-eyebrow {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; color: {TEXT_SECONDARY}; }}
.sky-strip-value {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.8rem; font-weight: 700; }}
.sky-strip-label {{ font-family: 'IBM Plex Sans', sans-serif; font-size: 0.95rem; font-weight: 500; color: {TEXT_SECONDARY}; margin-left: 8px; }}
.sky-strip {{
  position: relative;
  height: 14px;
  border-radius: 999px;
  background: linear-gradient(90deg, {GOOD} 0%, {MODERATE} 20%, {SENSITIVE} 30%, {UNHEALTHY} 40%, {VERY_UNHEALTHY} 60%, {HAZARDOUS} 100%);
  box-shadow: inset 0 1px 3px rgba(0,0,0,0.35);
}}
.sky-strip-marker {{ position: absolute; top: -8px; transform: translateX(-50%); }}
.sky-strip-marker-dot {{
  width: 18px; height: 18px; border-radius: 50%; border: 3px solid {BG_SURFACE};
  animation: marker-pulse 2.6s ease-in-out infinite;
}}
@keyframes marker-pulse {{
  0%, 100% {{ filter: brightness(1); }}
  50% {{ filter: brightness(1.25); }}
}}
@media (prefers-reduced-motion: reduce) {{
  .sky-strip-marker-dot {{ animation: none; }}
}}
.sky-strip-ticks {{ position: relative; height: 18px; margin-top: 8px; }}
.sky-strip-ticks span {{
  position: absolute; transform: translateX(-50%);
  font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem; color: {TEXT_SECONDARY};
}}

.panel-heading {{ display: flex; align-items: center; justify-content: space-between; padding: 14px 18px 4px 18px; flex-wrap: wrap; gap: 8px; }}

.map-legend {{ display: flex; gap: 18px; flex-wrap: wrap; align-items: center; padding: 10px 18px 16px 18px; }}
.map-legend-item {{ display: flex; align-items: center; gap: 6px; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: {TEXT_SECONDARY}; }}
.map-legend-dot {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; }}

.chip {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  padding: 3px 10px;
  border-radius: 999px;
  white-space: nowrap;
  display: inline-block;
}}

.alert-row {{
  display: flex; gap: 12px; align-items: center;
  background: {BG_SURFACE}b3;
  backdrop-filter: blur(14px);
  border: 1px solid {BORDER_SOFT};
  border-radius: 12px;
  padding: 12px 16px; margin-bottom: 8px;
  transition: transform 0.15s ease, border-color 0.15s ease;
}}
.alert-row:hover {{ transform: translateX(2px); border-color: {BORDER}; }}
.alert-row-dot {{
  width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0;
}}
.alert-row-body {{ flex: 1; }}
.alert-row-title {{ font-weight: 500; color: {TEXT_PRIMARY}; font-size: 0.92rem; }}
.alert-row-meta {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: {TEXT_SECONDARY}; margin-top: 3px; }}

.empty-state {{
  border: 1px dashed {BORDER}; border-radius: 12px; padding: 28px;
  text-align: center; color: {TEXT_SECONDARY}; font-size: 0.88rem;
}}

.status-dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; }}

.section-divider {{ border: none; border-top: 1px solid {BORDER}; margin: 1.1rem 0; }}
</style>
"""

# Tick positions (percent along the 0-500 scale) matching the gradient stops
_TICKS = [(0, "0"), (10, "50"), (20, "100"), (30, "150"), (40, "200"), (60, "300"), (100, "500")]


def render_sky_strip(avg_aqi):
    pct = max(0, min(100, (avg_aqi or 0) / 500 * 100))
    label, color = get_aqi_category(avg_aqi)
    ticks_html = "".join(
        f'<span style="left:{p}%;">{t}</span>' for p, t in _TICKS
    )
    return _compact(f"""
    <div class="sky-strip-wrap">
      <div class="sky-strip-header">
        <span class="sky-strip-eyebrow">Current Atmosphere &middot; Regional Average</span>
        <span class="sky-strip-value" style="color:{color};">{avg_aqi:.0f}<span class="sky-strip-label">{label}</span></span>
      </div>
      <div class="sky-strip">
        <div class="sky-strip-marker" style="left:{pct}%;">
          <div class="sky-strip-marker-dot" style="background:{color}; box-shadow: 0 0 16px 4px {color}99;"></div>
        </div>
      </div>
      <div class="sky-strip-ticks">{ticks_html}</div>
    </div>
    """)


def live_pulse(label="LIVE", color=None):
    color = color or GOOD
    return _compact(f"""
    <span class="live-pulse">
      <span class="live-pulse-dot" style="background:{color};"></span>
      <span class="eyebrow">{label}</span>
    </span>
    """)


def stat_card(label, value, sublabel="", color=None, icon=None):
    color = color or ACCENT
    icon_html = ICONS.get(icon, "") if icon else ""
    return _compact(f"""
    <div class="stat-card">
      <div class="stat-card-accent" style="background:{color};"></div>
      <div class="stat-card-top">
        <div class="stat-card-label">{label}</div>
        <div class="stat-card-icon" style="color:{color};">{icon_html}</div>
      </div>
      <div class="stat-card-value" style="color:{color};">{value}</div>
      <div class="stat-card-sub">{sublabel}</div>
    </div>
    """)


def stat_row(cards_html):
    return _compact(f'<div class="stat-row">{"".join(cards_html)}</div>')


def severity_chip(severity):
    color = get_severity_color(severity)
    try:
        sev = int(severity)
        return _compact(f'<span class="chip" style="background:{color}22;color:{color};border:1px solid {color}55;">Severity {sev}</span>')
    except (TypeError, ValueError):
        return _compact(f'<span class="chip" style="background:{TEXT_SECONDARY}22;color:{TEXT_SECONDARY};">Unrated</span>')


def alert_row(title, meta, color):
    return _compact(f"""
    <div class="alert-row">
      <div class="alert-row-dot" style="background:{color}; box-shadow: 0 0 8px 1px {color}88;"></div>
      <div class="alert-row-body">
        <div class="alert-row-title">{title}</div>
        <div class="alert-row-meta">{meta}</div>
      </div>
    </div>
    """)


def empty_state(text):
    return _compact(f'<div class="empty-state">{text}</div>')


def panel_heading(title):
    return _compact(f'<div class="panel-heading"><span class="eyebrow">{title}</span></div>')


def map_legend():
    items = [
        ("AQI / Report severity", None),
        ("Good", GOOD), ("Moderate", MODERATE), ("Sensitive", SENSITIVE),
        ("Unhealthy", UNHEALTHY), ("Very Unhealthy", VERY_UNHEALTHY), ("Hazardous", HAZARDOUS),
        ("Fire hotspot", EMBER),
    ]
    parts = []
    for label, color in items:
        if color is None:
            parts.append(f'<span class="map-legend-item" style="color:{TEXT_PRIMARY};font-weight:500;">{label}</span>')
        else:
            parts.append(f'<span class="map-legend-item"><span class="map-legend-dot" style="background:{color};"></span>{label}</span>')
    return _compact(f'<div class="map-legend">{"".join(parts)}</div>')
