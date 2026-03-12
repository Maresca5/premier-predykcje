"""
data.py – ładowanie i preprocessing danych (CSV, API, harmonogram).
"""
import pandas as pd
import numpy as np
import requests
import sqlite3
import unicodedata
import re
import os
from io import StringIO
from datetime import datetime
import streamlit as st

from config import LIGI, DB_FILE, BIEZACY_SEZON

def waga_poprzedniego(n_biezacy: int) -> float:
    return float(np.clip(0.8 - (n_biezacy / 30) * 0.6, 0.2, 0.8))

st.set_page_config(page_title="⚽ Football Analytics", page_icon="⚽", layout="wide", initial_sidebar_state="expanded")


# ── Global CSS ────────────────────────────────────────────
st.markdown("""
<style>
/* ═══════════════════════════════════════════════════════════
   CSS CUSTOM PROPERTIES
   Streamlit ustawia [data-theme="light"|"dark"] na <body>
   Domyślnie: dark
   ═══════════════════════════════════════════════════════════ */
:root, body[data-theme="dark"], .stApp[data-theme="dark"] {
    --bg-app:        #0d0f14;
    --bg-sidebar:    #111318;
    --bg-card:       #14161c;
    --bg-card2:      #1a1c24;
    --bg-input:      #1e2028;
    --border:        #1e2028;
    --border-light:  #252d40;
    --text-primary:  #e8e8e8;
    --text-secondary:#ccc;
    --text-muted:    #888;
    --text-dim:      #666;
    --text-dimmer:   #444;
    --accent:        #4CAF50;
    --accent-dim:    rgba(76,175,80,0.07);
    --accent-bg:     #0d1f0d;
    --accent-border: #2e7d32;
    --danger:        #F44336;
    --warn:          #FF9800;
    --info:          #2196F3;
    --expander-bg:   linear-gradient(90deg, #141824 0%, #0d0f14 100%);
    --btn-bg:        linear-gradient(135deg, #1a4a1a, #2a6b2a);
    --btn-hover:     linear-gradient(135deg, #2a6b2a, #388e3c);
    --metric-bg:     #14161c;
    --vb-card-bg:    linear-gradient(145deg, #0d1f0d, #111a11);
    --pill-bg:       #14161c;
    --onboard-bg:    linear-gradient(135deg, #0d1f3a, #0d1a0d);
    --onboard-h:     #64b5f6;
    --onboard-p:     #8899aa;
    --rynek-row-bdr: #1a1a24;
    --rynek-bg:      #1a1a24;
    --section-text:  #e0e0e0;
    --table-bg:      #14161c;
    --table-header:  #1a1c24;
    --table-text:    #e8e8e8;
    --table-border:  #252d40;
    --date-bar-bg:   #1a1c24;
    --match-card-bg: #14161c;
    --badge-text:    #ffffff;
    --hero-bg:       #0d1f0d;
    --stat-bg:       #0e1117;
    --inline-code-bg:#1e2028;
}

body[data-theme="light"], .stApp[data-theme="light"] {
    --bg-app:        #f5f6fa;
    --bg-sidebar:    #ffffff;
    --bg-card:       #ffffff;
    --bg-card2:      #f0f2f6;
    --bg-input:      #e8eaf0;
    --border:        #d1d5e0;
    --border-light:  #c4cad8;
    --text-primary:  #111318;
    --text-secondary:#2c2f3a;
    --text-muted:    #5a6070;
    --text-dim:      #7a8090;
    --text-dimmer:   #aab0bc;
    --accent:        #2e7d32;
    --accent-dim:    rgba(46,125,50,0.08);
    --accent-bg:     #f0f9f0;
    --accent-border: #4CAF50;
    --danger:        #c62828;
    --warn:          #e65100;
    --info:          #1565c0;
    --expander-bg:   linear-gradient(90deg, #f0f2f8 0%, #f5f6fa 100%);
    --btn-bg:        linear-gradient(135deg, #2e7d32, #388e3c);
    --btn-hover:     linear-gradient(135deg, #388e3c, #43a047);
    --metric-bg:     #ffffff;
    --vb-card-bg:    linear-gradient(145deg, #f0f9f0, #f5fff5);
    --pill-bg:       #f0f2f6;
    --onboard-bg:    linear-gradient(135deg, #e3f0ff, #f0fff0);
    --onboard-h:     #1565c0;
    --onboard-p:     #445566;
    --rynek-row-bdr: #e0e4ef;
    --rynek-bg:      #eef0f8;
    --section-text:  #111318;
    --table-bg:      #ffffff;
    --table-header:  #f0f2f6;
    --table-text:    #111318;
    --table-border:  #d1d5e0;
    --date-bar-bg:   #e8eaf0;
    --match-card-bg: #ffffff;
    --badge-text:    #ffffff;
    --hero-bg:       #f0f9f0;
    --stat-bg:       #f5f6fa;
    --inline-code-bg:#e8eaf0;
}

/* ── Tło i typografia ── */
[data-testid="stAppViewContainer"] { background: var(--bg-app) !important; }
[data-testid="stSidebar"]          { background: var(--bg-sidebar) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
h1, h2, h3 { font-family: 'Inter', sans-serif; letter-spacing: -0.02em; color: var(--text-primary) !important; }
p, li, span, label { color: var(--text-primary); }

/* ── Główna treść ── */
[data-testid="stMain"] { background: var(--bg-app) !important; }
[data-testid="stMain"] * { color: inherit; }

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-size: 0.82em; font-weight: 500; padding: 6px 14px;
    border-radius: 6px 6px 0 0; color: var(--text-dim) !important; transition: all .2s;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important;
    background: var(--accent-dim) !important;
}
[data-testid="stTabs"] button:hover { color: var(--text-muted) !important; background: rgba(128,128,128,0.06) !important; }

/* ── Metryki ── */
[data-testid="stMetric"] {
    background: var(--metric-bg) !important; border: 1px solid var(--border) !important;
    border-radius: 10px !important; padding: 14px 16px !important;
}
[data-testid="stMetricLabel"] { font-size: 0.72em !important; color: var(--text-dim) !important; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stMetricValue"] { font-size: 1.55em !important; font-weight: 700 !important; color: var(--text-primary) !important; }
[data-testid="stMetricDelta"]  { font-size: 0.78em !important; }

/* ── Przyciski ── */
[data-testid="stButton"] > button {
    background: var(--btn-bg) !important;
    color: #fff !important; border: 1px solid var(--accent-border) !important; border-radius: 8px !important;
    font-weight: 600 !important; transition: all .2s !important;
}
[data-testid="stButton"] > button:hover { background: var(--btn-hover) !important; }

/* ── Tabele dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 8px !important; }
[data-testid="stDataFrame"] table { background: var(--table-bg) !important; color: var(--table-text) !important; }
[data-testid="stDataFrame"] th { background: var(--table-header) !important; color: var(--table-text) !important; border-color: var(--table-border) !important; }
[data-testid="stDataFrame"] td { background: var(--table-bg) !important; color: var(--table-text) !important; border-color: var(--table-border) !important; }

/* ── Selectbox / inputs ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div,
[data-testid="stTextInput"] > div > div > input {
    background: var(--bg-input) !important;
    color: var(--text-primary) !important;
    border-color: var(--border) !important;
}
[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label,
[data-testid="stTextInput"] label,
[data-testid="stSlider"] label,
[data-testid="stNumberInput"] label,
[data-testid="stRadio"] label { color: var(--text-secondary) !important; }

/* ── Toggle ── */
[data-testid="stToggle"] label { color: var(--text-secondary) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 20px 0 !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] .stProgress > div > div { background: var(--accent) !important; }

/* ── Expander ── */
details { border: 1px solid var(--border) !important; border-radius: 8px !important; background: var(--bg-card) !important; }
details summary { font-weight: 600 !important; color: var(--text-secondary) !important; }
[data-testid="stExpander"] details summary {
    background: var(--expander-bg) !important;
    border-left: 2px solid var(--border-light) !important;
    padding-left: 8px !important;
    border-radius: 4px !important;
    transition: border-color 0.2s;
}
[data-testid="stExpander"] details[open] summary { border-left-color: var(--info) !important; }
[data-testid="stExpander"] details summary p { color: var(--text-secondary) !important; }

/* ── Caption / info / warning / error boxes ── */
[data-testid="stCaptionContainer"] { color: var(--text-muted) !important; }
[data-testid="stInfo"]    { background: color-mix(in srgb, var(--info) 10%, var(--bg-card)) !important; border-color: var(--info) !important; color: var(--text-primary) !important; }
[data-testid="stWarning"] { background: color-mix(in srgb, var(--warn) 10%, var(--bg-card)) !important; border-color: var(--warn) !important; color: var(--text-primary) !important; }
[data-testid="stError"]   { background: color-mix(in srgb, var(--danger) 10%, var(--bg-card)) !important; border-color: var(--danger) !important; color: var(--text-primary) !important; }
[data-testid="stSuccess"] { background: color-mix(in srgb, var(--accent) 10%, var(--bg-card)) !important; border-color: var(--accent) !important; color: var(--text-primary) !important; }

/* ── Tooltips helper ── */
.tooltip-label {
    display: inline-block; font-size: 0.72em; font-weight: 600;
    color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.06em;
    border-bottom: 1px dashed var(--border-light); cursor: help; margin-bottom: 2px;
}

/* ── Hero value bet card ── */
.vb-card {
    background: var(--vb-card-bg);
    border: 1px solid var(--accent-border); border-radius: 12px;
    padding: 16px 14px; text-align: center;
    transition: transform .2s, border-color .2s;
}
.vb-card:hover { transform: translateY(-2px); border-color: var(--accent); }
.vb-card .date   { font-size: 0.7em; color: var(--text-dim); margin-bottom: 4px; }
.vb-card .match  { font-size: 0.88em; font-weight: 700; color: var(--text-primary); margin-bottom: 8px; line-height: 1.3; }
.vb-card .bet    { font-size: 1.15em; font-weight: 800; color: var(--accent); }
.vb-card .meta   { font-size: 0.78em; color: var(--text-muted); margin-top: 5px; }
.vb-card .ev     { color: var(--accent); font-weight: 700; }
.vb-card .kelly  { font-size: 0.76em; color: var(--accent); margin-top: 4px; }

/* ── Stat pill ── */
.stat-pill {
    display: inline-block; background: var(--pill-bg); border: 1px solid var(--border);
    border-radius: 20px; padding: 3px 10px; font-size: 0.75em;
    color: var(--text-muted); margin: 2px;
}

/* ── Section header ── */
.section-header {
    font-size: 1.1em; font-weight: 700; color: var(--section-text);
    border-left: 3px solid var(--accent); padding-left: 10px; margin: 18px 0 12px 0;
}

/* ── Onboarding banner ── */
.onboard-banner {
    background: var(--onboard-bg);
    border: 1px solid var(--border-light); border-radius: 12px; padding: 20px 24px; margin-bottom: 20px;
}
.onboard-banner h3 { color: var(--onboard-h) !important; margin: 0 0 10px 0; font-size: 1.05em; }
.onboard-banner p  { color: var(--onboard-p) !important; font-size: 0.83em; margin: 0; line-height: 1.6; }

/* ── Hit rate badge ── */
.hr-badge-green  { color: var(--accent); font-weight: 700; }
.hr-badge-orange { color: var(--warn); font-weight: 700; }
.hr-badge-red    { color: var(--danger); font-weight: 700; }
.hr-badge-gray   { color: var(--text-dim); }

/* ── Rynek row ── */
.rynek-row { display: flex; align-items: center; gap: 8px; padding: 10px 0; border-bottom: 1px solid var(--rynek-row-bdr); }
.rynek-name { font-weight: 600; color: var(--text-primary); min-width: 140px; font-size: 0.88em; }
.rynek-bar-wrap { flex: 1; background: var(--rynek-bg); border-radius: 4px; height: 8px; overflow: hidden; }
.rynek-bar { height: 8px; border-radius: 4px; }
.rynek-stats { font-size: 0.75em; color: var(--text-dim); min-width: 120px; text-align: right; }

/* ── Date bar (nagłówki sobota/niedziela w liście meczy) ── */
.date-bar {
    background: var(--date-bar-bg) !important;
    color: var(--text-secondary) !important;
    padding: 4px 10px; border-radius: 6px; font-size: 0.8em;
    font-weight: 600; margin: 10px 0 4px 0;
}

/* ── Match card ── */
.match-card {
    background: var(--match-card-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}
.match-card * { color: inherit !important; }

/* ── Stat table (xG, SOT itp.) ── */
.stat-table { background: var(--stat-bg) !important; border-radius: 8px; }
.stat-table td, .stat-table th { color: var(--table-text) !important; background: transparent !important; }

/* ── inline code ── */
code { background: var(--inline-code-bg) !important; color: var(--text-secondary) !important; padding: 1px 5px; border-radius: 3px; }

/* ── Plotly / Altair chart backgrounds ── */
[data-testid="stPlotlyChart"] { background: transparent !important; }

/* ── Wszelkie niezdefiniowane białe tła w light mode ── */
body[data-theme="light"] .element-container,
body[data-theme="light"] [data-testid="stVerticalBlock"] {
    background: transparent !important;
}

/* ── Streamlit domyślny tekst ── */
body[data-theme="light"] .stMarkdown,
body[data-theme="light"] .stText,
body[data-theme="light"] p {
    color: var(--text-primary) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Auto-theme: Streamlit ustawia data-theme na body automatycznie ──────────
# Nasze CSS używa body[data-theme="light/dark"] - nie potrzebujemy JS.


# ===========================================================================

def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return str(name)
    name = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode("ASCII")
    for suffix in [" FC", " CF", " UD", " CD", " RCD", " AS", " Stade", " 1."]:
        name = name.replace(suffix, "")
    return name.strip()

NAZWY_MAP = {
    # ══════════════════════════════════════════════════════
    # PREMIER LEAGUE 2025/26
    # co.uk CSV → "Arsenal", "Man City", "Nott'm Forest" itd.
    # fd.org API shortName → mapped to co.uk value
    # ══════════════════════════════════════════════════════
    # Pełne nazwy (fd.org "name" field)
    "Arsenal FC":                    "Arsenal",
    "Aston Villa FC":                "Aston Villa",
    "AFC Bournemouth":               "Bournemouth",
    "Brentford FC":                  "Brentford",
    "Brighton & Hove Albion FC":     "Brighton",
    "Brighton & Hove Albion":        "Brighton",
    "Burnley FC":                    "Burnley",
    "Chelsea FC":                    "Chelsea",
    "Crystal Palace FC":             "Crystal Palace",
    "Everton FC":                    "Everton",
    "Fulham FC":                     "Fulham",
    "Leeds United FC":               "Leeds",
    "Leeds United":                  "Leeds",
    "Liverpool FC":                  "Liverpool",
    "Manchester City FC":            "Man City",
    "Manchester City":               "Man City",
    "Manchester United FC":          "Man United",
    "Manchester United":             "Man United",
    "Newcastle United FC":           "Newcastle",
    "Newcastle United":              "Newcastle",
    "Nottingham Forest FC":          "Nott'm Forest",
    "Nottingham Forest":             "Nott'm Forest",
    "Sunderland AFC":                "Sunderland",
    "Tottenham Hotspur FC":          "Tottenham",
    "Tottenham Hotspur":             "Tottenham",
    "West Ham United FC":            "West Ham",
    "West Ham United":               "West Ham",
    "Wolverhampton Wanderers FC":    "Wolves",
    "Wolverhampton Wanderers":       "Wolves",
    "Wolverhampton":                 "Wolves",
    # fd.org shortName (to co API faktycznie zwraca w polu "shortName")
    "Nottm Forest":                  "Nott'm Forest",
    "Newcastle Utd":                 "Newcastle",
    "Spurs":                         "Tottenham",
    "Man Utd":                       "Man United",
    "Man City":                      "Man City",
    "Leeds Utd":                     "Leeds",
    "West Ham Utd":                  "West Ham",
    "Wolves FC":                     "Wolves",
    "Wolves":                        "Wolves",
    # Drużyny które mogą pojawić się ze starego sezonu (2024/25)
    "Ipswich Town FC":               "Ipswich",
    "Ipswich Town":                  "Ipswich",
    "Ipswich":                       "Ipswich",
    "Southampton FC":                "Southampton",
    "Leicester City FC":             "Leicester",
    "Leicester City":                "Leicester",
    "Sheffield United FC":           "Sheffield Utd",
    "Sheffield United":              "Sheffield Utd",
    "Sheffield Utd":                 "Sheffield Utd",
    # Inne warianty EPL
    "Norwich City FC":               "Norwich",
    "Norwich City":                  "Norwich",
    "Watford FC":                    "Watford",
    "Middlesbrough FC":              "Middlesbrough",
    "Luton Town FC":                 "Luton",
    "Luton Town":                    "Luton",
    "Coventry City FC":              "Coventry",

    # ══════════════════════════════════════════════════════
    # LA LIGA
    # co.uk: "Barcelona", "Ath Madrid", "Ath Bilbao", "Sociedad",
    #        "Betis", "Celta", "Espanol", "Alaves", "Vallecano" itd.
    # ══════════════════════════════════════════════════════
    "FC Barcelona":                  "Barcelona",
    "Real Madrid CF":                "Real Madrid",
    "Atlético de Madrid":            "Ath Madrid",
    "Atlético Madrid":               "Ath Madrid",
    "Atletico Madrid":               "Ath Madrid",
    "Club Atlético de Madrid":       "Ath Madrid",
    "Athletic Club":                 "Ath Bilbao",
    "Athletic Bilbao":               "Ath Bilbao",
    "Athletic Club Bilbao":          "Ath Bilbao",
    "Real Sociedad de Fútbol":       "Sociedad",
    "Real Sociedad":                 "Sociedad",
    "Real Betis Balompié":           "Betis",
    "Real Betis":                    "Betis",
    "RC Celta de Vigo":              "Celta",
    "Celta Vigo":                    "Celta",
    "RCD Espanyol":                  "Espanol",
    "Espanyol":                      "Espanol",
    "Deportivo Alavés":              "Alaves",
    "Deportivo Alaves":              "Alaves",
    "Alavés":                        "Alaves",
    "Rayo Vallecano de Madrid":      "Vallecano",
    "Rayo Vallecano":                "Vallecano",
    "Getafe CF":                     "Getafe",
    "Sevilla FC":                    "Sevilla",
    "Villarreal CF":                 "Villarreal",
    "RCD Mallorca":                  "Mallorca",
    "Real Valladolid CF":            "Valladolid",
    "Real Valladolid":               "Valladolid",
    "UD Las Palmas":                 "Las Palmas",
    "Girona FC":                     "Girona",
    "Cadiz CF":                      "Cadiz",
    "Cadiz":                         "Cadiz",
    "Granada CF":                    "Granada",
    "UD Almeria":                    "Almeria",
    "Levante UD":                    "Levante",
    "Elche CF":                      "Elche",
    "CA Osasuna":                    "Osasuna",
    "Valencia CF":                   "Valencia",
    "Real Oviedo":                   "Oviedo",
    "CD Leganés":                    "Leganes",
    "Leganés":                       "Leganes",
    "Real Valladolid":               "Valladolid",
    "Deportivo La Coruña":           "La Coruna",
    "RC Deportivo":                  "La Coruna",

    # ══════════════════════════════════════════════════════
    # BUNDESLIGA
    # co.uk: "Bayern Munich", "Dortmund", "RB Leipzig", "Leverkusen",
    #        "Ein Frankfurt", "M'gladbach", "FC Koln", "St Pauli" itd.
    # ══════════════════════════════════════════════════════
    "FC Bayern München":             "Bayern Munich",
    "FC Bayern Munich":              "Bayern Munich",
    "Bayern München":                "Bayern Munich",
    "Bayern":                        "Bayern Munich",
    "Borussia Dortmund":             "Dortmund",
    "RB Leipzig":                    "RB Leipzig",
    "Leipzig":                       "RB Leipzig",
    "Bayer 04 Leverkusen":           "Leverkusen",
    "Bayer Leverkusen":              "Leverkusen",
    "Eintracht Frankfurt":           "Ein Frankfurt",
    "VfB Stuttgart":                 "Stuttgart",
    "VfL Wolfsburg":                 "Wolfsburg",
    "Borussia Mönchengladbach":      "M'gladbach",
    "Borussia Monchengladbach":      "M'gladbach",
    "Borussia M'gladbach":          "M'gladbach",
    "M'gladbach":                   "M'gladbach",
    "1. FC Union Berlin":            "Union Berlin",
    "SC Freiburg":                   "Freiburg",
    "1. FC Köln":                    "FC Koln",
    "FC Köln":                       "FC Koln",
    "FC Koln":                       "FC Koln",
    "1. FSV Mainz 05":               "Mainz",
    "FSV Mainz 05":                  "Mainz",
    "Mainz 05":                      "Mainz",
    "TSG Hoffenheim":                "Hoffenheim",
    "TSG 1899 Hoffenheim":           "Hoffenheim",
    "SV Werder Bremen":              "Werder Bremen",
    "Werder Bremen":                 "Werder Bremen",
    "VfL Bochum":                    "Bochum",
    "FC Augsburg":                   "Augsburg",
    "FC St. Pauli":                  "St Pauli",
    "St. Pauli":                     "St Pauli",
    "1. FC Heidenheim 1846":         "Heidenheim",
    "1. FC Heidenheim":              "Heidenheim",
    "Hamburger SV":                  "Hamburg",
    "HSV":                           "Hamburg",
    "Holstein Kiel":                 "Kiel",
    "SC Paderborn 07":               "Paderborn",
    "SpVgg Greuther Fürth":          "Greuther Furth",
    "Hertha BSC":                    "Hertha",
    "Hertha Berlin":                 "Hertha",
    "Fortuna Düsseldorf":            "Fortuna Dusseldorf",
    "SV Darmstadt 98":               "Darmstadt",

    # ══════════════════════════════════════════════════════
    # SERIE A
    # co.uk: "Milan", "Inter", "Juventus", "Napoli", "Roma" itd.
    # ══════════════════════════════════════════════════════
    "AC Milan":                      "Milan",
    "Internazionale":                "Inter",
    "FC Internazionale Milano":      "Inter",
    "Inter Milan":                   "Inter",
    "Juventus FC":                   "Juventus",
    "SSC Napoli":                    "Napoli",
    "AS Roma":                       "Roma",
    "SS Lazio":                      "Lazio",
    "Atalanta BC":                   "Atalanta",
    "ACF Fiorentina":                "Fiorentina",
    "Fiorentina":                    "Fiorentina",
    "Torino FC":                     "Torino",
    "Udinese Calcio":                "Udinese",
    "Bologna FC 1909":               "Bologna",
    "Bologna":                       "Bologna",
    "Empoli FC":                     "Empoli",
    "US Lecce":                      "Lecce",
    "Genoa CFC":                     "Genoa",
    "Cagliari Calcio":               "Cagliari",
    "Venezia FC":                    "Venezia",
    "Parma Calcio 1913":             "Parma",
    "Como 1907":                     "Como",
    "US Salernitana 1919":           "Salernitana",
    "AC Monza":                      "Monza",
    "Frosinone Calcio":              "Frosinone",
    "Hellas Verona FC":              "Verona",
    "Hellas Verona":                 "Verona",
    "AC Cesena":                     "Cesena",
    "US Cremonese":                  "Cremonese",
    "Spezia Calcio":                 "Spezia",
    "US Sassuolo Calcio":            "Sassuolo",

    # ══════════════════════════════════════════════════════
    # LIGUE 1
    # co.uk: "Paris SG", "Marseille", "Monaco", "Lyon", "Lille" itd.
    # ══════════════════════════════════════════════════════
    "Paris Saint-Germain FC":        "Paris SG",
    "Paris Saint-Germain":           "Paris SG",
    "PSG":                           "Paris SG",
    "Olympique de Marseille":        "Marseille",
    "Marseille":                     "Marseille",
    "AS Monaco FC":                  "Monaco",
    "AS Monaco":                     "Monaco",
    "Olympique Lyonnais":            "Lyon",
    "Olympique Lyonnais FC":         "Lyon",
    "LOSC Lille":                    "Lille",
    "Lille OSC":                     "Lille",
    "Stade Rennais FC":              "Rennes",
    "Stade Rennais":                 "Rennes",
    "OGC Nice":                      "Nice",
    "RC Lens":                       "Lens",
    "RC Strasbourg Alsace":          "Strasbourg",
    "RC Strasbourg":                 "Strasbourg",
    "FC Nantes":                     "Nantes",
    "Montpellier HSC":               "Montpellier",
    "FC Toulouse":                   "Toulouse",
    "Toulouse FC":                   "Toulouse",
    "Stade de Reims":                "Reims",
    "Reims":                         "Reims",
    "Stade Brestois 29":             "Brest",
    "Stade Brestois":                "Brest",
    "Clermont Foot 63":              "Clermont",
    "Clermont Foot":                 "Clermont",
    "AJ Auxerre":                    "Auxerre",
    "Le Havre AC":                   "Le Havre",
    "Havre AC":                      "Le Havre",
    "FC Metz":                       "Metz",
    "AS Saint-Étienne":              "St Etienne",
    "Saint-Etienne":                 "St Etienne",
    "Angers SCO":                    "Angers",
    "AC Ajaccio":                    "Ajaccio",
    "FC Lorient":                    "Lorient",
    "RC Paris":                      "Paris FC",    # ── Poprawki z debug (screenshoty 2026-03-09) ──────────────────────
    # EPL: fd.org zwraca skrócone shortName bez "& Hove Albion" i "Forest"
    "Brighton Hove":                 "Brighton",
    "Nottingham":                    "Nott'm Forest",
    # Bundesliga: fd.org shortName bez prefiksu
    "Bremen":                        "Werder Bremen",
    "Frankfurt":                     "Ein Frankfurt",
    # La Liga: fd.org używa znaków diakrytycznych i skrótów
    "Alavés":                        "Alaves",
    "Athletic":                      "Ath Bilbao",
    "Atleti":                        "Ath Madrid",
    "Barça":                         "Barcelona",
    # Serie A
    "AC Pisa":                       "Pisa",
    "Pisa SC":                       "Pisa",
    # Ligue 1
    "Olympique Lyon":                "Lyon",
}


_niezmapowane: set = set()

def map_nazwa(nazwa: str) -> str:
    if not isinstance(nazwa, str):
        return str(nazwa)
    if nazwa in NAZWY_MAP:
        return NAZWY_MAP[nazwa]
    def uproszcz(s):
        return s.replace(" ", "").replace("-", "").replace("'", "").lower()
    for key, val in NAZWY_MAP.items():
        if uproszcz(key) == uproszcz(nazwa):
            return val
    znorm = normalize_name(nazwa)
    if znorm in NAZWY_MAP:
        return NAZWY_MAP[znorm]
    _niezmapowane.add(nazwa)
    return nazwa

# ===========================================================================
# BAZA DANYCH – NOWA STRUKTURA
# ===========================================================================

def _pobierz_csv(league_code: str, sezon: str) -> pd.DataFrame:
    try:
        url = f"https://www.football-data.co.uk/mmz4281/{sezon}/{league_code}.csv"
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        for col in ["HC", "AC", "HY", "AY", "HR", "AR", "HST", "AST"]:
            if col not in df.columns:
                df[col] = np.nan
        df["total_gole"]   = df["FTHG"] + df["FTAG"]
        df["total_kartki"] = df["HY"] + df["AY"] + (df["HR"] + df["AR"]) * 2
        df["total_rozne"]  = df["HC"] + df["AC"]
        df["HST"] = pd.to_numeric(df["HST"], errors="coerce")
        df["AST"] = pd.to_numeric(df["AST"], errors="coerce")
        # Kursy bukmacherów – zachowaj B365, Pinnacle, Max i Avg do analizy ROI i line movement
        for col in ["B365H","B365D","B365A","PSH","PSD","PSA",
                    "PSCH","PSCD","PSCA",               # Pinnacle closing (line movement)
                    "MaxCH","MaxCD","MaxCA",             # Max closing (best fair odds)
                    "AvgCH","AvgCD","AvgCA",             # Avg closing (market consensus)
                    "BbAvH","BbAvD","BbAvA"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = np.nan
        return df.sort_values("Date")
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=900)
def load_historical(league_code: str) -> pd.DataFrame:
    df_now  = _pobierz_csv(league_code, "2526")
    df_prev = _pobierz_csv(league_code, "2425")
    if df_now.empty and df_prev.empty:
        return pd.DataFrame()
    if df_now.empty:
        return df_prev
    if df_prev.empty:
        return df_now
    n_now = len(df_now)
    w_prev = waga_poprzedniego(n_now)
    n_prev_target = min(int(n_now * w_prev / (1 - w_prev)), len(df_prev))
    df_prev_s = df_prev.tail(n_prev_target).copy()
    df_prev_s["_sezon"] = "poprzedni"
    df_now = df_now.copy()
    df_now["_sezon"] = "biezacy"
    return pd.concat([df_prev_s, df_now], ignore_index=True).sort_values("Date")


@st.cache_data(ttl=86400, show_spinner=False)
def load_h2h_csv(league_code: str, n_seasons: int = 4) -> pd.DataFrame:
    """Ładuje ostatnie n_seasons sezonów CSV dla potrzeb H2H (bez ważenia).
    Sezony: 2526, 2425, 2324, 2223."""
    sezony = ["2526", "2425", "2324", "2223"][:n_seasons]
    frames = []
    for s in sezony:
        df = _pobierz_csv(league_code, s)
        if not df.empty:
            df = df.copy()
            df["_sezon_h2h"] = s
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("Date", ascending=False)

def _get_fd_api_key() -> str | None:
    """Pobiera klucz football-data.org z secrets lub session_state."""
    try:
        return st.secrets["FOOTBALL_DATA_API_KEY"]
    except Exception:
        return st.session_state.get("_fd_api_key")

@st.cache_data(ttl=3600)
def load_schedule(fd_org_id: int, filename: str) -> pd.DataFrame:
    """Ładuje terminarz z football-data.org API (TTL 1h).
    Fallback na lokalny CSV jeśli brak klucza lub błąd API."""

    api_key = _get_fd_api_key()

    if api_key:
        try:
            import requests as _req
            headers = {"X-Auth-Token": api_key}

            # Wyznacz rok sezonu na podstawie daty:
            # Europejskie ligi startują w lipcu/sierpniu → sezon X/X+1
            # Jeśli jesteśmy po lipcu → bieżący sezon startował w tym roku
            # Jeśli jesteśmy przed lipcem → bieżący sezon startował rok temu
            from datetime import date as _date
            _today = _date.today()
            _target_year = _today.year if _today.month >= 7 else _today.year - 1

            # Pobierz mecze dla wyznaczonego roku sezonu
            url = f"https://api.football-data.org/v4/competitions/{fd_org_id}/matches?season={_target_year}"
            resp = _req.get(url, headers=headers, timeout=10)

            # Jeśli 404 lub brak danych → cofnij o rok (sezon jeszcze nie opublikowany)
            if resp.status_code == 404:
                _target_year -= 1
                url = f"https://api.football-data.org/v4/competitions/{fd_org_id}/matches?season={_target_year}"
                resp = _req.get(url, headers=headers, timeout=10)

            resp.raise_for_status()
            matches = resp.json().get("matches", [])

            # Jeśli pusty wynik → cofnij rok
            if not matches:
                _target_year -= 1
                url = f"https://api.football-data.org/v4/competitions/{fd_org_id}/matches?season={_target_year}"
                resp = _req.get(url, headers=headers, timeout=10)
                resp.raise_for_status()
                matches = resp.json().get("matches", [])

            if not matches:
                raise ValueError(f"Brak meczów dla sezonu {_target_year} i {_target_year-1}")

            rows = []
            for m in matches:
                ht_name = m["homeTeam"].get("shortName") or m["homeTeam"].get("name", "")
                at_name = m["awayTeam"].get("shortName") or m["awayTeam"].get("name", "")
                ht = map_nazwa(ht_name)
                at = map_nazwa(at_name)
                dt = pd.to_datetime(m["utcDate"]).tz_localize(None) if not m["utcDate"].endswith("Z") \
                    else pd.to_datetime(m["utcDate"]).tz_convert(None)
                status = m.get("status", "")
                is_played = status == "FINISHED"
                score = m.get("score", {}).get("fullTime", {})
                wynik_h = score.get("home")
                wynik_a = score.get("away")
                rows.append({
                    "round":     m.get("matchday", 0),
                    "date":      dt,
                    "home_team": ht,
                    "away_team": at,
                    "home_id":   m["homeTeam"].get("id"),
                    "away_id":   m["awayTeam"].get("id"),
                    "is_played": is_played,
                    "status":    status,
                    "wynik_h":   wynik_h,
                    "wynik_a":   wynik_a,
                    "fd_id":     m.get("id"),
                    "_season":   str(_target_year),
                })

            df = pd.DataFrame(rows)
            df["round"] = pd.to_numeric(df["round"], errors="coerce").fillna(0).astype(int)
            # Zapisz info o sezonie do session_state dla sidebara
            try:
                st.session_state["_fd_season_debug"] = f"{_target_year}/{str(_target_year+1)[2:]}"
            except Exception:
                pass
            return df.sort_values("date").reset_index(drop=True)

        except Exception as _e:
            st.sidebar.warning(f"⚠️ API terminarz: {_e} → CSV fallback")

    # ── Fallback: lokalny CSV (stare zachowanie) ──────────────────────
    try:
        df = pd.read_csv(filename)
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
        if "round" in df.columns:
            df["round"] = pd.to_numeric(df["round"], errors="coerce").fillna(0).astype(int)
        else:
            df = df.sort_values("date")
            df["date_only"] = df["date"].dt.date
            unique_dates = sorted(df["date_only"].unique())
            date_to_round = {d: i+1 for i, d in enumerate(unique_dates)}
            df["round"] = df["date_only"].map(date_to_round)
            df = df.drop("date_only", axis=1)
        # Dodaj brakujące kolumny dla kompatybilności
        for col in ["is_played", "wynik_h", "wynik_a", "status", "fd_id"]:
            if col not in df.columns:
                df[col] = None
        return df.sort_values("date").reset_index(drop=True)
    except Exception as e:
        st.error(f"Problem z terminarza {filename}: {e}")
        return pd.DataFrame()

# ===========================================================================
# FUNKCJE POMOCNICZE DLA KOLEJEK
# ===========================================================================
@st.cache_data(ttl=3600)

def load_standings(fd_org_id: int) -> dict:
    """Pobiera tabelę ligową z football-data.org API (TTL 1h).
    Zwraca dict: {team_id: {name, short, position, points, played,
                             won, draw, lost, gf, ga, gd, form}}
    form = string np. "W,L,W,W,D" (ostatnie 5 meczów, od najnowszego)
    Fallback: pusty dict przy błędzie lub braku klucza."""
    api_key = _get_fd_api_key()
    if not api_key:
        return {}
    try:
        import requests as _req
        resp = _req.get(
            f"https://api.football-data.org/v4/competitions/{fd_org_id}/standings",
            headers={"X-Auth-Token": api_key}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # Szukaj TOTAL standings (nie home/away)
        table = []
        for sg in data.get("standings", []):
            if sg.get("type") == "TOTAL":
                table = sg.get("table", [])
                break
        if not table and data.get("standings"):
            table = data["standings"][0].get("table", [])
        result = {}
        for row in table:
            team = row.get("team", {})
            tid = team.get("id")
            if not tid:
                continue
            # Forma: "W,W,L,D,W" → lista ostatnich 5 od najnowszego
            raw_form = row.get("form") or ""
            form_list = [x.strip() for x in raw_form.split(",") if x.strip()][-5:]
            result[tid] = {
                "name":     team.get("name", ""),
                "short":    team.get("shortName") or team.get("tla") or team.get("name", ""),
                "position": row.get("position", 0),
                "points":   row.get("points", 0),
                "played":   row.get("playedGames", 0),
                "won":      row.get("won", 0),
                "draw":     row.get("draw", 0),
                "lost":     row.get("lost", 0),
                "gf":       row.get("goalsFor", 0),
                "ga":       row.get("goalsAgainst", 0),
                "gd":       row.get("goalDifference", 0),
                "form":     form_list,   # lista ["W","L","W","W","D"]
                "crest":    team.get("crest") or f"https://crests.football-data.org/{tid}.png",
            }
        return result
    except Exception:
        return {}


def _standings_by_name(standings: dict) -> dict:
    """Pomocniczy: zwraca standings indeksowane po znormalizowanej nazwie
    (po map_nazwa) dla łatwego lookup w karcie meczu."""
    by_name = {}
    for tid, info in standings.items():
        # Próbuj wszystkie warianty nazwy
        for raw in [info["name"], info["short"]]:
            mapped = map_nazwa(raw)
            by_name[mapped] = info
            by_name[raw] = info
    return by_name


@st.cache_data(ttl=86400, show_spinner=False)
def load_h2h_api(team_id_h: int, team_id_a: int, api_key: str, n: int = 5) -> list:
    """Pobiera ostatnie n bezpośrednich spotkań dwóch drużyn z fd.org API.
    Używa endpointu /v4/teams/{id}/matches z filtrem na rywala.
    Zwraca listę dict: {date, home, away, score_h, score_a, winner}
    """
    if not api_key or not team_id_h or not team_id_a:
        return []
    try:
        import requests as _req
        headers = {"X-Auth-Token": api_key}
        # Pobierz mecze drużyny domowej (ostatnie 30, status=FINISHED)
        url = (f"https://api.football-data.org/v4/teams/{team_id_h}/matches"
               f"?status=FINISHED&limit=30")
        resp = _req.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        matches = resp.json().get("matches", [])
        # Filtruj tylko mecze z rywalem = team_id_a
        h2h_matches = []
        for m in matches:
            ht_id = m.get("homeTeam", {}).get("id")
            at_id = m.get("awayTeam", {}).get("id")
            if team_id_a in (ht_id, at_id):
                score = m.get("score", {}).get("fullTime", {})
                sh = score.get("home")
                sa = score.get("away")
                if sh is None or sa is None:
                    continue
                h2h_matches.append({
                    "date":    m.get("utcDate", "")[:10],
                    "home":    m.get("homeTeam", {}).get("shortName") or m.get("homeTeam", {}).get("name", "?"),
                    "away":    m.get("awayTeam", {}).get("shortName") or m.get("awayTeam", {}).get("name", "?"),
                    "home_id": ht_id,
                    "away_id": at_id,
                    "score_h": int(sh),
                    "score_a": int(sa),
                })
        # Posortuj malejąco po dacie, weź n ostatnich
        h2h_matches.sort(key=lambda x: x["date"], reverse=True)
        return h2h_matches[:n]
    except Exception:
        return []



def _forma_warning(h_form: list, a_form: list,
                   h_pos: int, a_pos: int, pred_typ: str) -> str | None:
    """Zwraca tekst ostrzeżenia jeśli forma przeczy predykcji modelu.
    pred_typ: '1' (wygrana gospodarza), 'X', '2' (wygrana gościa)"""
    if not h_form or not a_form:
        return None
    h_wins  = h_form.count("W")
    a_wins  = a_form.count("W")
    h_losses = h_form.count("L")
    a_losses = a_form.count("L")
    warnings = []
    # Model faworyzuje gospodarza, ale ma złą formę (≤1W z 5) i jest wyżej w tabeli
    if pred_typ == "1" and h_wins <= 1 and h_losses >= 3:
        warnings.append(f"Gospodarz ({h_wins}W/{h_losses}P z 5) w słabej formie")
    # Model faworyzuje gościa, ale ma złą formę
    if pred_typ == "2" and a_wins <= 1 and a_losses >= 3:
        warnings.append(f"Gość ({a_wins}W/{a_losses}P z 5) w słabej formie")
    # Różnica pozycji vs predykcja (tylko gdy standings dostępne)
    if h_pos > 0 and a_pos > 0:
        pos_diff = a_pos - h_pos  # > 0 = gospodarz wyżej w tabeli
        if pred_typ == "2" and pos_diff >= 10 and a_wins <= 2:
            warnings.append(f"Faworyzowany gość na poz.{a_pos} vs gospodarz poz.{h_pos}")
        if pred_typ == "1" and pos_diff <= -10 and h_wins <= 2:
            warnings.append(f"Faworyzowany gospodarz na poz.{h_pos} vs gość poz.{a_pos}")
    return " · ".join(warnings) if warnings else None


def get_current_round(schedule: pd.DataFrame) -> int:
    """
    Zwraca numer aktualnej kolejki.
    Logika: szuka kolejki gdzie WIEKSZOSC meczow jest w przyszlosci.
    Rozwiazuje problem Bundesligi: zalegly mecz z kolejki N nie blokuje
    pokazania kolejki N+8 jesli ona ma wiecej meczow w przyszlosci.
    """
    if schedule.empty:
        return 0
    dzisiaj = datetime.now().date()
    sch = schedule.copy()
    sch["_d"] = sch["date"].dt.date

    # Dla kazdej kolejki: ile meczow w przyszlosci vs ile juz bylo
    rundy = sorted(sch["round"].unique())
    for runda in rundy:
        mecze_rundy  = sch[sch["round"] == runda]
        n_total      = len(mecze_rundy)
        n_przyszle   = (mecze_rundy["_d"] >= dzisiaj).sum()
        n_przeszle   = n_total - n_przyszle
        # Jesli wiekszosc meczow tej kolejki jest w przyszlosci/dzisiaj -> to aktualna kolejka
        if n_przyszle >= n_przeszle and n_przyszle > 0:
            return int(runda)
    # Wszystkie mecze w przeszlosci -> ostatnia kolejka
    return int(schedule["round"].max())

def get_available_rounds(schedule: pd.DataFrame) -> list:
    """Zwraca liste dostepnych kolejek z opisem (do selectboxa)."""
    if schedule.empty:
        return []
    dzisiaj = datetime.now().date()
    sch = schedule.copy()
    sch["_d"] = sch["date"].dt.date
    result = []
    for runda in sorted(sch["round"].unique()):
        mecze_r  = sch[sch["round"] == runda]
        n_fut    = (mecze_r["_d"] >= dzisiaj).sum()
        n_past   = len(mecze_r) - n_fut
        min_date = mecze_r["_d"].min()
        max_date = mecze_r["_d"].max()
        if n_fut > 0 and n_past > 0:
            tag = "⚠️ zalegly"
        elif n_fut > 0:
            tag = "🔜 nadchodzaca"
        else:
            tag = "✅ zakonczona"
        label = f"Kolejka {int(runda)} · {min_date.strftime('%d.%m')}–{max_date.strftime('%d.%m')} · {tag}"
        result.append((int(runda), label))
    return result

def get_round_status(schedule: pd.DataFrame, round_num: int) -> str:
    """Zwraca status kolejki (przeszła, dzisiejsza, przyszła)"""
    if schedule.empty:
        return "Nieznana"
    
    dzisiaj = datetime.now().date()
    mecze_kolejki = schedule[schedule["round"] == round_num]
    
    if mecze_kolejki.empty:
        return "Nieznana"
    
    najblizsza_data = mecze_kolejki["date"].min().date()
    
    if najblizsza_data > dzisiaj:
        return "🔜 Następna"
    elif najblizsza_data == dzisiaj:
        return "⚽ Dzisiaj"
    else:
        return "📅 Przeszła"

# ===========================================================================
# STATYSTYKI
# ===========================================================================
TIME_DECAY_TAU = 30.0  # globalny fallback; per-liga ustawiane w LIGI dict




def weighted_mean(values: pd.Series, dates: pd.Series = None,
                  tau_days: float = TIME_DECAY_TAU) -> float:
    """Ważona średnia z wykładniczym decay (tau=21 dni).
    Mecz sprzed 3 tyg. ma wagę ~37% meczu z dziś. Szybsza reakcja na formę.
    Jeśli dates niedostępne – fallback do liniowego linspace."""
    if len(values) == 0:
        return 0.0
    if dates is not None and len(dates) == len(values):
        try:
            dates_dt = pd.to_datetime(dates)
            ref = dates_dt.max()
            days_ago = (ref - dates_dt).dt.total_seconds() / 86400
            weights = np.exp(-days_ago.values / tau_days)
            weights = np.clip(weights, 0.01, None)  # min waga 0.01
        except Exception:
            weights = np.linspace(1, 2, len(values))
    else:
        weights = np.linspace(1, 2, len(values))
    return float(np.average(values, weights=weights))

@st.cache_data

def oblicz_wszystkie_statystyki(df_json: str) -> pd.DataFrame:
    return _oblicz_statystyki_impl(df_json)

def _oblicz_statystyki_impl(df_json: str) -> pd.DataFrame:
    df = pd.read_json(df_json)
    if df.empty:
        return pd.DataFrame()
    druzyny = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
    dane = {}
    for d in druzyny:
        home = df[df["HomeTeam"] == d].tail(10)
        away = df[df["AwayTeam"] == d].tail(10)
        if len(home) < 2 or len(away) < 2:
            continue
        # Daty do exponential decay
        h_dates = home["Date"] if "Date" in home.columns else None
        a_dates = away["Date"] if "Date" in away.columns else None
        # SOT – celne strzały (jeśli dostępne w danych)
        home_sot = home["HST"].dropna() if "HST" in home.columns else pd.Series([], dtype=float)
        away_sot = away["AST"].dropna() if "AST" in away.columns else pd.Series([], dtype=float)
        h_sot_dates = home.loc[home["HST"].notna(), "Date"] if "HST" in home.columns and "Date" in home.columns else None
        a_sot_dates = away.loc[away["AST"].notna(), "Date"] if "AST" in away.columns and "Date" in away.columns else None
        dane[d] = {
            "Gole strzelone (dom)":    weighted_mean(home["FTHG"], h_dates),
            "Gole stracone (dom)":     weighted_mean(home["FTAG"], h_dates),
            "Gole strzelone (wyjazd)": weighted_mean(away["FTAG"], a_dates),
            "Gole stracone (wyjazd)":  weighted_mean(away["FTHG"], a_dates),
            "Różne (dom)":             weighted_mean(home["total_rozne"], h_dates),
            "Różne (wyjazd)":          weighted_mean(away["total_rozne"], a_dates),
            "Kartki (dom)":            weighted_mean(home["total_kartki"], h_dates),
            "Kartki (wyjazd)":         weighted_mean(away["total_kartki"], a_dates),
            # SOT (celne strzały)
            "SOT (dom)":    (weighted_mean(home.loc[home["HST"].notna(),"HST"], h_sot_dates)
                             if len(home_sot) >= 2 else None),
            "SOT (wyjazd)": (weighted_mean(away.loc[away["AST"].notna(),"AST"], a_sot_dates)
                             if len(away_sot) >= 2 else None),
            # Konwersja SOT→Gol (gole / strzały celne)
            "Konwersja (dom)":    (float(home["FTHG"].sum() / home_sot.sum())
                                   if home_sot.sum() > 0 else None),
            "Konwersja (wyjazd)": (float(away["FTAG"].sum() / away_sot.sum())
                                   if away_sot.sum() > 0 else None),
        }
    return pd.DataFrame(dane).T.round(2)

@st.cache_data
def oblicz_srednie_ligowe(df_json: str) -> dict:
    return _oblicz_srednie_impl(df_json)

def _oblicz_srednie_impl(df_json: str) -> dict:
    df = pd.read_json(df_json)
    if df.empty:
        return {"avg_home": 1.5, "avg_away": 1.2, "rho": -0.13, "n_biezacy": 0}
    n_biezacy = int((df.get("_sezon", pd.Series()) == "biezacy").sum()) if "_sezon" in df.columns else len(df)
    avg_h = float(df["FTHG"].mean())
    avg_a = float(df["FTAG"].mean())
    n = len(df)
    obs_00 = len(df[(df["FTHG"] == 0) & (df["FTAG"] == 0)]) / n
    obs_11 = len(df[(df["FTHG"] == 1) & (df["FTAG"] == 1)]) / n
    from scipy.stats import poisson as _p
    exp_00 = _p.pmf(0, avg_h) * _p.pmf(0, avg_a)
    exp_11 = _p.pmf(1, avg_h) * _p.pmf(1, avg_a)
    rho_00 = (obs_00 / exp_00 - 1) / (avg_h * avg_a) if exp_00 > 0 else -0.13
    rho_11 = -(obs_11 / exp_11 - 1) if exp_11 > 0 else -0.13
    rho = float(np.clip(np.mean([rho_00, rho_11]), -0.25, 0.0))
    # Średnie SOT ligowe (do konwersji SOT→gole w blendzie)
    avg_sot_h = float(df["HST"].dropna().mean()) if "HST" in df.columns and df["HST"].notna().sum() > 10 else None
    avg_sot_a = float(df["AST"].dropna().mean()) if "AST" in df.columns and df["AST"].notna().sum() > 10 else None
    return {"avg_home": avg_h, "avg_away": avg_a, "rho": rho,
            "n_biezacy": n_biezacy,
            "avg_sot_home": avg_sot_h, "avg_sot_away": avg_sot_a}

# Waga SOT w blendzie z golami – parametr globalny
SOT_BLEND_W = 0.30   # 0.0 = tylko gole, 0.30 = 70% gole + 30% SOT
