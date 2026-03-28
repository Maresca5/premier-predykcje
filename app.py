import streamlit as st
import pandas as pd
from scipy.stats import poisson
from datetime import datetime
import requests
from io import StringIO
import numpy as np
import unicodedata
import sqlite3
import os
import json
import itertools

try:
    import backtest as _bt
    _BT_OK = True
except Exception:
    _BT_OK = False

try:
    import odds_api as _oa
    _OA_OK = True
except ImportError:
    _OA_OK = False

def _kurs_dc_live(typ, oh, od, oa):
    """Kurs DC i impl dla live odds – identyczna logika jak backtest.py."""
    try:
        oh, od, oa = float(oh), float(od), float(oa)
        if oh <= 1 or od <= 1 or oa <= 1: return None, None
        s = 1/oh + 1/od + 1/oa
        ih=(1/oh)/s; id_=(1/od)/s; ia=(1/oa)/s
        if typ=="1":  return oh,           ih
        if typ=="X":  return od,           id_
        if typ=="2":  return oa,           ia
        if typ=="1X": idc=ih+id_; return round(1/idc,3), idc
        if typ=="X2": idc=id_+ia; return round(1/idc,3), idc
    except Exception:
        pass
    return None, None

# ===========================================================================
# KONFIGURACJA
# ===========================================================================
LIGI = {
    # tau: dni time-decay; fw: form_weight krok per W/L
    # fd_org_id: ID ligi w football-data.org API
    # cup_ids: ID rozgrywek pucharowych dla fatigue (UCL/UEL/UECL/krajowe puchary)
    "Premier League": {"csv_code": "E0",  "fd_org_id": 2021, "cup_ids": [2001,2002,2139,2021], "file": "terminarz_premier_2025.csv",  "tau": 30.0, "fw": 0.03},
    "La Liga":        {"csv_code": "SP1", "fd_org_id": 2014, "cup_ids": [2001,2002,2079],       "file": "terminarz_la_liga_2025.csv",  "tau": 28.0, "fw": 0.04},
    "Bundesliga":     {"csv_code": "D1",  "fd_org_id": 2002, "cup_ids": [2001,2002,2011],       "file": "terminarz_bundesliga_2025.csv","tau": 28.0, "fw": 0.04},
    "Serie A":        {"csv_code": "I1",  "fd_org_id": 2019, "cup_ids": [2001,2002,2080],       "file": "terminarz_serie_a_2025.csv",  "tau": 28.0, "fw": 0.04},
    "Ligue 1":        {"csv_code": "F1",  "fd_org_id": 2015, "cup_ids": [2001,2002,2081],       "file": "terminarz_ligue_1_2025.csv",  "tau": 21.0, "fw": 0.05},
}

DB_FILE    = "predykcje.db"

# Bieżący sezon (kod football-data.co.uk: "2526" = sezon 2025/26)
# Europejskie ligi: sezon startuje w lipcu/sierpniu
# Jeśli miesiąc >= 7 → rok bieżący, inaczej rok-1
def _oblicz_biezacy_sezon() -> str:
    _y = datetime.now().year if datetime.now().month >= 7 else datetime.now().year - 1
    return f"{str(_y)[2:]}{str(_y+1)[2:]}"

BIEZACY_SEZON = _oblicz_biezacy_sezon()   # np. "2526" dla sezonu 2025/26

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

def _fuzzy_normalize(s: str) -> str:
    """Agresywna normalizacja do fuzzy matchingu — usuwa akcenty, sufiksy klubowe, znaki specjalne."""
    import unicodedata as _ud, re as _re
    s = _ud.normalize("NFD", s)
    s = "".join(c for c in s if _ud.category(c) != "Mn")
    s = _re.sub(
        r"\b(FC|SC|CF|AC|AS|RC|CD|RCD|SL|SS|US|SK|FK|BK|VfB|VfL|SV|FSV|TSG|BSC|"
        r"RB|KV|KRC|AFC|RFC|GFC|OGC|SM|UD|SD|CP|CA|CR|CE|1\.|04|Stade)\b",
        "", s, flags=_re.IGNORECASE)
    s = _re.sub(r"[^\w\s]", " ", s)
    s = _re.sub(r"\s+", " ", s).strip().lower()
    return s

# Cache fuzzy: {znormalizowana_nazwa_obca: dopasowana_wartość_z_NAZWY_MAP}
_niezmapowane: set = set()
_fuzzy_cache: dict = {}

def map_nazwa(nazwa: str) -> str:
    if not isinstance(nazwa, str):
        return str(nazwa)
    # 1. Bezpośrednie trafienie
    if nazwa in NAZWY_MAP:
        return NAZWY_MAP[nazwa]
    # 2. Uproszczone porównanie (bez spacji/myślników)
    def _uproszcz(s):
        return s.replace(" ", "").replace("-", "").replace("'", "").lower()
    _u = _uproszcz(nazwa)
    for key, val in NAZWY_MAP.items():
        if _uproszcz(key) == _u:
            return val
    # 3. normalize_name fallback
    znorm = normalize_name(nazwa)
    if znorm in NAZWY_MAP:
        return NAZWY_MAP[znorm]
    # 4. Fuzzy matching jako auto-fallback (difflib)
    if nazwa in _fuzzy_cache:
        return _fuzzy_cache[nazwa]
    from difflib import get_close_matches as _gcm
    _all_values = list(set(NAZWY_MAP.values()))
    # Próba bezpośrednia
    _m = _gcm(nazwa, _all_values, n=1, cutoff=0.72)
    if not _m:
        # Próba po normalizacji (usuwa FC, akcenty itp.)
        _fn  = _fuzzy_normalize(nazwa)
        _fvs = {_fuzzy_normalize(v): v for v in _all_values}
        _m2  = _gcm(_fn, list(_fvs.keys()), n=1, cutoff=0.55)
        if _m2:
            _m = [_fvs[_m2[0]]]
    if _m:
        _fuzzy_cache[nazwa] = _m[0]
        return _m[0]
    # 5. Nie znaleziono — loguj i zwróć oryginalną
    _niezmapowane.add(nazwa)
    return nazwa

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
    # fd.org używa pełnych nazw — MUSZĄ być tutaj jawnie żeby uniknąć
    # fałszywego fuzzy matchingu (np. "RCD Espanyol de Barcelona" -> "Barcelona")
    # ══════════════════════════════════════════════════════
    "FC Barcelona":                  "Barcelona",
    "Futbol Club Barcelona":         "Barcelona",
    "Real Madrid CF":                "Real Madrid",
    "Real Madrid Club de Fútbol":    "Real Madrid",
    "Atlético de Madrid":            "Ath Madrid",
    "Atlético Madrid":               "Ath Madrid",
    "Atletico Madrid":               "Ath Madrid",
    "Club Atlético de Madrid":       "Ath Madrid",
    "Athletic Club":                 "Ath Bilbao",
    "Athletic Bilbao":               "Ath Bilbao",
    "Athletic Club Bilbao":          "Ath Bilbao",
    "Real Sociedad de Fútbol":       "Sociedad",
    "Real Sociedad de Futbol":       "Sociedad",
    "Real Sociedad":                 "Sociedad",
    "Real Betis Balompié":           "Betis",
    "Real Betis Balompie":           "Betis",
    "Real Betis":                    "Betis",
    "RC Celta de Vigo":              "Celta",
    "Celta Vigo":                    "Celta",
    "RCD Espanyol":                  "Espanol",
    "RCD Espanyol de Barcelona":     "Espanol",   # ← pełna nazwa z fd.org!
    "Espanyol de Barcelona":         "Espanol",
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
    "CD Leganes":                    "Leganes",
    "Leganés":                       "Leganes",
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
    "Paris Saint Germain":           "Paris SG",
    "Paris SG FC":                   "Paris SG",
    "Paris SG":                      "Paris SG",
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




# ===========================================================================
# BAZA DANYCH – NOWA STRUKTURA
# ===========================================================================
def init_db():
    """Inicjalizuje bazę danych z nową strukturą analityczną"""
    con = sqlite3.connect(DB_FILE)
    
    # Tabela predykcje (stara, dla kompatybilności)
    con.execute("""
        CREATE TABLE IF NOT EXISTS predykcje (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            liga    TEXT,
            home    TEXT,
            away    TEXT,
            round   INTEGER,
            typ     TEXT,
            fo_typ  REAL,
            p_home  REAL,
            p_draw  REAL,
            p_away  REAL,
            data    TEXT,
            UNIQUE(liga, home, away, round)
        )
    """)
    
    # NOWA TABELA: zdarzenia (każda noga osobno)
    con.execute("""
        CREATE TABLE IF NOT EXISTS zdarzenia (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            liga        TEXT,
            sezon       TEXT,
            kolejnosc   INTEGER,
            mecz        TEXT,
            home        TEXT,
            away        TEXT,
            rynek       TEXT,
            linia       REAL,
            typ         TEXT,
            p_model     REAL,
            fair_odds   REAL,
            wynik       TEXT,
            trafione    BOOLEAN,
            data        TEXT,
            UNIQUE(liga, sezon, kolejnosc, mecz, rynek, typ, linia)
        )
    """)
    # Migracja: dodaj kolumnę sezon do istniejących baz (bez niej ALTER TABLE nie przejdzie)
    try:
        con.execute("ALTER TABLE zdarzenia ADD COLUMN sezon TEXT")
        # Wypełnij stare rekordy (bez sezonu) wartością poprzedniego sezonu
        _prev = f"{str(int(BIEZACY_SEZON[:2])-1):02d}{BIEZACY_SEZON[:2]}"
        con.execute("UPDATE zdarzenia SET sezon=? WHERE sezon IS NULL", (_prev,))
        con.commit()
    except Exception:
        pass  # kolumna już istnieje
    
    # Tabela kupony (opcjonalna, dla historycznych danych)
    con.execute("""
        CREATE TABLE IF NOT EXISTS kupony (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            liga           TEXT,
            kolejnosc      INTEGER,
            typ_kuponu     TEXT,
            zdarzenia_json TEXT,
            ako            REAL,
            p_combo        REAL,
            data           TEXT
        )
    """)

    # Tabela bankroll – śledzenie realnego kapitału
    con.execute("""
        CREATE TABLE IF NOT EXISTS bankroll (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            data        TEXT NOT NULL,
            liga        TEXT,
            kolejnosc   INTEGER,
            opis        TEXT,
            kwota       REAL NOT NULL,
            typ         TEXT DEFAULT 'korekta',
            kapital_po  REAL,
            UNIQUE(data, liga, kolejnosc, opis)
        )
    """)

    con.commit()
    con.close()

# ===========================================================================
# ŁADOWANIE DANYCH – BLEND SEZONÓW
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
    (po map_nazwa) dla łatwego lookup w karcie meczu.
    Uwaga: używamy tylko pełnej nazwy (name), nie short — skróty jak
    'FCB', 'ESP' mogą przez fuzzy matching trafić w złą drużynę i
    mieszać cresty (np. Barcelona <-> Espanyol)."""
    by_name = {}
    for tid, info in standings.items():
        raw    = info["name"]   # tylko pełna nazwa, nie short
        mapped = map_nazwa(raw)
        by_name[mapped] = info
        by_name[raw]    = info
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

def oblicz_lambdy(h: str, a: str, srednie_df: pd.DataFrame,
                  srednie_lig: dict, forma_dict: dict,
                  sot_w: float = SOT_BLEND_W,
                  csv_code: str = "E0",
                  historical: pd.DataFrame = None,
                  mecz_data: object = None) -> tuple:
    """
    Zwraca (lam_h, lam_a, lam_r, lam_k, sot_aktywny, lam_sot).

    Asymetryczny shrink lambdy (potwierdzone backtestem na 3 datasetach):
      ALPHA_LAM_OFF = 0.10  – atak jest zmienny → shrinkujemy mocniej
      ALPHA_LAM_DEF = 0.20  – obrona jest stabilna → shrinkujemy słabiej
    Wynik: +1.5–2.6pp hit rate, +2–5pp ROI, Brier -0.002 na wszystkich 3 datasetach.
    Fatigue factor: korekta λ gdy drużyna grała <4 dni temu (-4%) lub >10 dni (+2%).
    """
    # ── Asymetryczne parametry shrinkage lambdy ──────────────────
    ALPHA_OFF = 0.10   # shrink składowej ofensywnej w kierunku avg_ligi
    ALPHA_DEF = 0.20   # shrink składowej defensywnej w kierunku avg_ligi

    avg_h = max(srednie_lig["avg_home"], 0.5)
    avg_a = max(srednie_lig["avg_away"], 0.5)
    atak_h   = srednie_df.loc[h, "Gole strzelone (dom)"]    / avg_h
    obrona_a = srednie_df.loc[a, "Gole stracone (wyjazd)"]  / avg_a
    atak_a   = srednie_df.loc[a, "Gole strzelone (wyjazd)"] / avg_a
    obrona_h = srednie_df.loc[h, "Gole stracone (dom)"]     / avg_h

    def form_weight(team: str) -> float:
        f = forma_dict.get(team, "")
        w = f.count("W"); l = f.count("L")
        return float(np.clip(1.0 + (w - l) * 0.03, 0.85, 1.15))

    lam_h_goals = avg_h * atak_h * obrona_a * form_weight(h)
    lam_a_goals = avg_a * atak_a * obrona_h * form_weight(a)

    # ── Asymetryczny shrink: lam_h = f(atak domu), lam_a = f(obrona domu)
    # lam_h zależy głównie od ataku drużyny domowej → alpha_off
    # lam_a zależy głównie od obrony drużyny domowej → alpha_def (mniejszy shrink)
    lam_h_goals = (1 - ALPHA_OFF) * lam_h_goals + ALPHA_OFF * avg_h
    lam_a_goals = (1 - ALPHA_DEF) * lam_a_goals + ALPHA_DEF * avg_a

    # SOT blend – tylko gdy oba mają dane i sot_w > 0
    sot_aktywny = False
    lam_h = lam_h_goals
    lam_a = lam_a_goals

    if sot_w > 0:
        sot_h = srednie_df.loc[h, "SOT (dom)"]   if "SOT (dom)"    in srednie_df.columns else None
        sot_a = srednie_df.loc[a, "SOT (wyjazd)"] if "SOT (wyjazd)" in srednie_df.columns else None
        avg_sot_h = srednie_lig.get("avg_sot_home", None)
        avg_sot_a = srednie_lig.get("avg_sot_away", None)
        if (sot_h is not None and sot_a is not None and
                avg_sot_h and avg_sot_a and
                not np.isnan(sot_h) and not np.isnan(sot_a)):
            conv_h = avg_h / avg_sot_h
            conv_a = avg_a / avg_sot_a
            lam_sot_h = sot_h * conv_h * obrona_a * form_weight(h)
            lam_sot_a = sot_a * conv_a * obrona_h * form_weight(a)
            # Asymetryczny shrink również dla składowej SOT
            lam_sot_h = (1 - ALPHA_OFF) * lam_sot_h + ALPHA_OFF * avg_h
            lam_sot_a = (1 - ALPHA_DEF) * lam_sot_a + ALPHA_DEF * avg_a
            lam_h = (1 - sot_w) * lam_h_goals + sot_w * lam_sot_h
            lam_a = (1 - sot_w) * lam_a_goals + sot_w * lam_sot_a
            sot_aktywny = True

    lam_r = (srednie_df.loc[h, "Różne (dom)"] + srednie_df.loc[a, "Różne (wyjazd)"]) / 2
    lam_k = (srednie_df.loc[h, "Kartki (dom)"] + srednie_df.loc[a, "Kartki (wyjazd)"]) / 2
    sot_h_raw = srednie_df.loc[h, "SOT (dom)"]   if "SOT (dom)"    in srednie_df.columns else None
    sot_a_raw = srednie_df.loc[a, "SOT (wyjazd)"] if "SOT (wyjazd)" in srednie_df.columns else None
    lam_sot_total = None
    if sot_h_raw is not None and sot_a_raw is not None:
        try:
            sh = float(sot_h_raw); sa = float(sot_a_raw)
            if not (np.isnan(sh) or np.isnan(sa)):
                lam_sot_total = sh + sa
        except (TypeError, ValueError):
            pass

    # ── Fatigue factor ────────────────────────────────────────────────────
    # Gdy historical i mecz_data dostarczone: sprawdź ile dni od ostatniego meczu
    # <4 dni → zmęczenie → lam ataku -4%, obrona -2% (łatwiej strzelić zmęczonej obronie)
    # >10 dni → świeżość → lam ataku +2%
    if historical is not None and mecz_data is not None:
        try:
            _ref = pd.Timestamp(mecz_data)
            def _dni_od_meczu(team):
                _tm = historical[
                    (historical["HomeTeam"] == team) | (historical["AwayTeam"] == team)
                ]["Date"].dropna()
                _tm = _tm[_tm < _ref]
                if _tm.empty: return None
                return (_ref - _tm.max()).days
            _days_h = _dni_od_meczu(h)
            _days_a = _dni_od_meczu(a)
            for _days, _is_home in [(_days_h, True), (_days_a, False)]:
                if _days is None: continue
                if _days < 4:
                    _fatigue = 0.96   # zmęczenie: -4% ataku
                    _def_boost = 1.02 # lekki boost strat (zmęczona obrona)
                elif _days > 10:
                    _fatigue = 1.02   # odpoczęty: +2%
                    _def_boost = 1.0
                else:
                    continue
                if _is_home:
                    lam_h = float(np.clip(lam_h * _fatigue, 0.3, 4.5))
                    lam_a = float(np.clip(lam_a * _def_boost, 0.3, 4.5))
                else:
                    lam_a = float(np.clip(lam_a * _fatigue, 0.3, 4.5))
                    lam_h = float(np.clip(lam_h * _def_boost, 0.3, 4.5))
        except Exception:
            pass  # fatigue zawsze opcjonalny

    return (float(np.clip(lam_h, 0.3, 4.5)),
            float(np.clip(lam_a, 0.3, 4.5)),
            lam_r, lam_k, sot_aktywny, lam_sot_total)

def oblicz_forme(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    forma = {}
    for d in pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel()):
        mecze = df[(df["HomeTeam"] == d) | (df["AwayTeam"] == d)].tail(5)
        wyniki = []
        for _, m in mecze.iterrows():
            if m["HomeTeam"] == d:
                wyniki.append("W" if m["FTHG"] > m["FTAG"] else ("L" if m["FTHG"] < m["FTAG"] else "D"))
            else:
                wyniki.append("W" if m["FTAG"] > m["FTHG"] else ("L" if m["FTAG"] < m["FTHG"] else "D"))
        forma[d] = "".join(wyniki)
    return forma

def tabela_ligowa(df: pd.DataFrame) -> pd.DataFrame:
    if "_sezon" in df.columns:
        df = df[df["_sezon"] == "biezacy"]
    if df.empty:
        return pd.DataFrame()
    table: dict = {}
    for _, m in df.iterrows():
        home, away = m["HomeTeam"], m["AwayTeam"]
        hg, ag = m["FTHG"], m["FTAG"]
        for t in [home, away]:
            if t not in table:
                table[t] = {"Pkt": 0, "GZ": 0, "GS": 0, "M": 0}
        table[home]["GZ"] += hg; table[home]["GS"] += ag; table[home]["M"] += 1
        table[away]["GZ"] += ag; table[away]["GS"] += hg; table[away]["M"] += 1
        if   hg > ag: table[home]["Pkt"] += 3
        elif hg < ag: table[away]["Pkt"] += 3
        else:         table[home]["Pkt"] += 1; table[away]["Pkt"] += 1
    res = pd.DataFrame(table).T
    res["RG"] = res["GZ"] - res["GS"]
    return res.sort_values(["Pkt", "RG", "GZ"], ascending=False)

# ===========================================================================
# MODEL
# ===========================================================================
def dixon_coles_adj(M: np.ndarray, lam_h: float, lam_a: float,
                    rho: float = -0.13) -> np.ndarray:
    M = M.copy()
    tau = {(0,0): 1 - lam_h*lam_a*rho, (1,0): 1 + lam_a*rho,
           (0,1): 1 + lam_h*rho,        (1,1): 1 - rho}
    for (i, j), t in tau.items():
        if i < M.shape[0] and j < M.shape[1]:
            M[i, j] *= max(t, 0.001)
    M /= M.sum()
    return M

# Shrinkage per liga – zoptymalizowany na backtestach 2 sezony x 5 lig (3098 meczow)
# E0:0.25 | I1:0.28 | SP1:0.38 | D1:0.40 | F1:0.45
KALIBRACJA_PER_LIGA = {"E0": 0.25, "SP1": 0.38, "D1": 0.40, "I1": 0.28, "F1": 0.45}
SHRINK_ALPHA  = 0.25
PROG_PEWNY    = 0.55
PROG_PODWOJNA = 0.55

def _get_shrink(csv_code: str, n_train: int = 200) -> float:
    """
    Dynamiczny shrinkage: przy malej liczbie meczow w bazie shrinkujemy mocniej.
    Formula: shrink = base + bonus * (1 - min(n_train, N_FULL) / N_FULL)
    Przy n_train=0  -> base + 0.20 (max ostroznosc)
    Przy n_train=150 -> base + 0.00 (pelna kalibracja)
    """
    N_FULL   = 150   # po ilu meczach shrinkage osiaga wartosc docelowa
    BONUS    = 0.20  # maksymalny dodatek na poczatku sezonu
    base     = KALIBRACJA_PER_LIGA.get(csv_code, SHRINK_ALPHA)
    dynamic  = BONUS * max(0.0, 1.0 - min(n_train, N_FULL) / N_FULL)
    return float(np.clip(base + dynamic, 0.0, 0.85))

def kalibruj_prawdopodobienstwa(p_home: float, p_draw: float, p_away: float,
                                csv_code: str = "E0", n_train: int = 200) -> tuple:
    a = _get_shrink(csv_code, n_train)
    p_h = (1-a)*p_home + a/3
    p_d = (1-a)*p_draw + a/3
    p_a = (1-a)*p_away + a/3
    s = p_h+p_d+p_a
    return p_h/s, p_d/s, p_a/s

def wybierz_typ(p_home: float, p_draw: float, p_away: float,
                csv_code: str = "E0", n_train: int = 200) -> tuple:
    p_home, p_draw, p_away = kalibruj_prawdopodobienstwa(
        p_home, p_draw, p_away, csv_code, n_train)
    p_1x = p_home + p_draw; p_x2 = p_away + p_draw
    # Bundesliga (D1): wyłącz singiele – backtest 2324/2425:
    #   Typ=1 overconfidence +23-33pp vs tylko +4-18pp dla 1X/X2
    #   Hit rate singlei 47-59% vs 58-76% dla 1X/X2
    is_bundesliga = (csv_code == "D1")
    if not is_bundesliga:
        if p_home >= PROG_PEWNY: return "1",  p_home
        if p_away >= PROG_PEWNY: return "2",  p_away
    if p_1x >= PROG_PODWOJNA or p_x2 >= PROG_PODWOJNA:
        return ("1X", p_1x) if p_1x >= p_x2 else ("X2", p_x2)
    probs = {"1": p_home, "X": p_draw, "2": p_away}
    t = max(probs, key=probs.get)
    return t, probs[t]

# ── Market Noise & Kelly helpers ──────────────────────────────────────
MARKET_NOISE_MAX = 0.25
KELLY_FRACTION   = 0.125         # 1/8 Kelly – default
KELLY_BANKROLL_DEFAULT = 1000.0
# Conservative Kelly – shrinkage prawdopodobieństwa TYLKO do Kelly/EV
# Symulacja (5 lig × 2 sezony): -15% nadwyżki + KF/2 → MaxDD 55%→33%
# p_kelly = 0.5 + (p_model - 0.5) * KELLY_PROB_SCALE
KELLY_PROB_SCALE = 0.85          # -15% nadwyżki powyżej 50%
KELLY_FRAC_SCALE = 0.50          # Half-Kelly na wyjście stawki

# Per-rynek ułamki Kelly – obniżone o 50% względem poprzednich wartości
# Analiza backtestu E0 2425: model zawyża p przy faworytach (p=0.80 → actual 50%)
# → mniejsze frakcje chronią przed agresywnym Kelly na niepewnych predykcjach
KELLY_FRACTIONS = {
    "1X2":    0.125,  # 1/8 Kelly – backtest wykazał optymalny balans ROI/DrawDown
    "AH":     0.125,
    "Gole":   0.075,  # niezweryfikowane historycznie + losowość bramkowa
    "BTTS":   0.075,
    "Rożne":  0.000,  # WYŁĄCZONE – lambda bramkowa ≠ rozkład rożnych (neg. dwumianowy)
    "Kartki": 0.000,  # WYŁĄCZONE – brak kalibracji statystyk kartkowych
    "SOT":    0.050,
}
MAX_EXPOSURE_PCT = 0.05   # max 5% bankrollu per mecz (suma wszystkich rynków)

def market_noise_check(p_model, p_impl):
    diff = abs(p_model - p_impl)
    noise = diff > MARKET_NOISE_MAX
    kierunek = "Model wyzej niz rynek" if p_model > p_impl else "Rynek wyzej niz model"
    kolor = "#F44336" if noise else ("#FF9800" if diff > 0.15 else "#4CAF50")
    zgodnosc = max(0.0, 1.0 - diff / MARKET_NOISE_MAX)
    return {"noise": noise, "diff": diff, "kierunek": kierunek,
            "kolor": kolor, "zgodnosc_pct": zgodnosc}

def kelly_stake(p_model, kurs_buk, bankroll=KELLY_BANKROLL_DEFAULT,
               fraction=None, rynek: str = "1X2",
               already_exposed: float = 0.0) -> dict:
    """
    Oblicza stawkę Kelly z:
    - Per-rynek ułamkiem (1X2=0.25, gole=0.15, kartki/rożne=0.10)
    - Max Exposure per Match: suma stawek na jeden mecz ≤ MAX_EXPOSURE_PCT * bankroll
    - already_exposed: ile już postawiono na ten mecz (z innych rynków)
    """
    try:
        # Wybierz ułamek per rynek (lub override)
        if fraction is None:
            fraction = KELLY_FRACTIONS.get(rynek, KELLY_FRACTION)

        # Kalibracja liniowa p_model przed Kelly
        # Conservative Kelly: p_kelly = 0.5 + (p_model - 0.5) * KELLY_PROB_SCALE
        # Symulacja 5 lig × 2 sezony: -15% nadwyżki + Half-Kelly → MaxDD 55%→33%
        # Stosujemy TYLKO do Kelly, nie do wyświetlanego p_model ani EV
        p_kelly = 0.5 + (float(p_model) - 0.5) * KELLY_PROB_SCALE
        p_kelly = max(0.01, min(0.99, p_kelly))
        fraction = fraction * KELLY_FRAC_SCALE  # Half-Kelly

        # Rynki z fraction=0 są wyłączone z Kelly (tylko informacyjne)
        if fraction == 0.0:
            return {"f_full":0,"f_frac":0,"stake_pln":0,"ev_per_unit":0,
                    "safe":False,"rynek":rynek,"capped":False,
                    "disabled":True,"fraction_used":0}

        b = kurs_buk - 1.0
        q = 1.0 - p_kelly
        if b <= 0 or p_kelly <= 0 or p_kelly >= 1:
            return {"f_full":0,"f_frac":0,"stake_pln":0,"ev_per_unit":0,
                    "safe":False,"rynek":rynek,"capped":False}
        f_full = max(0.0, (p_kelly * b - q) / b)
        f_frac = f_full * fraction
        stake  = bankroll * f_frac

        # Max Exposure per Match – cap jeśli przekracza limit
        max_allowed = bankroll * MAX_EXPOSURE_PCT - already_exposed
        capped = False
        if stake > max_allowed > 0:
            stake  = max_allowed
            f_frac = stake / bankroll
            capped = True
        elif max_allowed <= 0:
            stake  = 0.0
            f_frac = 0.0
            capped = True

        ev_puu = p_kelly * b - q
        # EV filter: nie stawiamy jeśli EV < 5% (backtest: EV 0-5% → ROI -64%)
        ev_ok = ev_puu >= 0.05
        return {"f_full":round(f_full,4),"f_frac":round(f_frac,4),
                "stake_pln":round(stake, 2),"ev_per_unit":round(ev_puu,4),
                "safe": f_frac > 0 and ev_puu > 0 and ev_ok,
                "ev_ok": ev_ok,
                "rynek":rynek,"capped":capped,
                "fraction_used":round(fraction,3),
                "p_kelly":round(p_kelly,4)}
    except Exception:
        return {"f_full":0,"f_frac":0,"stake_pln":0,"ev_per_unit":0,
                "safe":False,"rynek":rynek,"capped":False}

def due_to_score_flag(team, srednie_df, historical):
    try:
        if "SOT (dom)" not in srednie_df.columns or team not in srednie_df.index:
            return None
        sot_d = float(srednie_df.loc[team, "SOT (dom)"])
        sot_w = float(srednie_df.loc[team, "SOT (wyjazd)"])
        sot_avg = (sot_d + sot_w) / 2
        if np.isnan(sot_avg): return None
        mecze_t = historical[(historical["HomeTeam"]==team)|(historical["AwayTeam"]==team)].tail(3)
        if len(mecze_t) < 3: return None
        gole = [int(m["FTHG"]) if m["HomeTeam"]==team else int(m["FTAG"])
                for _, m in mecze_t.iterrows()]
        avg_g = sum(gole)/len(gole)
        exp_g = sot_avg * 0.30
        if avg_g < exp_g * 0.60:
            return {"active":True,"sot_avg":round(sot_avg,1),"gole_last3":gole,
                    "avg_gole":round(avg_g,2),"expected":round(exp_g,2),
                    "deficit":round(exp_g-avg_g,2),
                    "msg":f"SOT avg {sot_avg:.1f} → tylko {avg_g:.1f} gola/mecz (ost.3). Oczekiwane odbicie."}
    except Exception:
        pass
    return None

def wybierz_wynik(M: np.ndarray, lam_h: float, lam_a: float) -> tuple:
    idx_max = np.unravel_index(M.argmax(), M.shape)
    p_max = float(M[idx_max])
    top3  = np.sort(M.flatten())[::-1][:3]
    if float(top3[0] - top3[2]) < 0.04:
        g_h = min(int(round(lam_h)), M.shape[0] - 1)
        g_a = min(int(round(lam_a)), M.shape[1] - 1)
        return g_h, g_a, float(M[g_h, g_a])
    return int(idx_max[0]), int(idx_max[1]), p_max

def entropy_meczu(p_home: float, p_draw: float, p_away: float) -> float:
    ps = [p for p in [p_home, p_draw, p_away] if p > 0]
    return float(-sum(p * np.log2(p) for p in ps))

def chaos_label(entropy: float) -> tuple:
    pct = entropy / np.log2(3)
    if pct < 0.55:   return "Pewny",    "🔒", pct
    elif pct < 0.75: return "Klarowny", "🎯", pct
    elif pct < 0.90: return "Otwarty",  "⚡", pct
    else:            return "Chaos",    "🌀", pct


def sot_momentum(team: str, historical: pd.DataFrame) -> float | None:
    """
    SOT Momentum = (avg SOT ostatnie 3 mecze) / (avg SOT sezon).
    > 1.3 → drużyna 'puka do bramki' silniej niż zwykle (under-rewarded jeśli goli mało).
    < 0.7 → drużyna wygasła.
    Zwraca ratio lub None jeśli brak danych.
    """
    if historical.empty or "HST" not in historical.columns:
        return None
    try:
        home_m = historical[historical["HomeTeam"] == team][["Date", "FTHG", "HST"]].rename(
            columns={"HST": "sot", "FTHG": "gole"})
        away_m = historical[historical["AwayTeam"] == team][["Date", "FTAG", "AST"]].rename(
            columns={"AST": "sot", "FTAG": "gole"})
        all_m = pd.concat([home_m, away_m]).dropna(subset=["sot"])
        all_m = all_m.sort_values("Date") if "Date" in all_m.columns else all_m
        if len(all_m) < 5:
            return None
        season_avg = float(all_m["sot"].mean())
        recent_avg = float(all_m.tail(3)["sot"].mean())
        if season_avg < 0.5:
            return None
        return round(recent_avg / season_avg, 2)
    except Exception:
        return None


def quality_score(p_typ: float, entropy: float,
                  sot_mom_h: float | None, sot_mom_a: float | None,
                  ev: float = 0.0) -> tuple[int, str, str]:
    """
    Quality Score 1-5 gwiazdek dla zakładu.
    Kryteria:
      • p_typ    → siła sygnału modelu
      • entropy  → pewność (niski chaos)
      • SOT mom  → atak pod presją
      • EV       → wartość vs rynek
    Zwraca (gwiazdki: int, label: str, tooltip: str)
    """
    score = 0.0
    reasons = []

    # 1. Prawdopodobieństwo (0-2 pkt)
    if p_typ >= 0.75:
        score += 2.0; reasons.append("p≥75%")
    elif p_typ >= 0.62:
        score += 1.2; reasons.append("p≥62%")
    elif p_typ >= 0.55:
        score += 0.6; reasons.append("p≥55%")

    # 2. Entropy (chaos) (0-1.5 pkt)
    ent_norm = entropy / np.log2(3)
    if ent_norm < 0.55:
        score += 1.5; reasons.append("Chaos Low")
    elif ent_norm < 0.75:
        score += 0.9; reasons.append("Chaos Med")
    elif ent_norm < 0.90:
        score += 0.3; reasons.append("Chaos High")
    # chaos pełny: 0 pkt

    # 3. SOT Momentum (0-1 pkt) — atak pod presją
    mom_h = sot_mom_h or 1.0
    mom_a = sot_mom_a or 1.0
    best_mom = max(mom_h, mom_a)
    if best_mom >= 1.35:
        score += 1.0; reasons.append("🔥 SOT Momentum")
    elif best_mom >= 1.15:
        score += 0.5; reasons.append("SOT ↑")

    # 4. EV vs rynek (0-0.5 pkt)
    if ev >= 0.12:
        score += 0.5; reasons.append("EV≥12%")
    elif ev >= 0.06:
        score += 0.25; reasons.append("EV≥6%")

    # Normalizuj do 1-5 gwiazdek (max raw ≈ 5.0)
    stars = int(np.clip(round(score), 1, 5))
    star_str = "★" * stars + "☆" * (5 - stars)
    label = {5: "Wzorcowy", 4: "Silny", 3: "Dobry", 2: "Umiarkowany", 1: "Ryzykowny"}[stars]
    tooltip = " · ".join(reasons) if reasons else "brak sygnałów"
    return stars, star_str, label, tooltip


def entropy_kelly_multiplier(entropy: float) -> float:
    """
    Płynny bezpiecznik Kelly na podstawie chaosu macierzy.
    entropy ~0.5 (pewny) → 1.0×   (pełna stawka)
    entropy ~1.0 (otwarty) → 0.70×
    entropy ~1.3 (chaos) → 0.45×
    entropy ≥1.5 → 0.30×
    """
    # Liniowa interpolacja w kawałkach
    if entropy <= 0.5:  return 1.00
    if entropy <= 1.0:  return 1.00 - 0.30 * (entropy - 0.5) / 0.5   # 1.00 → 0.70
    if entropy <= 1.3:  return 0.70 - 0.25 * (entropy - 1.0) / 0.3   # 0.70 → 0.45
    return max(0.30, 0.45 - 0.15 * (entropy - 1.3) / 0.2)            # 0.45 → 0.30

def confidence_score(p_home: float, p_draw: float, p_away: float) -> tuple:
    vals  = sorted([p_home, p_draw, p_away], reverse=True)
    spread = vals[0] - vals[2]
    edge   = vals[0] - vals[1]
    if edge > 0.18:   return "High",     "🟢", f"Wyraźny faworyt (+{edge:.0%} nad 2. opcją)"
    elif edge > 0.08: return "Medium",   "🟡", f"Umiarkowana przewaga (+{edge:.0%})"
    else:             return "Coinflip", "🔴", f"Mecz bardzo wyrównany (spread {spread:.0%})"

def fair_odds(p: float) -> float:
    return round(1 / p, 2) if 0 < p <= 1 else 999.0

def predykcja_meczu(lam_h: float, lam_a: float, rho: float = -0.13, csv_code: str = "E0", n_train: int = 200) -> dict:
    max_gole = int(np.clip(np.ceil(max(lam_h, lam_a) + 5), 8, 12))
    M = dixon_coles_adj(
        np.outer(poisson.pmf(range(max_gole), lam_h),
                 poisson.pmf(range(max_gole), lam_a)),
        lam_h, lam_a, rho=rho
    )
    p_home = float(np.tril(M, -1).sum())
    p_draw = float(np.trace(M))
    p_away = float(np.triu(M, 1).sum())
    wynik_h, wynik_a, p_exact = wybierz_wynik(M, lam_h, lam_a)
    p_home_cal, p_draw_cal, p_away_cal = kalibruj_prawdopodobienstwa(p_home, p_draw, p_away, csv_code, n_train)
    typ, p_typ = wybierz_typ(p_home, p_draw, p_away, csv_code, n_train)
    conf_level, conf_emoji, conf_opis = confidence_score(p_home_cal, p_draw_cal, p_away_cal)
    ent = entropy_meczu(p_home_cal, p_draw_cal, p_away_cal)
    ch_label, ch_emoji, ch_pct = chaos_label(ent)
    return {
        "lam_h": lam_h, "lam_a": lam_a,
        "p_home": p_home_cal, "p_draw": p_draw_cal, "p_away": p_away_cal,
        "p_home_raw": p_home, "p_draw_raw": p_draw, "p_away_raw": p_away,
        "wynik_h": wynik_h, "wynik_a": wynik_a, "p_exact": p_exact,
        "fo_home": fair_odds(p_home_cal), "fo_draw": fair_odds(p_draw_cal), "fo_away": fair_odds(p_away_cal),
        "typ": typ, "p_typ": p_typ, "fo_typ": fair_odds(p_typ),
        "conf_level": conf_level, "conf_emoji": conf_emoji, "conf_opis": conf_opis,
        "entropy": ent, "chaos_label": ch_label, "chaos_emoji": ch_emoji, "chaos_pct": ch_pct,
        "macierz": M,
        "shrink_uzyte": _get_shrink(csv_code, n_train),
        "n_train": n_train,
        "ci_half": _get_shrink(csv_code, n_train) * 0.5,
    }

# ===========================================================================
# ALTERNATYWNE ZDARZENIA
# ===========================================================================
def alternatywne_zdarzenia(lam_h: float, lam_a: float, lam_r: float,
                            lam_k: float, rho: float,
                            prog_min: float = 0.55,
                            lam_sot: float = None) -> list:
    zdarzenia = []
    mg = int(np.clip(np.ceil(max(lam_h, lam_a) + 5), 8, 12))
    M = dixon_coles_adj(
        np.outer(poisson.pmf(range(mg), lam_h), poisson.pmf(range(mg), lam_a)),
        lam_h, lam_a, rho=rho
    )

    for linia in [1.5, 2.5, 3.5]:
        p_over = float(sum(M[i,j] for i in range(mg) for j in range(mg) if i+j > linia))
        p_under = 1 - p_over
        if p_over >= prog_min:
            zdarzenia.append(("⚽", f"Over {linia} goli",  p_over,  fair_odds(p_over),  "Gole", linia))
        if p_under >= prog_min:
            zdarzenia.append(("⚽", f"Under {linia} goli", p_under, fair_odds(p_under), "Gole", linia))

    p_btts   = float(1 - M[0,:].sum() - M[:,0].sum() + M[0,0])
    p_nobtts = 1 - p_btts
    if p_btts   >= prog_min: zdarzenia.append(("⚽", "BTTS – Tak",  p_btts,   fair_odds(p_btts),   "BTTS", 0))
    if p_nobtts >= prog_min: zdarzenia.append(("⚽", "BTTS – Nie",  p_nobtts, fair_odds(p_nobtts), "BTTS", 0))

    for linia in [7.5, 8.5, 9.5, 10.5]:
        p_over = float(1 - poisson.cdf(int(linia), lam_r))
        if p_over >= prog_min:
            zdarzenia.append(("🚩", f"Over {linia} rożnych", p_over, fair_odds(p_over), "Rożne", linia))

    for linia in [2.5, 3.5, 4.5]:
        p_over = float(1 - poisson.cdf(int(linia), lam_k))
        if p_over >= prog_min:
            zdarzenia.append(("🟨", f"Over {linia} kartek", p_over, fair_odds(p_over), "Kartki", linia))

    # Celne strzały (HST+AST) – Poisson, lam_sot przekazywane opcjonalnie
    # Filtr odds >= 1.30 – poniżej tej granicy rynek nie ma wartości
    SOT_MIN_ODDS = 1.30
    if lam_sot is not None and lam_sot > 0:
        for linia in [3.5, 4.5, 5.5, 6.5]:
            p_over = float(1 - poisson.cdf(int(linia), lam_sot))
            fo_sot = fair_odds(p_over)
            if p_over >= prog_min and fo_sot >= SOT_MIN_ODDS:
                zdarzenia.append(("🎯", f"Over {linia} celnych", p_over,
                                  fo_sot, "SOT", linia))

    return sorted(zdarzenia, key=lambda x: -x[2])

# ===========================================================================
# ZAPIS ZDARZEŃ DO BAZY (tracking skuteczności)
# ===========================================================================
def zapisz_zdarzenia(liga: str, kolejnosc: int, mecz: str, home: str, away: str,
                     rynek: str, typ: str, linia: float, p_model: float, fair_odds: float,
                     sezon: str = None):
    """Zapisuje pojedyncze zdarzenie do bazy (do późniejszej weryfikacji)"""
    if sezon is None:
        sezon = BIEZACY_SEZON
    init_db()
    con = sqlite3.connect(DB_FILE)
    try:
        con.execute(
            """INSERT OR IGNORE INTO zdarzenia 
               (liga, sezon, kolejnosc, mecz, home, away, rynek, linia, typ, p_model, fair_odds, data)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (liga, sezon, kolejnosc, mecz, home, away, rynek, linia, typ, p_model, fair_odds,
             datetime.now().strftime("%Y-%m-%d"))
        )
        con.commit()
    except Exception as e:
        st.warning(f"Błąd zapisu zdarzenia: {e}")
    finally:
        con.close()

def aktualizuj_wynik_zdarzenia(home: str, away: str, hist: pd.DataFrame):
    """Po meczu aktualizuje wszystkie zdarzenia dla danego meczu"""
    if "_sezon" in hist.columns:
        hist = hist[hist["_sezon"] == "biezacy"]
    
    match = hist[(hist["HomeTeam"] == home) & (hist["AwayTeam"] == away)]
    if match.empty:
        return
    
    row = match.iloc[-1]
    hg = int(row["FTHG"]); ag = int(row["FTAG"])
    wynik_1x2 = "1" if hg > ag else ("2" if ag > hg else "X")
    rzuty = int(row.get("HC", 0)) + int(row.get("AC", 0))
    kartki = int(row.get("HY", 0)) + int(row.get("AY", 0)) + \
             (int(row.get("HR", 0)) + int(row.get("AR", 0))) * 2
    
    init_db()
    con = sqlite3.connect(DB_FILE)
    
    # Pobierz wszystkie zdarzenia dla tego meczu
    zdarzenia = con.execute(
        "SELECT id, rynek, typ, linia FROM zdarzenia WHERE home=? AND away=? AND trafione IS NULL",
        (home, away)
    ).fetchall()
    
    for z in zdarzenia:
        zid, rynek, typ, linia = z
        trafione = False
        
        if rynek == "Gole":
            if "Over" in typ:
                trafione = (hg + ag) > linia
            elif "Under" in typ:
                trafione = (hg + ag) < linia
        elif rynek == "BTTS":
            if typ == "BTTS – Tak":
                trafione = (hg > 0 and ag > 0)
            else:
                trafione = (hg == 0 or ag == 0)
        elif rynek == "Rożne":
            trafione = rzuty > linia
        elif rynek == "Kartki":
            trafione = kartki > linia
        elif rynek == "1X2":
            if typ == "1": trafione = (wynik_1x2 == "1")
            elif typ == "X": trafione = (wynik_1x2 == "X")
            elif typ == "2": trafione = (wynik_1x2 == "2")
            elif typ == "1X": trafione = (wynik_1x2 in ("1", "X"))
            elif typ == "X2": trafione = (wynik_1x2 in ("X", "2"))
        elif rynek == "SOT":
            if "Over" in typ:
                # Celne strzały – potrzebujemy HST+AST
                hst = int(row.get("HST", 0)) if not pd.isna(row.get("HST", 0)) else 0
                ast = int(row.get("AST", 0)) if not pd.isna(row.get("AST", 0)) else 0
                total_sot = hst + ast
                trafione = total_sot > linia
        
        con.execute(
            "UPDATE zdarzenia SET wynik=?, trafione=? WHERE id=?",
            (f"{hg}:{ag}", trafione, zid)
        )
    
    con.commit()
    con.close()


def aktualizuj_wyniki_z_api(fd_org_id: int, api_key: str) -> int:
    """Pobiera wyniki zakończonych meczów z fd.org API i aktualizuje bazę.
    Szybsze niż CSV – dane dostępne ~15min po meczu.
    Zwraca liczbę zaktualizowanych meczów."""
    if not api_key:
        return 0
    try:
        import requests as _req
        from datetime import date as _date, timedelta as _td
        headers = {"X-Auth-Token": api_key}
        # Pobierz mecze z ostatnich 7 dni które mają status FINISHED
        _date_from = (_date.today() - _td(days=7)).isoformat()
        _date_to   = _date.today().isoformat()
        url = (f"https://api.football-data.org/v4/competitions/{fd_org_id}/matches"
               f"?status=FINISHED&dateFrom={_date_from}&dateTo={_date_to}")
        resp = _req.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        matches = resp.json().get("matches", [])

        init_db()
        con = sqlite3.connect(DB_FILE)
        n_updated = 0

        for m in matches:
            score = m.get("score", {}).get("fullTime", {})
            hg = score.get("home")
            ag = score.get("away")
            if hg is None or ag is None:
                continue
            hg, ag = int(hg), int(ag)
            ht_name = m.get("homeTeam", {}).get("shortName") or m.get("homeTeam", {}).get("name", "")
            at_name = m.get("awayTeam", {}).get("shortName") or m.get("awayTeam", {}).get("name", "")
            # Spróbuj znaleźć w bazie przez map_nazwa
            ht_mapped = map_nazwa(ht_name)
            at_mapped = map_nazwa(at_name)
            wynik_1x2 = "1" if hg > ag else ("2" if ag > hg else "X")

            zdarzenia = con.execute(
                "SELECT id, rynek, typ, linia FROM zdarzenia WHERE home=? AND away=? AND trafione IS NULL",
                (ht_mapped, at_mapped)
            ).fetchall()
            if not zdarzenia:
                # Spróbuj też oryginalne nazwy
                zdarzenia = con.execute(
                    "SELECT id, rynek, typ, linia FROM zdarzenia WHERE home=? AND away=? AND trafione IS NULL",
                    (ht_name, at_name)
                ).fetchall()

            for zid, rynek, typ, linia in zdarzenia:
                trafione = False
                if rynek == "1X2":
                    if typ == "1":  trafione = (wynik_1x2 == "1")
                    elif typ == "X": trafione = (wynik_1x2 == "X")
                    elif typ == "2": trafione = (wynik_1x2 == "2")
                    elif typ == "1X": trafione = (wynik_1x2 in ("1","X"))
                    elif typ == "X2": trafione = (wynik_1x2 in ("X","2"))
                elif rynek == "Gole":
                    trafione = ((hg+ag) > linia) if "Over" in typ else ((hg+ag) < linia)
                elif rynek == "BTTS":
                    trafione = (hg>0 and ag>0) if "Tak" in typ else (hg==0 or ag==0)
                con.execute(
                    "UPDATE zdarzenia SET wynik=?, trafione=? WHERE id=?",
                    (f"{hg}:{ag}", trafione, zid)
                )
                n_updated += 1

        con.commit()
        con.close()
        return n_updated // max(1, len(matches) or 1)  # approx mecze
    except Exception:
        return 0


# ===========================================================================
# STATYSTYKI SKUTECZNOŚCI
# ===========================================================================
def statystyki_skutecznosci(liga: str = None) -> pd.DataFrame:
    """Zwraca DataFrame ze statystykami skuteczności per rynek"""
    init_db()
    con = sqlite3.connect(DB_FILE)
    
    query = "SELECT rynek, typ, linia, p_model, fair_odds, trafione FROM zdarzenia WHERE trafione IS NOT NULL"
    params = []
    if liga:
        query += " AND liga=?"
        params.append(liga)
    
    df = pd.read_sql_query(query, con, params=params)
    con.close()
    
    if df.empty:
        return pd.DataFrame()
    
    # Grupuj po rynku
    stats = []
    for (rynek, typ, linia), group in df.groupby(['rynek', 'typ', 'linia']):
        # Buduj czytelną nazwę: "Gole Over 2.5", "Gole Under 2.5", "BTTS Tak" etc.
        # typ może być: "Over", "Under", "Tak", "Nie", "Over 2.5 goli", "Under 1.5 goli"
        # Normalizuj: wyciągnij Over/Under z długiej nazwy
        typ_norm = typ or ""
        if "Over" in typ_norm and typ_norm != "Over":
            typ_norm = "Over"
        elif "Under" in typ_norm and typ_norm != "Under":
            typ_norm = "Under"

        nazwa = f"{rynek}"
        if typ_norm in ("Tak", "Nie"):
            nazwa += f" {typ_norm}"
        elif typ_norm in ("Over", "Under") and linia and linia > 0:
            nazwa += f" {typ_norm} {float(linia):.1f}"
        elif linia and linia > 0:
            nazwa += f" {float(linia):.1f}"
        elif typ_norm:
            nazwa += f" {typ_norm}"
        
        trafione = group['trafione'].sum()
        wszystkie = len(group)
        skutecznosc = trafione / wszystkie if wszystkie > 0 else 0
        sr_p_model = group['p_model'].mean()
        sr_fair = group['fair_odds'].mean()
        
        roi = (trafione * (sr_fair - 1) - (wszystkie - trafione)) / wszystkie if wszystkie > 0 else 0
        # Brier Score dla zdarzeń binarnych: mean((p - y)^2)
        brier = float(((group["p_model"] - group["trafione"].astype(float)) ** 2).mean())
        # Brier Score losowego modelu dla tego rynku to sr_p_model*(1-sr_p_model)*2 ≈ 0.25 przy p=0.5
        # Referencja: model mówi zawsze sr_p_model → BS_ref = sr_p_model*(1-skutecznosc)^2 + (1-sr_p_model)*skutecznosc^2
        brier_ref = sr_p_model*(1-skutecznosc)**2 + (1-sr_p_model)*skutecznosc**2
        brier_skill = 1 - brier/brier_ref if brier_ref > 0 else 0  # >0 = lepszy od baseline

        stats.append({
            "Rynek": nazwa,
            "Typów": wszystkie,
            "Trafione": int(trafione),
            "Skuteczność": f"{skutecznosc:.1%}",
            "Śr. P model": f"{sr_p_model:.1%}",
            "Śr. Fair": f"{sr_fair:.2f}",
            "Brier ↓": round(brier, 3),
            "Skill": round(brier_skill, 2),
            "ROI": f"{roi:+.1%}",
            # wartości numeryczne do sortowania/wykresu
            "_roi_v": roi,
            "_brier_v": brier,
            "_skut_v": skutecznosc,
            "Kolor": "🟢" if roi > 0.03 else ("🟡" if roi > -0.03 else "🔴")
        })
    
    return pd.DataFrame(stats).sort_values("Typów", ascending=False)

def kalibracja_modelu(liga: str = None) -> pd.DataFrame:
    """Zwraca dane do wykresu kalibracji"""
    init_db()
    con = sqlite3.connect(DB_FILE)
    
    query = "SELECT p_model, trafione FROM zdarzenia WHERE trafione IS NOT NULL"
    params = []
    if liga:
        query += " AND liga=?"
        params.append(liga)
    
    df = pd.read_sql_query(query, con, params=params)
    con.close()
    
    if df.empty:
        return pd.DataFrame()
    
    # Podziel na przedziały
    bins = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 1.0]
    labels = ["50-55%", "55-60%", "60-65%", "65-70%", "70-75%", "75-80%", "80%+"]
    
    df['przedzial'] = pd.cut(df['p_model'], bins=bins, labels=labels, right=False)
    
    kalibracja = df.groupby('przedzial').agg(
        liczba=('trafione', 'count'),
        trafione=('trafione', 'sum'),
        p_srednia=('p_model', 'mean')
    ).reset_index()
    
    kalibracja['skutecznosc'] = kalibracja['trafione'] / kalibracja['liczba']
    kalibracja['rozbieznosc'] = kalibracja['skutecznosc'] - kalibracja['p_srednia']
    
    return kalibracja

# ===========================================================================
# METRYKI ANALITYCZNE – Brier, ECE, Rolling
# ===========================================================================
def metryki_globalne(liga: str = None) -> dict:
    """Zwraca globalne metryki modelu: Brier Score, ECE, sharpness."""
    init_db()
    con = sqlite3.connect(DB_FILE)
    q = "SELECT p_model, trafione FROM zdarzenia WHERE trafione IS NOT NULL"
    params = []
    if liga:
        q += " AND liga=?"; params.append(liga)
    df = pd.read_sql_query(q, con, params=params)
    con.close()
    if df.empty or len(df) < 5:
        return {}
    y   = df["trafione"].astype(float).values
    p   = df["p_model"].values
    n   = len(df)
    # Brier Score
    brier = float(np.mean((p - y) ** 2))
    # Brier Skill Score vs naive (baseline: zawsze przewiduj średnią)
    p_mean = y.mean()
    brier_naive = float(np.mean((p_mean - y) ** 2))
    bss = 1 - brier / brier_naive if brier_naive > 0 else 0.0
    # ECE – Expected Calibration Error (ważona średnia |p_bucket - hit_rate|)
    bins   = np.arange(0.5, 1.05, 0.05)
    labels = [f"{b:.0%}" for b in bins[:-1]]
    df2 = df.copy()
    df2["bucket"] = pd.cut(df2["p_model"], bins=bins, labels=labels, right=False)
    ece = 0.0
    for _, g in df2.groupby("bucket", observed=True):
        if len(g) == 0: continue
        ece += (len(g) / n) * abs(g["p_model"].mean() - g["trafione"].mean())
    # Sharpness – odchylenie standardowe prognoz (im wyższe, tym bardziej zdecydowany model)
    sharpness = float(np.std(p))
    return {
        "n": n, "brier": round(brier, 4), "bss": round(bss, 4),
        "ece": round(ece, 4), "sharpness": round(sharpness, 4),
        "avg_p": round(float(p.mean()), 3), "hit_rate": round(float(y.mean()), 3),
    }

def model_sharpness_vs_rynek(liga: str, oa_cached: dict, oa_module,
                              schedule: pd.DataFrame, srednie_df,
                              srednie_lig: dict, forma_dict: dict,
                              rho: float, n_biezacy: int, csv_code: str) -> dict:
    """
    Liczy agregat rozbieżności model vs rynek dla bieżącej kolejki.
    Sweet spot: srednia roznica 5-15% – tam leżą pieniądze.
    """
    if not oa_cached or schedule.empty or srednie_df.empty:
        return {}
    aktualna_k = get_current_round(schedule)
    mecze_k = schedule[schedule["round"] == aktualna_k]
    diffs = []
    sweet = 0
    noise = 0
    aligned = 0
    for _, mecz in mecze_k.iterrows():
        h = map_nazwa(mecz["home_team"])
        a = map_nazwa(mecz["away_team"])
        if h not in srednie_df.index or a not in srednie_df.index:
            continue
        try:
            lh, la, lr, lk, _, lsot = oblicz_lambdy(h, a, srednie_df, srednie_lig, forma_dict,
                               csv_code=LIGI[wybrana_liga]["csv_code"])
            pred = predykcja_meczu(lh, la, rho=rho, csv_code=csv_code, n_train=n_biezacy)
            o = oa_module.znajdz_kursy(h, a, oa_cached)
            if not o:
                continue
            s = 1/o["odds_h"] + 1/o["odds_d"] + 1/o["odds_a"]
            impl = {"1": (1/o["odds_h"])/s, "X": (1/o["odds_d"])/s, "2": (1/o["odds_a"])/s}
            # DC implied dla double chances
            impl["1X"] = impl["1"] + impl["X"]
            impl["X2"] = impl["X"] + impl["2"]
            p_impl = impl.get(pred["typ"], 0)
            diff = abs(pred["p_typ"] - p_impl)
            diffs.append({"mecz": f"{h}–{a}", "typ": pred["typ"],
                          "p_model": pred["p_typ"], "p_impl": p_impl, "diff": diff})
            if diff < 0.05:
                aligned += 1
            elif diff <= 0.15:
                sweet += 1
            else:
                noise += 1
        except Exception:
            continue
    if not diffs:
        return {}
    avg_diff = sum(d["diff"] for d in diffs) / len(diffs)
    return {
        "avg_diff": avg_diff,
        "n_mecze": len(diffs),
        "aligned": aligned,   # < 5% – model = rynek
        "sweet": sweet,       # 5-15% – sweet spot
        "noise": noise,       # > 15% – za duże różnice
        "details": diffs,
    }


def rolling_stats(liga: str = None, okno: int = 50) -> pd.DataFrame:
    """Zwraca rolling Brier Score i skuteczność per zdarzenie (posortowane chronologicznie)."""
    init_db()
    con = sqlite3.connect(DB_FILE)
    q = "SELECT data, kolejnosc, p_model, trafione FROM zdarzenia WHERE trafione IS NOT NULL"
    params = []
    if liga:
        q += " AND liga=?"; params.append(liga)
    q += " ORDER BY data ASC, kolejnosc ASC"
    df = pd.read_sql_query(q, con, params=params)
    con.close()
    if df.empty or len(df) < okno:
        return pd.DataFrame()
    df["y"]         = df["trafione"].astype(float)
    df["brier_pkt"] = (df["p_model"] - df["y"]) ** 2
    df["idx"]       = range(len(df))
    # Rolling średnia
    df["brier_roll"] = df["brier_pkt"].rolling(okno, min_periods=okno//2).mean()
    df["hit_roll"]   = df["y"].rolling(okno, min_periods=okno//2).mean()
    df["p_roll"]     = df["p_model"].rolling(okno, min_periods=okno//2).mean()
    return df[["idx", "data", "kolejnosc", "brier_roll", "hit_roll", "p_roll"]].dropna()

def brier_per_kolejka(liga: str = None) -> pd.DataFrame:
    """Średni Brier Score per kolejka – do wykresu trendów."""
    init_db()
    con = sqlite3.connect(DB_FILE)
    q = ("SELECT kolejnosc, AVG((p_model - CAST(trafione AS REAL)) * (p_model - CAST(trafione AS REAL))) as brier, "
         "AVG(CAST(trafione AS REAL)) as hit_rate, COUNT(*) as n "
         "FROM zdarzenia WHERE trafione IS NOT NULL")
    params = []
    if liga:
        q += " AND liga=?"; params.append(liga)
    q += " GROUP BY kolejnosc ORDER BY kolejnosc"
    df = pd.read_sql_query(q, con, params=params)
    con.close()
    return df


# ===========================================================================
# FORMA + KONTEKST (zastępuje AI komentarz)
# ===========================================================================
def _forma_badge(litera: str) -> str:
    """Zwraca span z kolorowym tłem dla W/D/L."""
    colors = {"W": ("#1b5e20","#4CAF50"), "D": ("#4a3800","#FFC107"), "L": ("#7f0000","#ef5350")}
    bg, fg = colors.get(litera, ("#1a1a1a","#888"))
    return (f"<span style='background:{bg};color:{fg};padding:2px 6px;"
            f"border-radius:4px;font-weight:700;font-size:0.82em;margin:1px'>{litera}</span>")

def _forma_html(forma_str: str) -> str:
    """Render formy jako kolorowe badge'e."""
    if not forma_str or forma_str == "?": return "<span style='color:#555'>brak danych</span>"
    return "".join(_forma_badge(c) for c in forma_str)

def _trend_forma(forma_str: str) -> tuple:
    """Zwraca (trend_opis, trend_color)."""
    if not forma_str or len(forma_str) < 3:
        return "za mało meczów", "#555"
    ostatnie2 = forma_str[-2:]
    w = forma_str.count("W"); l = forma_str.count("L")
    if ostatnie2 in ("WW",) or w >= 4:
        return "↗ rosnąca forma", "#4CAF50"
    if ostatnie2 in ("LL",) or l >= 3:
        return "↘ spadkowa forma", "#ef5350"
    if w == l:
        return "→ stabilna forma", "#FFC107"
    return "→ zmienna forma", "#888"

def _stat_bar(val, avg_liga, label, unit="", higher_better=True) -> str:
    """Mini pasek porównania z ligową średnią."""
    if val is None or avg_liga is None or avg_liga == 0:
        return f"<span style='color:#555;font-size:0.78em'>{label}: brak</span>"
    ratio = val / avg_liga
    pct = min(ratio * 50, 100)  # 50% = średnia ligi
    c = "#4CAF50" if (ratio > 1.1 and higher_better) or (ratio < 0.9 and not higher_better)         else ("#ef5350" if (ratio < 0.9 and higher_better) or (ratio > 1.1 and not higher_better)         else "#FFC107")
    bar_w = int(min(pct, 100))
    diff = val - avg_liga
    sign = "+" if diff >= 0 else ""
    return (f"<div style='margin:3px 0'>"
            f"<div style='display:flex;justify-content:space-between;margin-bottom:2px'>"
            f"<span style='font-size:0.75em;color:#666'>{label}</span>"
            f"<span style='font-size:0.75em;color:{c};font-weight:600'>"
            f"{val:.2f}{unit} <span style='color:#444'>({sign}{diff:.2f})</span></span></div>"
            f"<div style='background:var(--bg-card2);border-radius:3px;height:4px'>"
            f"<div style='background:{c};width:{bar_w}%;height:4px;border-radius:3px'></div>"
            f"</div></div>")

def render_forma_kontekst(home: str, away: str, pred: dict,
                           forma_dict: dict, srednie_df, srednie_lig: dict,
                           odds_buk: dict = None) -> str:
    """Generuje HTML panel z formą drużyn i kontekstem statystycznym."""
    fh = forma_dict.get(home, "?")
    fa = forma_dict.get(away, "?")
    lh = float(pred.get("lam_h", 1.2))
    la = float(pred.get("lam_a", 1.0))
    ph = float(pred.get("p_home", 0.33))
    pd_ = float(pred.get("p_draw", 0.33))
    pa = float(pred.get("p_away", 0.33))
    ent = float(pred.get("entropy", 0))
    avg_h_lig = srednie_lig.get("avg_home", 1.5)
    avg_a_lig = srednie_lig.get("avg_away", 1.2)

    # Statystyki per drużyna z srednie_df
    def _get(team, col, default=None):
        try:
            v = srednie_df.loc[team, col]
            return float(v) if v is not None and str(v) != "nan" else default
        except Exception: return default

    h_gs   = _get(home, "Gole strzelone (dom)")
    h_gc   = _get(home, "Gole stracone (dom)")
    a_gs   = _get(away, "Gole strzelone (wyjazd)")
    a_gc   = _get(away, "Gole stracone (wyjazd)")
    h_sot  = _get(home, "SOT (dom)")
    a_sot  = _get(away, "SOT (wyjazd)")
    h_konw = _get(home, "Konwersja (dom)")
    a_konw = _get(away, "Konwersja (wyjazd)")
    avg_sot_h_lig = srednie_lig.get("avg_sot_home")
    avg_sot_a_lig = srednie_lig.get("avg_sot_away")

    trend_h, tc_h = _trend_forma(fh)
    trend_a, tc_a = _trend_forma(fa)

    # Konflikt forma vs model
    konflikty = []
    if ph > 0.50 and fh.count("L") >= 3:
        konflikty.append(f"⚠️ {home}: słaba forma [{fh}] vs model faworyt (λ={lh:.2f}) – model widzi siłę statystyczną, nie serię")
    elif pa > 0.50 and fa.count("L") >= 3:
        konflikty.append(f"⚠️ {away}: słaba forma [{fa}] vs model faworyt (λ={la:.2f}) – model widzi siłę statystyczną, nie serię")
    if ph > 0.50 and fh.count("W") >= 4:
        konflikty.append(f"✅ {home}: forma [{fh}] zbieżna z modelem")
    if pa > 0.50 and fa.count("W") >= 4:
        konflikty.append(f"✅ {away}: forma [{fa}] zbieżna z modelem")

    # Deep value vs bukmacher
    dv_html = ""
    if odds_buk:
        try:
            typ = pred.get("typ","?")
            k_buk = {"1": float(odds_buk.get("odds_h",0) or 0),
                     "X": float(odds_buk.get("odds_d",0) or 0),
                     "2": float(odds_buk.get("odds_a",0) or 0)}.get(typ, 0)
            fo = float(pred.get("fo_typ", 0) or 0)
            if k_buk > 1 and fo > 1:
                rozb = (k_buk - fo) / fo
                if abs(rozb) >= 0.08:
                    dv_c = "#4CAF50" if rozb > 0 else "#ef5350"
                    dv_label = "⚡ Rynek płaci więcej niż fair" if rozb > 0 else "⚠️ Rynek poniżej fair"
                    dv_html = (f"<div style='margin-top:8px;background:var(--bg-card);border-left:3px solid {dv_c};"
                               f"padding:6px 10px;border-radius:0 4px 4px 0;font-size:0.8em'>"
                               f"<span style='color:{dv_c};font-weight:700'>{dv_label}:</span> "
                               f"fair={fo:.2f} · buk={k_buk:.2f} · różnica {rozb:+.0%}</div>")
        except Exception: pass

    # Entropia – jak przewidywalny mecz?
    if ent < 0.85:   chaos_label, chaos_c = "Łatwy do typowania", "#4CAF50"
    elif ent < 1.10: chaos_label, chaos_c = "Umiarkowanie otwarty", "#FFC107"
    else:            chaos_label, chaos_c = "Losowy mecz", "#ef5350"

    # Buduj HTML
    konflikty_html = ""
    if konflikty:
        konflikty_html = "".join(
            f"<div style='font-size:0.78em;color:#aaa;margin:3px 0;padding:4px 8px;"
            f"background:var(--bg-card);border-radius:4px'>{k}</div>"
            for k in konflikty)

    html = (
        f"<div style='background:var(--bg-app);border:1px solid var(--border);border-radius:8px;"
        f"padding:12px 14px;font-family:inherit'>"
        # Wiersz drużyn
        f"<div style='display:grid;grid-template-columns:1fr auto 1fr;gap:12px;align-items:start'>"
        # Lewa: dom
        f"<div>"
        f"<div style='font-size:0.8em;color:#555;margin-bottom:4px'>🏠 {home}</div>"
        f"<div style='margin-bottom:6px'>{_forma_html(fh)}</div>"
        f"<div style='font-size:0.75em;color:{tc_h}'>{trend_h}</div>"
        + _stat_bar(h_gs, avg_h_lig, "Gole strzel. (dom)")
        + _stat_bar(h_gc, avg_h_lig, "Gole strac. (dom)", higher_better=False)
        + ((_stat_bar(h_sot, avg_sot_h_lig, "SOT (dom)")) if h_sot and avg_sot_h_lig else "")
        + f"</div>"
        # Środek: szanse
        f"<div style='text-align:center;padding:0 8px'>"
        f"<div style='font-size:0.7em;color:#444;margin-bottom:4px'>MODEL</div>"
        f"<div style='font-size:1.1em;font-weight:800;color:#eee'>"
        f"<span style='color:#2196F3'>{ph:.0%}</span> · "
        f"<span style='color:#FF9800'>{pd_:.0%}</span> · "
        f"<span style='color:#E91E63'>{pa:.0%}</span></div>"
        f"<div style='font-size:0.65em;color:#444;margin-top:2px'>1 · X · 2</div>"
        f"<div style='margin-top:8px;font-size:0.7em;color:{chaos_c}'>{chaos_label}</div>"
        f"<div style='font-size:0.68em;color:#444'>H={ent:.2f}</div>"
        f"<div style='margin-top:8px'>"
        f"<div style='font-size:0.7em;color:#444'>λ domu</div>"
        f"<div style='font-size:0.95em;font-weight:700;color:#eee'>{lh:.2f}</div>"
        f"<div style='font-size:0.7em;color:#444;margin-top:4px'>λ gości</div>"
        f"<div style='font-size:0.95em;font-weight:700;color:#eee'>{la:.2f}</div>"
        f"</div></div>"
        # Prawa: wyjazd
        f"<div style='text-align:right'>"
        f"<div style='font-size:0.8em;color:#555;margin-bottom:4px'>{away} ✈️</div>"
        f"<div style='margin-bottom:6px;text-align:right'>{_forma_html(fa)}</div>"
        f"<div style='font-size:0.75em;color:{tc_a};text-align:right'>{trend_a}</div>"
        + _stat_bar(a_gs, avg_a_lig, "Gole strzel. (wyjazd)")
        + _stat_bar(a_gc, avg_a_lig, "Gole strac. (wyjazd)", higher_better=False)
        + ((_stat_bar(a_sot, avg_sot_a_lig, "SOT (wyjazd)")) if a_sot and avg_sot_a_lig else "")
        + f"</div>"
        f"</div>"
        # Dół: konflikty + deep value
        + (f"<div style='margin-top:8px;border-top:1px solid #1e2028;padding-top:8px'>{konflikty_html}</div>" if konflikty_html else "")
        + dv_html
        + f"</div>"
    )
    return html


# ===========================================================================
# HELPERS UI
# ===========================================================================
def koloruj(p: float) -> str:
    return "🟢" if p > 0.65 else ("🟡" if p > 0.50 else "🔴")

def oblicz_p(typ: str, linia: float, lam: float) -> float:
    return 1 - poisson.cdf(linia, lam) if typ == "Over" else poisson.cdf(linia, lam)

def badge_typ(typ: str) -> str:
    if   typ == "1X": styl = "background:linear-gradient(90deg,#2196F3,#FF9800)"
    elif typ == "X2": styl = "background:linear-gradient(90deg,#FF9800,#E91E63)"
    else:
        kolory = {"1": "#2196F3", "X": "#FF9800", "2": "#E91E63"}
        styl   = f"background:{kolory.get(typ, '#888')}"
    return (f"<span style='{styl};color:var(--text-primary);padding:2px 14px;"
            f"border-radius:12px;font-weight:bold;font-size:0.95em'>{typ}</span>")

def render_macierz_html(M: np.ndarray, home: str, away: str) -> str:
    max_val = M.max() if M.max() > 0 else 1
    best    = np.unravel_index(M.argmax(), M.shape)
    n       = M.shape[0]
    rows    = ["<table style='border-collapse:collapse;font-size:0.70em;width:100%;margin-top:6px'>",
               "<thead><tr>",
               f"<th style='padding:2px 5px;color:#999;font-size:0.85em'>{home[:7]}↓ {away[:7]}→</th>"]
    for j in range(n):
        rows.append(f"<th style='padding:2px 5px;text-align:center;color:#aaa'>{j}</th>")
    rows.append("</tr></thead><tbody>")
    for i in range(n):
        rows.append("<tr>")
        rows.append(f"<td style='padding:2px 5px;font-weight:bold;color:#aaa;text-align:center'>{i}</td>")
        for j in range(n):
            v = M[i,j]; ratio = v / max_val
            g = int(255 * (1 - ratio * 0.85))
            bold = "font-weight:bold;" if (i,j) == best else ""
            bord = "border:2px solid #E91E63;" if (i,j) == best else ""
            rows.append(f"<td style='padding:2px 5px;text-align:center;background:rgb(255,{g},{g});{bold}{bord}'>{v:.1%}</td>")
        rows.append("</tr>")
    rows.append("</tbody></table>")
    return "".join(rows)

def macierz_goli_p(lam_h, lam_a, rho, linia_int, typ_gole):
    mg = int(np.clip(np.ceil(max(lam_h, lam_a) + 5), 8, 12))
    M  = dixon_coles_adj(
        np.outer(poisson.pmf(range(mg), lam_h), poisson.pmf(range(mg), lam_a)),
        lam_h, lam_a, rho=rho
    )
    p_over = float(sum(M[i,j] for i in range(mg) for j in range(mg) if i+j > linia_int))
    return p_over if typ_gole == "Over" else 1 - p_over

# ===========================================================================
# NOWA FUNKCJA – OSTRZEŻENIA SĘDZIOWSKIE
# ===========================================================================
def ostrzezenie_sedziego(sedzia, sedziowie_df, prog_kartki=4.5):
    """Zwraca ostrzeżenie dotyczące sędziego lub None gdy brak danych.
    Dane sędziów dostępne tylko dla Premier League (kolumna Referee w CSV).
    """
    if sedziowie_df.empty or sedzia in ("Nieznany", "", None):
        return None
    if sedzia not in sedziowie_df["Sędzia"].values:
        return None
    sedz = sedziowie_df[sedziowie_df["Sędzia"] == sedzia].iloc[0]
    avg_cards = sedz["Total Kart/M ↓"]
    if avg_cards >= prog_kartki:
        return f"🚩 Wysokie ryzyko kartek – śr. {avg_cards:.1f}/mecz"
    elif avg_cards >= 3.5:
        return f"🟡 Umiarkowane ryzyko kartek ({avg_cards:.1f}/mecz)"
    else:
        return f"🟢 Niskie ryzyko kartek ({avg_cards:.1f}/mecz)"

# ===========================================================================
# DEEP DATA – Power Rankings + Sędziowie
# ===========================================================================
@st.cache_data
def deep_data_stats(df_json: str, druzyny_ligi: set = None) -> tuple:
    """Zwraca (power_df, sedziowie_df) dla tab Deep Data.
    druzyny_ligi: jeśli podany, pokazuje tylko drużyny bieżącego sezonu (bez spadkowiczów).
    """
    df = pd.read_json(df_json)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # ── Power Rankings ────────────────────────────────────────────────────
    druzyny_all = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
    druzyny = [d for d in druzyny_all if d in druzyny_ligi] if druzyny_ligi else druzyny_all
    power_rows = []
    for d in druzyny:
        h_df = df[df["HomeTeam"] == d]
        a_df = df[df["AwayTeam"] == d]
        all_m = pd.concat([
            h_df.assign(_gole_str=h_df["FTHG"], _gole_strac=h_df["FTAG"],
                        _sot=h_df.get("HST", np.nan), _kartki=h_df["total_kartki"],
                        _rozne=h_df["total_rozne"]),
            a_df.assign(_gole_str=a_df["FTAG"], _gole_strac=a_df["FTHG"],
                        _sot=a_df.get("AST", np.nan), _kartki=a_df["total_kartki"],
                        _rozne=a_df["total_rozne"])
        ])
        if len(all_m) < 3:
            continue
        gol_str   = all_m["_gole_str"].mean()
        gol_strac = all_m["_gole_strac"].mean()
        sot_sr    = all_m["_sot"].dropna().mean() if all_m["_sot"].notna().any() else None
        kart_sr   = all_m["_kartki"].mean()
        rozne_sr  = all_m["_rozne"].mean()
        konv      = (gol_str / sot_sr) if (sot_sr and sot_sr > 0) else None
        # xG-proxy: SOT × liga_średnia_konwersji (szacunkowa)
        xg_proxy  = (sot_sr * 0.11) if sot_sr else None  # ~11% konwersja PL
        # Faule popełnione / wymuszone (Factor of Opposition)
        fo_c_sr = None
        fo_s_sr = None
        shots_sr = None
        if "HF" in df.columns:
            fo_committed = pd.concat([h_df["HF"], a_df["AF"]]).dropna()
            fo_suffered  = pd.concat([h_df["AF"], a_df["HF"]]).dropna()
            fo_c_sr = fo_committed.mean() if not fo_committed.empty else None
            fo_s_sr = fo_suffered.mean()  if not fo_suffered.empty  else None
        if "HS" in df.columns:
            shots_vals = pd.concat([h_df["HS"], a_df["AS"]]).dropna()
            shots_sr   = shots_vals.mean() if not shots_vals.empty else None
        # Shots-to-Corners ratio
        s2c = round(shots_sr / rozne_sr, 2) if (shots_sr and rozne_sr and rozne_sr > 0) else None
        # Style Tags
        style_tags = []
        if fo_c_sr and fo_c_sr > 12 and kart_sr > 2.0:
            style_tags.append("🟥 Hot Heads")
        if s2c and s2c < 2.2:
            style_tags.append("📐 Corner Machine")
        if s2c and s2c > 3.2:
            style_tags.append("🎯 Gra Środkiem")
        if fo_c_sr and fo_c_sr < 9.5 and shots_sr and shots_sr < 10:
            style_tags.append("🚌 Bus Parker")
        # Forma ostatnie 5
        mecze5 = df[(df["HomeTeam"]==d)|(df["AwayTeam"]==d)].tail(5)
        form5_pts = 0
        for _, m5 in mecze5.iterrows():
            if m5["HomeTeam"] == d:
                if m5["FTHG"] > m5["FTAG"]: form5_pts += 3
                elif m5["FTHG"] == m5["FTAG"]: form5_pts += 1
            else:
                if m5["FTAG"] > m5["FTHG"]: form5_pts += 3
                elif m5["FTAG"] == m5["FTHG"]: form5_pts += 1
        power_rows.append({
            "Drużyna":         d,
            "M":               len(all_m),
            "Gole/M ↑":        round(gol_str, 2),
            "Strac./M ↓":      round(gol_strac, 2),
            "SOT/M":           round(sot_sr, 1) if sot_sr else "–",
            "Konwersja%":      f"{konv:.1%}" if konv else "–",
            "xG-proxy":        round(xg_proxy, 2) if xg_proxy else "–",
            "Kartki/M":        round(kart_sr, 1),
            "Rożne/M":         round(rozne_sr, 1),
            "Faule/M":         round(fo_c_sr, 1) if fo_c_sr else "–",
            "Faule wym./M":    round(fo_s_sr, 1) if fo_s_sr else "–",
            "S2C ratio":       s2c if s2c else "–",
            "Styl":            " ".join(style_tags) if style_tags else "–",
            "Forma (pkt/5M)":  form5_pts,
            "_gol_str":        gol_str,
            "_gol_strac":      gol_strac,
            "_forma":          form5_pts,
        })
    power_df = pd.DataFrame(power_rows)

    # ── Sędziowie ────────────────────────────────────────────────────────
    sedzio_df = pd.DataFrame()
    if "Referee" in df.columns:
        ref_grp = df.groupby("Referee").agg(
            Meczów=("Referee", "count"),
            Kartki_Y_avg=("HY", lambda x: (x + df.loc[x.index, "AY"]).mean()),
            Kartki_R_avg=("HR", lambda x: (x + df.loc[x.index, "AR"]).mean()),
            Gole_avg=("total_gole", "mean"),
        ).reset_index()
        # Łączna suma kartek (Yellow + 2*Red) per mecz
        ref_grp2 = []
        for ref, grp in df.groupby("Referee"):
            n = len(grp)
            if n < 3:
                continue
            y_avg  = (grp["HY"] + grp["AY"]).mean()
            r_avg  = (grp["HR"] + grp["AR"]).mean()
            tot_k  = y_avg + r_avg * 2
            g_avg  = grp["total_gole"].mean()
            ref_grp2.append({
                "Sędzia": ref, "Meczów": n,
                "Kartki Y/M": round(y_avg, 1),
                "Kartki R/M": round(r_avg, 2),
                "Total Kart/M ↓": round(tot_k, 1),
                "Gole/M": round(g_avg, 1),
                "_tot_k": tot_k,
            })
        sedzio_df = pd.DataFrame(ref_grp2).sort_values("_tot_k", ascending=False) if ref_grp2 else pd.DataFrame()

    return power_df, sedzio_df

# ===========================================================================
# ŁADOWANIE DANYCH I SIDEBAR
# ===========================================================================
# ── SIDEBAR ─────────────────────────────────────────────────────────────────
_liga_list = list(LIGI.keys())
if "._liga_override" in str(st.session_state):
    pass
_liga_default_idx = _liga_list.index(st.session_state.get("_liga_override", _liga_list[0])) \
    if st.session_state.get("_liga_override") in _liga_list else 0
wybrana_liga = st.sidebar.selectbox("🌍 Liga", _liga_list, index=_liga_default_idx)
debug_mode   = False  # ustawiane w expander Parametry niżej

# ── Kursy live (The Odds API) ─────────────────────────────────────────────
_CSV_CODE  = LIGI[wybrana_liga]["csv_code"]
_OA_DB     = "predykcje.db"
_oa_cached = {}


def oblicz_kontekst_meczu(h: str, a: str, historical: pd.DataFrame,
                           sbn: dict) -> dict:
    """
    Zwraca słownik z kontekstem meczu:
    - pozycje w tabeli, punkty, GD
    - match_type (Top_vs_Top, Top_vs_Bot itd.)
    - historyczne avg kartek/rożnych/fauli per drużyna
    - efekt wyniku do przerwy na kartki
    - motywacja (relegation battle, title race itd.)
    """
    ctx = {}

    # ── Pozycja z standings (API) ────────────────────────────────────────
    sh = sbn.get(h, {})
    sa = sbn.get(a, {})
    ctx["h_pos"]    = sh.get("position", 0)
    ctx["a_pos"]    = sa.get("position", 0)
    ctx["h_pts"]    = sh.get("points",   0)
    ctx["a_pts"]    = sa.get("points",   0)
    ctx["h_gd"]     = sh.get("gd",       0)
    ctx["a_gd"]     = sa.get("gd",       0)
    ctx["h_played"] = sh.get("played",   0)
    ctx["h_won"]    = sh.get("won",      0)
    ctx["h_draw"]   = sh.get("draw",     0)
    ctx["h_lost"]   = sh.get("lost",     0)
    ctx["a_won"]    = sa.get("won",      0)
    ctx["a_draw"]   = sa.get("draw",     0)
    ctx["a_lost"]   = sa.get("lost",     0)
    ctx["h_gf"]     = sh.get("gf",       0)
    ctx["h_ga"]     = sh.get("ga",       0)
    ctx["a_gf"]     = sa.get("gf",       0)
    ctx["a_ga"]     = sa.get("ga",       0)

    pos_h = ctx["h_pos"]
    pos_a = ctx["a_pos"]
    played = max(ctx["h_played"], 1)

    # ── Match type ───────────────────────────────────────────────────────
    n_teams = max(pos_h, pos_a, 20)
    top_cut    = max(1, round(n_teams * 0.30))   # top 30%
    bottom_cut = min(n_teams, round(n_teams * 0.70))  # bottom 30%
    h_is_top = 0 < pos_h <= top_cut
    a_is_top = 0 < pos_a <= top_cut
    h_is_bot = pos_h >= bottom_cut
    a_is_bot = pos_a >= bottom_cut
    if   h_is_top and a_is_top:  ctx["match_type"] = "Top vs Top"
    elif h_is_top and a_is_bot:  ctx["match_type"] = "Top vs Słaby"
    elif h_is_bot and a_is_top:  ctx["match_type"] = "Słaby vs Top"
    elif h_is_bot and a_is_bot:  ctx["match_type"] = "Dół tabeli"
    else:                        ctx["match_type"] = "Środek tabeli"

    ctx["pos_diff"] = abs(pos_h - pos_a) if pos_h and pos_a else 0

    # ── Motywacja ────────────────────────────────────────────────────────
    motyw = []
    if pos_h and pos_h <= 4:  motyw.append(f"🏆 {h[:10]} walczy o top-4")
    if pos_a and pos_a <= 4:  motyw.append(f"🏆 {a[:10]} walczy o top-4")
    if pos_h and pos_h <= 1:  motyw.append(f"👑 {h[:10]} lider")
    if pos_a and pos_a <= 1:  motyw.append(f"👑 {a[:10]} lider")
    if pos_h and pos_h >= n_teams - 2:  motyw.append(f"🔴 {h[:10]} w strefie spadkowej")
    if pos_a and pos_a >= n_teams - 2:  motyw.append(f"🔴 {a[:10]} w strefie spadkowej")
    if pos_h and pos_h >= n_teams - 5:  motyw.append(f"⚠️ {h[:10]} w pobliżu strefy spadkowej")
    if pos_a and pos_a >= n_teams - 5:  motyw.append(f"⚠️ {a[:10]} w pobliżu strefy spadkowej")
    ctx["motywacja"] = motyw

    # ── Statystyki z historical CSV ──────────────────────────────────────
    if not historical.empty:
        def _team_stats(team):
            hm = historical[historical["HomeTeam"] == team].copy()
            am = historical[historical["AwayTeam"] == team].copy()
            stats = {}
            # Kartki
            yc = []; rc = []; fo = []; fo_suffered = []; co = []
            if "HY" in hm.columns:
                yc += hm["HY"].dropna().tolist()
                yc += am["AY"].dropna().tolist()
                rc += hm["HR"].dropna().tolist()
                rc += am["AR"].dropna().tolist()
            if "HF" in hm.columns:
                fo           += hm["HF"].dropna().tolist()
                fo           += am["AF"].dropna().tolist()
                fo_suffered  += hm["AF"].dropna().tolist()
                fo_suffered  += am["HF"].dropna().tolist()
            if "HC" in hm.columns:
                co += hm["HC"].dropna().tolist()
                co += am["AC"].dropna().tolist()
            stats["avg_yc"]          = float(np.mean(yc))          if yc          else None
            stats["avg_rc"]          = float(np.mean(rc))          if rc          else None
            stats["avg_fo"]          = float(np.mean(fo))          if fo          else None
            stats["avg_fo_suffered"] = float(np.mean(fo_suffered)) if fo_suffered else None
            stats["avg_co"]          = float(np.mean(co))          if co          else None
            stats["n"]               = len(hm) + len(am)
            # Strzały
            sht = []; st = []
            if "HS" in hm.columns:
                sht += hm["HS"].dropna().tolist()
                sht += am["AS"].dropna().tolist()
                st  += hm["HST"].dropna().tolist() if "HST" in hm.columns else []
                st  += am["AST"].dropna().tolist() if "AST" in am.columns else []
            stats["avg_shots"] = float(np.mean(sht)) if sht else None
            stats["avg_sot"]   = float(np.mean(st))  if st  else None
            # Strzały przyjęte (shots faced) — do Pressure Tolerance
            sht_faced = []
            if "AS" in hm.columns:
                sht_faced += hm["AS"].dropna().tolist()  # goście strzelali na gospodarza
                sht_faced += am["HS"].dropna().tolist()  # gospodarz strzelał na gościa
            stats["avg_shots_faced"] = float(np.mean(sht_faced)) if sht_faced else None
            # Rożne przyjęte (corners conceded)
            co_conc = []
            if "AC" in hm.columns:
                co_conc += hm["AC"].dropna().tolist()  # AC = rożne gości gdy H gra u siebie
                co_conc += am["HC"].dropna().tolist()  # HC = rożne gospod. gdy H gra w gościach
            stats["avg_co_conceded"] = float(np.mean(co_conc)) if co_conc else None
            # Pressure Tolerance: corners_conceded / shots_faced
            if stats["avg_co_conceded"] and stats["avg_shots_faced"] and stats["avg_shots_faced"] > 0:
                stats["pressure_tolerance"] = round(stats["avg_co_conceded"] / stats["avg_shots_faced"], 3)
            else:
                stats["pressure_tolerance"] = None
            # Shots-to-Corners ratio
            avg_co_val = stats["avg_co"] or 0
            stats["s2c_ratio"] = round(stats["avg_shots"] / avg_co_val, 2) \
                if (stats["avg_shots"] and avg_co_val > 0) else None
            # ── Volatility Index (CV = std/mean) ─────────────────────────
            # Wyższe CV = bardziej nieprzewidywalna drużyna → Kelly powinien obniżać stawkę
            if len(co) >= 4:
                co_arr = np.array(co)
                stats["vol_co_cv"]   = round(co_arr.std() / co_arr.mean(), 3) if co_arr.mean() > 0 else None
                stats["vol_co_std"]  = round(co_arr.std(), 2)
            else:
                stats["vol_co_cv"] = stats["vol_co_std"] = None
            if len(yc) >= 4:
                yc_arr = np.array(yc)
                stats["vol_yc_cv"]   = round(yc_arr.std() / yc_arr.mean(), 3) if yc_arr.mean() > 0 else None
                stats["vol_yc_std"]  = round(yc_arr.std(), 2)
            else:
                stats["vol_yc_cv"] = stats["vol_yc_std"] = None
            # ── Style Tags ───────────────────────────────────────────────
            tags = []
            _yc  = stats["avg_yc"]  or 0
            _fo  = stats["avg_fo"]  or 0
            _fos = stats["avg_fo_suffered"] or 0
            _sht = stats["avg_shots"] or 0
            _co  = stats["avg_co"]  or 0
            _s2c = stats["s2c_ratio"] or 0
            _pt  = stats["pressure_tolerance"] or 0
            if _fo > 12 and _yc > 2.0:
                tags.append("🟥 Hot Heads")
            if _fos > 12.5:
                tags.append("🎭 Foul Magnet")
            if _s2c > 0 and _s2c < 2.2:
                tags.append("📐 Corner Machine")
            if _s2c > 3.2:
                tags.append("🎯 Central Attackers")
            if _fo < 9.5 and _sht < 10:
                tags.append("🚌 Bus Parker")
            if _pt >= 0.43:
                tags.append("🛡️ Aktywna obrona")  # wybijają na rożne
            if not tags:
                tags.append("⚖️ Balanced")
            stats["style_tags"] = tags
            return stats


        h_stats = _team_stats(h)
        a_stats = _team_stats(a)
        ctx["h_stats"] = h_stats
        ctx["a_stats"] = a_stats

        # ── Łączne przewidywane kartki/rożne/faule dla tego meczu ────────
        if h_stats["avg_yc"] and a_stats["avg_yc"]:
            ctx["lambda_yc"] = h_stats["avg_yc"] + a_stats["avg_yc"]
        if h_stats["avg_fo"] and a_stats["avg_fo"]:
            ctx["lambda_fo"] = h_stats["avg_fo"] + a_stats["avg_fo"]
        if h_stats["avg_co"] and a_stats["avg_co"]:
            ctx["lambda_co"] = h_stats["avg_co"] + a_stats["avg_co"]

        # ── Aggression Clash – CPF weighted (r=0.835 z danymi EPL) ───────
        # fo * cpf = oczekiwane kartki z fauli — najsilniejszy predictor
        h_cpf = (h_stats["avg_yc"] / h_stats["avg_fo"]) if (h_stats.get("avg_fo") and h_stats["avg_fo"] > 0) else None
        a_cpf = (a_stats["avg_yc"] / a_stats["avg_fo"]) if (a_stats.get("avg_fo") and a_stats["avg_fo"] > 0) else None
        ctx["h_cpf"] = round(h_cpf, 3) if h_cpf else None
        ctx["a_cpf"] = round(a_cpf, 3) if a_cpf else None
        if h_cpf and a_cpf and h_stats.get("avg_fo") and a_stats.get("avg_fo"):
            ctx["agg_clash"] = round(
                h_stats["avg_fo"] * h_cpf + a_stats["avg_fo"] * a_cpf, 2
            )
            # Liga avg CPF ~0.17 EPL → expected cards z fauli avg ~3.6
            _lig_cpf_avg = 3.6
            ctx["agg_clash_vs_avg"] = round((ctx["agg_clash"] - _lig_cpf_avg) / _lig_cpf_avg * 100, 1)
            ctx["agg_clash_alert"]  = ctx["agg_clash_vs_avg"] >= 15

        # ── Tempo Index (r=0.42-0.46 z rożnymi) ─────────────────────────
        # shots + corners + fouls obu drużyn
        if all(h_stats.get(k) for k in ["avg_shots","avg_co","avg_fo"]) and \
           all(a_stats.get(k) for k in ["avg_shots","avg_co","avg_fo"]):
            ctx["tempo_index"] = round(
                h_stats["avg_shots"] + h_stats["avg_co"] + h_stats["avg_fo"] +
                a_stats["avg_shots"] + a_stats["avg_co"] + a_stats["avg_fo"], 1
            )
            # EPL avg tempo ~60 (13.6+5.3+10.6 + 11.0+4.6+11.0)
            _lig_tempo_avg = 56.0
            ctx["tempo_vs_avg"] = round((ctx["tempo_index"] - _lig_tempo_avg) / _lig_tempo_avg * 100, 1)

        # ── Home advantage w rożnych (+0.73 corner/mecz) ─────────────────
        if "HC" in historical.columns:
            ctx["home_co_adv"] = round(historical["HC"].mean() - historical["AC"].mean(), 2)

        # ── Factor of Opposition – CLASH score ───────────────────────────
        if h_stats.get("avg_fo") and a_stats.get("avg_fo_suffered"):
            clash_h = (h_stats["avg_fo"] + a_stats["avg_fo_suffered"]) / 2
        else:
            clash_h = None
        if a_stats.get("avg_fo") and h_stats.get("avg_fo_suffered"):
            clash_a = (a_stats["avg_fo"] + h_stats["avg_fo_suffered"]) / 2
        else:
            clash_a = None
        if clash_h and clash_a:
            clash_total = clash_h + clash_a
            _lig_fo_avg = historical["HF"].mean() + historical["AF"].mean() \
                if "HF" in historical.columns else 21.0
            ctx["clash_total"]   = round(clash_total, 1)
            ctx["clash_vs_avg"]  = round((clash_total - _lig_fo_avg) / _lig_fo_avg * 100, 1)
            ctx["clash_alert"]   = ctx["clash_vs_avg"] >= 15

        # ── Corners per Shot (Shots→Corners efficiency) ───────────────────
        if h_stats.get("avg_shots") and h_stats.get("avg_co") and h_stats["avg_shots"] > 0:
            ctx["h_cps"] = round(h_stats["avg_co"] / h_stats["avg_shots"] * 100, 1)
        if a_stats.get("avg_shots") and a_stats.get("avg_co") and a_stats["avg_shots"] > 0:
            ctx["a_cps"] = round(a_stats["avg_co"] / a_stats["avg_shots"] * 100, 1)

        # ── Pressure Tolerance Matchup ────────────────────────────────────
        # shots_H × PT_A = ile rożnych wyciągnie H z obrony A (r=0.271)
        h_pt = h_stats.get("pressure_tolerance")
        a_pt = a_stats.get("pressure_tolerance")
        ctx["h_pt"] = h_pt
        ctx["a_pt"] = a_pt
        if h_stats.get("avg_shots") and a_pt:
            ctx["h_exp_co_pt"] = round(h_stats["avg_shots"] * a_pt, 2)
        if a_stats.get("avg_shots") and h_pt:
            ctx["a_exp_co_pt"] = round(a_stats["avg_shots"] * h_pt, 2)
        if ctx.get("h_exp_co_pt") and ctx.get("a_exp_co_pt"):
            ctx["exp_co_pt_total"] = round(ctx["h_exp_co_pt"] + ctx["a_exp_co_pt"], 1)
            # Porównaj z prostą średnią lambda_co — różnica = przewaga informacyjna
            if "lambda_co" in ctx:
                ctx["co_pt_vs_lambda"] = round(ctx["exp_co_pt_total"] - ctx["lambda_co"], 2)

        # ── Referee Strictness (YC/Foul) ─────────────────────────────────
        if "Referee" in historical.columns and "HF" in historical.columns:
            _ref_stats = {}
            for _ref, _grp in historical.groupby("Referee"):
                _tf = (_grp["HF"] + _grp["AF"]).sum()
                _ty = (_grp["HY"] + _grp["AY"]).sum()
                if _tf >= 10:
                    _ref_stats[_ref] = {
                        "yc_per_foul": round(_ty / _tf, 3),
                        "avg_yc":      round((_grp["HY"] + _grp["AY"]).mean(), 2),
                        "avg_fo":      round((_grp["HF"] + _grp["AF"]).mean(), 1),
                        "n":           len(_grp),
                    }
            ctx["referee_stats_map"] = _ref_stats

        # ── Volatility Index ──────────────────────────────────────────────
        # CV (coefficient of variation) = std/mean — nieprzewidywalność drużyny
        ctx["h_vol_co"] = h_stats.get("vol_co_cv")
        ctx["h_vol_yc"] = h_stats.get("vol_yc_cv")
        ctx["a_vol_co"] = a_stats.get("vol_co_cv")
        ctx["a_vol_yc"] = a_stats.get("vol_yc_cv")
        # Matchup volatility — jeśli obie drużyny nieprzewidywalne → Kelly ↓
        if h_stats.get("vol_yc_cv") and a_stats.get("vol_yc_cv"):
            ctx["vol_yc_matchup"] = round((h_stats["vol_yc_cv"] + a_stats["vol_yc_cv"]) / 2, 3)
            ctx["vol_co_matchup"] = round(((h_stats.get("vol_co_cv") or 0) + (a_stats.get("vol_co_cv") or 0)) / 2, 3)
            # High volatility warning: avg CV > 0.65
            ctx["vol_warning_yc"] = ctx["vol_yc_matchup"] > 0.65
            ctx["vol_warning_co"] = ctx["vol_co_matchup"] > 0.55

        # ── Game State Sensitivity ────────────────────────────────────────
        # Dla każdego stanu HTR (H/D/A) liczymy avg rożne i żółte per drużyna
        def _game_state_stats(team):
            gs = {}
            for state, label in [("H", "Prowadzi"), ("D", "Remis"), ("A", "Przegrywa")]:
                hm_s = historical[(historical["HomeTeam"] == team) & (historical["HTR"] == state)]
                am_s = historical[(historical["AwayTeam"] == team) & (historical["HTR"] == {
                    "H": "A", "D": "D", "A": "H"}[state])]
                co_vals = pd.concat([
                    hm_s["HC"] if "HC" in hm_s.columns else pd.Series(dtype=float),
                    am_s["AC"] if "AC" in am_s.columns else pd.Series(dtype=float)
                ]).dropna()
                yc_vals = pd.concat([
                    hm_s["HY"] if "HY" in hm_s.columns else pd.Series(dtype=float),
                    am_s["AY"] if "AY" in am_s.columns else pd.Series(dtype=float)
                ]).dropna()
                n = len(co_vals)
                gs[label] = {
                    "n":    n,
                    "co":   round(co_vals.mean(), 1) if n >= 2 else None,
                    "yc":   round(yc_vals.mean(), 1) if n >= 2 else None,
                }
            return gs
        if "HTR" in historical.columns:
            ctx["h_game_state"] = _game_state_stats(h)
            ctx["a_game_state"] = _game_state_stats(a)

        # ── Korekta kontekstowa λ_yc (na podstawie korelacji z danych) ──
        # Top vs Top: +0.67 kartki vs avg (z EPL 24/25: 4.73 vs avg 4.06)
        # Dół tabeli: -1.06 kartki
        # Top vs Słaby: -0.24
        # Środek: base
        if "lambda_yc" in ctx:
            _adj = {"Top vs Top": +0.67, "Dół tabeli": -1.06,
                    "Top vs Słaby": -0.24, "Słaby vs Top": -0.24,
                    "Środek tabeli": 0.0}
            ctx["lambda_yc_adj"] = ctx["lambda_yc"] + _adj.get(ctx["match_type"], 0)

        # ── Home/Away split ───────────────────────────────────────────────
        if not historical.empty:
            def _home_away_split(team):
                hm = historical[historical["HomeTeam"] == team]
                am = historical[historical["AwayTeam"] == team]
                h_pts = ((hm["FTR"]=="H")*3 + (hm["FTR"]=="D")*1).sum()
                a_pts = ((am["FTR"]=="A")*3 + (am["FTR"]=="D")*1).sum()
                h_n = len(hm); a_n = len(am)
                return {
                    "h_ppg":    round(h_pts/h_n, 2) if h_n else None,
                    "a_ppg":    round(a_pts/a_n, 2) if a_n else None,
                    "home_adv": round(h_pts/h_n - a_pts/a_n, 2) if h_n and a_n else None,
                    "h_shots":  round(hm["HS"].mean(), 1) if h_n and "HS" in hm.columns else None,
                    "a_shots":  round(am["AS"].mean(), 1) if a_n and "AS" in am.columns else None,
                }
            ctx["h_ha"] = _home_away_split(h)
            ctx["a_ha"] = _home_away_split(a)

        # ── Sezonowość ────────────────────────────────────────────────────
        if "Date" in historical.columns:
            try:
                _hist_dates  = pd.to_datetime(historical["Date"], dayfirst=True, errors="coerce")
                _curr_month  = pd.Timestamp.now().month
                _month_grp   = historical.copy()
                _month_grp["_m"] = _hist_dates.dt.month
                _month_stats = _month_grp.groupby("_m")[["total_kartki","total_rozne"]].mean()
                if _curr_month in _month_stats.index:
                    _ms = _month_stats.loc[_curr_month]
                    _ov_yc = historical["total_kartki"].mean() if "total_kartki" in historical.columns else 3.8
                    _ov_co = historical["total_rozne"].mean()  if "total_rozne"  in historical.columns else 9.8
                    ctx["season_yc_avg"]  = round(float(_ms["total_kartki"]), 2)
                    ctx["season_co_avg"]  = round(float(_ms["total_rozne"]),  2)
                    ctx["season_yc_diff"] = round(float(_ms["total_kartki"]) - _ov_yc, 2)
                    ctx["season_co_diff"] = round(float(_ms["total_rozne"])  - _ov_co, 2)
            except Exception:
                pass

        # ── Finishing Quality (gole/SOT) ──────────────────────────────────
        if not historical.empty and "HST" in historical.columns:
            def _finishing(team):
                hm  = historical[historical["HomeTeam"] == team]
                am  = historical[historical["AwayTeam"] == team]
                gole = pd.concat([hm["FTHG"], am["FTAG"]]).sum()
                sot  = pd.concat([hm["HST"], am["AST"]]).sum()
                return round(gole / sot, 3) if sot > 0 else None
            ctx["h_finishing"] = _finishing(h)
            ctx["a_finishing"] = _finishing(a)

        # ── Efekt poprzedniego wyniku ─────────────────────────────────────
        if not historical.empty:
            def _last_result(team):
                tm = historical[(historical["HomeTeam"]==team)|(historical["AwayTeam"]==team)]
                if tm.empty: return None
                last = tm.iloc[-1]
                if last["HomeTeam"] == team:
                    return "W" if last["FTR"]=="H" else ("D" if last["FTR"]=="D" else "L")
                else:
                    return "W" if last["FTR"]=="A" else ("D" if last["FTR"]=="D" else "L")
            ctx["h_last_result"] = _last_result(h)
            ctx["a_last_result"] = _last_result(a)

        # ── Narracja meczu ────────────────────────────────────────────────
        def _build_narrative(ctx, h, a):
            parts = []
            hs  = ctx.get("h_stats", {}); as_ = ctx.get("a_stats", {})
            h_sht  = hs.get("avg_shots", 0) or 0
            a_sht  = as_.get("avg_shots", 0) or 0
            lam_co = ctx.get("lambda_co", 0) or 0
            lam_yc = ctx.get("lambda_yc_adj") or ctx.get("lambda_yc", 0) or 0
            exp_co = ctx.get("exp_co_pt_total") or lam_co

            # 1. Charakter gry
            if h_sht + a_sht > 26:
                parts.append(f"Otwarta gra — obie drużyny generują dużo strzałów ({h_sht:.0f}+{a_sht:.0f}/M), spodziewaj się tempa")
            elif h_sht + a_sht < 18:
                parts.append(f"Mecz zapowiada się defensywnie — mało strzałów po obu stronach ({h_sht:.0f}+{a_sht:.0f}/M)")
            else:
                parts.append(f"Wyrównany charakter gry, bez wyraźnej dominacji ofensywnej")

            # 2. Rożne
            if exp_co > 11.5:
                parts.append(f"Dużo rożnych w perspektywie — styl obrony obu drużyn sprzyja dośrodkowaniom (exp ~{exp_co:.1f})")
            elif exp_co < 8.5:
                parts.append(f"Poniżej średniej rożnych — drużyny bronią pasywnie lub grają środkiem pola")

            # 3. Kartki
            if lam_yc:
                agg_alert = ctx.get("agg_clash_alert", False)
                h2h_hot   = (ctx.get("h2h_stats") or {}).get("hot_rivalry", False)
                if agg_alert and h2h_hot:
                    parts.append(f"Wysokie ryzyko kartek — agresywny profil + historycznie gorące starcia (λ={lam_yc:.1f})")
                elif agg_alert:
                    parts.append(f"Profil faulowania wskazuje na podwyższoną liczbę kartek (λ={lam_yc:.1f})")
                elif h2h_hot:
                    parts.append(f"H2H jest nerwowe — te drużyny zazwyczaj obfitują w kartki")
                elif lam_yc < 3.2:
                    parts.append(f"Spokojniejszy mecz pod względem dyscypliny (λ={lam_yc:.1f})")

            # 4. Home advantage
            h_ha = ctx.get("h_ha") or {}; a_ha = ctx.get("a_ha") or {}
            h_adv = h_ha.get("home_adv"); a_adv = a_ha.get("home_adv")
            if h_adv and h_adv > 0.7:
                parts.append(f"{h} to twierdza u siebie (+{h_adv:.2f} pkt/M) — efekt własnego boiska jest realny")
            elif h_adv and h_adv < 0:
                parts.append(f"{h} paradoksalnie gra lepiej na wyjeździe niż u siebie")
            if a_adv and a_adv < -0.3:
                parts.append(f"{a} wyraźnie słabszy w roli gościa ({a_ha.get('a_ppg',0):.2f} pkt/M wyjazdowo)")

            # 5. Finishing
            h_fin = ctx.get("h_finishing"); a_fin = ctx.get("a_finishing")
            if h_fin and a_fin:
                if h_fin > 0.38 and a_fin < 0.28:
                    parts.append(f"{h} wykańcza znacznie skuteczniej ({h_fin:.0%} goli/SOT vs {a_fin:.0%} {a})")
                elif a_fin > 0.38 and h_fin < 0.28:
                    parts.append(f"{a} ma wyraźną przewagę w wykańczaniu akcji ({a_fin:.0%} vs {h_fin:.0%})")

            # 6. Sezonowość
            s_diff = ctx.get("season_yc_diff")
            if s_diff and abs(s_diff) >= 0.4:
                direction = "wyższy" if s_diff > 0 else "niższy"
                parts.append(f"Aktualny okres sezonu historycznie daje {direction} poziom kartek ({s_diff:+.1f} vs roczna średnia)")

            # 7. Motywacja
            motyw = ctx.get("motywacja", [])
            if motyw:
                parts.append(" · ".join(motyw[:2]))

            # 8. Volatility
            if ctx.get("vol_warning_yc") or ctx.get("vol_warning_co"):
                items = []
                if ctx.get("vol_warning_yc"): items.append("kartki")
                if ctx.get("vol_warning_co"): items.append("rożne")
                parts.append(f"⚠️ Wysoka zmienność ({', '.join(items)}) — wyniki tych drużyn mocno się wahają")

            return parts

        ctx["narrative"] = _build_narrative(ctx, h, a)


        # ── H2H alternatywne statystyki ──────────────────────────────────
        _h2h_all = historical[
            ((historical["HomeTeam"] == h) & (historical["AwayTeam"] == a)) |
            ((historical["HomeTeam"] == a) & (historical["AwayTeam"] == h))
        ].copy()
        if not _h2h_all.empty:
            h2h_stats = {}
            if "HY" in _h2h_all.columns:
                h2h_stats["avg_yc"]    = (_h2h_all["HY"] + _h2h_all["AY"]).mean()
                h2h_stats["avg_rc"]    = (_h2h_all["HR"] + _h2h_all["AR"]).mean()
                # H2H avg cards — r=0.713 z przyszłymi kartkami (najsilniejszy historyczny predictor)
                h2h_stats["avg_cards"] = round((_h2h_all["HY"] + _h2h_all["AY"]).mean(), 2)
                h2h_stats["max_cards"] = int((_h2h_all["HY"] + _h2h_all["AY"]).max())
                h2h_stats["hot_rivalry"] = h2h_stats["avg_cards"] >= 4.5
            if "HF" in _h2h_all.columns:
                h2h_stats["avg_fo"]  = (_h2h_all["HF"] + _h2h_all["AF"]).mean()
            if "HC" in _h2h_all.columns:
                h2h_stats["avg_co"]  = (_h2h_all["HC"] + _h2h_all["AC"]).mean()
            h2h_stats["n"] = len(_h2h_all)
            ctx["h2h_stats"] = h2h_stats
        else:
            ctx["h2h_stats"] = {}

        # ── Over/Under 2.5 implied probability z kursu bukmachera ────────
        # Jeśli mamy B365>2.5 w historical dla tej pary
        _last_ou = _h2h_all[["B365>2.5","B365<2.5"]].dropna().tail(3)                    if "B365>2.5" in _h2h_all.columns else pd.DataFrame()
        if not _last_ou.empty:
            ctx["ou_implied_over"]  = round((1/_last_ou["B365>2.5"].mean()), 3)
            ctx["ou_implied_under"] = round((1/_last_ou["B365<2.5"].mean()), 3)
    else:
        ctx["h_stats"]  = {}
        ctx["a_stats"]  = {}
        ctx["h2h_stats"] = {}

    return ctx


def _oa_get_key():
    try: return st.secrets["ODDS_API_KEY"]
    except Exception: return st.session_state.get("_odds_key")

_oa_key = _oa_get_key()

with st.sidebar.expander("📅 Terminarz (football-data.org)", expanded=not bool(_get_fd_api_key())):
    _fd_key = _get_fd_api_key()
    if not _fd_key:
        _fd_mk = st.text_input("Klucz football-data.org", type="password",
            placeholder="lub dodaj FOOTBALL_DATA_API_KEY do secrets", key="_fd_key_inp")
        if _fd_mk:
            st.session_state["_fd_api_key"] = _fd_mk
            st.rerun()
    else:
        # Pokaż sezon z aktualnie załadowanego terminarza
        _fd_sch = st.session_state.get("_fd_season_debug", "")
        st.markdown(
            f"<div style='font-size:0.78em;color:#4CAF50'>✅ Klucz aktywny – terminarz z API</div>"
            f"<div style='font-size:0.72em;color:#555;margin-top:2px'>"
            f"Odświeżanie co 1h · fallback na CSV{' · sezon: ' + _fd_sch if _fd_sch else ''}</div>",
            unsafe_allow_html=True)
        if st.button("🔄 Wyczyść cache terminarza", use_container_width=True, key="_fd_clear"):
            load_schedule.clear()
            st.success("Cache wyczyszczony – odświeżam...")
        # Debug: pokaż niezmapowane nazwy z API
        if _niezmapowane:
            st.markdown(
                "<div style='font-size:0.72em;color:#FF9800;margin-top:6px'>"
                "⚠️ Niezmapowane nazwy z API – dodaj do NAZWY_MAP:</div>",
                unsafe_allow_html=True)
            for _nm in sorted(_niezmapowane):
                st.code(_nm, language=None)

with st.sidebar.expander("💰 Kursy bukmacherskie", expanded=not bool(_oa_key)):
    if not _oa_key:
        _mk = st.text_input("Klucz The Odds API", type="password",
            placeholder="lub dodaj ODDS_API_KEY do secrets", key="_odds_key_inp")
        if _mk:
            st.session_state["_odds_key"] = _mk
            _oa_key = _mk
    if _OA_OK and _oa_key:
        _stats = _oa.get_usage_stats(_OA_DB)
        _rem   = _stats.get("requests_remaining")
        _last  = _stats.get("last_per_liga", {}).get(_CSV_CODE)
        if _rem is not None:
            _bc = "#4CAF50" if _rem > 200 else ("#FF9800" if _rem > 50 else "#F44336")
            st.markdown(
                f"<div style='font-size:0.78em;color:#888;margin-bottom:4px'>"
                f"Requestów: <b style='color:{_bc}'>{_rem}/500</b></div>"
                f"<div style='background:var(--bg-card2);border-radius:3px;height:4px'>"
                f"<div style='background:{_bc};width:{min(int(_rem/5),100)}%;height:4px;border-radius:3px'></div></div>",
                unsafe_allow_html=True)
        if _last:
            try:
                _ldt = datetime.fromisoformat(_last.replace("Z", "+00:00"))
                _age = (datetime.now(_ldt.tzinfo) - _ldt).total_seconds() / 3600
                st.caption(f"Aktualizacja: {_age:.0f}h temu")
            except Exception: pass
        _cr, _cf = st.columns(2)
        if _cr.button("🔄 Odśwież", use_container_width=True, key="_oa_ref"):
            with st.spinner("Pobieranie..."):
                _res = _oa.fetch_odds(_CSV_CODE, _OA_DB, _oa_key)
            if _res["ok"] and not _res.get("from_cache"):
                st.success(f"✅ {_res['n_events']} meczów | {_res['requests_remaining']} req")
            elif _res.get("from_cache"):
                st.info(f"Cache ({_res['age_h']}h). Za {_res['next_refresh_h']}h.")
            else:
                st.error(_res["error"])
        if _cf.button("⚡ Wymuś", use_container_width=True, key="_oa_force"):
            with st.spinner("Pobieranie..."):
                _res = _oa.fetch_odds(_CSV_CODE, _OA_DB, _oa_key, force=True)
            if _res["ok"]: st.success(f"✅ {_res['n_events']} meczów")
            else:          st.error(_res["error"])
        _oa_cached = _oa.get_cached_odds(_CSV_CODE, _OA_DB)

        # ── Debug: pokaż co jest w cache kursów (diagnoza PSG i innych) ──
        if _oa_cached and st.session_state.get("_oa_debug_show"):
            with st.expander(f"🔍 Cache odds ({len(_oa_cached)} meczów)", expanded=True):
                for (hh, aa) in list(_oa_cached.keys())[:15]:
                    _mapped_h = _oa.map_api_to_model(hh)
                    _mapped_a = _oa.map_api_to_model(aa)
                    st.caption(f"`{hh}` → `{_mapped_h}` | `{aa}` → `{_mapped_a}`")
        st.checkbox("🔍 Debug cache", key="_oa_debug_show", value=False)


        # ── Auto-odświeżanie kursów (raz dziennie, rano) ─────────────────
        # Oszczędzamy limity: max 1 request/liga/dobę, tylko między 6:00-22:00
        # Klucz: dzień + liga, żeby nie odpytywać przy każdym rerunie Streamlit
        _auto_key = f"_oa_auto_{_CSV_CODE}_{datetime.now().strftime('%Y%m%d')}"
        _auto_hour = datetime.now().hour
        _auto_done = st.session_state.get(_auto_key, False)
        _stats_auto = _oa.get_usage_stats(_OA_DB)
        _rem_auto = _stats_auto.get("requests_remaining") or 500
        _last_auto = _stats_auto.get("last_per_liga", {}).get(_CSV_CODE)
        # Sprawdź czy dane są stare (>6h)
        _data_stale = True
        if _last_auto:
            try:
                _ldt_a = datetime.fromisoformat(_last_auto.replace("Z", "+00:00"))
                _age_a = (datetime.now(_ldt_a.tzinfo) - _ldt_a).total_seconds() / 3600
                _data_stale = _age_a > 6
            except Exception:
                pass
        if (not _auto_done and _data_stale and
                6 <= _auto_hour <= 22 and
                _rem_auto > 50):
            try:
                _res_auto = _oa.fetch_odds(_CSV_CODE, _OA_DB, _oa_key)
                if _res_auto.get("ok") and not _res_auto.get("from_cache"):
                    st.session_state[_auto_key] = True
                    _oa_cached = _oa.get_cached_odds(_CSV_CODE, _OA_DB)
                    st.toast(f"🔄 Kursy auto-zaktualizowane: {_res_auto['n_events']} meczów "
                             f"[{wybrana_liga}] · {_res_auto['requests_remaining']} req/mies.", icon="✅")
                elif _res_auto.get("from_cache"):
                    st.session_state[_auto_key] = True  # cache OK, nie odpytuj ponownie
            except Exception:
                pass

    # Stary digest usunięty: wysyłał per-liga przy każdym rerun Streamlit
        st.caption("ℹ️ Plik `odds_api.py` nie znaleziony.")

# ── Bankroll & Kelly ────────────────────────────────────────────────────────
if "bankroll_per_liga" not in st.session_state:
    st.session_state["bankroll_per_liga"] = {l: 1000.0 for l in LIGI.keys()}
_br_per_liga = st.session_state["bankroll_per_liga"]

# Zawsze oblicz _br_input przed expander (potrzebny niżej do caption)
_br_input = _br_per_liga.get(wybrana_liga, 1000.0) if st.session_state.get("_br_mode_toggle") else float(st.session_state.get("bankroll", 1000.0))

# Quick bankroll - zawsze widoczny
_br_quick = st.sidebar.number_input(
    "💰 Bankroll (zł)", min_value=10.0, max_value=1_000_000.0,
    value=float(st.session_state.get("bankroll", 1000.0)),
    step=100.0, key="_br_quick_sb",
    help="Kwota do obliczeń Kelly")
st.session_state["bankroll"] = _br_quick
for _lk in LIGI.keys():
    _br_per_liga[_lk] = _br_quick
st.session_state["bankroll_per_liga"] = _br_per_liga

with st.sidebar.expander("⚙️ Zaawansowane Kelly", expanded=False):
    _br_input = _br_quick  # używamy quick widget z góry
    _br_mode = st.toggle("💡 Osobny bankroll per liga", value=False, key="_br_mode_toggle")
    if _br_mode:
        st.markdown(
            "<div style='font-size:0.74em;color:#555;margin-bottom:6px'>"
            "Bankroll oddzielnie dla każdej ligi:</div>",
            unsafe_allow_html=True)
        for _br_liga in list(LIGI.keys()):
            _br_per_liga[_br_liga] = st.number_input(
                f"{_br_liga[:12]}",
                min_value=10.0, max_value=1_000_000.0,
                value=float(_br_per_liga.get(_br_liga, 1000.0)),
                step=100.0, key=f"_br_{_br_liga[:6]}",
                help=f"Bankroll dla {_br_liga}")
        st.session_state["bankroll_per_liga"] = _br_per_liga
        _br_input = _br_per_liga.get(wybrana_liga, 1000.0)
        st.session_state["bankroll"] = _br_input
        st.caption(f"Aktywna ({wybrana_liga[:12]}): **{_br_input:.0f} zł**")
    else:
        _br_input = st.number_input(
            "Bankroll (zł)", min_value=10.0, max_value=1_000_000.0,
            value=float(st.session_state.get("bankroll", 1000.0)),
            step=100.0, key="_br_widget",
            help="Kwota do obliczeń Kelly – wspólna dla wszystkich lig")
        st.session_state["bankroll"] = _br_input
        for _lk in LIGI.keys():
            _br_per_liga[_lk] = _br_input
        st.session_state["bankroll_per_liga"] = _br_per_liga

    _kelly_frac_options = {
        "🛡️ Konserwatywny (1/16)": 0.0625,
        "⚖️ Standardowy (1/8)":    0.125,
        "🔥 Agresywny (1/4)":      0.25,
    }
    _kelly_frac_label = st.select_slider(
        "Poziom ryzyka",
        options=list(_kelly_frac_options.keys()),
        value=st.session_state.get("kelly_frac_label", "⚖️ Standardowy (1/8)"),
        key="_kf_slider")
    st.session_state["kelly_frac_label"] = _kelly_frac_label
    st.session_state["kelly_frac"]       = _kelly_frac_options[_kelly_frac_label]
    _kelly_info_val = _br_input * st.session_state["kelly_frac"] * 0.3
    st.caption(f"Typowa stawka ~**{_kelly_info_val:.0f} zł** · Conservative Kelly · max 5%/mecz · EV≥5%")

with st.spinner(f"⚙️ Model Dixon-Coles analizuje dane {wybrana_liga}..."):
    historical = load_historical(LIGI[wybrana_liga]["csv_code"])
    schedule   = load_schedule(LIGI[wybrana_liga]["fd_org_id"], LIGI[wybrana_liga]["file"])
    standings  = load_standings(LIGI[wybrana_liga]["fd_org_id"])
    _sbn       = _standings_by_name(standings) if standings else {}

# ── Debug mapowania nazw ────────────────────────────────────────────────────
# Porównaj nazwy z terminarza (fd.org) z nazwami z historycznych CSV (co.uk)
# Pokaż tylko gdy debug_mode LUB są niezgodności
if not schedule.empty and not historical.empty:
    _couk_teams = set()
    for _col in ["HomeTeam", "AwayTeam"]:
        if _col in historical.columns:
            _couk_teams.update(historical[_col].dropna().unique())
    _sch_teams = set()
    for _col in ["home_team", "away_team"]:
        if _col in schedule.columns:
            _sch_teams.update(schedule[_col].dropna().unique())
    # Nazwy z terminarza których NIE MA w CSV
    _missing_in_csv = sorted(_sch_teams - _couk_teams)
    if _missing_in_csv:
        st.warning(
            f"⚠️ **Mapowanie nazw [{wybrana_liga}]** — "
            f"{len(_missing_in_csv)} nazw z terminarza nie pasuje do CSV. "
            f"Predykcje dla tych meczów będą niepełne.")
        with st.expander("🔍 Pokaż niezgodności", expanded=True):
            _dc1, _dc2 = st.columns(2)
            _dc1.markdown("**Nazwa z fd.org (terminarz)**")
            _dc2.markdown("**Najbliższe w co.uk CSV**")
            for _bad in _missing_in_csv:
                # Znajdź najbliższe dopasowanie z co.uk
                from difflib import get_close_matches
                _close = get_close_matches(_bad, _couk_teams, n=2, cutoff=0.4)
                _dc1.code(_bad)
                _dc2.code(", ".join(_close) if _close else "— brak podobnych —")

# Auto-aktualizacja wynikow: przy każdym wczytaniu synchronizuj trafione
# BEZ session_state cache - odpala się zawsze gdy są mecze bez wyników
# (szybkie: tylko SELECT + UPDATE dla meczów z trafione IS NULL)
if not historical.empty:
    init_db()
    _con_au = sqlite3.connect(DB_FILE)
    _mecze_bez_wyniku = _con_au.execute(
        "SELECT DISTINCT home, away FROM zdarzenia WHERE liga=? AND trafione IS NULL",
        (wybrana_liga,)
    ).fetchall()
    _con_au.close()
    _n_updated = 0
    for _h_au, _a_au in _mecze_bez_wyniku:
        _rows_before = sqlite3.connect(DB_FILE).execute(
            "SELECT COUNT(*) FROM zdarzenia WHERE home=? AND away=? AND trafione IS NOT NULL",
            (_h_au, _a_au)).fetchone()[0]
        aktualizuj_wynik_zdarzenia(_h_au, _a_au, historical)
        _rows_after = sqlite3.connect(DB_FILE).execute(
            "SELECT COUNT(*) FROM zdarzenia WHERE home=? AND away=? AND trafione IS NOT NULL",
            (_h_au, _a_au)).fetchone()[0]
        if _rows_after > _rows_before:
            _n_updated += 1
    if _n_updated > 0:
        st.toast(f"✅ Auto-zaktualizowano wyniki {_n_updated} meczów z football-data.co.uk", icon="⚽")

# ── Liga Switcher (szybka nawigacja) ────────────────────────────────────────
_LIGA_FLAGS = {
    "Premier League": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "La Liga":        "🇪🇸",
    "Bundesliga":     "🇩🇪",
    "Serie A":        "🇮🇹",
    "Ligue 1":        "🇫🇷",
}
# Ikony lig z football-data.org (te same co herby drużyn)
_LIGA_LOGOS = {
    "Premier League": "https://crests.football-data.org/PL.png",
    "La Liga":        "https://crests.football-data.org/PD.png",
    "Bundesliga":     "https://crests.football-data.org/BL1.png",
    "Serie A":        "https://crests.football-data.org/SA.png",
    "Ligue 1":        "https://crests.football-data.org/FL1.png",
}
def _liga_img(liga_name, size="28px", style=""):
    """Zwraca <img> z logo ligi lub flagę emoji jako fallback."""
    url = _LIGA_LOGOS.get(liga_name)
    flag = _LIGA_FLAGS.get(liga_name, "🌍")
    if url:
        return (f"<img src='{url}' style='width:{size};height:{size};"
                f"object-fit:contain;vertical-align:middle;{style}' "
                f"onerror=\"this.outerHTML='{flag}'\">")
    return f"<span style='font-size:{size}'>{flag}</span>"
# ── Liga Switcher – poziomy pasek ───────────────────────────────────────────
# Obsługa query param _ls przy wczytaniu strony
_qs_ls = st.query_params.get("_ls")
if _qs_ls is not None:
    try:
        _qs_liga = list(LIGI.keys())[int(_qs_ls)]
        if _qs_liga != wybrana_liga:
            st.session_state["_liga_override"] = _qs_liga
            st.query_params.clear()
            st.rerun()
    except (IndexError, ValueError):
        pass

# HTML <a href target="_self"> – otwiera w TEJ SAMEJ karcie, poziomo na mobile
_ls_pills = []
for _li, _ln in enumerate(LIGI.keys()):
    _active = _ln == wybrana_liga
    _bg     = "#1e3a5f" if _active else "#1e293b"
    _border = "#4CAF50" if _active else "#334d6e"
    _col    = "#ffffff" if _active else "#e2e8f0"
    _fw     = "700"     if _active else "500"
    _short  = {"Premier League":"EPL","La Liga":"La Liga",
                "Bundesliga":"Bund.","Serie A":"Serie A","Ligue 1":"Ligue 1"}.get(_ln, _ln)
    _logo_url = _LIGA_LOGOS.get(_ln, "")
    _flag_fb  = _LIGA_FLAGS.get(_ln, "🌍")
    _icon_html = (
        f"<img src='{_logo_url}' style='width:28px;height:28px;"
        f"object-fit:contain;filter:none' "
        f"onerror=\"this.outerHTML='<span style=font-size:1.3em>{_flag_fb}</span>'\">"
        if _logo_url else f"<span style='font-size:1.4em'>{_flag_fb}</span>"
    )
    _ls_pills.append(
        f"<a href='?_ls={_li}' target='_self' "
        f"style='display:inline-flex;flex-direction:column;align-items:center;"
        f"gap:3px;padding:7px 12px;border-radius:14px;border:1.5px solid {_border};"
        f"background:{_bg};color:{_col};font-size:0.85em;font-weight:{_fw};"
        f"text-decoration:none;white-space:nowrap;flex-shrink:0;'>"
        f"{_icon_html}"
        f"<span style='font-size:0.62em;color:{_col}'>{_short}</span>"
        f"</a>"
    )
st.markdown(
    "<div style='display:flex;flex-direction:row;flex-wrap:nowrap;"
    "overflow-x:auto;gap:8px;margin-bottom:12px;padding:4px 2px;"
    "-webkit-overflow-scrolling:touch;scrollbar-width:none'>"
    + "".join(_ls_pills) +
    "</div>",
    unsafe_allow_html=True)

# ── Hero Header ────────────────────────────────────────────────────────────
_hc1, _hc2, _hc3 = st.columns([5, 2, 2])
with _hc1:
    _flag_h = _liga_img(wybrana_liga, size="36px", style="margin-right:8px;vertical-align:middle")
    st.markdown(
        f"<h1 style='margin:0 0 2px 0;font-size:1.9em;font-weight:800;color:var(--text-primary);"
        f"display:flex;align-items:center;gap:6px'>"
        f"{_flag_h} {wybrana_liga}</h1>"
        f"<p style='margin:0;color:#444;font-size:0.8em;letter-spacing:0.04em'>"
        f"DIXON-COLES MODEL · SOT BLEND · WALK-FORWARD BACKTEST</p>",
        unsafe_allow_html=True)
with _hc2:
    if not schedule.empty:
        aktualna_kolejka = get_current_round(schedule)
        liczba_meczy = len(schedule[schedule["round"] == aktualna_kolejka])
        status = get_round_status(schedule, aktualna_kolejka)
        st.metric(f"{status} kolejka", f"#{aktualna_kolejka}", f"↑ {liczba_meczy} meczów")
with _hc3:
    # Szybki KPI: hit rate z bazy
    try:
        _con_hero = __import__('sqlite3').connect(DB_FILE)
        _hr_row = _con_hero.execute(
            "SELECT COUNT(*), SUM(trafione) FROM zdarzenia WHERE liga=? AND sezon=? AND trafione IS NOT NULL AND rynek='1X2'",
            (wybrana_liga, BIEZACY_SEZON)).fetchone()
        _con_hero.close()
        _hr_n = int(_hr_row[0]) if _hr_row and _hr_row[0] else 0
        _hr_v = (_hr_row[1]/_hr_row[0]*100) if _hr_row and _hr_row[0] else 0
        if _hr_n > 0:
            _hr_col = "#4CAF50" if _hr_v >= 62 else ("#FF9800" if _hr_v >= 55 else "#F44336")
            st.markdown(
                f"<div style='background:var(--bg-card);border:1px solid var(--border);border-radius:10px;"
                f"padding:12px 16px;text-align:center'>"
                f"<div style='font-size:0.68em;color:#555;text-transform:uppercase;letter-spacing:.05em'>Hit Rate (1X2)</div>"
                f"<div style='font-size:2em;font-weight:800;color:{_hr_col};line-height:1.1'>{_hr_v:.1f}%</div>"
                f"<div style='font-size:0.68em;color:#444'>{_hr_n} typów · sezon 25/26</div>"
                f"</div>", unsafe_allow_html=True)
    except Exception:
        pass

if not historical.empty:
    srednie_df  = oblicz_wszystkie_statystyki(historical.to_json())
    srednie_lig = oblicz_srednie_ligowe(historical.to_json())
    forma_dict  = oblicz_forme(historical)
    tabela      = tabela_ligowa(historical)
    rho         = srednie_lig["rho"]
    n_biezacy   = srednie_lig["n_biezacy"]
    w_prev      = waga_poprzedniego(n_biezacy)


    # telegram_bot.py nadal używany do alertów real-time (send_message, digest).


    # Pobierz dane do ostrzeżeń sędziowskich
    # Drużyny aktualnie grające w lidze = z terminarza bieżącego sezonu
    if not schedule.empty:
        druzyny_ligi = set(
            list(schedule["home_team"].map(map_nazwa).dropna()) +
            list(schedule["away_team"].map(map_nazwa).dropna())
        )
    else:
        druzyny_ligi = set()
    power_df, sedziowie_df = deep_data_stats(historical.to_json(), druzyny_ligi)

    # Sidebar: postęp sezonu + parametry
    if not schedule.empty:
        aktualna_kolejka  = get_current_round(schedule)
        wszystkie_kolejki = sorted(schedule["round"].unique())
        pozycja = wszystkie_kolejki.index(aktualna_kolejka) + 1 if aktualna_kolejka in wszystkie_kolejki else 0
        st.sidebar.caption(f"📅 Kolejka **#{aktualna_kolejka}** · sezon 25/26 · {n_biezacy} meczów")
        st.sidebar.progress(pozycja / len(wszystkie_kolejki), text=f"{pozycja}/{len(wszystkie_kolejki)}")

        # ── Auto-backfill predykcji dla wszystkich minionych kolejek ─────────
        # Odpala się gdy zmieni się liczba meczów w CSV (nowy gameweek w football-data)
        # Uzupełnia kolejki gdzie brakuje predykcji LUB brakuje wyników
        # Klucz time-based (co 30 min) żeby odpalał się nawet gdy CSV nie zmienił rozmiaru
        import time as _time
        _bf_slot = int(_time.time() // 1800)  # nowy slot co 30 min
        _bf_key = f"backfill_{wybrana_liga}_{aktualna_kolejka}_{_bf_slot}"
        if _bf_key not in st.session_state and not srednie_df.empty:
            _dzisiaj_bf = datetime.now().date()
            _n_bf = 0
            init_db()
            for _runda_bf in sorted(schedule["round"].unique()):
                if _runda_bf >= aktualna_kolejka:
                    continue  # tylko minione kolejki
                # Sprawdź stan tej kolejki w DB
                _con_bf = sqlite3.connect(DB_FILE)
                _existing = _con_bf.execute(
                    "SELECT COUNT(*) FROM zdarzenia WHERE liga=? AND kolejnosc=? AND rynek='1X2'",
                    (wybrana_liga, int(_runda_bf))).fetchone()[0]
                _null_results = _con_bf.execute(
                    "SELECT COUNT(*) FROM zdarzenia WHERE liga=? AND kolejnosc=? AND trafione IS NULL",
                    (wybrana_liga, int(_runda_bf))).fetchone()[0]
                _con_bf.close()
                if _existing > 0 and _null_results == 0:
                    continue  # kompletna kolejka – pomiń
                # Brakuje predykcji LUB wyników – uzupełnij
                _mecze_rundy_bf = schedule[schedule["round"] == _runda_bf]
                if _existing == 0:
                    # Generuj predykcje dla tej kolejki
                    for _, _mr_bf in _mecze_rundy_bf.iterrows():
                        _hbf = map_nazwa(_mr_bf["home_team"])
                        _abf = map_nazwa(_mr_bf["away_team"])
                        if _hbf not in srednie_df.index or _abf not in srednie_df.index:
                            continue
                        try:
                            _lhbf, _labf, _lrbf, _lkbf, _, _lsbf = oblicz_lambdy(
                                _hbf, _abf, srednie_df, srednie_lig, forma_dict,
                                csv_code=LIGI[wybrana_liga]["csv_code"])
                            _pbf = predykcja_meczu(_lhbf, _labf, rho=rho,
                                                   csv_code=LIGI[wybrana_liga]["csv_code"],
                                                   n_train=n_biezacy)
                            _mbf = f"{_hbf} – {_abf}"
                            zapisz_zdarzenia(wybrana_liga, int(_runda_bf), _mbf,
                                             _hbf, _abf, "1X2", _pbf["typ"], 0.0,
                                             _pbf["p_typ"], _pbf["fo_typ"])
                            for _em, _nz, _pz, _foz, _kz, _lz in alternatywne_zdarzenia(
                                    _lhbf, _labf, _lrbf, _lkbf, rho, lam_sot=_lsbf):
                                zapisz_zdarzenia(wybrana_liga, int(_runda_bf), _mbf,
                                                 _hbf, _abf, _kz, _nz, _lz, _pz, _foz)
                            _n_bf += 1
                        except Exception:
                            continue
                # Zaktualizuj wyniki dla tej kolejki (nowe predykcje lub stare bez wyników)
                for _, _mr_bf in _mecze_rundy_bf.iterrows():
                    _hbf2 = map_nazwa(_mr_bf["home_team"])
                    _abf2 = map_nazwa(_mr_bf["away_team"])
                    aktualizuj_wynik_zdarzenia(_hbf2, _abf2, historical)
            if _n_bf > 0:
                st.toast(f"📦 Uzupełniono predykcje dla {_n_bf} meczów z poprzednich kolejek", icon="📊")
            st.session_state[_bf_key] = _n_bf
    _shrink_now   = _get_shrink(LIGI[wybrana_liga]["csv_code"], n_biezacy)
    _shrink_base  = KALIBRACJA_PER_LIGA.get(LIGI[wybrana_liga]["csv_code"], SHRINK_ALPHA)
    _shrink_bonus = _shrink_now - _shrink_base
    _shrink_info  = f"shrink {_shrink_now:.2f}" + (f" (+{_shrink_bonus:.2f})" if _shrink_bonus > 0.01 else "")
    with st.sidebar.expander("⚙️ Parametry modelu", expanded=False):
        st.caption(f"ρ = `{rho:.4f}` · waga prev `{w_prev:.2f}` · {_shrink_info}")
        st.caption("Dane: football-data.co.uk")
        debug_mode = st.checkbox("🔧 Debug nazw", value=False, key="_dbg")

    # ── Model Sharpness vs Rynek (sidebar) – tu historical i srednie_df są dostępne
    if _OA_OK and _oa_key and _oa_cached and not schedule.empty:
        _ms = model_sharpness_vs_rynek(
            wybrana_liga, _oa_cached, _oa,
            schedule, srednie_df, srednie_lig, forma_dict,
            rho, n_biezacy, LIGI[wybrana_liga]["csv_code"])
        if _ms:
            with st.sidebar.expander("📊 Model Sharpness", expanded=True):
                _ad = _ms["avg_diff"]
                _sc = "#4CAF50" if 0.05 <= _ad <= 0.15 else ("#FF9800" if _ad < 0.05 else "#F44336")
                _label = "✅ Sweet Spot" if 0.05 <= _ad <= 0.15 else ("⚠️ Zbyt blisko" if _ad < 0.05 else "⚠️ Zbyt daleko")
                st.markdown(
                    f"<div style='background:var(--stat-bg);border:1px solid var(--border);border-radius:8px;padding:10px'>"
                    f"<div style='font-size:1.4em;font-weight:bold;color:{_sc};text-align:center'>{_ad:.1%}</div>"
                    f"<div style='font-size:0.78em;color:{_sc};text-align:center;margin-bottom:6px'>{_label}</div>"
                    f"<div style='display:flex;justify-content:space-around;font-size:0.72em;color:#666'>"
                    f"<div>🔵 Zbieżne<br><b style='color:#888'>{_ms['aligned']}</b></div>"
                    f"<div>🟢 Sweet<br><b style='color:#4CAF50'>{_ms['sweet']}</b></div>"
                    f"<div>🔴 Noise<br><b style='color:#F44336'>{_ms['noise']}</b></div>"
                    f"</div></div>",
                    unsafe_allow_html=True)
                st.caption("Sweet spot: różnica 5–15% model vs rynek")

    # ── Ekran startowy: Najważniejsze okazje kolejki ─────────────────────
    if not schedule.empty and not srednie_df.empty and _oa_cached:
        _start_kolejka = get_current_round(schedule)
        _start_mecze   = schedule[schedule["round"] == _start_kolejka]
        _start_top = []
        for _, _sm in _start_mecze.iterrows():
            _sh = map_nazwa(_sm["home_team"]); _sa = map_nazwa(_sm["away_team"])
            if _sh not in srednie_df.index or _sa not in srednie_df.index: continue
            try:
                _slh, _sla, _slr, _slk, _, _slsot = oblicz_lambdy(
                    _sh, _sa, srednie_df, srednie_lig, forma_dict,
                                  csv_code=LIGI[wybrana_liga]["csv_code"])
                _sp = predykcja_meczu(_slh, _sla, rho=rho,
                                       csv_code=LIGI[wybrana_liga]["csv_code"], n_train=n_biezacy)
                _so = _oa.znajdz_kursy(_sh, _sa, _oa_cached) if _OA_OK and _oa_key else None
                _skdc = None
                if _so:
                    _skdc, _sidc = _kurs_dc_live(_sp["typ"], _so["odds_h"], _so["odds_d"], _so["odds_a"])
                _sev = _sp["p_typ"] * (_skdc or _sp["fo_typ"]) - 1
                _smn = market_noise_check(_sp["p_typ"], _sidc) if _so and _skdc else None
                _kurs_ok = (_skdc is None) or (_skdc <= 3.50)
                # SOT Momentum per drużyna
                _smh = sot_momentum(_sh, historical)
                _sma = sot_momentum(_sa, historical)
                _start_top.append({
                    "mecz": f"{_sh} – {_sa}",
                    "h": _sh, "a": _sa,
                    "typ": _sp["typ"], "p": _sp["p_typ"],
                    "fo": _sp["fo_typ"], "kurs_buk": _skdc,
                    "ev": _sev, "is_val": _sev >= 0.04,
                    "noise": _smn["noise"] if _smn else False,
                    "data": _sm.get("date",""),
                    "entropy": _sp["entropy"],
                    "sot_mom_h": _smh, "sot_mom_a": _sma,
                })
            except Exception: continue
        _top_val = sorted([x for x in _start_top if x["is_val"] and not x["noise"]],
                          key=lambda x: -x["ev"])[:4]
        if _top_val:
            st.markdown(
                "<div class='section-header'>🏆 Value Bets tej kolejki"
                "<span style='font-size:.65em;color:#555;font-weight:400;margin-left:10px'>"
                "EV ≥ 4% · model vs rynek</span></div>",
                unsafe_allow_html=True)
            _tv_cols = st.columns(max(len(_top_val), 1))
            for _tvc, _tv in zip(_tv_cols, _top_val):
                _kb  = st.session_state.get("bankroll", 1000.0)
                _kf2 = st.session_state.get("kelly_frac", 0.125)
                # Dynamic Kelly: mnożnik entropy
                _ent   = _tv.get("entropy", 1.0)
                _ekm   = entropy_kelly_multiplier(_ent)
                _kl    = kelly_stake(_tv["p"], _tv["kurs_buk"] or _tv["fo"],
                                     bankroll=_kb, fraction=_kf2 * _ekm)
                _kurs_str  = f"{_tv['kurs_buk']:.2f}" if _tv["kurs_buk"] else f"{_tv['fo']:.2f}✦"
                _ev_pct    = f"{_tv['ev']:+.1%}"
                _buk_p     = 1 / (_tv['kurs_buk'] or _tv['fo'])
                _edge_pp   = (_tv['p'] - _buk_p) * 100
                # Quality Score
                _qs, _qs_str, _qs_label, _qs_tip = quality_score(
                    _tv["p"], _ent,
                    _tv.get("sot_mom_h"), _tv.get("sot_mom_a"),
                    ev=_tv["ev"])
                # SOT Momentum badge
                _smh, _sma = _tv.get("sot_mom_h"), _tv.get("sot_mom_a")
                _sot_badge = ""
                if _smh and _smh >= 1.35:
                    _sot_badge += f"<span class='stat-pill' title='Gospodarz {_smh:.0%} więcej strzałów celnych niż sezonowo'>🔥 Atak H +{(_smh-1)*100:.0f}%</span>"
                elif _smh and _smh >= 1.15:
                    _sot_badge += f"<span class='stat-pill' title='SOT Momentum Gospodarz'>↑SOT H +{(_smh-1)*100:.0f}%</span>"
                if _sma and _sma >= 1.35:
                    _sot_badge += f"<span class='stat-pill' title='Gość {_sma:.0%} więcej strzałów celnych niż sezonowo'>🔥 Atak A +{(_sma-1)*100:.0f}%</span>"
                elif _sma and _sma >= 1.15:
                    _sot_badge += f"<span class='stat-pill' title='SOT Momentum Gość'>↑SOT A +{(_sma-1)*100:.0f}%</span>"
                # EV sygnalizacja
                if _tv['ev'] >= 0.15:
                    _sig_color = "#2e7d32"; _sig_label = "🟢 Wysoki EV"
                elif _tv['ev'] >= 0.05:
                    _sig_color = "#f57c00"; _sig_label = "🟡 Umiarkowany EV"
                else:
                    _sig_color = "#c62828"; _sig_label = "🔴 Niski EV"
                # Quality color
                _qs_c = {5:"#1a6b1a", 4:"#2e7d32", 3:"#f57c00", 2:"#888", 1:"#c62828"}.get(_qs, "#888")
                # Kelly z korektą entropy
                _kelly_str = ""
                if _kl['safe']:
                    _ekm_note = f" ×{_ekm:.2f}" if _ekm < 0.95 else ""
                    _kelly_str = (f"<div class='kelly'>💰 Kelly: <b>{int(_kl['stake_pln'])} zł</b>"
                                  f"<span style='color:#666;font-size:0.85em'>{_ekm_note}</span></div>")
                _tvc.markdown(
                    f"<div class='vb-card' style='border-color:{_sig_color}'>"
                    # Quality Score — górny prawy
                    f"<div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:2px'>"
                    f"<div class='date'>{str(_tv['data'])[:10] if _tv['data'] else ''}</div>"
                    f"<div title='Quality Score: {_qs_tip}' style='font-size:0.78em;color:{_qs_c};"
                    f"font-weight:700;letter-spacing:1px'>{_qs_str}"
                    f"<span style='font-size:0.72em;font-weight:400;margin-left:3px;color:{_qs_c}88'>{_qs_label}</span>"
                    f"</div></div>"
                    # Herby + nazwy
                    f"<div style='display:flex;align-items:center;justify-content:center;gap:6px;margin:4px 0'>"
                    f"{(chr(60)+'img src='+chr(39)+_sbn.get(_tv.get('h',''),{}).get('crest','')+chr(39)+' style='+chr(39)+'width:22px;height:22px;object-fit:contain;vertical-align:middle'+chr(39)+' onerror='+chr(34)+'this.style.display='+chr(39)+'none'+chr(39)+chr(34)+chr(62)) if _sbn.get(_tv.get('h',''),{}).get('crest') else ''}"
                    f"<span style='font-size:0.78em;color:#999'>{_tv.get('h','')}</span>"
                    f"<span style='color:#444;font-size:0.75em'>–</span>"
                    f"<span style='font-size:0.78em;color:#999'>{_tv.get('a','')}</span>"
                    f"{(chr(60)+'img src='+chr(39)+_sbn.get(_tv.get('a',''),{}).get('crest','')+chr(39)+' style='+chr(39)+'width:22px;height:22px;object-fit:contain;vertical-align:middle'+chr(39)+' onerror='+chr(34)+'this.style.display='+chr(39)+'none'+chr(39)+chr(34)+chr(62)) if _sbn.get(_tv.get('a',''),{}).get('crest') else ''}"
                    f"</div>"
                    f"<div class='bet' style='color:{_sig_color}'>{_tv['typ']} @ {_kurs_str}</div>"
                    f"<div class='meta'>"
                    f"<span title='Prawdopodobieństwo modelu Dixon-Coles'>Model: {_tv['p']:.0%}</span>"
                    f" &nbsp;·&nbsp; "
                    f"<span class='ev' style='color:{_sig_color}' title='Expected Value'>EV: {_ev_pct}</span>"
                    f"</div>"
                    f"<div style='margin-top:6px;display:flex;flex-wrap:wrap;gap:4px'>"
                    f"<span class='stat-pill' style='border-color:{_sig_color}33;color:{_sig_color}'>{_sig_label}</span>"
                    f"<span class='stat-pill' title='Edge = różnica między p modelu a implied prob bukmachera'>EDGE +{_edge_pp:.1f}pp</span>"
                    f"{_sot_badge}"
                    f"</div>"
                    f"{_kelly_str}"
                    f"</div>",
                    unsafe_allow_html=True)
            st.markdown(
                "<p style='font-size:0.72em;color:#444;margin:8px 0 16px 0'>"
                "✦ = fair odds (brak danych bukmachera) · "
                "EV = p × kurs − 1 · "
                "★★★★★ = Quality Score (p + Chaos + SOT Momentum + EV) · "
                "Kelly ×N = korekta na Chaos Index</p>",
                unsafe_allow_html=True)
            st.markdown("---")

    # TABS
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⚽ Mecze",
        "📊 Value Bets",
        "📈 Skuteczność",
        "🌍 Multi-Liga",
        "🧪 Backtest",
    ])

    # =========================================================================
    # TAB 2 – RANKING ZDARZEŃ (przeniesiony)
    # =========================================================================
    with tab1:
        st.subheader("⚽ Analiza meczu – aktualna kolejka")
        st.caption("Szczegółowa analiza każdego meczu. Rozwiń mecz → sprawdź rynki → zapisz do trackingu.")

        # ── Briefing kolejki – Stability Score ─────────────────────────────
        tgl1, tgl2 = st.columns(2)
        with tgl1: pokaz_komentarz = st.toggle("📊 Forma & Kontekst", value=True)
        with tgl2: pokaz_macierz  = st.toggle("🔢 Macierz wyników", value=False)
        st.markdown(
            "<div style='font-size:0.70em;color:#4b5563;margin:2px 0 8px;line-height:1.6'>"
            "⭐ <b style='color:#6b7280'>Quality Score</b>: "
            "<span style='color:#1a6b1a'>★★★★★</span> Elita (p≥75%, stabilny) &nbsp;·&nbsp; "
            "<span style='color:#2e7d32'>★★★★</span> Mocny &nbsp;·&nbsp; "
            "<span style='color:#f57c00'>★★★</span> Średni &nbsp;·&nbsp; "
            "<span style='color:#c62828'>★★</span> Słaby &nbsp;·&nbsp; "
            "<span style='color:#880e4f'>★</span> Unikaj"
            "</div>",
            unsafe_allow_html=True)

        if not schedule.empty and not srednie_df.empty:
            _auto_kolejka = get_current_round(schedule)
            _dostepne_rundy = get_available_rounds(schedule)
            if _dostepne_rundy:
                _rundy_labels = [lbl for _, lbl in _dostepne_rundy]
                _rundy_vals   = [r   for r,  _   in _dostepne_rundy]
                # Domyslny indeks = automatycznie wykryta kolejka
                _def_idx = _rundy_vals.index(_auto_kolejka) if _auto_kolejka in _rundy_vals else 0
                _sel_label = st.selectbox(
                    "📅 Kolejka",
                    _rundy_labels,
                    index=_def_idx,
                    key="tab1_kolejka_sel",
                    help="Wybierz kolejkę ręcznie jeśli auto-wykrycie jest błędne (np. zaległe mecze Bundesligi)"
                )
                aktualna_kolejka = _rundy_vals[_rundy_labels.index(_sel_label)]
            else:
                aktualna_kolejka = _auto_kolejka
            mecze = schedule[schedule["round"] == aktualna_kolejka]

            if not mecze.empty:
                st.caption(f"Kolejka #{aktualna_kolejka} – {len(mecze)} meczów")

                # Grupuj mecze według dnia
                DAYS_PL = {
                    "Monday":"Poniedziałek","Tuesday":"Wtorek","Wednesday":"Środa",
                    "Thursday":"Czwartek","Friday":"Piątek","Saturday":"Sobota","Sunday":"Niedziela"
                }
                mecze_sorted = mecze.sort_values("date")
                mecze_sorted["_date_only"] = mecze_sorted["date"].dt.date
                grupy_dni = mecze_sorted.groupby("_date_only", sort=True)

                ikony_t = {"1":"🔵","X":"🟠","2":"🔴","1X":"🟣","X2":"🟣"}
                dzien_ikony = {
                    "Sobota":"🟩","Niedziela":"🟦","Poniedziałek":"⬜",
                    "Wtorek":"⬜","Środa":"⬜","Czwartek":"⬜","Piątek":"🟨"
                }

                for data_dnia, mecze_dnia in grupy_dni:
                    dzien_en = data_dnia.strftime("%A")
                    dzien_pl = DAYS_PL.get(dzien_en, dzien_en)
                    data_fmt = data_dnia.strftime("%d.%m.%Y")
                    ikona_d  = dzien_ikony.get(dzien_pl, "⬜")
                    n_dnia   = len(mecze_dnia)

                    # Nagłówek dnia
                    st.markdown(
                        f"<div style='background:var(--date-bar-bg);"
                        f"padding:8px 16px;border-radius:8px;margin:18px 0 8px;"
                        f"border-left:3px solid #4CAF50'>"
                        f"<span style='font-size:0.95em;font-weight:bold;color:#ddd'>"
                        f"{ikona_d} {dzien_pl}, {data_fmt}</span>"
                        f"<span style='float:right;color:#555;font-size:0.80em'>{n_dnia} {'mecz' if n_dnia==1 else 'mecze' if n_dnia<5 else 'meczów'}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    kol_a, kol_b = st.columns(2)
                    for idx, (_, mecz) in enumerate(mecze_dnia.iterrows()):
                        h = map_nazwa(mecz["home_team"])
                        a = map_nazwa(mecz["away_team"])
                        if h not in srednie_df.index or a not in srednie_df.index:
                            continue
                        # ID drużyn z terminarza (potrzebne do H2H API)
                        _mecz_h_id = mecz.get("home_id")
                        _mecz_a_id = mecz.get("away_id")

                        lam_h, lam_a, lam_r, lam_k, sot_ok, lam_sot = oblicz_lambdy(h, a, srednie_df, srednie_lig, forma_dict,
                                                                    csv_code=LIGI[wybrana_liga]["csv_code"],
                                                                    historical=historical,
                                                                    mecz_data=mecz.get("date"))
                        pred = predykcja_meczu(lam_h, lam_a, rho=rho, csv_code=LIGI[wybrana_liga]["csv_code"], n_train=n_biezacy)
                        data_meczu = mecz["date"].strftime("%d.%m %H:%M") if pd.notna(mecz["date"]) else ""

                        kolumna = kol_a if idx % 2 == 0 else kol_b
                        conf_i  = "🟢" if pred["conf_level"]=="High" else ("🟡" if pred["conf_level"]=="Medium" else "🔴")
                        sot_badge = " 🎯SOT" if sot_ok else ""

                        sedzia = mecz.get("Referee", "Nieznany") if "Referee" in mecz else "Nieznany"
                        sedzia_ostr = ostrzezenie_sedziego(sedzia, sedziowie_df)

                        # Pro Match Card
                        _conf_pct_l = {"High": 82, "Medium": 55, "Coinflip": 35}.get(pred["conf_level"], 40)
                        _conf_col_l = {"High": "#4CAF50", "Medium": "#FF9800", "Coinflip": "#F44336"}.get(pred["conf_level"], "#888")
                        _conf_icon  = {"High": "🟢", "Medium": "🟡", "Coinflip": "🔴"}.get(pred["conf_level"], "⚪")
                        _typ_bg     = {"1": "#1b5e20", "X": "#4a3000", "2": "#0d2a4a"}.get(pred["typ"], "#1a1a2e")
                        _typ_fg     = {"1": "#4CAF50", "X": "#FFC107", "2": "#42a5f5"}.get(pred["typ"], "#aaa")
                        # Herby z standings
                        _mc_h_crest = _sbn.get(h, {}).get("crest", "")
                        _mc_a_crest = _sbn.get(a, {}).get("crest", "")
                        _mc_h_img = (f"<img src='{_mc_h_crest}' style='width:28px;height:28px;"
                                     f"object-fit:contain;vertical-align:middle' "
                                     f"onerror=\"this.style.display='none'\">") if _mc_h_crest else "🏟️"
                        _mc_a_img = (f"<img src='{_mc_a_crest}' style='width:28px;height:28px;"
                                     f"object-fit:contain;vertical-align:middle' "
                                     f"onerror=\"this.style.display='none'\">") if _mc_a_crest else "🏟️"
                        with kolumna:
                            # ── Karta nagłówkowa – zawsze widoczna ──────
                            st.markdown(
                                f"<div style='background:var(--stat-bg);border:1px solid var(--border);"
                                f"border-top:3px solid {_conf_col_l};"
                                f"border-radius:10px 10px 0 0;padding:11px 14px 10px'>"
                                # Wiersz 1: herb + nazwa | data | nazwa + herb
                                f"<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:7px'>"
                                f"<div style='display:flex;align-items:center;gap:7px;flex:1;min-width:0'>"
                                f"{_mc_h_img}"
                                f"<b style='color:var(--text-primary);font-size:0.93em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>{h}</b>"
                                f"</div>"
                                f"<span style='color:#4b5563;font-size:0.70em;padding:0 6px;flex-shrink:0'>{data_meczu}</span>"
                                f"<div style='display:flex;align-items:center;gap:7px;flex:1;min-width:0;justify-content:flex-end'>"
                                f"<b style='color:var(--text-primary);font-size:0.93em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;direction:rtl'>{a}</b>"
                                f"{_mc_a_img}"
                                f"</div>"
                                f"</div>"
                                # Wiersz 2: p% duże + badge typ + pasek confidence
                                f"<div style='display:flex;align-items:center;gap:8px;margin-top:6px'>"
                                f"<span style='font-size:1.35em;font-weight:900;color:{_conf_col_l};"
                                f"min-width:52px;text-align:center;letter-spacing:-0.5px'>"
                                f"{pred['p_typ']:.0%}</span>"
                                f"<div style='flex:1'>"
                                f"<div style='display:flex;align-items:center;gap:6px;margin-bottom:4px'>"
                                f"<span style='background:{_typ_bg};color:{_typ_fg};font-size:0.75em;"
                                f"font-weight:700;padding:2px 10px;border-radius:10px'>"
                                f"{pred['typ']} @ {pred['fo_typ']:.2f}</span>"
                                f"<span style='color:#4b5563;font-size:0.68em'>{_conf_icon} {pred['conf_level']}</span>"
                                f"</div>"
                                f"<div style='background:var(--bg-card2);border-radius:3px;height:4px'>"
                                f"<div style='background:{_conf_col_l};width:{_conf_pct_l}%;height:4px;border-radius:3px'></div>"
                                f"</div>"
                                f"</div>"
                                f"</div>",
                                unsafe_allow_html=True)
                            # ── Expander ze szczegółami (łączy się wizualnie z kartą) ──
                            with st.expander("▾ szczegóły", expanded=False):
                                _conf_pct = _conf_pct_l
                                _conf_col = _conf_col_l
                                _conf_lbl = {"High": "Wysoka pewność", "Medium": "Umiarkowana", "Coinflip": "Wyrównany"}.get(pred["conf_level"], "")

                                # ── 3 zakładki progressive disclosure ────────
                                _ctx = oblicz_kontekst_meczu(h, a, historical, _sbn)

                                _dt1, _dt2, _dt3, _dt4 = st.tabs(["📊 Analiza", "📈 Statystyki", "⚔️ H2H", "🔍 Kontekst"])

                                with _dt1:
                                    # Wynik modelowy
                                    st.markdown(
                                        f"<div style='text-align:center;font-size:1.7em;font-weight:bold;margin:4px 0'>"
                                        f"⚽ {pred['wynik_h']}:{pred['wynik_a']}"
                                        f"<span style='font-size:0.5em;color:#888;margin-left:8px'>({pred['p_exact']:.1%})</span></div>",
                                        unsafe_allow_html=True,
                                    )

                                    conf_colors  = {"High":"#4CAF50","Medium":"#FF9800","Coinflip":"#F44336"}
                                    chaos_colors = {"Pewny":"#4CAF50","Klarowny":"#8BC34A","Otwarty":"#FF9800","Chaos":"#F44336"}
                                    conf_c = conf_colors.get(pred["conf_level"], "#888")
                                    ch_c   = chaos_colors.get(pred["chaos_label"], "#888")
                                    bar_w  = int(pred["chaos_pct"] * 100)
                                    # Pre-compute Quality Score dla pro card
                                    _pc_smh = sot_momentum(h, historical)
                                    _pc_sma = sot_momentum(a, historical)
                                    _pc_qs, _pc_qs_str, _pc_qs_label, _pc_qs_tip = quality_score(
                                        pred["p_typ"], pred["entropy"], _pc_smh, _pc_sma, ev=0.0)
                                    _pc_qs_c = {5:"#1a6b1a",4:"#2e7d32",3:"#f57c00",2:"#888",1:"#c62828"}.get(_pc_qs,"#888")
                                    _pc_qs_html = (
                                        f"<div style='margin-top:8px;display:flex;flex-wrap:wrap;gap:4px;align-items:center'>"
                                        f"<span title='Quality Score: {_pc_qs_tip}' style='font-size:0.85em;color:{_pc_qs_c};"
                                        f"font-weight:700;letter-spacing:1px'>{_pc_qs_str} "
                                        f"<span style='font-size:0.72em;font-weight:400'>{_pc_qs_label}</span></span>"
                                        + (f"<span class='stat-pill' title='Gospodarz {_pc_smh:.0%} więcej SOT niż sezonowo'>🔥 Atak H +{(_pc_smh-1)*100:.0f}%</span>" if _pc_smh and _pc_smh >= 1.35 else "")
                                        + (f"<span class='stat-pill'>↑SOT H +{(_pc_smh-1)*100:.0f}%</span>" if _pc_smh and 1.15 <= _pc_smh < 1.35 else "")
                                        + (f"<span class='stat-pill' title='Gość {_pc_sma:.0%} więcej SOT niż sezonowo'>🔥 Atak A +{(_pc_sma-1)*100:.0f}%</span>" if _pc_sma and _pc_sma >= 1.35 else "")
                                        + (f"<span class='stat-pill'>↑SOT A +{(_pc_sma-1)*100:.0f}%</span>" if _pc_sma and 1.15 <= _pc_sma < 1.35 else "")
                                        + f"</div>"
                                    )
                                    st.markdown(
                                        f"<div style='text-align:center;margin-bottom:4px'>"
                                        f"Typ: {badge_typ(pred['typ'])}&nbsp;&nbsp;"
                                        f"<span style='font-size:0.88em;color:#888'>Fair Odds: <b>{pred['fo_typ']:.2f}</b> "
                                        f"({pred['p_typ']:.1%}"
                                        + (f"±{pred['ci_half']:.0%}" if pred.get('ci_half',0)>0.01 else "") +
                                        ")</span>"
                                        f"</div>"
                                        f"<div style='text-align:center;font-size:0.80em;color:{conf_c};margin-bottom:6px'>"
                                        f"{pred['conf_emoji']} <b>{pred['conf_level']}</b> · {pred['conf_opis']}"
                                        f"</div>"
                                        f"<div style='margin:0 8px 8px 8px'>"
                                        f"<div style='font-size:0.78em;color:#888'>"
                                        f"{pred['chaos_emoji']} Chaos Index: <b style='color:{ch_c}'>{pred['chaos_label']}</b>"
                                        f" ({pred['entropy']:.2f} bits)</div>"
                                        f"<div style='background:var(--bg-card2);border-radius:4px;height:4px;margin-top:4px'>"
                                        f"<div style='background:{ch_c};width:{bar_w}%;height:4px;border-radius:4px'></div>"
                                        f"</div>"
                                        f"{_pc_qs_html}"
                                        f"</div></div>",
                                        unsafe_allow_html=True,
                                    )

                                    # 1/X/2 prawdopodobieństwa
                                    st.markdown(
                                        f"<div style='display:flex;justify-content:space-around;margin:6px 0 2px'>"
                                        f"<div style='text-align:center'>"
                                        f"<div style='font-size:0.70em;color:#888;text-transform:uppercase;letter-spacing:1px'>1</div>"
                                        f"<div style='font-size:1.10em;font-weight:bold;color:#4CAF50'>{pred['p_home']:.0%}</div>"
                                        f"<div style='font-size:0.68em;color:#555'>fair {pred['fo_home']:.2f}</div></div>"
                                        f"<div style='text-align:center'>"
                                        f"<div style='font-size:0.70em;color:#888;text-transform:uppercase;letter-spacing:1px'>X</div>"
                                        f"<div style='font-size:1.10em;font-weight:bold;color:#FF9800'>{pred['p_draw']:.0%}</div>"
                                        f"<div style='font-size:0.68em;color:#555'>fair {pred['fo_draw']:.2f}</div></div>"
                                        f"<div style='text-align:center'>"
                                        f"<div style='font-size:0.70em;color:#888;text-transform:uppercase;letter-spacing:1px'>2</div>"
                                        f"<div style='font-size:1.10em;font-weight:bold;color:#2196F3'>{pred['p_away']:.0%}</div>"
                                        f"<div style='font-size:0.68em;color:#555'>fair {pred['fo_away']:.2f}</div></div>"
                                        f"</div>",
                                        unsafe_allow_html=True,
                                    )

                                    sot_info = " · 🎯 SOT blend aktywny" if sot_ok else " · gole only"
                                    st.markdown(
                                        f"<div style='text-align:center;font-size:0.78em;color:#555;margin-top:2px'>"
                                        f"λ {h[:8]}: <b style='color:#aaa'>{lam_h:.2f}</b> &nbsp;|&nbsp; "
                                        f"λ {a[:8]}: <b style='color:#aaa'>{lam_a:.2f}</b> &nbsp;|&nbsp; "
                                        f"Σ: <b style='color:#aaa'>{lam_h+lam_a:.2f}</b>"
                                        f"<span style='color:#4CAF50'>{sot_info}</span></div>",
                                        unsafe_allow_html=True,
                                    )

                                    # ── Match DNA strip ───────────────────────────────────────────
                                    # Kompaktowe sygnały wyróżniające ten mecz vs średnią ligową
                                    # Pojawia się tylko gdy są istotne odchylenia — nie zaśmieca widoku
                                    try:
                                        _ctx_q = oblicz_kontekst_meczu(h, a, historical, _sbn) \
                                            if not historical.empty else {}
                                        _hs_q  = _ctx_q.get("h_stats", {})
                                        _as_q  = _ctx_q.get("a_stats", {})

                                        # Liga avg jako punkt odniesienia
                                        _lig_yc = (historical["HY"].mean() + historical["AY"].mean()) \
                                            if "HY" in historical.columns else 3.8
                                        _lig_co = (historical["HC"].mean() + historical["AC"].mean()) \
                                            if "HC" in historical.columns else 9.8
                                        _lig_fo = (historical["HF"].mean() + historical["AF"].mean()) \
                                            if "HF" in historical.columns else 21.5

                                        _dna_pills = []

                                        # Styl gry — tagi (max 2, po jednym na drużynę)
                                        _h_tags_q = _hs_q.get("style_tags", [])
                                        _a_tags_q = _as_q.get("style_tags", [])
                                        _tag_c_map = {
                                            "🟥 Hot Heads":        "#ef4444",
                                            "🎭 Foul Magnet":      "#f97316",
                                            "📐 Corner Machine":   "#3b82f6",
                                            "🎯 Central Attackers":"#8b5cf6",
                                            "🚌 Bus Parker":       "#6b7280",
                                            "🛡️ Aktywna obrona":  "#06b6d4",
                                        }
                                        for _dtag in _h_tags_q:
                                            if _dtag == "⚖️ Balanced": continue
                                            _dtc = _tag_c_map.get(_dtag, "#6b7280")
                                            _dna_pills.append((
                                                f"{_dtag}",
                                                f"{h[:9]}",
                                                _dtc
                                            ))
                                        for _dtag in _a_tags_q:
                                            if _dtag == "⚖️ Balanced": continue
                                            _dtc = _tag_c_map.get(_dtag, "#6b7280")
                                            _dna_pills.append((
                                                f"{_dtag}",
                                                f"{a[:9]}",
                                                _dtc
                                            ))

                                        # Kartki — jeśli matchup ≥15% powyżej średniej
                                        _lam_yc_q = _ctx_q.get("lambda_yc_adj") or _ctx_q.get("lambda_yc")
                                        if _lam_yc_q and _lig_yc > 0:
                                            _yc_diff_pct = (_lam_yc_q - _lig_yc) / _lig_yc
                                            if _yc_diff_pct >= 0.15:
                                                _dna_pills.append((f"🟨 +{_yc_diff_pct:.0%} kartki", f"λ={_lam_yc_q:.1f}", "#f59e0b"))
                                            elif _yc_diff_pct <= -0.15:
                                                _dna_pills.append((f"🟨 {_yc_diff_pct:.0%} kartki", f"λ={_lam_yc_q:.1f}", "#6b7280"))

                                        # Rożne — Pressure Tolerance matchup
                                        _exp_co_q  = _ctx_q.get("exp_co_pt_total")
                                        _lam_co_q  = _ctx_q.get("lambda_co")
                                        _ref_co    = _exp_co_q or _lam_co_q
                                        if _ref_co and _lig_co > 0:
                                            _co_diff_pct = (_ref_co - _lig_co) / _lig_co
                                            if _co_diff_pct >= 0.15:
                                                _dna_pills.append((f"🚩 +{_co_diff_pct:.0%} rożne", f"exp={_ref_co:.1f}", "#3b82f6"))
                                            elif _co_diff_pct <= -0.15:
                                                _dna_pills.append((f"🚩 {_co_diff_pct:.0%} rożne", f"exp={_ref_co:.1f}", "#6b7280"))

                                        # H2H gorące
                                        _h2hs_q = _ctx_q.get("h2h_stats", {})
                                        if _h2hs_q.get("hot_rivalry"):
                                            _dna_pills.append((f"🔥 Gorące H2H", f"avg {_h2hs_q['avg_cards']:.1f}🟨", "#ef4444"))

                                        # CLASH alert
                                        if _ctx_q.get("clash_alert"):
                                            _dna_pills.append(("⚔️ Foul Clash", f"+{_ctx_q['clash_vs_avg']:.0f}%", "#ef4444"))

                                        # Home fortress lub travel team
                                        _h_ha_q = _ctx_q.get("h_ha") or {}
                                        _h_adv_q = _h_ha_q.get("home_adv")
                                        if _h_adv_q and _h_adv_q > 0.7:
                                            _dna_pills.append(("🏰 Twierdza", f"+{_h_adv_q:.2f}ppg", "#22c55e"))
                                        elif _h_adv_q and _h_adv_q < -0.2:
                                            _dna_pills.append(("✈️ Lepszy wyjazdowo", f"{_h_adv_q:+.2f}ppg", "#f97316"))

                                        # Volatility warning
                                        if _ctx_q.get("vol_warning_yc"):
                                            _dna_pills.append(("⚠️ Zmienny", f"CV={_ctx_q.get('vol_yc_matchup', 0):.2f}", "#f97316"))

                                        if _dna_pills:
                                            _pills_html = "".join([
                                                f"<span title='{sub}' style='"
                                                f"background:{col}18;border:1px solid {col}55;"
                                                f"border-radius:20px;padding:2px 8px;margin:2px;"
                                                f"font-size:0.68em;color:{col};font-weight:600;"
                                                f"display:inline-flex;align-items:center;gap:4px;"
                                                f"white-space:nowrap'>"
                                                f"{label}"
                                                f"<span style='opacity:0.65;font-size:0.85em'>{sub}</span>"
                                                f"</span>"
                                                for label, sub, col in _dna_pills[:7]
                                            ])
                                            st.markdown(
                                                f"<div style='margin:6px 0 4px;display:flex;"
                                                f"flex-wrap:wrap;gap:2px;align-items:center'>"
                                                f"<span style='font-size:0.62em;color:#374151;"
                                                f"text-transform:uppercase;letter-spacing:.08em;"
                                                f"margin-right:2px;flex-shrink:0'>DNA</span>"
                                                + _pills_html +
                                                f"</div>",
                                                unsafe_allow_html=True
                                            )
                                    except Exception:
                                        pass

                                    # Kursy live + Strefa Decyzji z Gauge
                                    _kurs_live_1x2 = None
                                    if _OA_OK and _oa_key and _oa_cached:
                                        _o = _oa.znajdz_kursy(h, a, _oa_cached)
                                        if _o:
                                            _kurs_live_1x2 = _o
                                            _bk_lbl = _o.get("bookmaker", "Pinnacle")
                                            _kdc, _idc = _kurs_dc_live(pred["typ"], _o["odds_h"], _o["odds_d"], _o["odds_a"])
                                            if _kdc:
                                                _edge   = pred["p_typ"] - _idc
                                                _ev_val = pred["p_typ"] * (_kdc - 1) - (1 - pred["p_typ"])
                                                _ev_c   = "#4CAF50" if _ev_val > 0 else "#F44336"
                                                _is_val = _ev_val >= 0.04 and pred["p_typ"] >= PROG_PEWNY
                                                _vbadge = " <b style='color:#4CAF50'>✓ VALUE</b>" if _is_val else ""
                                                _bankroll = st.session_state.get("bankroll", KELLY_BANKROLL_DEFAULT)
                                                _kelly = kelly_stake(pred["p_typ"], _kdc, bankroll=_bankroll)

                                                # Real-time alerty powodowały spam przy każdym rerun Streamlit.
                                                # Powiadomienia idą raz dziennie przez send_morning_digest.
                                                _mn2 = market_noise_check(pred["p_typ"], _idc)
                                                if _mn2["noise"]:
                                                    st.markdown(
                                                        f"<div style='background:var(--bg-card);border:1px solid #F44336;"
                                                        f"border-radius:5px;padding:4px 10px;margin:3px 0;"
                                                        f"font-size:0.76em;color:#F44336'>"
                                                        f"⚠️ <b>Market Noise</b> – różnica: "
                                                        f"<b>{_mn2['diff']:.0%}</b> · Sprawdź skład!</div>",
                                                        unsafe_allow_html=True)

                                                st.markdown(
                                                    f"<div style='background:var(--bg-card);border:1px solid "
                                                    f"{'#2a6b2a' if _is_val else '#1e2a3a'};"
                                                    f"border-radius:8px;padding:9px 14px;margin:4px 0'>"
                                                    f"<div style='font-size:0.74em;color:#555;margin-bottom:4px'>"
                                                    f"📊 {_bk_lbl} — kursy live</div>"
                                                    f"<div style='display:flex;justify-content:space-around;margin-bottom:5px'>"
                                                    f"<div style='text-align:center'><div style='font-size:0.65em;color:#666'>1</div>"
                                                    f"<div style='font-weight:bold;color:#aaa'>{_o['odds_h']:.2f}</div></div>"
                                                    f"<div style='text-align:center'><div style='font-size:0.65em;color:#666'>X</div>"
                                                    f"<div style='font-weight:bold;color:#aaa'>{_o['odds_d']:.2f}</div></div>"
                                                    f"<div style='text-align:center'><div style='font-size:0.65em;color:#666'>2</div>"
                                                    f"<div style='font-weight:bold;color:#aaa'>{_o['odds_a']:.2f}</div></div>"
                                                    f"</div>"
                                                    f"<div style='border-top:1px solid #1e2a3a;padding-top:5px;font-size:0.82em'>"
                                                    f"Typ: {badge_typ(pred['typ'])} &nbsp;"
                                                    f"Fair: <b>{pred['fo_typ']:.2f}</b> | "
                                                    f"Buk: <b>{_kdc:.2f}</b> | "
                                                    f"<abbr title='Edge = p modelu minus implied prob bukmachera. Pozytywny = przewaga nad rynkiem.' "
                                                    f"style='cursor:help;border-bottom:1px dotted #555'>Edge</abbr>: "
                                                    f"<span style='color:{_ev_c}'><b>{_edge:+.1%}</b></span> | "
                                                    f"<span style='color:{_ev_c}'><b>EV {_ev_val:+.1%}</b></span>"
                                                    f"{_vbadge}</div>"
                                                    f"</div>",
                                                    unsafe_allow_html=True)

                                                if _is_val and _kelly["safe"]:
                                                    _ev_col = "#4CAF50" if _kelly["ev_per_unit"] >= 0.10 else "#FF9800"
                                                    # Gauge – margines bezpieczeństwa
                                                    _fair_p = pred["p_typ"]
                                                    _buk_p  = 1 / _kdc
                                                    _gauge_max = 0.20
                                                    _gauge_pct = min(100, max(0, int((_fair_p - _buk_p) / _gauge_max * 50) + 50))
                                                    st.markdown(
                                                        f"<div style='margin:6px 0;padding:8px 12px;"
                                                        f"background:var(--bg-card);border:1px solid var(--border);border-radius:8px'>"
                                                        f"<div style='font-size:0.68em;color:#555;margin-bottom:5px;"
                                                        f"text-transform:uppercase;letter-spacing:.06em'>📐 Margines bezpieczeństwa</div>"
                                                        f"<div style='display:flex;align-items:center;"
                                                        f"justify-content:space-between;font-size:0.72em;color:#666;margin-bottom:3px'>"
                                                        f"<span>Kurs buka</span><span>Kurs Fair</span></div>"
                                                        f"<div style='position:relative;height:12px;"
                                                        f"background:linear-gradient(90deg,#F44336 0%,#FF9800 40%,#4CAF50 100%);"
                                                        f"border-radius:6px;margin-bottom:4px'>"
                                                        f"<div style='position:absolute;top:-2px;left:{_gauge_pct}%;"
                                                        f"transform:translateX(-50%);width:4px;height:16px;"
                                                        f"background:#fff;border-radius:2px;box-shadow:0 0 4px #fff8'></div>"
                                                        f"<div style='position:absolute;top:-2px;left:50%;"
                                                        f"transform:translateX(-50%);width:2px;height:16px;"
                                                        f"background:#888;border-radius:1px'></div>"
                                                        f"</div>"
                                                        f"<div style='display:flex;justify-content:space-between;font-size:0.70em'>"
                                                        f"<span style='color:#888'>{_buk_p:.1%} impl.</span>"
                                                        f"<span style='color:{_ev_col};font-weight:700'>Edge {_edge:+.1%}</span>"
                                                        f"<span style='color:#888'>{_fair_p:.1%} model</span>"
                                                        f"</div></div>",
                                                        unsafe_allow_html=True)
                                                    st.markdown(
                                                        "<div style='margin:4px 0;padding:10px 14px;"
                                                        "background:linear-gradient(135deg,#021a08 0%,#011208 100%);"
                                                        f"border-left:3px solid {_ev_col};border-radius:0 6px 6px 0;"
                                                        f"box-shadow:0 0 12px {_ev_col}22'>"
                                                        "<div style='font-size:0.68em;color:#555;text-transform:uppercase;"
                                                        "letter-spacing:.06em;margin-bottom:5px'>📍 Strefa Decyzji</div>"
                                                        "<div style='display:flex;align-items:center;"
                                                        "justify-content:space-between;flex-wrap:wrap;gap:8px'>"
                                                        "<div>"
                                                        f"<span style='font-size:1.6em;font-weight:800;color:{_ev_col}'>"
                                                        f"{_kelly['stake_pln']:.0f} zł</span>"
                                                        "<span style='font-size:0.78em;color:#888;margin-left:6px'>"
                                                        "stawka Kelly 1/8</span>"
                                                        "</div>"
                                                        "<div style='text-align:right;font-size:0.82em'>"
                                                        f"<div style='color:#888'>{_kelly['f_frac']:.1%} bankrollu</div>"
                                                        f"<div style='color:{_ev_col};font-weight:600'>"
                                                        f"EV {_kelly['ev_per_unit']:+.3f}</div>"
                                                        "</div></div></div>",
                                                        unsafe_allow_html=True)
                                    elif _OA_OK and _oa_key and not _oa_cached:
                                        st.caption("📊 Brak kursów — kliknij 'Odśwież kursy' w sidebarze.")

                                    # Forma & Kontekst
                                    if pokaz_komentarz:
                                        _odds_buk_kom = _kurs_live_1x2 if _kurs_live_1x2 else None
                                        _fk_html = render_forma_kontekst(
                                            h, a, pred, forma_dict, srednie_df, srednie_lig,
                                            odds_buk=_odds_buk_kom)
                                        st.markdown(_fk_html, unsafe_allow_html=True)

                                    # Alternatywne rynki
                                    with st.expander("📊 Alternatywne rynki (p ≥ 55%)", expanded=False):
                                        alt = alternatywne_zdarzenia(lam_h, lam_a, lam_r, lam_k, rho, lam_sot=lam_sot)
                                        if alt:
                                            cat_colors = {"Gole":"#2196F3","BTTS":"#9C27B0","Rożne":"#FF9800","Kartki":"#F44336","1X2":"#4CAF50","SOT":"#00BCD4"}
                                            rows_alt = []
                                            for emoji, nazwa, p, fo, kat, linia_z in alt[:8]:
                                                kc = cat_colors.get(kat, "#888")
                                                bw = int(p * 100)
                                                fc = "#4CAF50" if fo <= 1.60 else ("#FF9800" if fo <= 2.00 else "#aaa")
                                                rows_alt.append(
                                                    f"<tr><td style='padding:4px 8px;font-size:0.88em'>{emoji} {nazwa}</td>"
                                                    f"<td style='padding:4px 8px;width:110px'>"
                                                    f"<div style='display:flex;align-items:center;gap:5px'>"
                                                    f"<div style='flex:1;background:var(--bg-card2);border-radius:3px;height:5px'>"
                                                    f"<div style='background:{kc};width:{bw}%;height:5px;border-radius:3px'></div></div>"
                                                    f"<span style='color:{kc};font-size:0.82em;min-width:30px'>{p:.0%}</span></div></td>"
                                                    f"<td style='padding:4px 8px;text-align:right;color:{fc};font-weight:bold;font-size:0.88em'>{fo:.2f}</td></tr>"
                                                )
                                            st.markdown(
                                                f"<table style='width:100%;border-collapse:collapse'>"
                                                f"<thead><tr style='color:#555;font-size:0.75em;text-transform:uppercase'>"
                                                f"<th style='padding:4px 8px;text-align:left'>Rynek</th>"
                                                f"<th style='padding:4px 8px;text-align:left'>P</th>"
                                                f"<th style='padding:4px 8px;text-align:right'>Fair</th></tr></thead>"
                                                f"<tbody>{''.join(rows_alt)}</tbody></table>"
                                                f"<p style='color:#444;font-size:0.72em;margin:4px 0 0'>⚠️ Rożne/kartki – Poisson bez korelacji. Orientacyjnie.</p>",
                                                unsafe_allow_html=True,
                                            )
                                        else:
                                            st.caption("Brak zdarzeń powyżej progu 55%.")

                                with _dt2:
                                    # Oczekiwane statystyki – kompaktowy pasek
                                    _sot_d = f"{lam_sot:.1f}" if (lam_sot and lam_sot > 0) else "–"
                                    st.markdown(
                                        f"<div style='display:flex;justify-content:space-around;"
                                        f"background:var(--bg-card2);border-radius:6px;padding:5px 4px;margin:4px 0'>"
                                        f"<div style='text-align:center'>"
                                        f"<div style='font-size:0.62em;color:#555'>⚽ Śr. gole</div>"
                                        f"<div style='font-size:0.90em;font-weight:bold;color:#aaa'>{lam_h+lam_a:.2f}</div>"
                                        f"<div style='font-size:0.58em;color:#444'>{h[:5]}:{lam_h:.1f} {a[:5]}:{lam_a:.1f}</div></div>"
                                        f"<div style='text-align:center'>"
                                        f"<div style='font-size:0.62em;color:#555'>🚩 Śr. rożne</div>"
                                        f"<div style='font-size:0.90em;font-weight:bold;color:#aaa'>{lam_r:.1f}</div>"
                                        f"<div style='font-size:0.58em;color:#444'>obie drużyny</div></div>"
                                        f"<div style='text-align:center'>"
                                        f"<div style='font-size:0.62em;color:#555'>🟨 Śr. kartki</div>"
                                        f"<div style='font-size:0.90em;font-weight:bold;color:#aaa'>{lam_k:.1f}</div>"
                                        f"<div style='font-size:0.58em;color:#444'>Y=1 R=2</div></div>"
                                        f"<div style='text-align:center'>"
                                        f"<div style='font-size:0.62em;color:#555'>🎯 Śr. SOT</div>"
                                        f"<div style='font-size:0.90em;font-weight:bold;color:#aaa'>{_sot_d}</div>"
                                        f"<div style='font-size:0.58em;color:#444'>celne strzały</div></div>"
                                        f"</div>",
                                        unsafe_allow_html=True,
                                    )

                                    if sedzia_ostr:
                                        st.caption(f"🟨 **Sędzia:** {sedzia} – {sedzia_ostr}")
                                    elif sedzia not in ("Nieznany", "", None):
                                        # Pokaż YC/foul strictness jeśli mamy dane
                                        _ref_map = _ctx.get("referee_stats_map", {})
                                        _ref_data = _ref_map.get(sedzia)
                                        if _ref_data:
                                            _ycf = _ref_data["yc_per_foul"]
                                            _ref_label = "🔴 Aptekarz" if _ycf >= 0.19 else ("🟡 Surowy" if _ycf >= 0.17 else "🟢 Wyrozumiały")
                                            st.caption(
                                                f"🟨 **Sędzia:** {sedzia} · "
                                                f"YC/foul: **{_ycf:.3f}** {_ref_label} · "
                                                f"avg {_ref_data['avg_yc']:.1f} 🟨/M ({_ref_data['n']}M)"
                                            )
                                        else:
                                            st.caption(f"🟨 **Sędzia:** {sedzia}")

                                    # Macierz wyników (toggle)
                                    if pokaz_macierz:
                                        try:
                                            _M_arr = pred.get("macierz")
                                            if _M_arr is not None:
                                                st.markdown(
                                                    render_macierz_html(_M_arr, h, a),
                                                    unsafe_allow_html=True)
                                        except Exception:
                                            pass

                                    # Ostatnie 5 meczów drużyn
                                    try:
                                        _hist_h2h = historical.copy()
                                        def _ostatnie_mecze(team, n=5):
                                            _hm = _hist_h2h[_hist_h2h["HomeTeam"] == team].copy()
                                            _hm["res"]    = _hm.apply(lambda r: "W" if r["FTHG"]>r["FTAG"] else ("L" if r["FTHG"]<r["FTAG"] else "D"), axis=1)
                                            _hm["wynik"]  = _hm.apply(lambda r: f"{int(r['FTHG'])}:{int(r['FTAG'])}", axis=1)
                                            _hm["rywal"]  = _hm["AwayTeam"]; _hm["strona"] = "H"
                                            _am = _hist_h2h[_hist_h2h["AwayTeam"] == team].copy()
                                            _am["res"]    = _am.apply(lambda r: "W" if r["FTAG"]>r["FTHG"] else ("L" if r["FTAG"]<r["FTHG"] else "D"), axis=1)
                                            _am["wynik"]  = _am.apply(lambda r: f"{int(r['FTHG'])}:{int(r['FTAG'])}", axis=1)
                                            _am["rywal"]  = _am["HomeTeam"]; _am["strona"] = "A"
                                            return pd.concat([
                                                _hm[["Date","res","wynik","rywal","strona"]],
                                                _am[["Date","res","wynik","rywal","strona"]]
                                            ]).sort_values("Date", ascending=False).head(n)
                                        _h_last = _ostatnie_mecze(h, 5)
                                        _a_last = _ostatnie_mecze(a, 5)

                                        def _team_hist_html(team_name, df_last, crest_url=""):
                                            _crest_tag = (
                                                f"<img src='{crest_url}' style='width:16px;height:16px;"
                                                f"object-fit:contain;vertical-align:middle;margin-right:4px' "
                                                f"onerror=\"this.style.display='none'\">"
                                            ) if crest_url else ""
                                            rows = ""
                                            for _, _r in df_last.iterrows():
                                                _rc  = {"W":"#4CAF50","D":"#FF9800","L":"#F44336"}.get(_r["res"],"#888")
                                                _dt  = _r["Date"].strftime("%d.%m") if pd.notna(_r["Date"]) else "?"
                                                _rywal_short = str(_r["rywal"])[:11]
                                                _side = _r["strona"]
                                                rows += (
                                                    f"<div style='display:flex;align-items:center;gap:5px;"
                                                    f"padding:4px 0;border-bottom:1px solid #111827'>"
                                                    f"<span style='color:#4b5563;font-size:0.70em;min-width:30px'>{_dt}</span>"
                                                    f"<span style='color:#374151;font-size:0.68em;min-width:14px;text-align:center'>{_side}</span>"
                                                    f"<span style='color:#9ca3af;font-size:0.75em;flex:1;overflow:hidden;"
                                                    f"text-overflow:ellipsis;white-space:nowrap'>{_rywal_short}</span>"
                                                    f"<span style='background:{_rc}22;color:{_rc};font-size:0.70em;"
                                                    f"font-weight:700;padding:1px 5px;border-radius:6px;min-width:16px;text-align:center'>{_r['res']}</span>"
                                                    f"<span style='color:#6b7280;font-size:0.70em;min-width:26px;text-align:right'>{_r['wynik']}</span>"
                                                    f"</div>"
                                                )
                                            return (
                                                f"<div style='flex:1;min-width:0'>"
                                                f"<div style='display:flex;align-items:center;gap:4px;"
                                                f"font-size:0.78em;color:#e5e7eb;font-weight:600;margin-bottom:5px;"
                                                f"padding-bottom:4px;border-bottom:1px solid #1f2937'>"
                                                f"{_crest_tag}{team_name}</div>"
                                                f"{rows}"
                                                f"</div>"
                                            )

                                        _h_crest_url = _sbn.get(h, {}).get("crest", "")
                                        _a_crest_url = _sbn.get(a, {}).get("crest", "")
                                        if not _h_last.empty or not _a_last.empty:
                                            st.markdown(
                                                f"<div style='display:flex;gap:10px;margin:4px 0 10px'>"
                                                + _team_hist_html(h, _h_last, _h_crest_url)
                                                + f"<div style='width:1px;background:#1f2937;flex-shrink:0'></div>"
                                                + _team_hist_html(a, _a_last, _a_crest_url)
                                                + f"</div>",
                                                unsafe_allow_html=True)
                                    except Exception:
                                        pass

                                with _dt3:
                                    # H2H – 5 ostatnich spotkań
                                    _h2h_api = []
                                    _h2h_source = "csv"
                                    _fd_key_h2h = st.session_state.get("fd_api_key", "")
                                    if _fd_key_h2h and _mecz_h_id and _mecz_a_id:
                                        try:
                                            _h2h_api = load_h2h_api(
                                                int(_mecz_h_id), int(_mecz_a_id),
                                                _fd_key_h2h, n=5)
                                            if _h2h_api:
                                                _h2h_source = "api"
                                        except Exception:
                                            pass

                                    _h2h_unified = []
                                    if _h2h_source == "api" and _h2h_api:
                                        for _am in _h2h_api:
                                            _h2h_unified.append({
                                                "date": _am["date"],
                                                "home": _am["home"], "away": _am["away"],
                                                "score_h": _am["score_h"], "score_a": _am["score_a"],
                                            })
                                    else:
                                        _h2h_all = load_h2h_csv(LIGI[wybrana_liga]["csv_code"], n_seasons=4)
                                        if _h2h_all.empty:
                                            _h2h_all = historical
                                        _h2h_csv = _h2h_all[
                                            (((_h2h_all["HomeTeam"]==h) & (_h2h_all["AwayTeam"]==a)) |
                                             ((_h2h_all["HomeTeam"]==a) & (_h2h_all["AwayTeam"]==h)))
                                        ].sort_values("Date", ascending=False).head(5)
                                        for _, _cm in _h2h_csv.iterrows():
                                            _h2h_unified.append({
                                                "date": _cm["Date"].strftime("%Y-%m-%d") if pd.notna(_cm["Date"]) else "?",
                                                "home": _cm["HomeTeam"], "away": _cm["AwayTeam"],
                                                "score_h": int(_cm["FTHG"]), "score_a": int(_cm["FTAG"]),
                                            })

                                    if _h2h_unified:
                                        _h_wins = sum(1 for m in _h2h_unified
                                                      if (m["home"]==h and m["score_h"]>m["score_a"]) or
                                                         (m["away"]==h and m["score_a"]>m["score_h"]))
                                        _draws  = sum(1 for m in _h2h_unified if m["score_h"]==m["score_a"])
                                        _a_wins = len(_h2h_unified) - _h_wins - _draws
                                        _src_badge = "🌐 API" if _h2h_source == "api" else "📂 CSV"
                                        st.markdown(
                                            f"<div style='font-size:0.74em;margin-bottom:8px;display:flex;gap:12px'>"
                                            f"<span style='color:#4CAF50'>● {h[:12]} {_h_wins}W</span>"
                                            f"<span style='color:#FF9800'>● {_draws}R</span>"
                                            f"<span style='color:#F44336'>● {_a_wins}W {a[:12]}</span>"
                                            f"<span style='color:#374151;margin-left:auto'>{_src_badge}</span>"
                                            f"</div>",
                                            unsafe_allow_html=True)
                                        _h2h_rows = []
                                        for _hm in _h2h_unified:
                                            _hg, _ag = _hm["score_h"], _hm["score_a"]
                                            _hw, _aw = _hm["home"], _hm["away"]
                                            _dt2h = _hm["date"][:7] if len(_hm["date"]) >= 7 else _hm["date"]
                                            _res_c = "#4CAF50" if _hg > _ag else ("#F44336" if _hg < _ag else "#FF9800")
                                            _bold_h = "font-weight:700;color:var(--text-primary)" if _hw == h else "color:#aaa"
                                            _bold_a = "font-weight:700;color:var(--text-primary)" if _aw == h else "color:#aaa"
                                            _h2h_rows.append(
                                                f"<tr style='border-bottom:1px solid #111827'>"
                                                f"<td style='padding:5px 6px;color:#4b5563;font-size:0.75em'>{_dt2h}</td>"
                                                f"<td style='padding:5px 6px;font-size:0.78em;{_bold_h};text-align:right'>{_hw[:14]}</td>"
                                                f"<td style='padding:5px 8px;text-align:center;color:{_res_c};"
                                                f"font-weight:700;font-size:0.88em;white-space:nowrap'>{_hg}:{_ag}</td>"
                                                f"<td style='padding:5px 6px;font-size:0.78em;{_bold_a}'>{_aw[:14]}</td>"
                                                f"</tr>"
                                            )
                                        st.markdown(
                                            f"<table style='width:100%;border-collapse:collapse'>"
                                            f"{''.join(_h2h_rows)}</table>",
                                            unsafe_allow_html=True)
                                    else:
                                        st.caption("Brak danych H2H dla tej pary drużyn.")


                                with _dt4:
                                    # ── Kontekst meczu ──────────────────────────────────────────────
                                    _pos_h = _ctx.get('h_pos', 0)
                                    _pos_a = _ctx.get('a_pos', 0)
                                    _pts_h = _ctx.get('h_pts', 0)
                                    _pts_a = _ctx.get('a_pts', 0)
                                    _gd_h  = _ctx.get('h_gd', 0)
                                    _gd_a  = _ctx.get('a_gd', 0)
                                    _mt    = _ctx.get('match_type', '–')
                                    _motyw = _ctx.get('motywacja', [])

                                    _mt_colors = {
                                            'Top vs Top':      ('#7c3aed', '#3b1f6b'),
                                            'Top vs Słaby':    ('#2563eb', '#1e3a5f'),
                                            'Słaby vs Top':    ('#dc2626', '#4c1d1d'),
                                            'Dół tabeli':      ('#d97706', '#4a3000'),
                                            'Środek tabeli':   ('#374151', '#1f2937'),
                                    }
                                    _mt_fg, _mt_bg = _mt_colors.get(_mt, ('#9ca3af', '#1f2937'))

                                    # Badge match type + motywacja
                                    _motyw_html = ' '.join(
                                            f"<div style='font-size:0.70em;color:#FF9800;margin-top:2px'>{m}</div>"
                                            for m in _motyw
                                    )
                                    st.markdown(
                                            f"<div style='background:{_mt_bg};border:1px solid {_mt_fg}44;border-radius:8px;"
                                            f"padding:8px 12px;margin-bottom:10px'>"
                                            f"<span style='color:{_mt_fg};font-size:0.80em;font-weight:700'>{_mt}</span>"
                                            + _motyw_html +
                                            "</div>",
                                            unsafe_allow_html=True)

                                    # Porównanie pozycji
                                    if _pos_h and _pos_a:
                                        _pd = abs(_pos_h - _pos_a)
                                        _pd_col = '#4CAF50' if _pd >= 10 else ('#FF9800' if _pd >= 5 else '#9ca3af')
                                        _h_pkt_str  = f'{_pts_h} pkt · GD {_gd_h:+d}'
                                        _a_pkt_str  = f'{_pts_a} pkt · GD {_gd_a:+d}'
                                        st.markdown(
                                            f"<div style='background:var(--bg-card2);border-radius:8px;padding:10px 14px;margin-bottom:8px'>"
                                            f"<div style='display:grid;grid-template-columns:1fr auto 1fr;gap:8px;align-items:center'>"
                                            f"<div style='text-align:center'>"
                                            f"<div style='font-size:2.2em;font-weight:900;color:var(--text-primary)'>{_pos_h}</div>"
                                            f"<div style='font-size:0.66em;color:#6b7280'>pozycja</div>"
                                            f"<div style='font-size:0.70em;color:#9ca3af;margin-top:2px'>{_h_pkt_str}</div>"
                                            f"</div>"
                                            f"<div style='text-align:center'>"
                                            f"<div style='font-size:0.64em;color:{_pd_col};font-weight:700'>Δ pozycji</div>"
                                            f"<div style='font-size:1.5em;font-weight:800;color:{_pd_col}'>{_pd}</div>"
                                            f"</div>"
                                            f"<div style='text-align:center'>"
                                            f"<div style='font-size:2.2em;font-weight:900;color:var(--text-primary)'>{_pos_a}</div>"
                                            f"<div style='font-size:0.66em;color:#6b7280'>pozycja</div>"
                                            f"<div style='font-size:0.70em;color:#9ca3af;margin-top:2px'>{_a_pkt_str}</div>"
                                            f"</div>"
                                            f"</div></div>",
                                            unsafe_allow_html=True)

                                    # ── Styl gry: kartki / faule / rożne / strzały ──────────────────
                                    _hs   = _ctx.get('h_stats', {})
                                    _as   = _ctx.get('a_stats', {})
                                    _h2hs = _ctx.get('h2h_stats', {})

                                    if _hs and _as:
                                        st.markdown(
                                            f"<div style='display:flex;justify-content:space-between;font-size:0.68em;"
                                            f"color:#4b5563;font-weight:600;margin-bottom:4px'>"
                                            f"<span>{h[:12]}</span><span>Styl gry (śr./mecz)</span><span>{a[:12]}</span></div>",
                                            unsafe_allow_html=True)

                                        def _srow(label, hv, av, fmt='{:.1f}', hib='neutral'):
                                            if hv is None or av is None: return ''
                                            hs_v = fmt.format(hv); as_v = fmt.format(av)
                                            if hib == 'bad':
                                                hc = '#F44336' if hv > av*1.15 else ('#4CAF50' if hv < av*0.85 else '#9ca3af')
                                                ac = '#F44336' if av > hv*1.15 else ('#4CAF50' if av < hv*0.85 else '#9ca3af')
                                            elif hib == 'good':
                                                hc = '#4CAF50' if hv > av*1.10 else ('#F44336' if hv < av*0.90 else '#9ca3af')
                                                ac = '#4CAF50' if av > hv*1.10 else ('#F44336' if av < hv*0.90 else '#9ca3af')
                                            else: hc = ac = '#9ca3af'
                                            tot = hv+av; bh = int(hv/tot*100) if tot>0 else 50; ba = 100-bh
                                            return (
                                                f"<div style='margin:5px 0'>"
                                                f"<div style='display:flex;justify-content:space-between;font-size:0.70em;margin-bottom:2px'>"
                                                f"<span style='color:{hc};font-weight:700'>{hs_v}</span>"
                                                f"<span style='color:#4b5563'>{label}</span>"
                                                f"<span style='color:{ac};font-weight:700'>{as_v}</span></div>"
                                                f"<div style='display:flex;height:4px;border-radius:2px;overflow:hidden'>"
                                                f"<div style='width:{bh}%;background:{hc}88'></div>"
                                                f"<div style='width:{ba}%;background:{ac}88'></div>"
                                                f"</div></div>"
                                            )

                                        rows = ''
                                        rows += _srow('🟨 Żółte kartki',   _hs.get('avg_yc'),    _as.get('avg_yc'),    hib='bad')
                                        rows += _srow('🟥 Czerwone kartki', _hs.get('avg_rc'),    _as.get('avg_rc'),    fmt='{:.2f}', hib='bad')
                                        rows += _srow('🦵 Faule',           _hs.get('avg_fo'),    _as.get('avg_fo'),    hib='bad')
                                        rows += _srow('🛡️ Faule wymuszone', _hs.get('avg_fo_suffered'), _as.get('avg_fo_suffered'), hib='neutral')
                                        rows += _srow('🚩 Rożne',           _hs.get('avg_co'),    _as.get('avg_co'),    hib='neutral')
                                        rows += _srow('🎯 Strzały',         _hs.get('avg_shots'), _as.get('avg_shots'), hib='good')
                                        rows += _srow('🎯 Celne strzały',   _hs.get('avg_sot'),   _as.get('avg_sot'),   hib='good')

                                        # ── Style Tags ──────────────────────────────────────────────
                                        _tag_colors = {
                                            "🟥 Hot Heads":        "#ef444420",
                                            "🎭 Foul Magnet":      "#f9731620",
                                            "📐 Corner Machine":   "#3b82f620",
                                            "🎯 Central Attackers":"#8b5cf620",
                                            "🚌 Bus Parker":       "#6b728020",
                                            "⚖️ Balanced":         "#22c55e20",
                                        }
                                        _tag_border = {
                                            "🟥 Hot Heads":        "#ef4444",
                                            "🎭 Foul Magnet":      "#f97316",
                                            "📐 Corner Machine":   "#3b82f6",
                                            "🎯 Central Attackers":"#8b5cf6",
                                            "🚌 Bus Parker":       "#6b7280",
                                            "⚖️ Balanced":         "#22c55e",
                                        }
                                        def _tag_html(tags):
                                            if not tags:
                                                return "<span style='color:#555;font-size:0.75em'>brak danych</span>"
                                            out = ""
                                            for t in tags:
                                                bg = _tag_colors.get(t, "#33333320")
                                                bd = _tag_border.get(t, "#555")
                                                out += (f"<span style='background:{bg};border:1px solid {bd}55;"
                                                        f"border-radius:6px;padding:2px 8px;margin:2px;font-size:0.75em;"
                                                        f"display:inline-block;color:{bd};font-weight:600'>{t}</span>")
                                            return out

                                        _h_tags = _hs.get('style_tags', [])
                                        _a_tags = _as.get('style_tags', [])
                                        _cps_h = _ctx.get('h_cps')
                                        _cps_a = _ctx.get('a_cps')

                                        st.markdown(
                                            "<div style='background:var(--bg-card2);border-radius:8px;padding:10px 12px'>" + rows +
                                            "<div style='margin-top:10px;padding-top:8px;border-top:1px solid var(--border)'>"
                                            f"<div style='font-size:0.70em;color:#6b7280;font-weight:600;margin-bottom:4px'>🏷️ Styl gry</div>"
                                            f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:6px'>"
                                            f"<div><div style='font-size:0.68em;color:#aaa;margin-bottom:3px'>{h}</div>{_tag_html(_h_tags)}"
                                            + (f"<div style='font-size:0.68em;color:#6b7280;margin-top:3px'>C/shot: {_cps_h:.1f}% <span style='color:#444'>(rożne/strzał)</span></div>" if _cps_h else "")
                                            + f"</div>"
                                            f"<div><div style='font-size:0.68em;color:#aaa;margin-bottom:3px'>{a}</div>{_tag_html(_a_tags)}"
                                            + (f"<div style='font-size:0.68em;color:#6b7280;margin-top:3px'>C/shot: {_cps_a:.1f}% <span style='color:#444'>(rożne/strzał)</span></div>" if _cps_a else "")
                                            + "</div></div></div></div>",
                                            unsafe_allow_html=True)

                                        # ── CLASH – Factor of Opposition ────────────────────────────
                                        _clash = _ctx.get('clash_total')
                                        _clash_pct = _ctx.get('clash_vs_avg')
                                        _clash_alert = _ctx.get('clash_alert', False)
                                        if _clash is not None:
                                            _clash_c = "#ef4444" if _clash_alert else ("#f97316" if _clash_pct and _clash_pct > 5 else "#6b7280")
                                            st.markdown(
                                                f"<div style='background:{_clash_c}12;border:1px solid {_clash_c}44;"
                                                f"border-radius:8px;padding:10px 14px;margin-top:8px'>"
                                                f"<div style='font-size:0.70em;color:#6b7280;font-weight:600;margin-bottom:4px'>"
                                                f"⚔️ Factor of Opposition – CLASH</div>"
                                                f"<div style='display:flex;align-items:center;gap:12px'>"
                                                f"<div style='font-size:1.4em;font-weight:800;color:{_clash_c}'>{_clash:.1f}</div>"
                                                f"<div>"
                                                f"<div style='font-size:0.75em;color:#aaa'>faule (popełn. + wymuszone) / mecz</div>"
                                                f"<div style='font-size:0.72em;color:{_clash_c};font-weight:600'>"
                                                + (f"🔥 {_clash_pct:+.1f}% vs avg ligowej — Wysoki potencjał kartek!" if _clash_alert
                                                   else f"{_clash_pct:+.1f}% vs avg ligowej")
                                                + "</div></div></div></div>",
                                                unsafe_allow_html=True)

                                    # ── Game State Sensitivity ─────────────────────────────────────
                                    _hgs = _ctx.get('h_game_state')
                                    _ags = _ctx.get('a_game_state')
                                    if _hgs and _ags:
                                        def _gs_row(state_label, hg, ag):
                                            hco = hg.get('co'); hyc = hg.get('yc'); hn = hg.get('n', 0)
                                            aco = ag.get('co'); ayc = ag.get('yc'); an = ag.get('n', 0)
                                            if hn < 2 and an < 2:
                                                return ""
                                            hco_s = f"{hco:.1f}🚩 {hyc:.1f}🟨" if hco and hyc else "–"
                                            aco_s = f"{aco:.1f}🚩 {ayc:.1f}🟨" if aco and ayc else "–"
                                            return (
                                                f"<div style='display:grid;grid-template-columns:1fr 80px 1fr;"
                                                f"gap:4px;align-items:center;padding:3px 0;border-bottom:1px solid #ffffff08'>"
                                                f"<div style='font-size:0.75em;color:#e2e8f0;text-align:right'>{hco_s} <span style='color:#555;font-size:0.8em'>({hn}M)</span></div>"
                                                f"<div style='font-size:0.68em;color:#6b7280;text-align:center;font-weight:600'>{state_label}</div>"
                                                f"<div style='font-size:0.75em;color:#e2e8f0'><span style='color:#555;font-size:0.8em'>({an}M)</span> {aco_s}</div>"
                                                f"</div>"
                                            )
                                        _gs_html = ""
                                        _gs_html += _gs_row("Prowadzi →", _hgs.get("Prowadzi",{}), _ags.get("Prowadzi",{}))
                                        _gs_html += _gs_row("← Remis →",  _hgs.get("Remis",{}),    _ags.get("Remis",{}))
                                        _gs_html += _gs_row("← Przegrywa", _hgs.get("Przegrywa",{}), _ags.get("Przegrywa",{}))
                                        if _gs_html:
                                            st.markdown(
                                                f"<div style='background:var(--bg-card2);border-radius:8px;"
                                                f"padding:10px 12px;margin-top:8px'>"
                                                f"<div style='font-size:0.70em;color:#6b7280;font-weight:600;margin-bottom:6px'>"
                                                f"📊 Wrażliwość na wynik do przerwy (🚩 rożne · 🟨 kartki)</div>"
                                                f"<div style='display:grid;grid-template-columns:1fr 80px 1fr;"
                                                f"gap:2px;margin-bottom:4px'>"
                                                f"<div style='font-size:0.65em;color:#555;text-align:right'>{h}</div>"
                                                f"<div></div>"
                                                f"<div style='font-size:0.65em;color:#555'>{a}</div></div>"
                                                + _gs_html + "</div>",
                                                unsafe_allow_html=True)

                                    # ── Predykcja Poisson: kartki / faule / rożne ───────────────────
                                    _lam_yc  = _ctx.get('lambda_yc_adj') or _ctx.get('lambda_yc')
                                    _lam_raw = _ctx.get('lambda_yc')
                                    _lam_fo  = _ctx.get('lambda_fo')
                                    _lam_co  = _ctx.get('lambda_co')
                                    _agg_clash     = _ctx.get('agg_clash')
                                    _agg_clash_pct = _ctx.get('agg_clash_vs_avg')
                                    _agg_alert     = _ctx.get('agg_clash_alert', False)
                                    _tempo         = _ctx.get('tempo_index')
                                    _tempo_pct     = _ctx.get('tempo_vs_avg')
                                    _h2hs_cards    = (_h2hs or {}).get('avg_cards')
                                    _h2hs_hot      = (_h2hs or {}).get('hot_rivalry', False)
                                    _h2hs_max      = (_h2hs or {}).get('max_cards')

                                    if _lam_yc is not None:
                                        from scipy.stats import poisson as _pois
                                        _adj  = (_lam_yc - _lam_raw) if _lam_raw else 0
                                        _albl = f'(korekta {_adj:+.2f})' if abs(_adj) > 0.05 else '(baza)'
                                        _p3  = 1 - _pois.cdf(2, _lam_yc)
                                        _p4  = 1 - _pois.cdf(3, _lam_yc)
                                        _p5  = 1 - _pois.cdf(4, _lam_yc)
                                        _pf25 = (1 - _pois.cdf(24, _lam_fo)) if _lam_fo else None
                                        _pc9  = (1 - _pois.cdf(8,  _lam_co)) if _lam_co else None

                                        def _pill(label, prob, thr=0.55):
                                            c = '#4CAF50' if prob>=thr else ('#FF9800' if prob>=0.40 else '#6b7280')
                                            return (
                                                f"<div style='display:inline-flex;flex-direction:column;align-items:center;"
                                                f"background:{c}18;border:1px solid {c}44;border-radius:8px;padding:6px 10px;margin:3px'>"
                                                f"<span style='font-size:1.05em;font-weight:800;color:{c}'>{prob:.0%}</span>"
                                                f"<span style='font-size:0.62em;color:#6b7280'>{label}</span>"
                                                f"</div>"
                                            )

                                        pills = _pill('≥3 kartki', _p3) + _pill('≥4 kartki', _p4) + _pill('≥5 kartek', _p5)
                                        if _pf25 is not None: pills += _pill('≥25 fauli', _pf25)
                                        if _pc9  is not None: pills += _pill('≥9 rożnych', _pc9)

                                        # ── Wskaźniki kontekstowe ────────────────────────────────────
                                        _ctx_signals = ""
                                        _exp_co_pt   = _ctx.get('exp_co_pt_total')
                                        _co_pt_diff  = _ctx.get('co_pt_vs_lambda')
                                        _vol_warn_yc = _ctx.get('vol_warning_yc', False)
                                        _vol_warn_co = _ctx.get('vol_warning_co', False)
                                        _vol_yc_m    = _ctx.get('vol_yc_matchup')
                                        _vol_co_m    = _ctx.get('vol_co_matchup')

                                        # Aggression Clash (CPF weighted, r=0.835)
                                        if _agg_clash is not None:
                                            _ac_c = "#ef4444" if _agg_alert else ("#f97316" if _agg_clash_pct and _agg_clash_pct > 8 else "#6b7280")
                                            _ac_lbl = f"🔥 +{_agg_clash_pct:.0f}% vs liga" if _agg_alert else (f"{_agg_clash_pct:+.0f}% vs liga" if _agg_clash_pct else "")
                                            _ctx_signals += (
                                                f"<div style='background:{_ac_c}12;border:1px solid {_ac_c}33;"
                                                f"border-radius:8px;padding:6px 10px;margin:3px;display:inline-flex;"
                                                f"flex-direction:column;align-items:center'>"
                                                f"<span style='font-size:0.65em;color:#6b7280;font-weight:600'>⚔️ Agresja CPF</span>"
                                                f"<span style='font-size:1.0em;font-weight:800;color:{_ac_c}'>{_agg_clash:.2f}</span>"
                                                f"<span style='font-size:0.62em;color:{_ac_c}'>{_ac_lbl}</span>"
                                                f"</div>"
                                            )

                                        # Pressure Tolerance → korekta rożnych
                                        if _exp_co_pt is not None:
                                            _pt_diff_s = ""
                                            _ptc = "#6b7280"
                                            if _co_pt_diff is not None:
                                                _ptc = "#3b82f6" if _co_pt_diff > 0.5 else ("#ef4444" if _co_pt_diff < -0.5 else "#6b7280")
                                                _pt_diff_s = f"{'↑' if _co_pt_diff>0 else '↓'}{abs(_co_pt_diff):.1f} vs avg"
                                            _ctx_signals += (
                                                f"<div style='background:{_ptc}12;border:1px solid {_ptc}33;"
                                                f"border-radius:8px;padding:6px 10px;margin:3px;display:inline-flex;"
                                                f"flex-direction:column;align-items:center'>"
                                                f"<span style='font-size:0.65em;color:#6b7280;font-weight:600'>🛡️ Rożne (PT)</span>"
                                                f"<span style='font-size:1.0em;font-weight:800;color:{_ptc}'>{_exp_co_pt:.1f} exp</span>"
                                                f"<span style='font-size:0.62em;color:{_ptc}'>{_pt_diff_s}</span>"
                                                f"</div>"
                                            )

                                        # Tempo Index (r=0.42 z rożnymi)
                                        if _tempo is not None:
                                            _tc = "#3b82f6" if (_tempo_pct and _tempo_pct > 10) else "#6b7280"
                                            if _tempo_pct is None or abs(_tempo_pct) < 2:
                                                _tl = "≈ avg liga"
                                            elif _tempo_pct > 0:
                                                _tl = f"+{_tempo_pct:.0f}% vs liga"
                                            else:
                                                _tl = f"{_tempo_pct:.0f}% vs liga"
                                            _ctx_signals += (
                                                f"<div style='background:{_tc}12;border:1px solid {_tc}33;"
                                                f"border-radius:8px;padding:6px 10px;margin:3px;display:inline-flex;"
                                                f"flex-direction:column;align-items:center'>"
                                                f"<span style='font-size:0.65em;color:#6b7280;font-weight:600'>🎬 Tempo</span>"
                                                f"<span style='font-size:1.0em;font-weight:800;color:{_tc}'>{_tempo:.0f}</span>"
                                                f"<span style='font-size:0.62em;color:{_tc}'>{_tl}</span>"
                                                f"</div>"
                                            )

                                        # H2H hot rivalry
                                        if _h2hs_cards is not None:
                                            _hrc = "#ef4444" if _h2hs_hot else "#6b7280"
                                            _hrl = "🔥 Gorące H2H" if _h2hs_hot else "H2H historia"
                                            _ctx_signals += (
                                                f"<div style='background:{_hrc}12;border:1px solid {_hrc}33;"
                                                f"border-radius:8px;padding:6px 10px;margin:3px;display:inline-flex;"
                                                f"flex-direction:column;align-items:center'>"
                                                f"<span style='font-size:0.65em;color:#6b7280;font-weight:600'>{_hrl}</span>"
                                                f"<span style='font-size:1.0em;font-weight:800;color:{_hrc}'>{_h2hs_cards:.1f} 🟨/M</span>"
                                                + (f"<span style='font-size:0.62em;color:{_hrc}'>max {_h2hs_max} kart</span>" if _h2hs_max else "")
                                                + f"</div>"
                                            )

                                        # Volatility warning (ostrożność Kelly)
                                        _vol_html = ""
                                        if _vol_warn_yc and _vol_yc_m:
                                            _vol_html += (
                                                f"<div style='background:#f9731618;border:1px solid #f9731644;"
                                                f"border-radius:6px;padding:4px 10px;margin:2px;font-size:0.72em;color:#f97316'>"
                                                f"⚠️ Wysoka zmienność 🟨 (CV={_vol_yc_m:.2f}) — obniż stawkę Kelly!</div>"
                                            )
                                        if _vol_warn_co and _vol_co_m:
                                            _vol_html += (
                                                f"<div style='background:#f9731618;border:1px solid #f9731644;"
                                                f"border-radius:6px;padding:4px 10px;margin:2px;font-size:0.72em;color:#f97316'>"
                                                f"⚠️ Wysoka zmienność 🚩 (CV={_vol_co_m:.2f}) — obniż stawkę Kelly!</div>"
                                            )

                                        st.markdown(
                                            f"<div style='background:var(--stat-bg);border-radius:8px;padding:10px 12px;margin-top:8px'>"
                                            f"<div style='font-size:0.72em;color:#6b7280;font-weight:600;margin-bottom:6px'>"
                                            f"⚡ Rynki alternatywne (Poisson) · λ kartki={_lam_yc:.1f} {_albl}</div>"
                                            f"<div style='display:flex;flex-wrap:wrap;gap:2px'>" + pills + "</div>"
                                            + (f"<div style='margin-top:8px;padding-top:6px;border-top:1px solid #ffffff08'>"
                                               f"<div style='font-size:0.65em;color:#6b7280;font-weight:600;margin-bottom:4px'>"
                                               f"📐 Wskaźniki kontekstowe matchup</div>"
                                               f"<div style='display:flex;flex-wrap:wrap;gap:2px'>{_ctx_signals}</div>"
                                               + (_vol_html or "")
                                               + "</div>"
                                               if _ctx_signals else "")
                                            + "</div>",
                                            unsafe_allow_html=True)

                                    # ── H2H alternatywne statystyki ─────────────────────────────────
                                    if _h2hs and _h2hs.get('n', 0) > 0:
                                        _n = _h2hs['n']
                                        _h2h_rows = ''
                                        for _k, _lbl in [('avg_yc','🟨 kartki'), ('avg_fo','🦵 faule'), ('avg_co','🚩 rożne')]:
                                            if _h2hs.get(_k) is not None:
                                                _h2h_rows += (
                                                    f"<div style='display:inline-flex;flex-direction:column;align-items:center;"
                                                    f"background:#1e293b;border-radius:8px;padding:5px 10px;margin:2px'>"
                                                    f"<span style='font-size:1.0em;font-weight:700;color:#e2e8f0'>{_h2hs[_k]:.1f}</span>"
                                                    f"<span style='font-size:0.62em;color:#6b7280'>{_lbl}/mecz</span></div>"
                                                )
                                        if _h2h_rows:
                                            st.markdown(
                                                f"<div style='margin-top:8px'>"
                                                f"<div style='font-size:0.70em;color:#6b7280;font-weight:600;margin-bottom:4px'>"
                                                f"⚔️ H2H alternatywne ({_n} spotkań)</div>"
                                                f"<div style='display:flex;flex-wrap:wrap;gap:2px'>" + _h2h_rows + "</div></div>",
                                                unsafe_allow_html=True)

                                    # ── Efekt wyniku HT na kartki w lidze ──────────────────────────
                                    if not historical.empty and 'HY' in historical.columns and 'HTHG' in historical.columns:
                                        try:
                                            _dfh = historical.copy()
                                            _dfh['HTD'] = _dfh['HTHG'] - _dfh['HTAG']
                                            _dfh['TYC'] = _dfh['HY']   + _dfh['AY']
                                            _ht_grp = _dfh.groupby('HTD')['TYC'].mean()
                                            _ht_html = ''
                                            for _d, _avg in sorted(_ht_grp.items()):
                                                if abs(_d) <= 2:
                                                    _lbl2 = {-2:'HT 0:2',-1:'HT 0:1',0:'HT 0:0',1:'HT 1:0',2:'HT 2:0'}.get(int(_d), str(_d))
                                                    _w2   = int(_avg / 6 * 100)
                                                    _c2   = '#4CAF50' if _avg >= 4.5 else ('#FF9800' if _avg >= 3.5 else '#6b7280')
                                                    _ht_html += (
                                                        f"<div style='display:flex;align-items:center;gap:8px;margin:3px 0'>"
                                                        f"<span style='font-size:0.68em;color:#6b7280;width:46px'>{_lbl2}</span>"
                                                        f"<div style='flex:1;background:#1f2937;border-radius:2px;height:5px'>"
                                                        f"<div style='width:{_w2}%;background:{_c2};height:5px;border-radius:2px'></div></div>"
                                                        f"<span style='font-size:0.68em;color:{_c2};font-weight:700;width:32px'>{_avg:.1f} YC</span>"
                                                        f"</div>"
                                                    )
                                            if _ht_html:
                                                st.markdown(
                                                    f"<div style='margin-top:8px'>"
                                                    f"<div style='font-size:0.70em;color:#6b7280;font-weight:600;margin-bottom:4px'>"
                                                    f"📊 Kartki vs wynik do przerwy (liga)</div>" + _ht_html + "</div>",
                                                    unsafe_allow_html=True)
                                        except Exception:
                                            pass

                                    # ── Narracja meczu ──────────────────────────────────────────
                                    _narrative = _ctx.get("narrative", [])
                                    if _narrative:
                                        _narr_html = "".join([
                                            f"<div style='display:flex;gap:8px;align-items:flex-start;"
                                            f"padding:5px 0;border-bottom:1px solid #ffffff06'>"
                                            f"<span style='color:#3b82f6;font-size:0.8em;margin-top:1px;flex-shrink:0'>▸</span>"
                                            f"<span style='font-size:0.78em;color:#cbd5e1;line-height:1.4'>{p}</span>"
                                            f"</div>"
                                            for p in _narrative
                                        ])
                                        st.markdown(
                                            f"<div style='background:linear-gradient(135deg,#0f172a,#1e293b);"
                                            f"border:1px solid #3b82f633;border-radius:10px;"
                                            f"padding:12px 14px;margin-top:10px'>"
                                            f"<div style='font-size:0.70em;color:#3b82f6;font-weight:700;"
                                            f"letter-spacing:0.05em;margin-bottom:8px'>"
                                            f"🔭 PROFIL MECZU — czego można się spodziewać</div>"
                                            + _narr_html +
                                            f"</div>",
                                            unsafe_allow_html=True
                                        )
                # Auto-zapis predykcji – uruchamia sie raz per liga+kolejka
                # Klucz session_state zapobiega powtornemu zapisowi przy kazdym rerunie
                _save_key = f"saved_{wybrana_liga}_{aktualna_kolejka}"
                if _save_key not in st.session_state:
                    n_saved = 0
                    for _, mecz_s in mecze.iterrows():
                        h_s = map_nazwa(mecz_s["home_team"])
                        a_s = map_nazwa(mecz_s["away_team"])
                        if h_s not in srednie_df.index or a_s not in srednie_df.index:
                            continue
                        lhs, las, lrs, lks, _sot_sv, _lsot_sv = oblicz_lambdy(h_s, a_s, srednie_df, srednie_lig, forma_dict,
                                                 csv_code=LIGI[wybrana_liga]["csv_code"])
                        pred_s = predykcja_meczu(lhs, las, rho=rho, csv_code=LIGI[wybrana_liga]["csv_code"], n_train=n_biezacy)
                        mecz_str_s = f"{h_s} – {a_s}"
                        zapisz_zdarzenia(wybrana_liga, int(aktualna_kolejka), mecz_str_s, h_s, a_s,
                                         "1X2", pred_s["typ"], 0.0, pred_s["p_typ"], pred_s["fo_typ"])
                        n_saved += 1
                        for emo, nazwa_z, pz, foz, katz, liniz in alternatywne_zdarzenia(lhs, las, lrs, lks, rho, lam_sot=_lsot_sv):
                            zapisz_zdarzenia(wybrana_liga, int(aktualna_kolejka), mecz_str_s, h_s, a_s,
                                             katz, nazwa_z, liniz, pz, foz)
                    st.session_state[_save_key] = n_saved
                    if n_saved > 0:
                        st.caption(f"💾 Auto-zapisano predykcje dla {n_saved} meczów kolejki #{aktualna_kolejka}")
                if st.button("🔄 Aktualizuj wyniki (po meczach)", key="update_wyniki_btn",
                             help="Zapisuje predykcje dla WSZYSTKICH przeszłych kolejek i aktualizuje wyniki."):
                    with st.spinner("Zapisuję predykcje dla wszystkich kolejek sezonu..."):
                        # KROK 1: Zapisz predykcje dla wszystkich przeszłych kolejek
                        # Iteruj przez WSZYSTKIE kolejki w terminarzu które już się odbyły
                        _dzisiaj = datetime.now().date()
                        _wszystkie_rundy = sorted(schedule["round"].unique())
                        _n_zapisanych = 0
                        for _runda in _wszystkie_rundy:
                            _mecze_rundy = schedule[schedule["round"] == _runda]
                            # Sprawdź czy ta kolejka już się odbyła (większość meczów w przeszłości)
                            _n_przeszle = sum(
                                1 for _, _mr in _mecze_rundy.iterrows()
                                if pd.Timestamp(_mr["date"]).date() < _dzisiaj
                            )
                            if _n_przeszle == 0:
                                continue  # przyszła kolejka – pomiń
                            for _, mecz_r in _mecze_rundy.iterrows():
                                _hr = map_nazwa(mecz_r["home_team"])
                                _ar = map_nazwa(mecz_r["away_team"])
                                if _hr not in srednie_df.index or _ar not in srednie_df.index:
                                    continue
                                # Sprawdź czy już zapisane
                                _con_chk = sqlite3.connect(DB_FILE)
                                _juz = _con_chk.execute(
                                    "SELECT COUNT(*) FROM zdarzenia WHERE liga=? AND kolejnosc=? AND home=? AND away=?",
                                    (wybrana_liga, int(_runda), _hr, _ar)).fetchone()[0]
                                _con_chk.close()
                                if _juz > 0:
                                    continue  # już zapisane – pomiń
                                try:
                                    _lhr, _lar, _lrr, _lkr, _, _lsotr = oblicz_lambdy(
                                        _hr, _ar, srednie_df, srednie_lig, forma_dict,
                              csv_code=LIGI[wybrana_liga]["csv_code"])
                                    _predr = predykcja_meczu(_lhr, _lar, rho=rho,
                                                              csv_code=LIGI[wybrana_liga]["csv_code"],
                                                              n_train=n_biezacy)
                                    _mecz_str_r = f"{_hr} – {_ar}"
                                    zapisz_zdarzenia(wybrana_liga, int(_runda), _mecz_str_r,
                                                     _hr, _ar, "1X2", _predr["typ"], 0.0,
                                                     _predr["p_typ"], _predr["fo_typ"])
                                    for _em, _nz, _pz, _foz, _kz, _lz in alternatywne_zdarzenia(
                                            _lhr, _lar, _lrr, _lkr, rho, lam_sot=_lsotr):
                                        zapisz_zdarzenia(wybrana_liga, int(_runda), _mecz_str_r,
                                                         _hr, _ar, _kz, _nz, _lz, _pz, _foz)
                                    _n_zapisanych += 1
                                except Exception:
                                    continue

                        # KROK 2: Zaktualizuj wyniki – najpierw z API (szybkie), potem CSV (fallback)
                        _fd_key_upd = _get_fd_api_key()
                        _fd_org_id_upd = LIGI[wybrana_liga]["fd_org_id"]
                        _api_updated = 0
                        if _fd_key_upd:
                            with st.spinner("🌐 Pobieranie wyników z football-data.org API..."):
                                _api_updated = aktualizuj_wyniki_z_api(_fd_org_id_upd, _fd_key_upd)

                        # Fallback: CSV dla meczów których API nie pokryło
                        _con_u = sqlite3.connect(DB_FILE)
                        _mecze_db = _con_u.execute(
                            "SELECT DISTINCT home, away FROM zdarzenia WHERE liga=? AND trafione IS NULL",
                            (wybrana_liga,)).fetchall()
                        _con_u.close()
                        n_updated = _api_updated
                        for _hdb, _adb in _mecze_db:
                            _con_b = sqlite3.connect(DB_FILE)
                            _before = _con_b.execute(
                                "SELECT COUNT(*) FROM zdarzenia WHERE home=? AND away=? AND trafione IS NOT NULL",
                                (_hdb, _adb)).fetchone()[0]
                            _con_b.close()
                            aktualizuj_wynik_zdarzenia(_hdb, _adb, historical)
                            _con_a = sqlite3.connect(DB_FILE)
                            _after = _con_a.execute(
                                "SELECT COUNT(*) FROM zdarzenia WHERE home=? AND away=? AND trafione IS NOT NULL",
                                (_hdb, _adb)).fetchone()[0]
                            _con_a.close()
                            if _after > _before:
                                n_updated += 1

                    # Podsumowanie
                    _msg_parts = []
                    if _n_zapisanych > 0:
                        _msg_parts.append(f"💾 Zapisano predykcje dla {_n_zapisanych} nowych meczów")
                    if n_updated > 0:
                        _msg_parts.append(f"🔄 Zaktualizowano wyniki: {n_updated} meczów")
                    if _msg_parts:
                        st.success(" · ".join(_msg_parts))
                    else:
                        st.info("✅ Wszystkie dane aktualne – brak nowych wyników do pobrania.")
                    st.rerun()
            else:
                st.info("Brak meczów w tej kolejce")
        else:
            st.warning("Brak danych")


        # ── Analizy bieżącej kolejki (Monte Carlo + Model vs Rynek) ──
        st.divider()
        with st.expander("🎲 Monte Carlo – symulacja wyników kolejki", expanded=False):
            st.caption("Symulacja 10 000 scenariuszy na podstawie prawdopodobieństw modelu.")
            # ── Monte Carlo ────────────────────────────────────────────────────────
            st.markdown("### 🎲 Symulacja Monte Carlo – wyniki kolejki")
            st.caption("Symulacja 10 000 scenariuszy bieżącej kolejki na podstawie prawdopodobieństw modelu.")

            if not schedule.empty and not srednie_df.empty:
                _mc_kolejka = get_current_round(schedule)
                _mc_mecze   = schedule[schedule["round"] == _mc_kolejka]

                # Zbierz predykcje dla wszystkich meczów kolejki
                _mc_preds = []
                for _, _mc_m in _mc_mecze.iterrows():
                    _mh = map_nazwa(_mc_m["home_team"])
                    _ma = map_nazwa(_mc_m["away_team"])
                    if _mh not in srednie_df.index or _ma not in srednie_df.index:
                        continue
                    try:
                        _mlh, _mla, _mlr, _mlk, _, _mls = oblicz_lambdy(
                            _mh, _ma, srednie_df, srednie_lig, forma_dict,
                                      csv_code=LIGI[wybrana_liga]["csv_code"])
                        _mp = predykcja_meczu(_mlh, _mla, rho=rho,
                                              csv_code=LIGI[wybrana_liga]["csv_code"],
                                              n_train=n_biezacy)
                        _mc_preds.append({
                            "mecz": f"{_mh}–{_ma}",
                            "typ": _mp["typ"],
                            "p_typ": _mp["p_typ"],
                            "fo_typ": _mp["fo_typ"],
                        })
                    except Exception:
                        continue

                _n_min_mc = 5
                if len(_mc_preds) >= _n_min_mc:
                    N_SIM = 10_000
                    rng   = np.random.default_rng(42)

                    # Dla każdego scenariusza: ile typów trafia? jaki łączny PnL (flat)?
                    _mc_hits_all = np.zeros(N_SIM)
                    _mc_pnl_all  = np.zeros(N_SIM)
                    for _mcp in _mc_preds:
                        _hits = rng.random(N_SIM) < _mcp["p_typ"]
                        _mc_hits_all += _hits.astype(float)
                        _mc_pnl_all  += np.where(_hits, _mcp["fo_typ"] - 1, -1)

                    _n_typow  = len(_mc_preds)
                    _p_plus   = float((_mc_pnl_all > 0).mean())
                    _p_breakeven = float((_mc_pnl_all >= 0).mean())
                    _exp_hits = float(_mc_hits_all.mean())
                    _exp_pnl  = float(_mc_pnl_all.mean())
                    _p10      = float(np.percentile(_mc_pnl_all, 10))
                    _p90      = float(np.percentile(_mc_pnl_all, 90))
                    _med_hits = float(np.median(_mc_hits_all))

                    # Disclaimer próbki
                    _n_hist = n_biezacy
                    _wiarygodne = _n_hist >= 50
                    if not _wiarygodne:
                        st.warning(
                            f"⚠️ Tylko {_n_hist} meczów w bazie – symulacja jest orientacyjna. "
                            f"Wiarygodność rośnie po ~50+ meczach historycznych.")

                    # KPI
                    _mc1, _mc2, _mc3, _mc4 = st.columns(4)
                    _plus_c = "normal" if _p_plus >= 0.55 else "inverse"
                    _mc1.metric("🎯 Szansa na plus",   f"{_p_plus:.0%}",
                                delta_color=_plus_c)
                    _mc2.metric("📊 Oczek. trafione",  f"{_exp_hits:.1f}/{_n_typow}")
                    _mc3.metric("💹 Oczek. PnL (flat)", f"{_exp_pnl:+.2f} j")
                    _mc4.metric("📉 Zakres P10–P90",
                                f"{_p10:+.1f} / {_p90:+.1f} j",
                                help="10% najgorszych / 10% najlepszych scenariuszy")

                    # Histogram PnL
                    _hist_bins = np.arange(
                        float(np.floor(_mc_pnl_all.min())),
                        float(np.ceil(_mc_pnl_all.max())) + 1, 1)
                    _hist_counts, _hist_edges = np.histogram(_mc_pnl_all, bins=_hist_bins)
                    _max_count = _hist_counts.max() if _hist_counts.max() > 0 else 1
                    _bar_h = 60
                    _bar_w_total = 360
                    _n_bars = len(_hist_counts)
                    _bw = max(2, _bar_w_total // max(_n_bars, 1))
                    _svg_w = _n_bars * (_bw + 1) + 60
                    _svg_h = _bar_h + 30
                    _bar_svgs = []
                    for _bi, (_bc, _be) in enumerate(zip(_hist_counts, _hist_edges)):
                        _bh = int(_bc / _max_count * _bar_h)
                        _bx = 40 + _bi * (_bw + 1)
                        _by = _bar_h - _bh + 5
                        _bc_col = "#4CAF50" if _be >= 0 else "#F44336"
                        _bar_svgs.append(
                            f"<rect x='{_bx}' y='{_by}' width='{_bw}' height='{_bh}' "
                            f"fill='{_bc_col}' fill-opacity='0.8' rx='1'/>"
                        )
                    # Zero line
                    _zero_x = 40 + int((_n_bars * abs(_hist_edges[0]) /
                                         (abs(_hist_edges[-1] - _hist_edges[0]) + 0.001)))
                    _zero_x = max(40, min(_zero_x, 40 + _n_bars * (_bw + 1)))
                    _bar_svgs.append(
                        f"<line x1='{_zero_x}' y1='5' x2='{_zero_x}' y2='{_bar_h + 5}' "
                        f"stroke='#fff' stroke-width='1.5' stroke-dasharray='3,2' opacity='0.4'/>"
                        f"<text x='{_zero_x}' y='{_bar_h + 20}' text-anchor='middle' "
                        f"font-size='9' fill='#555'>0</text>"
                    )
                    _bar_svgs.append(
                        f"<text x='38' y='{_bar_h + 20}' text-anchor='end' "
                        f"font-size='9' fill='#555'>{_hist_edges[0]:.0f}</text>"
                        f"<text x='{40 + _n_bars*(_bw+1)}' y='{_bar_h + 20}' "
                        f"text-anchor='start' font-size='9' fill='#555'>{_hist_edges[-1]:.0f}j</text>"
                    )
                    st.markdown(
                        f"<div style='margin:8px 0'>"
                        f"<div style='font-size:0.78em;color:#666;margin-bottom:4px'>"
                        f"Rozkład PnL (10 000 scenariuszy) · 🟢 plus · 🔴 minus</div>"
                        f"<svg width='{_svg_w}' height='{_svg_h}' "
                        f"style='background:var(--stat-bg);border-radius:6px;display:block'>"
                        f"{''.join(_bar_svgs)}</svg>"
                        f"<div style='font-size:0.74em;color:#555;margin-top:3px'>"
                        f"Szansa na breakeven lub lepiej: <b style='color:#4CAF50'>{_p_breakeven:.0%}</b> · "
                        f"N symulacji: 10 000 · N typów 1X2: {_n_typow}</div>"
                        f"</div>",
                        unsafe_allow_html=True)

                    # Per mecz: p_typ i szansa trafienia w MC
                    with st.expander("📋 Szczegóły per mecz", expanded=False):
                        for _mcp in sorted(_mc_preds, key=lambda x: -x["p_typ"]):
                            _p_str = f"{_mcp['p_typ']:.0%}"
                            _p_c = "#4CAF50" if _mcp["p_typ"] >= 0.65 else ("#FF9800" if _mcp["p_typ"] >= 0.58 else "#888")
                            st.markdown(
                                f"<div style='display:flex;justify-content:space-between;"
                                f"padding:4px 0;border-bottom:1px solid #1a1a2e;font-size:0.84em'>"
                                f"<span style='color:#ccc'>{_mcp['mecz']}</span>"
                                f"<span style='color:#888'>{_mcp['typ']}</span>"
                                f"<span style='color:{_p_c};font-weight:bold'>{_p_str}</span>"
                                f"<span style='color:#555'>fair {_mcp['fo_typ']:.2f}</span>"
                                f"</div>",
                                unsafe_allow_html=True)
                else:
                    st.info(f"Monte Carlo wymaga min. {_n_min_mc} meczów z danymi. "
                            f"Dostępne: {len(_mc_preds)}/{len(_mc_mecze)}.")
            else:
                st.warning("Brak danych harmonogramu.")



        if _OA_OK and _oa_key and _oa_cached:
            with st.expander("📊 Model vs Rynek – Sharpness per mecz", expanded=False):
                st.caption("Różnica 5–15% → potencjalna przewaga modelu nad rynkiem.")
                # ── Sharpness vs Rynek detail w tab5 ─────────────────────────────────
                if _OA_OK and _oa_key and _oa_cached and not schedule.empty:
                    _ms5 = model_sharpness_vs_rynek(
                        wybrana_liga, _oa_cached, _oa,
                        schedule, srednie_df, srednie_lig, forma_dict,
                        rho, n_biezacy, LIGI[wybrana_liga]["csv_code"])
                    if _ms5 and _ms5.get("details"):
                        st.divider()
                        st.markdown("### 🎯 Model vs Rynek – Sharpness per mecz")
                        st.caption("Szukasz meczu gdzie różnica 5-15% → tam jest potencjalna przewaga.")
                        _det = sorted(_ms5["details"], key=lambda x: x["diff"], reverse=True)
                        _sh_rows = []
                        for d in _det:
                            _dc = "#4CAF50" if 0.05 <= d["diff"] <= 0.15 else ("#888" if d["diff"] < 0.05 else "#F44336")
                            _di = "🟢" if 0.05 <= d["diff"] <= 0.15 else ("🔵" if d["diff"] < 0.05 else "🔴")
                            _bar = int(min(d["diff"] / 0.25, 1.0) * 100)
                            _sh_rows.append(
                                f"<tr style='border-bottom:1px solid #1a1a2e'>"
                                f"<td style='padding:6px 10px;color:#ccc;font-size:0.83em'>{d['mecz']}</td>"
                                f"<td style='padding:6px 8px;text-align:center;color:#888;font-size:0.82em'>{d['typ']}</td>"
                                f"<td style='padding:6px 8px;text-align:center;color:#2196F3;font-size:0.82em'>{d['p_model']:.0%}</td>"
                                f"<td style='padding:6px 8px;text-align:center;color:#888;font-size:0.82em'>{d['p_impl']:.0%}</td>"
                                f"<td style='padding:6px 8px'>"
                                f"<div style='display:flex;align-items:center;gap:5px'>"
                                f"<div style='background:{_dc};width:{_bar}px;height:6px;border-radius:2px;min-width:3px'></div>"
                                f"<span style='color:{_dc};font-weight:bold;font-size:0.82em'>{_di} {d['diff']:.0%}</span>"
                                f"</div></td>"
                                f"</tr>"
                            )
                        st.markdown(
                            f"<div style='border-radius:8px;border:1px solid var(--border);overflow:hidden'>"
                            f"<table style='width:100%;border-collapse:collapse'>"
                            f"<thead><tr style='background:#1e1e2e;color:#555;font-size:0.72em;text-transform:uppercase'>"
                            f"<th style='padding:6px 10px;text-align:left'>Mecz</th>"
                            f"<th style='padding:6px 8px;text-align:center'>Typ</th>"
                            f"<th style='padding:6px 8px;text-align:center'>P model</th>"
                            f"<th style='padding:6px 8px;text-align:center'>P rynek</th>"
                            f"<th style='padding:6px 8px;text-align:left'>Różnica</th>"
                            f"</tr></thead><tbody>{''.join(_sh_rows)}</tbody></table></div>",
                            unsafe_allow_html=True)
                        st.caption("🟢 Sweet spot (5-15%) · 🔵 Zbieżne (<5%) · 🔴 Market Noise (>15%)")

                # (wysyła automatycznie o DIGEST_HOUR, wszystkie ligi, raz dziennie)

    # =========================================================================
        st.divider()

    with tab2:
        st.subheader("📊 Ranking zdarzeń kolejki")
        st.caption("Centrum systemu – Value Bets, Safe Haven i Shot Kings dla najbliższej kolejki.")
        
        if not schedule.empty and not srednie_df.empty:
            aktualna_kolejka = get_current_round(schedule)
            mecze = schedule[schedule["round"] == aktualna_kolejka]
            status = get_round_status(schedule, aktualna_kolejka)
            
            st.info(f"📅 Kolejka #{aktualna_kolejka} ({status})")

            with st.spinner("Generowanie rankingu..."):
                # ... reszta kodu TAB 1 bez zmian ...
                wszystkie_zd = []
                shot_kings = []
                
                for _, mecz in mecze.iterrows():
                    h = map_nazwa(mecz["home_team"])
                    a = map_nazwa(mecz["away_team"])
                    if h not in srednie_df.index or a not in srednie_df.index:
                        continue
                    
                    lam_h, lam_a, lam_r, lam_k, sot_ok, lam_sot = oblicz_lambdy(h, a, srednie_df, srednie_lig, forma_dict,
                                                                    csv_code=LIGI[wybrana_liga]["csv_code"],
                                                                    historical=historical,
                                                                    mecz_data=mecz.get("date"))
                    pred = predykcja_meczu(lam_h, lam_a, rho=rho, csv_code=LIGI[wybrana_liga]["csv_code"], n_train=n_biezacy)
                    mecz_str = f"{h} – {a}"

                    def _ev(p_val, fo_val):
                        return round(p_val * fo_val - 1.0, 3)

                    # Kurs live 1X2 z The Odds API (jeśli dostępny)
                    _kurs_live_1x2 = None
                    if _OA_OK and _oa_key and _oa_cached:
                        _o_r = _oa.znajdz_kursy(h, a, _oa_cached)
                        if _o_r:
                            _kurs_live_1x2 = _o_r

                    # Line movement signal (PSH→PSCH z historical)
                    # Mocny steam na favoryta = rynek skrócił kurs o >0.25 → ostrzeżenie
                    _lm_signal = None
                    try:
                        _hist_lm = historical[historical["_sezon"] == "biezacy"] \
                            if "_sezon" in historical.columns else historical
                        _lm_row = _hist_lm[
                            (_hist_lm["HomeTeam"] == h) & (_hist_lm["AwayTeam"] == a)
                        ]
                        if not _lm_row.empty:
                            _lr = _lm_row.iloc[-1]
                            _psh = _lr.get("PSH"); _psch = _lr.get("PSCH")
                            if _psh and _psch and not (pd.isna(_psh) or pd.isna(_psch)):
                                _move = float(_psh) - float(_psch)  # + = kurs wzrósł (steam away), - = steam home
                                if _move < -0.25:
                                    _lm_signal = ("home", abs(_move))
                                elif _move > 0.25:
                                    _lm_signal = ("away", abs(_move))
                    except Exception:
                        pass

                    # Forma – najpierw z standings API, fallback z historical CSV
                    _h_stand = _sbn.get(h, {})
                    _a_stand = _sbn.get(a, {})
                    _h_form  = [str(x) for x in (_h_stand.get("form", []) or [])
                                if x and str(x) not in ("nan","None","")]
                    _a_form  = [str(x) for x in (_a_stand.get("form", []) or [])
                                if x and str(x) not in ("nan","None","")]
                    _h_pos   = int(_h_stand.get("position", 0) or 0)
                    _a_pos   = int(_a_stand.get("position", 0) or 0)
                    # Fallback: policz formę z historical CSV gdy API nie daje
                    if not _h_form or not _a_form:
                        try:
                            _hist_fc = historical[historical["_sezon"] == "biezacy"] \
                                if "_sezon" in historical.columns else historical
                            def _form_from_csv(team: str, n: int = 5) -> list:
                                _hm = _hist_fc[_hist_fc["HomeTeam"] == team].copy()
                                _hm["res"] = _hm.apply(lambda r: "W" if r["FTHG"]>r["FTAG"] else ("L" if r["FTHG"]<r["FTAG"] else "D"), axis=1)
                                _am = _hist_fc[_hist_fc["AwayTeam"] == team].copy()
                                _am["res"] = _am.apply(lambda r: "W" if r["FTAG"]>r["FTHG"] else ("L" if r["FTAG"]<r["FTHG"] else "D"), axis=1)
                                _all = pd.concat([
                                    _hm[["Date","res"]], _am[["Date","res"]]
                                ]).sort_values("Date").tail(n)
                                return list(_all["res"])
                            if not _h_form:
                                _h_form = _form_from_csv(h)
                            if not _a_form:
                                _a_form = _form_from_csv(a)
                        except Exception:
                            pass
                    _form_warn = _forma_warning(
                        _h_form, _a_form, _h_pos, _a_pos, pred["typ"])

                    # Typ główny – dopisz kurs live i kelly
                    # Śledź ekspozycję per mecz od 1X2
                    _br_mecz2 = st.session_state.get("bankroll", KELLY_BANKROLL_DEFAULT)
                    _exposed_mecz2 = 0.0
                    if pred["p_typ"] >= 0.58 and pred["fo_typ"] >= 1.30:
                        ev = _ev(pred["p_typ"], pred["fo_typ"])
                        _kbuk = None
                        if _kurs_live_1x2:
                            _kbuk, _ = _kurs_dc_live(pred["typ"],
                                _kurs_live_1x2["odds_h"], _kurs_live_1x2["odds_d"], _kurs_live_1x2["odds_a"])
                        _kel = kelly_stake(pred["p_typ"], _kbuk if _kbuk else pred["fo_typ"],
                                           bankroll=_br_mecz2, rynek="1X2",
                                           already_exposed=_exposed_mecz2)
                        if _kel["safe"]:
                            _exposed_mecz2 += _kel["stake_pln"]
                        wszystkie_zd.append({
                            "Mecz": mecz_str, "Rynek": "1X2", "Typ": pred["typ"],
                            "P": pred["p_typ"], "Fair": pred["fo_typ"], "KursBuk": _kbuk,
                            "EV": ev,
                            "Kelly_stake": _kel["stake_pln"] if _kel["safe"] else None,
                            "Kelly_capped": _kel.get("capped", False),
                            "Kelly_frac_used": _kel.get("fraction_used", 0.125),
                            "p_kelly_used": _kel.get("p_kelly"),
                            "Kelly_unverified": False, "Kategoria": "1X2",
                            "LineMovement": _lm_signal,
                            "FormWarning": _form_warn,
                            "HomeForm": _h_form, "AwayForm": _a_form,
                            "HomePos": _h_pos, "AwayPos": _a_pos,
                        })

                    # Alternatywne zdarzenia – z per-rynek frakcją Kelly i max exposure
                    alt = alternatywne_zdarzenia(lam_h, lam_a, lam_r, lam_k, rho, prog_min=0.55, lam_sot=lam_sot)
                    _br_alt = _br_mecz2
                    _exp_mecz = _exposed_mecz2  # kontynuuje z 1X2
                    for emoji, nazwa, p, fo, kat, linia in alt:
                        if fo >= 1.30:
                            ev = _ev(p, fo)
                            _kel_alt = kelly_stake(p, fo, bankroll=_br_alt,
                                                   rynek=kat, already_exposed=_exp_mecz)
                            if _kel_alt["safe"]:
                                _exp_mecz += _kel_alt["stake_pln"]
                            wszystkie_zd.append({
                                "Mecz": mecz_str, "Rynek": kat, "Typ": nazwa,
                                "P": p, "Fair": fo, "KursBuk": None, "EV": ev,
                                "Kelly_stake": _kel_alt["stake_pln"] if _kel_alt["safe"] else None,
                                "Kelly_capped": _kel_alt.get("capped", False),
                                "Kelly_frac_used": _kel_alt.get("fraction_used", 0.050),
                                "p_kelly_used": _kel_alt.get("p_kelly"),
                                "Kelly_unverified": True, "Kategoria": kat
                            })
                    
                    # Shot Kings – tylko jeśli fair odds ≥ 1.30
                    if lam_sot and lam_sot > 0:
                        p_over_55 = 1 - poisson.cdf(5, lam_sot) if lam_sot else 0
                        fo_over_55 = fair_odds(p_over_55) if p_over_55 > 0 else 0
                        if fo_over_55 >= 1.30:
                            shot_kings.append({
                                "Mecz": mecz_str,
                                "Oczekiwane SOT": round(lam_sot, 1),
                                "Over 5.5 SOT": p_over_55,
                                "Fair": fo_over_55
                            })

            if wszystkie_zd:
                df_rank = pd.DataFrame(wszystkie_zd)
                # Zapisz Kelly bets do session_state – paper trading to stąd czyta
                _kelly_bets_ss = [
                    z for z in wszystkie_zd
                    if z.get("Kelly_stake") and z.get("Kelly_stake", 0) > 0
                ]
                st.session_state["kelly_bets_kolejka"] = _kelly_bets_ss
                st.session_state["kelly_bets_kolejka_nr"] = int(aktualna_kolejka)

                # ── Przełącznik widoku: Główne / Alternatywne / Wszystkie ──────
                _t2c1, _t2c2 = st.columns([2, 3])
                with _t2c1:
                    _widok = st.radio(
                        "Widok rynków",
                        ["🎯 Główne (1X2)", "⚡ Alternatywne", "📋 Wszystkie"],
                        horizontal=True, key="tab2_widok",
                        help="Główne = tylko 1X2 | Alternatywne = gole, kartki, rożne, SOT"
                    )
                with _t2c2:
                    _bankroll_t2 = st.session_state.get("bankroll", 1000.0)
                    st.caption(f"💼 Bankroll: **{_bankroll_t2:.0f} zł** · Kelly 1/4 · ✦ = fair odds (brak live)")

                if _widok == "🎯 Główne (1X2)":
                    df_rank = df_rank[df_rank["Kategoria"] == "1X2"]
                elif _widok == "⚡ Alternatywne":
                    df_rank = df_rank[df_rank["Kategoria"] != "1X2"]
                # else: Wszystkie - df_rank bez zmian

                # ── TOP PICKS – esencja dla "30-sekundowego użytkownika" ─────
                _top_src = df_rank[
                    (df_rank["EV"] >= 0.04) &
                    (df_rank["P"] >= PROG_PEWNY) &
                    (df_rank["Kelly_stake"].notna()) &
                    (df_rank["Kelly_stake"] > 0)
                ].sort_values("EV", ascending=False).head(3)

                if not _top_src.empty:
                    _tp_html = ""
                    for _, _tr in _top_src.iterrows():
                        _tp_ev   = _tr["EV"]
                        _tp_p    = _tr["P"]
                        _tp_fair = _tr["KursBuk"] or _tr["Fair"]
                        _tp_kel  = _tr["Kelly_stake"]
                        _tp_typ  = _tr["Typ"]
                        _tp_mecz = _tr["Mecz"]
                        _tp_kat  = _tr["Kategoria"]
                        _tp_ev_c = "#4CAF50" if _tp_ev >= 0.08 else "#FF9800"
                        _tp_unv  = _tr.get("Kelly_unverified", False)
                        _tp_kurs_lbl = f"{_tp_fair:.2f}" if _tp_fair else "–"
                        _tp_html += (
                            f"<div style='background:linear-gradient(135deg,#052010,#041a0c);"
                            f"border:1px solid {_tp_ev_c}44;border-left:3px solid {_tp_ev_c};"
                            f"border-radius:8px;padding:8px 12px;margin-bottom:6px;"
                            f"display:flex;justify-content:space-between;align-items:center;gap:8px'>"
                            f"<div style='flex:1;min-width:0'>"
                            f"<div style='font-size:0.72em;color:#6b7280;margin-bottom:1px'>"
                            f"{_tp_mecz} · <span style='color:#aaa'>{_tp_kat}</span></div>"
                            f"<div style='font-size:0.9em;font-weight:700;color:#e2e8f0'>{_tp_typ}</div>"
                            f"</div>"
                            f"<div style='text-align:right;flex-shrink:0'>"
                            f"<div style='font-size:1.0em;font-weight:800;color:{_tp_ev_c}'>EV {_tp_ev:+.1%}</div>"
                            f"<div style='font-size:0.72em;color:#6b7280'>"
                            f"p={_tp_p:.0%} · {'✦' if _tp_unv else ''}{_tp_kurs_lbl} · "
                            f"<span style='color:#4CAF50'>{_tp_kel:.0f} zł</span></div>"
                            f"</div></div>"
                        )
                    st.markdown(
                        f"<div style='background:#021a08;border:1px solid #4CAF5033;"
                        f"border-radius:10px;padding:10px 12px;margin-bottom:14px'>"
                        f"<div style='font-size:0.68em;color:#4CAF50;font-weight:700;"
                        f"letter-spacing:.08em;margin-bottom:8px'>🏆 TOP PICKS KOLEJKI</div>"
                        + _tp_html +
                        f"</div>",
                        unsafe_allow_html=True
                    )

                st.markdown("### 🔥 Value Bets kolejki")
                st.caption(f"Wszystkie zdarzenia z EV > 0 · posortowane wg EV · limit {MAX_EXPOSURE_PCT:.0%}/mecz")

                # ── Filtry ──────────────────────────────────────────────
                _fc1, _fc2, _fc3 = st.columns([2, 2, 2])
                with _fc1:
                    _ev_min_filtr = st.slider(
                        "Min EV", 0.0, 0.20, 0.0, 0.01,
                        format="+%.0f%%",
                        key="_vb_ev_min",
                        help="Pokaż tylko zdarzenia z EV ≥ progu")
                with _fc2:
                    _kurs_min_filtr = st.slider(
                        "Min kurs", 1.0, 3.0, 1.0, 0.05,
                        format="%.2f",
                        key="_vb_kurs_min",
                        help="Pokaż tylko zdarzenia z kursem ≥ wartości")
                with _fc3:
                    _kat_filtr = st.selectbox(
                        "Kategoria",
                        ["Wszystkie", "1X2", "Gole", "BTTS", "SOT"],
                        key="_vb_kat",
                        help="Filtruj po typie rynku")

                value_bets = df_rank[df_rank["EV"] > 0].sort_values("EV", ascending=False)

                # Zastosuj filtry
                if _ev_min_filtr > 0:
                    value_bets = value_bets[value_bets["EV"] >= _ev_min_filtr]
                if _kurs_min_filtr > 1.0:
                    _kurs_col = value_bets.get("KursBuk") if "KursBuk" in value_bets.columns else None
                    if _kurs_col is not None:
                        value_bets = value_bets[
                            value_bets["KursBuk"].apply(
                                lambda k: float(k) >= _kurs_min_filtr if k and str(k) not in ("nan","None","") else False
                            ) |
                            value_bets["Fair"].apply(lambda f: float(f) >= _kurs_min_filtr if f else False)
                        ]
                if _kat_filtr != "Wszystkie":
                    value_bets = value_bets[value_bets["Kategoria"] == _kat_filtr]
                if not value_bets.empty:
                    for _, row in value_bets.iterrows():
                        _ks  = row.get("Kelly_stake")
                        _kb  = row.get("KursBuk")
                        _cap = row.get("Kelly_capped", False)
                        _unv = row.get("Kelly_unverified", False)
                        _frac = row.get("Kelly_frac_used", 0.125)
                        _ec  = "#4CAF50" if row["EV"] > 0.05 else "#FF9800"
                        _kd  = f"{_kb:.2f}" if _kb else f"{row['Fair']:.2f}✦"
                        try:
                            _ks = float(_ks) if _ks is not None else None
                            if _ks is not None and (_ks != _ks or _ks <= 0): _ks = None
                        except (TypeError, ValueError):
                            _ks = None
                        if _ks:
                            _ki  = "🔒" if _cap else ("⚠️" if _unv else "🏦")
                            _kc  = "#FF9800" if (_cap or _unv) else "#4CAF50"
                            _kelly_html = (
                                f"<br><span style='color:{_kc};font-size:0.84em'>"
                                f"{_ki} {_ks:.0f} zł · f={_frac:.3f}</span>")
                        else:
                            _kelly_html = ""

                        # ── Marża bukmachera ────────────────────────────────
                        _margin_html = ""
                        try:
                            _fair_p = float(row["P"])
                            _fo_v   = float(row["Fair"]) if row["Fair"] > 1 else 0
                            _buk_v  = float(_kb) if _kb and _kb > 1 else _fo_v
                            if _buk_v > 1 and _fair_p > 0:
                                # Marża = implied_buk / implied_fair - 1
                                _impl_buk  = 1 / _buk_v
                                _impl_fair = _fair_p
                                _margin_pct = (_impl_fair - _impl_buk) / _impl_fair
                                _mc = "#4CAF50" if _margin_pct > 0 else "#F44336"
                                _margin_lbl = (
                                    f"Edge: <b style='color:{_mc}'>{_margin_pct:+.1%}</b>"
                                    f" &nbsp;·&nbsp; "
                                    f"Marża buk: <b style='color:#FF9800'>"
                                    f"{max(0,(1/_buk_v)/(1/_fo_v)-1) if _fo_v > 1 else 0:.1%}</b>"
                                    if _fo_v > 1 else
                                    f"Edge: <b style='color:{_mc}'>{_margin_pct:+.1%}</b>")
                                _margin_html = (
                                    f"<div style='font-size:0.75em;color:#666;margin-top:3px'>"
                                    f"p={_fair_p:.1%} · Fair={_fo_v:.2f} · {_margin_lbl}"
                                    f"</div>")
                        except Exception:
                            pass

                        _lm = row.get("LineMovement")
                        _lm_html = ""
                        if isinstance(_lm, tuple) and len(_lm) == 2:
                            _lm_dir, _lm_size = _lm
                            _lm_team = row["Mecz"].split(" – ")[0] if _lm_dir == "home" else row["Mecz"].split(" – ")[-1]
                            _lm_html = (
                                f"<div style='font-size:0.74em;color:#FF9800;margin-top:3px'>"
                                f"⚡ Steam: rynek shortuje <b>{_lm_team}</b> "
                                f"({_lm_size:.2f} pkt) · sprawdź skład/kontuzje"
                                f"</div>")
                        # Forma z standings
                        _fw = row.get("FormWarning")
                        _hf = row.get("HomeForm", [])
                        _af = row.get("AwayForm", [])
                        _hp = row.get("HomePos", 0)
                        _ap = row.get("AwayPos", 0)
                        _hf_str = "".join(_hf) if isinstance(_hf, list) else (str(_hf) if _hf and str(_hf) != "nan" else "")
                        _af_str = "".join(_af) if isinstance(_af, list) else (str(_af) if _af and str(_af) != "nan" else "")
                        _hp = int(_hp) if _hp and str(_hp) != "nan" else 0
                        _ap = int(_ap) if _ap and str(_ap) != "nan" else 0
                        _form_html = ""
                        if _hf_str or _af_str:
                            _teams = row["Mecz"].split(" – ")
                            _hn = _teams[0] if len(_teams) > 0 else ""
                            _an = _teams[1] if len(_teams) > 1 else ""
                            _pos_h = f"<span style='color:#888;font-size:0.72em'>#{_hp}</span> " if _hp else ""
                            _pos_a = f"<span style='color:#888;font-size:0.72em'>#{_ap}</span> " if _ap else ""
                            _form_html = (
                                f"<div style='display:flex;gap:12px;margin-top:4px;flex-wrap:wrap'>"
                                f"<span style='font-size:0.74em;color:#aaa'>"
                                f"{_pos_h}<b style='color:#ccc'>{_hn}</b>: {_forma_html(_hf_str)}</span>"
                                f"<span style='font-size:0.74em;color:#aaa'>"
                                f"{_pos_a}<b style='color:#ccc'>{_an}</b>: {_forma_html(_af_str)}</span>"
                                f"</div>")
                        _fw_html = ""
                        if _fw and str(_fw) != "nan":
                            _fw_html = (
                                f"<div style='font-size:0.74em;color:#FF5722;margin-top:3px'>"
                                f"⚠️ Forma vs model: {_fw}</div>")

                        # ── High EV + High Chaos warning ────────────────────
                        _chaos_warn_html = ""
                        _vb_ev_val = float(row["EV"]) if row["EV"] else 0
                        _vb_p_val  = float(row["P"])  if row["P"]  else 0
                        # Chaos przybliżamy z p_typ — niskie p = chaos
                        # entropy wysoka gdy p_home ≈ p_draw ≈ p_away
                        # Proxy: jeśli p < 0.62 i EV > 0.08 = podejrzanie dobry deal
                        _vb_is_high_ev    = _vb_ev_val >= 0.08
                        _vb_is_uncertain  = _vb_p_val < 0.62
                        if _vb_is_high_ev and _vb_is_uncertain:
                            _chaos_warn_html = (
                                f"<div style='font-size:0.74em;color:#f97316;margin-top:3px;"
                                f"background:#f9731610;border-radius:4px;padding:2px 6px'>"
                                f"⚠️ Wysokie EV, ale model niepewny (p={_vb_p_val:.0%}) — zweryfikuj skład</div>"
                            )
                        _typ_badge_color = {"1":"#1b5e20","X":"#4a3000","2":"#0d2a4a"}.get(row["Typ"],"#1a1a2e")
                        _typ_text_color  = {"1":"#4CAF50","X":"#FFC107","2":"#42a5f5"}.get(row["Typ"],"#aaa")
                        st.markdown(
                            f"<div style='background:var(--bg-card);border-left:3px solid {_ec};"
                            f"border-radius:8px;padding:10px 14px;margin:5px 0'>"
                            # Wiersz 1: mecz + badge typu
                            f"<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:5px'>"
                            f"<b style='color:var(--text-primary);font-size:0.90em'>{row['Mecz']}</b>"
                            f"<span style='background:{_typ_badge_color};color:{_typ_text_color};"
                            f"font-size:0.76em;font-weight:700;padding:2px 9px;border-radius:12px;"
                            f"margin-left:8px;flex-shrink:0'>{row['Typ']} · {row['Rynek']}</span>"
                            f"</div>"
                            # Wiersz 2: p · kurs · EV · Kelly
                            f"<div style='display:flex;align-items:center;justify-content:space-between'>"
                            f"<span style='color:#888;font-size:0.80em'>"
                            f"<span style='color:#aaa'>p</span> {row['P']:.0%}"
                            f" &nbsp;·&nbsp; <span style='color:#aaa'>kurs</span> <b style='color:var(--text-primary)'>{_kd}</b>"
                            f" &nbsp;·&nbsp; <b style='color:{_ec}'>EV {row['EV']:+.3f}</b>"
                            f"</span>"
                            f"{_kelly_html}"
                            f"</div>"
                            # Wiersz 3: marża bukmachera
                            f"{_margin_html}"
                            f"{_lm_html}"
                            f"{_fw_html}"
                            f"{_chaos_warn_html}"
                            f"{_form_html}"
                            f"</div>",
                            unsafe_allow_html=True)
                else:
                    st.info("Brak value bets w tej kolejce")
    with tab3:
        st.markdown("<div class='section-header'>📈 Skuteczność modelu · sezon 2025/26</div>",
                    unsafe_allow_html=True)

        # ── 4 duże KPI widoczne od razu ─────────────────────────────────────
        _mg_top = metryki_globalne(wybrana_liga)
        init_db()
        _con_kpi = sqlite3.connect(DB_FILE)
        _kpi_row = _con_kpi.execute(
            "SELECT COUNT(*), "
            "SUM(CASE WHEN trafione=1 THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN trafione IS NOT NULL THEN 1 ELSE 0 END) "
            "FROM zdarzenia WHERE liga=? AND sezon=? AND rynek='1X2'",
            (wybrana_liga, BIEZACY_SEZON)).fetchone()
        _kpi_ev_row = _con_kpi.execute(
            "SELECT "
            "SUM(CASE WHEN trafione=1 THEN (fair_odds-1) ELSE -1.0 END), "
            "SUM(CASE WHEN trafione IS NOT NULL THEN 1 ELSE 0 END) "
            "FROM zdarzenia WHERE liga=? AND sezon=? AND trafione IS NOT NULL",
            (wybrana_liga, BIEZACY_SEZON)).fetchone()
        _con_kpi.close()

        _kpi_n    = int(_kpi_row[0]) if _kpi_row and _kpi_row[0] else 0
        _kpi_traf = int(_kpi_row[1]) if _kpi_row and _kpi_row[1] else 0
        _kpi_rozl = int(_kpi_row[2]) if _kpi_row and _kpi_row[2] else 0
        _kpi_hit  = _kpi_traf / _kpi_rozl if _kpi_rozl else 0
        _kpi_roi  = float(_kpi_ev_row[0]) / float(_kpi_ev_row[1]) if _kpi_ev_row and _kpi_ev_row[1] else None
        _kpi_brier = _mg_top["brier"] if _mg_top else None
        _kpi_bss   = _mg_top["bss"]   if _mg_top else None

        # Kolorowane karty KPI zamiast szarych st.metric
        def _kpi_card(label, value, sub, color, icon):
            return (
                f"<div style='background:linear-gradient(135deg,{color}18,{color}08);"
                f"border:1px solid {color}44;border-radius:10px;padding:14px 16px;"
                f"text-align:center;min-height:90px'>"
                f"<div style='font-size:1.6em'>{icon}</div>"
                f"<div style='font-size:1.55em;font-weight:800;color:{color};margin:2px 0'>{value}</div>"
                f"<div style='font-size:0.72em;color:#9ca3af;font-weight:600'>{label}</div>"
                f"<div style='font-size:0.65em;color:#6b7280;margin-top:2px'>{sub}</div>"
                f"</div>"
            )

        _c1, _c2, _c3, _c4 = st.columns(4)
        _hr_col  = "#4CAF50" if _kpi_hit >= 0.62 else ("#FF9800" if _kpi_hit >= 0.55 else "#F44336")
        _roi_col = "#4CAF50" if (_kpi_roi is not None and _kpi_roi > 0) else "#F44336"
        _br_col  = "#4CAF50" if (_kpi_brier is not None and _kpi_brier < 0.22) else "#FF9800"
        _n_col   = "#60a5fa"

        _hr_sub   = f"z {_kpi_rozl} rozliczonych" if _kpi_rozl else "brak danych"
        _roi_sub  = f"na zakładach EV>0" if _kpi_roi is not None else "brak danych"
        _br_sub   = f"BSS {_kpi_bss:+.3f}" if _kpi_bss is not None else "brak danych"
        _n_sub    = f"{_kpi_traf} trafione"

        _c1.markdown(_kpi_card("Hit Rate 1X2",
            f"{_kpi_hit:.1%}" if _kpi_rozl else "–", _hr_sub, _hr_col, "🎯"), unsafe_allow_html=True)
        _c2.markdown(_kpi_card("ROI (fair odds)",
            f"{_kpi_roi:+.1%}" if _kpi_roi is not None else "–", _roi_sub, _roi_col, "💰"), unsafe_allow_html=True)
        _c3.markdown(_kpi_card("Brier Score ↓",
            f"{_kpi_brier:.4f}" if _kpi_brier else "–", _br_sub, _br_col, "📐"), unsafe_allow_html=True)
        _c4.markdown(_kpi_card("Typów łącznie",
            str(_kpi_n), _n_sub, _n_col, "🏆"), unsafe_allow_html=True)

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # Legenda inline (bez expander)
        with st.expander("ℹ️ Co oznaczają te metryki?", expanded=False):
            _leg1, _leg2, _leg3, _leg4 = st.columns(4)
            _leg1.markdown("**🎯 Hit Rate**  \n% trafnych predykcji (1/X/2).  \nDobry: **≥62%**. Losowy ~45–50%.")
            _leg2.markdown("**💰 ROI (fair)**  \nZysk/strata na zł postawionym na fair kursie.  \nDobry: **>0%**.")
            _leg3.markdown("**📐 Brier Score ↓**  \nDokładność p-ów. Niższy = lepszy.  \nLosowy ~0.25. BSS>0 = bije losowość.")
            _leg4.markdown("**🎯 ECE ↓**  \nKalibracja: czy p=60% trafia 60% czasu.  \n<0.05 = dobrze skalibrowany.")
            st.markdown(
                "**💰 Symulacja Kelly** pokazuje jak wyglądałby bankroll gdybyś od początku sezonu "
                "stawiał tylko na mecze gdzie EV≥5%. Frakcja 1/8 Kelly = bardzo zachowawcze zarządzanie kapitałem.")

        st.divider()

        # Pokaz dane per kolejka – historia nie znika, grupuje sie automatycznie
        init_db()
        _con_t4 = sqlite3.connect(DB_FILE)
        _hist_kolejki = pd.read_sql_query(
            """SELECT kolejnosc, COUNT(*) as n_typow,
               SUM(CASE WHEN trafione=1 THEN 1 ELSE 0 END) as n_traf,
               AVG(CASE WHEN trafione IS NOT NULL THEN CAST(trafione AS FLOAT) END) as hit_rate,
               MIN(data) as data_kolejki
               FROM zdarzenia WHERE liga=? AND rynek='1X2'
               GROUP BY kolejnosc ORDER BY kolejnosc""",
            _con_t4, params=(wybrana_liga,)
        )
        _con_t4.close()

        if not _hist_kolejki.empty:
            _hist_kolejki = _hist_kolejki[_hist_kolejki["n_typow"] > 0]
            n_z_wynikiem  = _hist_kolejki[_hist_kolejki["hit_rate"].notna()]
            if not n_z_wynikiem.empty:
                st.markdown("**📅 Historia wyników per kolejka (1X2)**")
                _rows_hk = []
                for _, rk in _hist_kolejki.iterrows():
                    _hr_raw = rk["hit_rate"]
                    _hr_ok  = (_hr_raw is not None) and not pd.isna(_hr_raw)
                    hr      = float(_hr_raw) if _hr_ok else None
                    hr_c    = "#4CAF50" if _hr_ok and hr >= 0.62 else ("#FF9800" if _hr_ok and hr >= 0.55 else ("#888" if not _hr_ok else "#F44336"))
                    hr_str  = f"{hr:.0%}" if _hr_ok else "–"
                    wyn_str = f"{int(rk['n_traf'])}/{int(rk['n_typow'])}" if _hr_ok else f"–/{int(rk['n_typow'])}"
                    status_ico = "✅" if _hr_ok and hr >= 0.62 else ("⚠️" if _hr_ok and hr >= 0.50 else ("⏳" if not _hr_ok else "❌"))
                    _bg     = "rgba(76,175,80,0.04)" if _hr_ok and hr>=0.62 else ("rgba(255,152,0,0.03)" if _hr_ok and hr>=0.50 else "#0d0f14")
                    _bar_w  = int(hr*100) if _hr_ok else 0
                    _rows_hk.append(
                        f"<div style='display:grid;grid-template-columns:60px 60px 100px 1fr 90px;"
                        f"align-items:center;padding:8px 12px;background:{_bg};"
                        f"border-bottom:1px solid #131520;font-size:0.84em'>"
                        f"<span style='font-weight:700;color:#ccc'>#{int(rk['kolejnosc'])}</span>"
                        f"<span style='color:#555'>{int(rk['n_typow'])}</span>"
                        f"<span style='color:#888'>{wyn_str}</span>"
                        f"<div style='padding:0 12px'>"
                        f"<div style='background:var(--bg-card2);border-radius:3px;height:6px'>"
                        f"<div style='width:{_bar_w}%;background:{hr_c};height:6px;border-radius:3px'></div>"
                        f"</div></div>"
                        f"<span style='text-align:right;font-weight:700;color:{hr_c}'>{status_ico} {hr_str}</span>"
                        f"</div>"
                    )
                # Nagłówek tabeli
                st.markdown(
                    "<div style='display:grid;grid-template-columns:60px 60px 100px 1fr 90px;"
                    "gap:0;padding:6px 12px;font-size:0.68em;color:#444;"
                    "text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid #1e2028'>"
                    "<span>Kolejka</span><span>Typów</span><span>Wynik</span>"
                    "<span style='padding-left:4px'>Progress</span><span style='text-align:right'>Hit Rate</span>"
                    "</div>",
                    unsafe_allow_html=True)
                st.markdown(
                    f"<div style='border:1px solid var(--border);border-radius:10px;overflow:hidden'>"
                    f"{''.join(_rows_hk)}</div>",
                    unsafe_allow_html=True)
                st.caption("⏳ = predykcje zapisane, wyniki jeszcze nie dostępne w football-data.co.uk")

        # ── Aktualizuj wszystkie kolejki jednym kliknięciem ───────────────
        _t4c1, _t4c2 = st.columns([2, 3])
        with _t4c1:
            if st.button("🔄 Aktualizuj wszystkie kolejki", key="update_all_rounds",
                         help="Pobiera wyniki ze wszystkich nierozliczonych kolejek football-data.co.uk"):
                with st.spinner("Aktualizuję wyniki..."):
                    _all_mecze = sqlite3.connect(DB_FILE).execute(
                        "SELECT DISTINCT home, away FROM zdarzenia WHERE liga=? AND trafione IS NULL",
                        (wybrana_liga,)).fetchall()
                    sqlite3.connect(DB_FILE).close()
                    _n_up = 0
                    for _hh, _aa in _all_mecze:
                        _con_b = sqlite3.connect(DB_FILE)
                        _b4 = _con_b.execute(
                            "SELECT COUNT(*) FROM zdarzenia WHERE home=? AND away=? AND trafione IS NOT NULL",
                            (_hh, _aa)).fetchone()[0]
                        _con_b.close()
                        aktualizuj_wynik_zdarzenia(_hh, _aa, historical)
                        _con_a = sqlite3.connect(DB_FILE)
                        _af = _con_a.execute(
                            "SELECT COUNT(*) FROM zdarzenia WHERE home=? AND away=? AND trafione IS NOT NULL",
                            (_hh, _aa)).fetchone()[0]
                        _con_a.close()
                        if _af > _b4:
                            _n_up += 1
                    st.success(f"✅ Zaktualizowano {_n_up} meczów")
                    st.rerun()
        with _t4c2:
            st.caption("Pobiera dane z football-data.co.uk. Uruchom po każdej serii meczów "
                       "(dane dostępne ~24h po meczu).")

        st.divider()

        # ── Szczegóły kolejki – kliknij w tabelę powyżej ─────────────────
        if not _hist_kolejki.empty:
            _kolejki_list = sorted(_hist_kolejki["kolejnosc"].astype(int).tolist(), reverse=True)
            _sel_kol = st.selectbox(
                "🔍 Sprawdź szczegóły kolejki",
                options=_kolejki_list,
                format_func=lambda k: f"Kolejka #{k}",
                key="tab4_sel_kolejka"
            )
            if _sel_kol:
                _con_det = sqlite3.connect(DB_FILE)
                _det_df  = pd.read_sql_query(
                    """SELECT mecz, home, away, rynek, typ, linia, p_model, fair_odds,
                              trafione, data
                       FROM zdarzenia
                       WHERE liga=? AND kolejnosc=?
                       ORDER BY rynek, mecz""",
                    _con_det, params=(wybrana_liga, int(_sel_kol))
                )
                _con_det.close()

                if not _det_df.empty:
                    _kol_row = _hist_kolejki[_hist_kolejki["kolejnosc"] == _sel_kol]
                    _kol_hr  = float(_kol_row["hit_rate"].iloc[0]) if not _kol_row.empty and _kol_row["hit_rate"].notna().any() else None
                    _kol_n   = int(_kol_row["n_typow"].iloc[0]) if not _kol_row.empty else 0
                    _kol_t   = int(_kol_row["n_traf"].iloc[0]) if not _kol_row.empty else 0
                    _hr_str  = f"{_kol_hr:.0%}" if _kol_hr is not None else "⏳"
                    st.markdown(
                        f"**Kolejka #{_sel_kol}** · {_kol_t}/{_kol_n} trafione (1X2) · Hit Rate: **{_hr_str}**")

                    # Grupuj mecze
                    for _mecz_str, _grp in _det_df.groupby("mecz"):
                        _home_r = _grp["home"].iloc[0]
                        _away_r = _grp["away"].iloc[0]
                        # Główny typ 1X2
                        _1x2 = _grp[_grp["rynek"] == "1X2"]
                        _has_result = _1x2["trafione"].notna().any() if not _1x2.empty else False
                        _traf_ico   = ""
                        if not _1x2.empty and _1x2["trafione"].notna().any():
                            _traf_ico = "✅" if int(_1x2["trafione"].iloc[0]) == 1 else "❌"

                        with st.expander(f"{_traf_ico} {_mecz_str}", expanded=False):
                            # Wiersze per rynek
                            _det_rows = []
                            for _, dr in _grp.iterrows():
                                _trf = dr["trafione"]
                                _ico = "✅" if _trf == 1 else ("❌" if _trf == 0 else "⏳")
                                _ico_c = "#4CAF50" if _trf == 1 else ("#F44336" if _trf == 0 else "#888")
                                _lin = f" {dr['linia']:.1f}" if dr["linia"] and float(dr["linia"]) > 0 else ""
                                _rynek_opis = f"{dr['rynek']}{_lin}"
                                _p_str = f"{float(dr['p_model']):.0%}" if dr["p_model"] else "–"
                                _fo_str = f"{float(dr['fair_odds']):.2f}" if dr["fair_odds"] else "–"
                                _det_rows.append(
                                    f"<tr>"
                                    f"<td style='padding:4px 8px;font-size:0.86em;color:#aaa'>{_rynek_opis}</td>"
                                    f"<td style='padding:4px 8px;font-size:0.86em;font-weight:bold'>{dr['typ']}</td>"
                                    f"<td style='padding:4px 8px;text-align:center;color:#2196F3;font-size:0.84em'>{_p_str}</td>"
                                    f"<td style='padding:4px 8px;text-align:center;color:#888;font-size:0.84em'>{_fo_str}</td>"
                                    f"<td style='padding:4px 8px;text-align:center;color:{_ico_c};font-weight:bold'>{_ico}</td>"
                                    f"</tr>"
                                )
                            st.markdown(
                                f"<table style='width:100%;border-collapse:collapse'>"
                                f"<thead><tr style='color:#555;font-size:0.72em;text-transform:uppercase'>"
                                f"<th style='padding:4px 8px'>Rynek</th>"
                                f"<th style='padding:4px 8px'>Typ</th>"
                                f"<th style='padding:4px 8px;text-align:center'>P model</th>"
                                f"<th style='padding:4px 8px;text-align:center'>Fair</th>"
                                f"<th style='padding:4px 8px;text-align:center'>Wynik</th>"
                                f"</tr></thead>"
                                f"<tbody>{''.join(_det_rows)}</tbody></table>",
                                unsafe_allow_html=True
                            )

        st.divider()
        st.divider()

        with st.expander("💰 Symulacja Kelly – Equity Curve", expanded=False):
            # ── Equity Curve Kelly – generowana na bieżąco z historical ─────────
            # Źródło prawdy: historical CSV (rozegrane mecze bieżącego sezonu)
            # Predykcje generowane walk-forward per kolejka z danych do K-1
            # NIE polega na tym czy coś zostało zapisane ręcznie do zdarzenia DB

            _hist_curr_eq = historical[historical["_sezon"] == "biezacy"].copy() \
                if "_sezon" in historical.columns else historical.copy()

            # Połącz z terminarzem żeby mieć kolejność (round number)
            _sched_played = schedule[
                schedule["date"].apply(lambda d: pd.Timestamp(d).date() < datetime.now().date())
            ].copy() if not schedule.empty else pd.DataFrame()

            if not _hist_curr_eq.empty and not _sched_played.empty and len(_hist_curr_eq) >= 3:
                # Zbuduj mapę mecz -> kolejka z terminarza
                _round_map = {}
                for _, _sr in _sched_played.iterrows():
                    _sh = map_nazwa(_sr["home_team"])
                    _sa = map_nazwa(_sr["away_team"])
                    _round_map[(_sh, _sa)] = int(_sr["round"])

                # Zbierz predykcje + wyniki per mecz z historical
                _eq_rows = []
                for _, _hr in _hist_curr_eq.iterrows():
                    _hh = str(_hr.get("HomeTeam", ""))
                    _aa = str(_hr.get("AwayTeam", ""))
                    if not _hh or not _aa:
                        continue
                    if _hh not in srednie_df.index or _aa not in srednie_df.index:
                        continue

                    # Kolejka z terminarza, fallback szacowanie z pozycji
                    _rnd = _round_map.get((_hh, _aa))
                    if _rnd is None:
                        # Spróbuj odwrotnie (home/away mogą być zamienione w CSV)
                        _rnd = _round_map.get((_aa, _hh))
                    if _rnd is None:
                        # Ostatni fallback: szacuj z daty (kolejka ≈ numer meczu / 10)
                        try:
                            _dt_eq = pd.to_datetime(_hr.get("Date", ""), dayfirst=True, errors="coerce")
                            _rnd = int((_hist_curr_eq.index.get_loc(_hr.name) // 10) + 1) if not pd.isna(_dt_eq) else 1
                        except Exception:
                            _rnd = 1

                    # Wynik rzeczywisty
                    try:
                        _hg_eq = int(_hr["FTHG"]); _ag_eq = int(_hr["FTAG"])
                    except Exception:
                        continue
                    _wynik_1x2 = "1" if _hg_eq > _ag_eq else ("2" if _ag_eq > _hg_eq else "X")

                    # Predykcja modelu (z danych globalnych sezonu — uproszczenie walk-forward)
                    try:
                        _lh_eq, _la_eq = oblicz_lambdy(
                            _hh, _aa, srednie_df, srednie_lig, forma_dict,
                            csv_code=LIGI[wybrana_liga]["csv_code"])[:2]
                        _pred_eq = predykcja_meczu(
                            _lh_eq, _la_eq, rho=rho,
                            csv_code=LIGI[wybrana_liga]["csv_code"], n_train=n_biezacy)
                    except Exception:
                        continue

                    _typ_eq = _pred_eq["typ"]
                    _p_eq   = _pred_eq["p_typ"]
                    _fo_eq  = _pred_eq["fo_typ"]

                    # Trafiony?
                    if _typ_eq == "1":  _traf_eq = (_wynik_1x2 == "1")
                    elif _typ_eq == "X": _traf_eq = (_wynik_1x2 == "X")
                    elif _typ_eq == "2": _traf_eq = (_wynik_1x2 == "2")
                    elif _typ_eq == "1X": _traf_eq = (_wynik_1x2 in ("1","X"))
                    elif _typ_eq == "X2": _traf_eq = (_wynik_1x2 in ("X","2"))
                    else: _traf_eq = False

                    # Kurs bukmachera (MaxC > Pinnacle > B365)
                    def _get_k(col_h, col_d, col_a, row, typ):
                        try:
                            oh = float(row.get(col_h) or 0)
                            od = float(row.get(col_d) or 0)
                            oa = float(row.get(col_a) or 0)
                            if min(oh, od, oa) <= 1.01: return None
                            s = 1/oh + 1/od + 1/oa
                            ih, id_, ia = (1/oh)/s, (1/od)/s, (1/oa)/s
                            if typ == "1":  return oh
                            if typ == "X":  return od
                            if typ == "2":  return oa
                            if typ == "1X": return round(1/(ih+id_), 3)
                            if typ == "X2": return round(1/(id_+ia), 3)
                        except Exception: return None

                    _k_eq = (_get_k("MaxCH","MaxCD","MaxCA", _hr, _typ_eq) or
                             _get_k("PSCH","PSCD","PSCA",  _hr, _typ_eq) or
                             _get_k("PSH","PSD","PSA",     _hr, _typ_eq) or
                             _get_k("B365H","B365D","B365A", _hr, _typ_eq) or
                             _fo_eq)

                    _eq_rows.append({
                        "kolejnosc": _rnd,
                        "home":      _hh,
                        "away":      _aa,
                        "typ":       _typ_eq,
                        "p_model":   _p_eq,
                        "fair_odds": _fo_eq,
                        "kurs_buk":  _k_eq,
                        "trafione":  int(_traf_eq),
                    })

                _eq_df = pd.DataFrame(_eq_rows).sort_values(["kolejnosc", "home"])

                # Też scalamy z DB (jeśli są ręcznie zapisane z innym p_model)
                # Dajemy priorytet danym on-the-fly — DB służy tylko jako uzupełnienie
                _con_eq = sqlite3.connect(DB_FILE)
                _eq_db = pd.read_sql_query(
                    """SELECT kolejnosc, home, away, typ, trafione, p_model, fair_odds
                       FROM zdarzenia
                       WHERE liga=? AND rynek='1X2' AND trafione IS NOT NULL AND sezon=?
                       ORDER BY kolejnosc, id""",
                    _con_eq, params=(wybrana_liga, BIEZACY_SEZON))
                _con_eq.close()

                # Użyj on-the-fly jako główne źródło (ma kompletne dane)
                # _eq_db jako info o liczbie zapisanych (do caption)
                _n_db = len(_eq_db)
                _n_live = len(_eq_df)

                st.caption(
                    f"📊 Dane: **{_n_live}** meczów z historical (on-the-fly)"
                    + (f" · {_n_db} w bazie DB" if _n_db > 0 else " · brak zapisów w DB — kliknij 'Aktualizuj wyniki'")
                )

            else:
                _eq_df = pd.DataFrame()

            if len(_eq_df) >= 3:

                # Kelly parametry
                _KS      = float(st.session_state.get("bankroll", 1000.0))
                _KF_base = float(st.session_state.get("kelly_frac", 0.125))
                _KF      = _KF_base * KELLY_FRAC_SCALE
                _KMAX    = 0.05; _KEV = 0.05; _KMAX_ODDS = 3.50

                def _conservative_p(p):
                    return 0.5 + (float(p) - 0.5) * KELLY_PROB_SCALE

                def _kurs_efektywny(row):
                    k = row.get("kurs_buk")
                    if k is not None and not pd.isna(k): return float(k)
                    fo = row.get("fair_odds")
                    if fo is not None and not pd.isna(fo): return float(fo)
                    return None
                _eq_df["kurs_eff"] = _eq_df.apply(_kurs_efektywny, axis=1)

                def _calc_ev(r):
                    try:
                        k = r["kurs_eff"]
                        if k is None or pd.isna(k): return None
                        k = float(k)
                        if not (1.35 <= k <= _KMAX_ODDS): return None
                        return float(r["p_model"]) * k - 1.0
                    except Exception:
                        return None
                _eq_df["ev"] = _eq_df.apply(_calc_ev, axis=1)

                # Źródła kursów caption
                _n_csv_ok        = int(_eq_df["kurs_buk"].notna().sum())
                _n_fair_fallback = int(_eq_df["kurs_buk"].isna().sum())
                _src_parts = [f"📊 Źródła kursów: **{_n_csv_ok}** z football-data (MaxC/Pinnacle/B365)"]
                if _n_fair_fallback > 0:
                    _src_parts.append(f"**{_n_fair_fallback}** fair odds (brak kursu w CSV)")
                st.caption("  ·  ".join(_src_parts))
                _bk   = _KS
                _bk_flat = _KS   # równoległa flat dla porównania
                _bk_vals  = []
                _flat_vals = []
                _kelly_typy = 0; _kelly_traf = 0; _kelly_pnl = 0.0
                _peak_k = _KS; _max_dd_k = 0.0

                for _kol, _grp in _eq_df.groupby("kolejnosc"):
                    # Flat – wszystkie typy 1 jednostka, kurs realny (buk > fair fallback)
                    for _, _r in _grp.iterrows():
                        _kf_real = _r["kurs_buk"] if (_r["kurs_buk"] is not None and not pd.isna(_r["kurs_buk"])) else float(_r["fair_odds"])
                        _pv = float(_kf_real) - 1 if _r["trafione"] == 1 else -1
                        _bk_flat += _pv
                        _flat_vals.append(round(_bk_flat, 2))

                    # Kelly – top 3 z EV≥5% i kursem 1.35–3.50 · Conservative Kelly
                    _cand = _grp[_grp["ev"].notna() & (_grp["ev"] >= _KEV)].nlargest(3, "ev")
                    for _, _r in _cand.iterrows():
                        _pt  = _conservative_p(_r["p_model"])   # shrinkage -15%
                        _k   = float(_r["kurs_eff"])
                        _b   = _k - 1
                        _f   = min(max((_pt*_b-(1-_pt))/_b, 0.0) * _KF, _KMAX)
                        _stw = round(_bk * _f, 2)
                        if _stw > 0:
                            _wyg = _r["trafione"] == 1
                            _bk += _stw*(_k-1) if _wyg else -_stw
                            _kelly_pnl += _stw*(_k-1) if _wyg else -_stw
                            _kelly_typy += 1
                            _kelly_traf += int(_wyg)
                        _bk = max(_bk, 0.01)
                        if _bk > _peak_k: _peak_k = _bk
                        _dd = (_peak_k - _bk) / _peak_k * 100
                        if _dd > _max_dd_k: _max_dd_k = _dd
                    _bk_vals.append(round(_bk, 2))  # per kolejka

                _roi_kelly = (_bk - _KS) / _KS * 100
                _roi_flat  = (_bk_flat - _KS) / _KS * 100
                _kol_nums  = sorted(_eq_df["kolejnosc"].unique())
                _kc = "#4CAF50" if _bk >= _KS else "#F44336"
                _fc = "#888" if _bk_flat >= _KS else "#F44336"
                _kelly_hr  = f"{_kelly_traf/_kelly_typy:.0%}" if _kelly_typy else "–"

                st.markdown(
                    "<div class='section-header'>💰 Symulacja Kelly – bieżący sezon"
                    "<span style='font-size:.65em;color:#555;font-weight:400;margin-left:10px'>"
                    "top 3 typy/kolejkę · Pinnacle/B365 1.35–3.50 · Conservative Kelly"
                    "</span></div>",
                    unsafe_allow_html=True)
                _ss_ml_key = f"ml_bk_{wybrana_liga}"
                st.session_state[_ss_ml_key] = {
                    "bk": round(_bk, 2),
                    "bk_flat": round(_bk_flat, 2),
                    "roi": round(_roi_kelly, 2),
                    "roi_flat": round(_roi_flat, 2),
                    "n": _kelly_typy,
                    "dd": round(_max_dd_k, 2),
                    "bk_vals": _bk_vals,
                    "kol_nums": list(_kol_nums),
                }


                _ek1, _ek2, _ek3, _ek4 = st.columns(4)
                _ek1.metric("💰 Bankroll", f"{_bk:.0f} zł",
                            delta=f"{_roi_kelly:+.1f}%",
                            delta_color="normal",
                            help="Końcowy bankroll po wszystkich typach Kelly od początku sezonu")
                _ek2.metric("📋 Typów Kelly", _kelly_typy,
                            delta=f"Hit {_kelly_hr}",
                            delta_color="off",
                            help="Tylko mecze gdzie model miał EV≥5% vs kurs bukmachera")
                _ek3.metric("📉 Max Drawdown", f"{_max_dd_k:.1f}%",
                            delta="wysoki" if _max_dd_k > 20 else "ok",
                            delta_color="inverse" if _max_dd_k > 20 else "normal",
                            help="Największy spadek bankrollu od szczytu")
                _ek4.metric("📊 Flat (porównanie)", f"{_bk_flat:.0f} zł",
                            delta=f"{_roi_flat:+.1f}%",
                            delta_color="normal" if _roi_flat >= 0 else "inverse",
                            help="Flat betting – 1 jednostka na każdy typ 1X2 (fair odds bez marży)")

                if _bk_vals and len(_kol_nums) == len(_bk_vals):
                    import pandas as _pd_eq
                    _chart_df = _pd_eq.DataFrame({
                        "Kolejka": _kol_nums,
                        "Kelly (zł)": _bk_vals,
                    }).set_index("Kolejka")
                    st.line_chart(_chart_df, height=220, color=[_kc])

                if _kelly_typy == 0:
                    st.caption("⚠️ Brak typów Kelly – dane o kursach bukmachera niedostępne "
                               "w historical (brak kolumn PSH/B365H).")
                else:
                    _missing_kols = _eq_df.groupby("kolejnosc").apply(
                        lambda g: g["kurs_buk"].isna().all()).sum()
                    _kols_total = _eq_df["kolejnosc"].nunique()
                    st.caption(
                        f"Start: 1 000 zł · {_kelly_typy} typów z EV≥5% · kurs 1.35–3.50 · "
                        f"PnL Kelly: {_kelly_pnl:+.0f} zł · "
                        f"Flat (fair odds, bez marży): {_roi_flat:+.1f}%"
                    )
                    # Kontekst statystyczny – N za małe?
                    _n_warn_color = "#f57c00" if _kelly_typy < 150 else "#4CAF50"
                    _n_warn_txt = (
                        f"⚠️ Próba statystyczna: **{_kelly_typy} typów** – "
                        f"{'za mało do wiarygodnej oceny Kelly (min. ~150)' if _kelly_typy < 150 else 'wystarczająca próba'}. "
                        f"Kolejek bez kursów bukmachera: **{_missing_kols}/{_kols_total}** "
                        f"(historyczne Pinnacle/B365 niedostępne retroaktywnie)."
                    )
                    st.markdown(
                        f"<div style='font-size:0.78em;color:{_n_warn_color};"
                        f"background:var(--bg-card);border:1px solid {_n_warn_color}44;"
                        f"border-radius:6px;padding:8px 12px;margin-top:6px'>{_n_warn_txt}</div>",
                        unsafe_allow_html=True)


                # ── Szczegóły typów Kelly – gdzie model się mylił ─────────────
                if _kelly_typy > 0:
                    with st.expander(f"🔍 Szczegóły typów Kelly ({_kelly_typy} zakładów)", expanded=False):
                        _kelly_rows = []
                        for _kol2, _grp2 in _eq_df.groupby("kolejnosc"):
                            _cand2 = _grp2[_grp2["ev"].notna() & (_grp2["ev"] >= _KEV)].nlargest(3, "ev")
                            for _, _r2 in _cand2.iterrows():
                                # kurs_eff = closing odds CSV lub fair odds jako fallback
                                # NIE pomijamy meczów bez closing (najnowsze kolejki)
                                _ke2 = _r2.get("kurs_eff")
                                _kb2 = _r2.get("kurs_buk")
                                if _ke2 is None or pd.isna(_ke2): continue
                                _ke2 = float(_ke2)
                                if _ke2 <= 1.0: continue
                                _pt2   = float(_r2["p_model"])
                                _ev2   = float(_r2["ev"])
                                _b2    = _ke2 - 1
                                _f2    = min(max((_pt2*_b2-(1-_pt2))/_b2, 0.0) * _KF, _KMAX)
                                _stw2  = round(1000 * _f2, 2)
                                _traf2 = _r2["trafione"] == 1
                                _pnl2  = round(_stw2*(_ke2-1), 2) if _traf2 else round(-_stw2, 2)
                                _fair_flag = " ✦" if (_kb2 is None or pd.isna(_kb2)) else ""
                                _kelly_rows.append({
                                    "Kolejka": int(_kol2),
                                    "Mecz":    f"{_r2['home']} – {_r2['away']}",
                                    "Typ":     _r2["typ"],
                                    "P model": f"{_pt2:.0%}",
                                    "Kurs":    f"{_ke2:.2f}{_fair_flag}",
                                    "EV":      f"{_ev2:+.1%}",
                                    "Stawka*": f"{_stw2:.0f} zł",
                                    "Wynik":   "✅ TAK" if _traf2 else "❌ NIE",
                                    "PnL*":    f"{_pnl2:+.0f} zł",
                                })
                        if _kelly_rows:
                            import pandas as _pd_kd
                            _kd_df = _pd_kd.DataFrame(_kelly_rows)
                            st.dataframe(
                                _kd_df,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Wynik": st.column_config.TextColumn("Wynik", width="small"),
                                    "PnL*":  st.column_config.TextColumn("PnL*",  width="small"),
                                }
                            )
                            _wins   = sum(1 for r in _kelly_rows if "TAK" in r["Wynik"])
                            _losses = len(_kelly_rows) - _wins
                            st.caption(
                                f"✅ Trafione: {_wins} · ❌ Chybione: {_losses} · "
                                f"*Stawki szacunkowe od 1000 zł · ✦ = fair odds (CSV bez closing)"
                            )


        st.divider()

        # ── Skuteczność rynków alternatywnych ──────────────────────────────
        st.markdown(
            "<div class='section-header'>🎯 Skuteczność – rynki alternatywne"
            "<span style='font-size:.65em;color:#555;font-weight:400;margin-left:10px'>"
            "bez 1/X/2/1X/X2 · gole, BTTS, rożne, kartki, SOT"
            "</span></div>",
            unsafe_allow_html=True)
        try:
            _alt_con = sqlite3.connect(DB_FILE)
            _alt_df = pd.read_sql_query(
                """SELECT rynek, typ, p_model, fair_odds, trafione
                   FROM zdarzenia
                   WHERE liga=? AND rynek != '1X2' AND trafione IS NOT NULL
                   AND sezon=?
                   ORDER BY id""",
                _alt_con, params=(wybrana_liga, BIEZACY_SEZON))
            _alt_con.close()

            if len(_alt_df) < 5:
                st.info("Za mało danych rynków alternatywnych (minimum 5 rozliczonych zdarzeń).")
            else:
                # Grupuj per rynek
                _alt_summary = []
                for _kat, _grp in _alt_df.groupby("rynek"):
                    _n    = len(_grp)
                    _traf = int(_grp["trafione"].sum())
                    _hr   = _traf / _n
                    _avg_fo = _grp["fair_odds"].mean()
                    # Flat PnL (stawka 1 jednostka)
                    _pnl_flat = sum(
                        float(r["fair_odds"]) - 1 if r["trafione"] == 1 else -1
                        for _, r in _grp.iterrows()
                    )
                    _yield_flat = _pnl_flat / _n * 100
                    # Avg prob modelu
                    _avg_p = _grp["p_model"].mean()
                    # Brier score (kalibracja)
                    _brier = (((_grp["p_model"] - _grp["trafione"]) ** 2).mean())
                    _alt_summary.append({
                        "Rynek": _kat,
                        "n": _n,
                        "Hit%": _hr,
                        "AvgP": _avg_p,
                        "AvgFO": _avg_fo,
                        "Yield%": _yield_flat,
                        "PnL": _pnl_flat,
                        "Brier": _brier,
                    })

                if not _alt_summary:
                    st.info("Brak danych.")
                else:
                    _alt_summary.sort(key=lambda x: -x["n"])

                    # KPI wiersz – łącznie
                    _alt_total_n    = sum(r["n"] for r in _alt_summary)
                    _alt_total_traf = sum(int(r["Hit%"] * r["n"]) for r in _alt_summary)
                    _alt_total_pnl  = sum(r["PnL"] for r in _alt_summary)
                    _alt_total_yield = _alt_total_pnl / _alt_total_n * 100 if _alt_total_n else 0

                    _ac1, _ac2, _ac3, _ac4 = st.columns(4)
                    _ac1.metric("Łącznie typów", f"{_alt_total_n}")
                    _ac2.metric("Hit rate", f"{_alt_total_traf/_alt_total_n:.1%}" if _alt_total_n else "–")
                    _alt_pnl_c = "normal" if _alt_total_pnl >= 0 else "inverse"
                    _ac3.metric("PnL (flat)", f"{_alt_total_pnl:+.1f} j.", delta_color=_alt_pnl_c)
                    _ac4.metric("Yield", f"{_alt_total_yield:+.1f}%", delta_color=_alt_pnl_c)

                    # Tabela per rynek
                    _alt_rows = []
                    for r in _alt_summary:
                        _yc = "#4CAF50" if r["Yield%"] >= 0 else "#F44336"
                        _hc = "#4CAF50" if r["Hit%"] >= (1 / r["AvgFO"]) else "#aaa"
                        _alt_rows.append(
                            f"<tr>"
                            f"<td style='padding:9px 12px;font-weight:600;color:#ddd'>{r['Rynek']}</td>"
                            f"<td style='padding:9px 10px;text-align:center;color:#888'>{r['n']}</td>"
                            f"<td style='padding:9px 10px;text-align:center;color:{_hc};font-weight:700'>{r['Hit%']:.1%}</td>"
                            f"<td style='padding:9px 10px;text-align:center;color:#888'>{r['AvgP']:.1%}</td>"
                            f"<td style='padding:9px 10px;text-align:center;color:#888'>{r['AvgFO']:.2f}</td>"
                            f"<td style='padding:9px 10px;text-align:center;color:{_yc};font-weight:700'>{r['Yield%']:+.1f}%</td>"
                            f"<td style='padding:9px 10px;text-align:center;color:{_yc}'>{r['PnL']:+.1f}</td>"
                            f"<td style='padding:9px 10px;text-align:center;color:#555;font-size:0.82em'>{r['Brier']:.3f}</td>"
                            f"</tr>"
                        )
                    st.markdown(
                        "<div style='overflow-x:auto;border-radius:8px;border:1px solid var(--border);margin-top:8px'>"
                        "<table style='width:100%;border-collapse:collapse'>"
                        "<thead><tr style='background:var(--bg-card);color:#444;font-size:0.7em;text-transform:uppercase'>"
                        "<th style='padding:8px 12px'>Rynek</th>"
                        "<th style='padding:8px 12px;text-align:center'>Typów</th>"
                        "<th style='padding:8px 12px;text-align:center'>Hit%</th>"
                        "<th style='padding:8px 12px;text-align:center'>Avg P</th>"
                        "<th style='padding:8px 12px;text-align:center'>Avg FO</th>"
                        "<th style='padding:8px 12px;text-align:center'>Yield (flat)</th>"
                        "<th style='padding:8px 12px;text-align:center'>PnL</th>"
                        "<th style='padding:8px 12px;text-align:center'>Brier ↓</th>"
                        f"</tr></thead><tbody>{''.join(_alt_rows)}</tbody></table></div>",
                        unsafe_allow_html=True)
                    st.caption("Hit% zielony = powyżej break-even (1/AvgFO). PnL w jednostkach flat (stawka 1/typ). Brier: im niższy tym lepiej skalibrowany model.")

                    # Mini wykres: Hit% vs Break-even per rynek
                    try:
                        import plotly.graph_objects as _go_alt
                        _fig_alt = _go_alt.Figure()
                        _rynki_sorted = sorted(_alt_summary, key=lambda x: x["Yield%"], reverse=True)
                        _labels = [r["Rynek"] for r in _rynki_sorted]
                        _yields = [r["Yield%"] for r in _rynki_sorted]
                        _colors_alt = ["#4CAF50" if y >= 0 else "#F44336" for y in _yields]
                        _fig_alt.add_trace(_go_alt.Bar(
                            x=_labels, y=_yields,
                            marker_color=_colors_alt,
                            text=[f"{y:+.1f}%" for y in _yields],
                            textposition="outside",
                            hovertemplate="<b>%{x}</b><br>Yield: %{y:+.1f}%<extra></extra>"
                        ))
                        _fig_alt.add_hline(y=0, line_dash="dot", line_color="#444", line_width=1)
                        _fig_alt.update_layout(
                            paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
                            height=220, margin=dict(l=30, r=20, t=20, b=60),
                            xaxis=dict(color="#555", gridcolor="#1a1c24", tickangle=-20),
                            yaxis=dict(title="Yield %", color="#555", gridcolor="#1a1c24"),
                            font=dict(color="#888"), showlegend=False)
                        st.plotly_chart(_fig_alt, use_container_width=True,
                                        config={"displayModeBar": False})
                    except Exception:
                        pass  # wykres opcjonalny

                    # ── Skuteczność per linia (expanders) ────────────────────
                    st.markdown(
                        "<div style='margin:16px 0 8px 0;font-size:0.78em;color:#555;"
                        "text-transform:uppercase;letter-spacing:.07em'>"
                        "📐 Skuteczność per linia – rozwiń kategorię</div>",
                        unsafe_allow_html=True)

                    # Grupuj rynek → typy (linie)
                    # Wzorzec: rynek = "Gole" → typ = "Over 2.5", "Under 1.5" itp.
                    # Grupujemy po (rynek, typ) i liczymy n/hit/yield
                    _linia_df = _alt_df.copy()
                    _linia_df["p_model"]   = pd.to_numeric(_linia_df["p_model"],   errors="coerce")
                    _linia_df["fair_odds"] = pd.to_numeric(_linia_df["fair_odds"], errors="coerce")
                    _linia_df = _linia_df.dropna(subset=["p_model", "fair_odds"])

                    # Kategorie rynków do wyświetlenia (kolejność)
                    _kat_icons = {
                        "Gole":    "⚽", "BTTS":   "🤝",
                        "Rożne":   "🚩", "Kartki": "🟨",
                        "SOT":     "🎯",
                    }
                    _kat_kolejnosc = ["Gole", "BTTS", "Rożne", "Kartki", "SOT"]

                    for _kat in _kat_kolejnosc:
                        _kat_data = _linia_df[_linia_df["rynek"] == _kat]
                        if _kat_data.empty:
                            continue
                        _kat_n    = len(_kat_data)
                        _kat_traf = int(_kat_data["trafione"].sum())
                        _kat_hr   = _kat_traf / _kat_n
                        _kat_pnl  = sum(
                            float(r["fair_odds"]) - 1 if r["trafione"] == 1 else -1
                            for _, r in _kat_data.iterrows()
                        )
                        _kat_yield = _kat_pnl / _kat_n * 100
                        _kat_yc   = "#4CAF50" if _kat_yield >= 0 else "#F44336"
                        _kat_ico  = _kat_icons.get(_kat, "📊")

                        _exp_lbl = (
                            f"{_kat_ico} {_kat}  ·  "
                            f"{_kat_n} typów  ·  "
                            f"Hit {_kat_hr:.0%}  ·  "
                            f"Yield {_kat_yield:+.1f}%"
                        )
                        with st.expander(_exp_lbl, expanded=False):
                            # Per-linia tabela
                            _linia_rows_data = []
                            for _typ_val, _grp_t in _kat_data.groupby("typ"):
                                _tn = len(_grp_t)
                                if _tn < 2:
                                    continue
                                _tt = int(_grp_t["trafione"].sum())
                                _th = _tt / _tn
                                _tfo = _grp_t["fair_odds"].mean()
                                _tbe = 1 / _tfo  # break-even
                                _tpnl = sum(
                                    float(r["fair_odds"]) - 1 if r["trafione"] == 1 else -1
                                    for _, r in _grp_t.iterrows()
                                )
                                _tyield = _tpnl / _tn * 100
                                _tbrier = float(
                                    ((_grp_t["p_model"] - _grp_t["trafione"]) ** 2).mean()
                                )
                                _linia_rows_data.append({
                                    "typ": _typ_val, "n": _tn,
                                    "hit": _th, "be": _tbe,
                                    "fo": _tfo, "yield": _tyield,
                                    "pnl": _tpnl, "brier": _tbrier,
                                })

                            # Sortuj po Yield malejąco
                            _linia_rows_data.sort(key=lambda x: -x["yield"])

                            if not _linia_rows_data:
                                st.caption("Za mało danych (min. 2 typy per linia).")
                            else:
                                _tr_html = ""
                                for _r in _linia_rows_data:
                                    _yc2  = "#4CAF50" if _r["yield"] >= 0 else "#F44336"
                                    _hc2  = "#4CAF50" if _r["hit"] >= _r["be"] else "#aaa"
                                    _best = "background:#0a1a0a;" if _r["yield"] == max(x["yield"] for x in _linia_rows_data) else ""
                                    _pnl_sym = "▲" if _r["pnl"] >= 0 else "▼"
                                    _tr_html += (
                                        f"<tr style='{_best}'>"
                                        f"<td style='padding:8px 12px;font-weight:600;color:#e5e7eb'>{_r['typ']}</td>"
                                        f"<td style='padding:8px 10px;text-align:center;color:#6b7280'>{_r['n']}</td>"
                                        f"<td style='padding:8px 10px;text-align:center;color:{_hc2};font-weight:700'>{_r['hit']:.0%}</td>"
                                        f"<td style='padding:8px 10px;text-align:center;color:#4b5563;font-size:0.82em'>{_r['be']:.0%}</td>"
                                        f"<td style='padding:8px 10px;text-align:center;color:#9ca3af'>{_r['fo']:.2f}</td>"
                                        f"<td style='padding:8px 10px;text-align:center;color:{_yc2};font-weight:700'>{_r['yield']:+.1f}%</td>"
                                        f"<td style='padding:8px 10px;text-align:center;color:{_yc2}'>{_pnl_sym} {abs(_r['pnl']):.1f}</td>"
                                        f"<td style='padding:8px 10px;text-align:center;color:#374151;font-size:0.8em'>{_r['brier']:.3f}</td>"
                                        f"</tr>"
                                    )
                                st.markdown(
                                    "<div style='overflow-x:auto;border-radius:8px;"
                                    "border:1px solid var(--border)'>"
                                    "<table style='width:100%;border-collapse:collapse'>"
                                    "<thead><tr style='background:var(--bg-app);color:#444;"
                                    "font-size:0.70em;text-transform:uppercase'>"
                                    "<th style='padding:6px 12px;text-align:left'>Linia</th>"
                                    "<th style='padding:6px 10px;text-align:center'>Typów</th>"
                                    "<th style='padding:6px 10px;text-align:center'>Hit%</th>"
                                    "<th style='padding:6px 10px;text-align:center'>Break-even</th>"
                                    "<th style='padding:6px 10px;text-align:center'>Avg FO</th>"
                                    "<th style='padding:6px 10px;text-align:center'>Yield</th>"
                                    "<th style='padding:6px 10px;text-align:center'>PnL</th>"
                                    f"<th style='padding:6px 10px;text-align:center'>Brier ↓</th>"
                                    f"</tr></thead><tbody>{_tr_html}</tbody></table></div>",
                                    unsafe_allow_html=True)
                                st.caption(
                                    "Tło zielone = najlepsza linia w kategorii. "
                                    "Hit% zielony = powyżej break-even. "
                                    "Min. 2 typy per linia.")

        except Exception as _e_alt:
            st.caption(f"Błąd sekcji alt rynków: {_e_alt}")


        st.divider()

        with st.expander("📉 Kalibracja & Rolling Performance", expanded=False):
            # ── Kalibracja (połączone z Skutecznością) ─────────────────────
            st.divider()
            st.subheader("📉 Kalibracja modelu & Rolling Performance")
            st.caption("Kalibracja: czy model mówi 65% → trafia ~65%? Rolling: jak ewoluuje jakość modelu w czasie.")

            st.markdown("### 📈 Rolling Performance (okno 50 zdarzeń)")
            rk1, rk2 = st.columns(2)
            with rk1:
                okno_r = st.slider("Okno rolling (duże)", 20, 100, 50, 10, key="roll_win")
            with rk2:
                okno_short = st.slider("Okno rolling (krótkie)", 5, 25, 12, 1, key="roll_win_short",
                                       help="Krótkie okno (10-15) pokazuje szybkie zmiany skuteczności")
            roll_df       = rolling_stats(wybrana_liga, okno=okno_r)
            roll_df_short = rolling_stats(wybrana_liga, okno=okno_short)

            if not roll_df.empty:
                rc1, rc2 = st.columns(2)
                with rc1:
                    st.markdown("**Brier Score rolling ↓** *(niższy = lepszy)*")
                    chart_brier = roll_df[["idx","brier_roll"]].rename(
                        columns={"idx":"Zdarzenie","brier_roll":"Brier Score"}
                    ).set_index("Zdarzenie")
                    st.line_chart(chart_brier, height=200, color="#F44336")
                    last_b = roll_df["brier_roll"].iloc[-1]
                    trend_b = roll_df["brier_roll"].iloc[-1] - roll_df["brier_roll"].iloc[max(0,len(roll_df)-10)]
                    st.caption(f"Ostatni rolling Brier: **{last_b:.4f}** "
                               f"({'↓ poprawa' if trend_b < 0 else '↑ pogorszenie'} o {abs(trend_b):.4f})")
                with rc2:
                    st.markdown("**Hit Rate rolling** *(model vs rzeczywistość)*")
                    chart_hit = roll_df[["idx","hit_roll","p_roll"]].rename(
                        columns={"idx":"Zdarzenie","hit_roll":"Hit Rate rzeczywisty","p_roll":"P model (avg)"}
                    ).set_index("Zdarzenie")
                    st.line_chart(chart_hit, height=200)
                    last_hit = roll_df["hit_roll"].iloc[-1]
                    last_p   = roll_df["p_roll"].iloc[-1]
                    gap = last_hit - last_p
                    gap_col = "✅" if abs(gap) < 0.03 else ("⚠️" if abs(gap) < 0.08 else "❌")
                    st.caption(f"{gap_col} Hit Rate: **{last_hit:.1%}** | P model: **{last_p:.1%}** | "
                               f"Gap: **{gap:+.1%}**")
            else:
                st.info(f"Rolling dostępny po min. {okno_r//2} zdarzeniach z wynikami.")

            # Krótkie okno rolling – szybkie zmiany formy modelu
            if not roll_df_short.empty:
                st.divider()
                st.markdown(f"**🔍 Hit Rate rolling krótkie (okno {okno_short})** – szybkie zmiany skuteczności")
                _short1, _short2 = st.columns([3, 1])
                with _short1:
                    _chart_short = roll_df_short[["idx","hit_roll","p_roll"]].rename(
                        columns={"idx":"Zdarzenie","hit_roll":"Hit Rate","p_roll":"P model"}
                    ).set_index("Zdarzenie")
                    st.line_chart(_chart_short, height=160)
                with _short2:
                    _last_hit_s = roll_df_short["hit_roll"].iloc[-1]
                    _last_p_s   = roll_df_short["p_roll"].iloc[-1]
                    _gap_s      = _last_hit_s - _last_p_s
                    _trend_dir  = "📈" if _gap_s > 0.03 else ("📉" if _gap_s < -0.03 else "➡️")
                    st.metric("Hit Rate", f"{_last_hit_s:.1%}",
                              delta=f"{_gap_s:+.1%} vs model",
                              delta_color="normal" if _gap_s >= 0 else "inverse")
                    st.caption(f"{_trend_dir} P model: {_last_p_s:.1%}")

            bpk_df = brier_per_kolejka(wybrana_liga)
            if not bpk_df.empty and len(bpk_df) >= 2:
                st.divider()
                st.markdown("### 📊 Brier Score per kolejka")
                bpk_chart = bpk_df[["kolejnosc","brier","hit_rate"]].rename(
                    columns={"kolejnosc":"Kolejka","brier":"Brier Score","hit_rate":"Hit Rate"}
                ).set_index("Kolejka")
                st.line_chart(bpk_chart, height=180)
                worst_k  = bpk_df.loc[bpk_df["brier"].idxmax(), "kolejnosc"]
                best_k   = bpk_df.loc[bpk_df["brier"].idxmin(), "kolejnosc"]
                st.caption(f"Najlepsza kolejka: **{int(best_k)}** ({bpk_df['brier'].min():.4f}) · "
                           f"Najgorsza: **{int(worst_k)}** ({bpk_df['brier'].max():.4f})")

            st.divider()
            st.markdown("### 🎯 Confidence Calibration")
            st.caption("Czy model mówi 65% → trafia ~65%? Każdy wiersz to 'bucket' zdarzeń o podobnym prawdopodobieństwie.")
            kal_df = kalibracja_modelu(wybrana_liga if wybrana_liga != "Wszystkie" else None)
            # Minimalna próbka
            _n_total_kal = 0
            _con_kal_n = sqlite3.connect(DB_FILE)
            _row_kal_n = _con_kal_n.execute(
                "SELECT COUNT(*) FROM zdarzenia WHERE liga=? AND trafione IS NOT NULL",
                (wybrana_liga,)).fetchone()
            _con_kal_n.close()
            _n_total_kal = _row_kal_n[0] if _row_kal_n else 0
            _min_wiarygodne = 300
            _prog_kal = min(_n_total_kal / _min_wiarygodne, 1.0)
            _prog_c = "#4CAF50" if _prog_kal >= 1.0 else ("#FF9800" if _prog_kal >= 0.5 else "#F44336")
            st.markdown(
                f"<div style='margin-bottom:8px'>"
                f"<div style='font-size:0.78em;color:#888;margin-bottom:3px'>"
                f"Próbka do wiarygodnej kalibracji: "
                f"<b style='color:{_prog_c}'>{_n_total_kal}/{_min_wiarygodne} zdarzeń "
                f"({_prog_kal:.0%})</b></div>"
                f"<div style='background:var(--bg-card2);border-radius:3px;height:5px'>"
                f"<div style='background:{_prog_c};width:{int(_prog_kal*100)}%;height:5px;border-radius:3px'></div>"
                f"</div></div>",
                unsafe_allow_html=True)

            if not kal_df.empty:
                kal_df = kal_df.dropna(subset=["skutecznosc"])

                rows_k = []
                for _, rk in kal_df.iterrows():
                    diff = rk["rozbieznosc"]
                    p_m  = rk["p_srednia"]
                    hit  = rk["skutecznosc"]
                    n    = int(rk["liczba"])
                    diff_col  = "#4CAF50" if abs(diff) < 0.05 else ("#FF9800" if abs(diff) < 0.12 else "#F44336")
                    diff_icon = "✅" if abs(diff) < 0.05 else ("⚠️" if abs(diff) < 0.12 else "❌")
                    # Wizualny pasek: model (niebieski) vs hit rate (zielony)
                    bar_model = int(p_m * 100)
                    bar_hit   = int(hit * 100) if hit else 0
                    bar_html  = (
                        f"<div style='position:relative;background:var(--bg-card2);border-radius:3px;height:12px;width:120px;overflow:hidden'>"
                        f"<div style='position:absolute;background:#2196F3;opacity:0.5;height:12px;width:{bar_model}%'></div>"
                        f"<div style='position:absolute;background:#4CAF50;opacity:0.8;height:12px;width:{bar_hit}%'>"
                        f"</div></div>"
                    )
                    rows_k.append(
                        f"<tr style='border-bottom:1px solid #1a1a2e'>"
                        f"<td style='padding:7px 10px;font-weight:bold;color:#ddd'>{rk['przedzial']}</td>"
                        f"<td style='padding:7px 10px;text-align:center;color:#666'>{n}</td>"
                        f"<td style='padding:7px 6px'>{bar_html}</td>"
                        f"<td style='padding:7px 8px;text-align:center;color:#2196F3;font-size:0.88em'>{p_m:.1%}</td>"
                        f"<td style='padding:7px 8px;text-align:center;color:#4CAF50;font-size:0.88em'>{hit:.1%}</td>"
                        f"<td style='padding:7px 10px;text-align:center;color:{diff_col};font-weight:bold'>"
                        f"{diff_icon} {diff:+.1%}</td>"
                        f"</tr>"
                    )
                st.markdown(
                    f"<div style='overflow-x:auto;border-radius:8px;border:1px solid var(--border)'>"
                    f"<table style='width:100%;border-collapse:collapse;font-size:0.86em'>"
                    f"<thead><tr style='background:#1e1e2e;color:#666;font-size:0.74em;text-transform:uppercase'>"
                    f"<th style='padding:7px 10px;text-align:left'>Bucket P</th>"
                    f"<th style='padding:7px 10px;text-align:center'>N</th>"
                    f"<th style='padding:7px 10px;text-align:left'>Model 🔵 / Hit 🟢</th>"
                    f"<th style='padding:7px 8px;text-align:center'>P model</th>"
                    f"<th style='padding:7px 8px;text-align:center'>Hit Rate</th>"
                    f"<th style='padding:7px 10px;text-align:center'>Δ</th>"
                    f"</tr></thead><tbody>{''.join(rows_k)}</tbody></table></div>",
                    unsafe_allow_html=True,
                )

                st.divider()
                st.markdown("**📊 Reliability Curve** *(model vs rzeczywistość)*")
                w_rc5, h_rc5, pad_rc5 = 580, 340, 55
                def rc5_px(xv, yv):
                    px = pad_rc5 + (xv - 0.45) / 0.55 * (w_rc5 - 2 * pad_rc5)
                    py = h_rc5 - pad_rc5 - (yv - 0.45) / 0.55 * (h_rc5 - 2 * pad_rc5)
                    return px, py
                diag5 = [rc5_px(t, t) for t in [0.5, 0.65, 0.80, 0.95]]
                diag5_line = " ".join(f"{p[0]:.0f},{p[1]:.0f}" for p in diag5)
                circles5 = []
                for _, rk in kal_df.iterrows():
                    xv5 = rk["p_srednia"]; yv5 = rk["skutecznosc"]; n5 = int(rk["liczba"])
                    diff5 = rk["rozbieznosc"]
                    px5, py5 = rc5_px(xv5, yv5)
                    r5 = min(max(int(n5 * 1.2), 6), 22)
                    col5 = "#4CAF50" if abs(diff5) < 0.05 else ("#FF9800" if abs(diff5) < 0.12 else "#F44336")
                    circles5.append(
                        f"<circle cx='{px5:.0f}' cy='{py5:.0f}' r='{r5}' fill='{col5}' "
                        f"fill-opacity='0.85' stroke='white' stroke-width='1.5'>"
                        f"<title>Przedzial: {rk['przedzial']} | Model: {xv5:.1%} | Hit: {yv5:.1%} | N={n5}</title>"
                        f"</circle>"
                        f"<text x='{px5 + r5 + 4:.0f}' y='{py5 + 4:.0f}' font-size='9' fill='#ccc' "
                        f"font-family='sans-serif'>{rk['przedzial']} (n={n5})</text>"
                    )
                svg5 = (
                    f'<svg width="{w_rc5}" height="{h_rc5}" '
                    f'style="background:var(--stat-bg);border-radius:8px;display:block;margin:auto">'
                    f'<polyline points="{diag5_line}" fill="none" stroke="#444" '
                    f'stroke-width="1.5" stroke-dasharray="6,4"/>'
                    f'<text x="{w_rc5 - pad_rc5 + 4}" y="{pad_rc5 - 4}" font-size="9" fill="#555" '
                    f'font-family="sans-serif">idealny model</text>'
                    f'{"".join(circles5)}'
                    f'<text x="{w_rc5 // 2}" y="{h_rc5 - 6}" text-anchor="middle" '
                    f'font-size="10" fill="#888" font-family="sans-serif">P modelu →</text>'
                    f'<text x="12" y="{h_rc5 // 2}" text-anchor="middle" font-size="10" fill="#888" '
                    f'font-family="sans-serif" transform="rotate(-90,12,{h_rc5 // 2})">Hit Rate →</text>'
                    f'<circle cx="{w_rc5 - 130}" cy="20" r="6" fill="#4CAF50" stroke="white" stroke-width="1"/>'
                    f'<text x="{w_rc5 - 120}" y="25" font-size="9" fill="#aaa" font-family="sans-serif">Dobrze skalibrowany (&lt;5%)</text>'
                    f'<circle cx="{w_rc5 - 130}" cy="38" r="6" fill="#FF9800" stroke="white" stroke-width="1"/>'
                    f'<text x="{w_rc5 - 120}" y="43" font-size="9" fill="#aaa" font-family="sans-serif">Umiarkowany (5–12%)</text>'
                    f'<circle cx="{w_rc5 - 130}" cy="56" r="6" fill="#F44336" stroke="white" stroke-width="1"/>'
                    f'<text x="{w_rc5 - 120}" y="61" font-size="9" fill="#aaa" font-family="sans-serif">Słabo skalibrowany (&gt;12%)</text>'
                    f'</svg>'
                )
                st.markdown(svg5, unsafe_allow_html=True)
                st.divider()
                avg_bias = float(kal_df["rozbieznosc"].mean())
                if abs(avg_bias) < 0.02:
                    st.success(f"✅ Model dobrze skalibrowany (średni bias {avg_bias:+.1%})")
                elif avg_bias > 0:
                    st.warning(f"⚠️ Model zbyt ostrożny – niedoszacowuje szanse (bias {avg_bias:+.1%}). "
                               f"Rozważ obniżenie progu pewności.")
                else:
                    st.warning(f"⚠️ Model zbyt pewny siebie – przeszacowuje szanse (bias {avg_bias:+.1%}). "
                               f"Rozważ podwyższenie progu pewności.")
                st.caption("Punkty powyżej przekątnej = model był zbyt ostrożny. Poniżej = zbyt pewny siebie.")
            else:
                with st.container(border=True):
                    st.markdown("**📭 Brak danych do reliability curve.**")
                    st.markdown("Potrzebne są co najmniej **30 zdarzeń z wynikami** w każdym przedziale p. Wróć po kilku kolejkach trackingu.")

            # ── EV vs Actual Yield ───────────────────────────────────────────────────
            st.divider()
            st.markdown(
                "<div class='section-header'>📈 EV vs Actual Yield – sprzężenie zwrotne modelu"
                "<span style='font-size:.65em;color:#555;font-weight:400;margin-left:10px'>"
                "czy model faktycznie ma edge tam gdzie widzi EV?"
                "</span></div>",
                unsafe_allow_html=True)
            st.caption(
                "Dla każdego przedziału EV: ile model obiecywał vs ile faktycznie zarobiono "
                "(fair odds). Jeśli Actual Yield < 0 przy EV > 15% → zwiększ próg EV dla tej ligi.")

            try:
                import sqlite3 as _sq3, pandas as _pd_ev
                _db_ev = _sq3.connect(DB_FILE)
                _ev_df = _pd_ev.read_sql_query(
                    """SELECT p_model, fair_odds, trafione, typ, kolejnosc
                       FROM zdarzenia
                       WHERE liga = ?
                         AND rynek = '1X2'
                         AND trafione IS NOT NULL
                         AND fair_odds IS NOT NULL
                         AND p_model IS NOT NULL
                         AND fair_odds >= 1.35""",
                    _db_ev, params=(wybrana_liga,))
                _db_ev.close()

                if len(_ev_df) < 20:
                    st.info("📭 Za mało danych (minimum 20 typów z wynikami). Wróć po kolejnych kolejkach.")
                else:
                    # Oblicz EV przewidywany (fair odds)
                    _ev_df["ev_pred"] = _ev_df["p_model"].astype(float) * _ev_df["fair_odds"].astype(float) - 1
                    # Oblicz rzeczywisty yield na fair odds
                    _ev_df["yield_act"] = _ev_df.apply(
                        lambda r: float(r["fair_odds"]) - 1 if int(r["trafione"]) == 1 else -1.0, axis=1)
                    # trafione może być 0/1 lub "TAK"/"NIE"
                    _ev_df["hit"] = _ev_df["trafione"].apply(
                        lambda x: 1 if x in (1, True, "TAK", "1", 1.0) else 0)
                    _ev_df["yield_act"] = _ev_df.apply(
                        lambda r: float(r["fair_odds"]) - 1 if r["hit"] == 1 else -1.0, axis=1)

                    # Biny EV
                    _bins   = [(-0.50, 0.00, "< 0%"),
                               (0.00,  0.04, "0–4%"),
                               (0.04,  0.08, "4–8%"),
                               (0.08,  0.12, "8–12%"),
                               (0.12,  0.20, "12–20%"),
                               (0.20,  0.35, "20–35%"),
                               (0.35,  9.99, "> 35%")]

                    _rows = []
                    for lo, hi, label in _bins:
                        _sub = _ev_df[(_ev_df["ev_pred"] >= lo) & (_ev_df["ev_pred"] < hi)]
                        if len(_sub) == 0: continue
                        _rows.append({
                            "EV bin": label,
                            "N":      len(_sub),
                            "Avg EV": _sub["ev_pred"].mean(),
                            "Yield":  _sub["yield_act"].mean(),
                            "Hit%":   _sub["hit"].mean(),
                        })
                    _ev_bin_df = _pd_ev.DataFrame(_rows)

                    if len(_ev_bin_df) >= 2:
                        _ec1, _ec2 = st.columns([3, 2])

                        with _ec1:
                            # Wykres słupkowy EV vs Yield
                            import altair as _alt_ev
                            _chart_data = _pd_ev.DataFrame({
                                "EV bin":   _ev_bin_df["EV bin"].tolist() * 2,
                                "Wartość":  _ev_bin_df["Avg EV"].tolist() + _ev_bin_df["Yield"].tolist(),
                                "Seria":    ["Avg EV (model)"] * len(_ev_bin_df) + ["Actual Yield (fair)"] * len(_ev_bin_df),
                                "N":        _ev_bin_df["N"].tolist() * 2,
                            })
                            _color_scale = _alt_ev.Scale(
                                domain=["Avg EV (model)", "Actual Yield (fair)"],
                                range=["#4A90D9", "#E8602C"])
                            _ev_chart = (
                                _alt_ev.Chart(_chart_data)
                                .mark_bar(opacity=0.85)
                                .encode(
                                    x=_alt_ev.X("EV bin:N", sort=[r[2] for r in _bins],
                                                axis=_alt_ev.Axis(labelAngle=0)),
                                    y=_alt_ev.Y("Wartość:Q", axis=_alt_ev.Axis(format=".0%"),
                                                title="EV / Yield"),
                                    color=_alt_ev.Color("Seria:N", scale=_color_scale,
                                                        legend=_alt_ev.Legend(orient="top")),
                                    xOffset="Seria:N",
                                    tooltip=["EV bin", "Seria",
                                             _alt_ev.Tooltip("Wartość:Q", format=".1%"),
                                             _alt_ev.Tooltip("N:Q", title="Liczba typów")],
                                )
                                .properties(height=260,
                                            title="EV przewidywany vs rzeczywisty yield (fair odds)")
                            )
                            # Linia zerowa
                            _zero_line = (_alt_ev.Chart(_pd_ev.DataFrame({"y": [0]}))
                                          .mark_rule(color="#888", strokeDash=[4,4])
                                          .encode(y="y:Q"))
                            st.altair_chart(_ev_chart + _zero_line, use_container_width=True)

                        with _ec2:
                            # Tabela z oceną
                            st.markdown("**Tabela per bin EV**")
                            for _, _row in _ev_bin_df.iterrows():
                                _diff = _row["Yield"] - _row["Avg EV"]
                                if _row["Yield"] >= 0.02:
                                    _ocena = "✅ Edge"
                                    _col   = "#4CAF50"
                                elif _row["Yield"] >= -0.05:
                                    _ocena = "🟡 Neutral"
                                    _col   = "#FF9800"
                                else:
                                    _ocena = "🔴 Brak edge"
                                    _col   = "#F44336"
                                st.markdown(
                                    f"<div style='background:var(--bg-card2);border-radius:8px;padding:8px 12px;"
                                    f"margin-bottom:6px;border-left:3px solid {_col}'>"
                                    f"<b>{_row['EV bin']}</b> · N={int(_row['N'])} · "
                                    f"EV={_row['Avg EV']:+.1%} → Yield={_row['Yield']:+.1%} "
                                    f"<span style='color:{_col}'>{_ocena}</span>"
                                    f"</div>",
                                    unsafe_allow_html=True)

                        # Wniosek automatyczny
                        _best_bin = _ev_bin_df.loc[_ev_bin_df["Yield"].idxmax()]
                        _worst_bin = _ev_bin_df.loc[_ev_bin_df["Yield"].idxmin()]
                        _profitable = _ev_bin_df[_ev_bin_df["Yield"] >= 0.02]
                        if len(_profitable) > 0:
                            _min_ev_edge = _profitable["Avg EV"].min()
                            st.success(
                                f"💡 **Rzeczywisty edge zaczyna się od EV ≥ {_min_ev_edge:.0%}** "
                                f"(bin '{_best_bin['EV bin']}': yield {_best_bin['Yield']:+.1%} na {int(_best_bin['N'])} typach). "
                                f"Rozważ podniesienie filtru EV do {_min_ev_edge:.0%} dla tej ligi.")
                        else:
                            st.warning(
                                f"⚠️ Żaden bin EV nie pokazuje dodatniego yieldu na fair odds. "
                                f"Model jest overconfident – shrinkage pomógłby. "
                                f"Najlepszy bin: '{_best_bin['EV bin']}' (yield {_best_bin['Yield']:+.1%}).")
                    else:
                        st.info("Za mało różnych przedziałów EV żeby zbudować wykres.")
            except Exception as _e_ev:
                st.caption(f"EV vs Yield: {_e_ev}")

            # ── Value Distribution – Yield per przedział kursowy ────────────────────
            st.divider()
            st.markdown(
                "<div class='section-header'>📊 Value Distribution – Yield per przedział kursowy"
                "<span style='font-size:.65em;color:#555;font-weight:400;margin-left:10px'>"
                "gdzie model faktycznie zarabia?"
                "</span></div>",
                unsafe_allow_html=True)
            st.caption("Histogram pokazuje skuteczność i yield dla każdego przedziału kursowego. "
                       "Zielony = dodatni yield na fair odds. Pomaga wybrać optymalny zakres kursów do gry.")

            try:
                import sqlite3 as _sq3v, pandas as _pdv
                _dbv = _sq3v.connect(DB_FILE)
                _vdf = _pdv.read_sql_query(
                    """SELECT p_model, fair_odds, trafione, typ
                       FROM zdarzenia
                       WHERE liga = ?
                         AND rynek = '1X2'
                         AND trafione IS NOT NULL
                         AND fair_odds IS NOT NULL
                         AND fair_odds >= 1.10""",
                    _dbv, params=(wybrana_liga,))
                _dbv.close()

                if len(_vdf) < 20:
                    st.info("📭 Za mało danych. Wróć po kilku kolejkach trackingu.")
                else:
                    _vdf["hit"] = _vdf["trafione"].apply(
                        lambda x: 1 if x in (1, True, "TAK", "1", 1.0) else 0)
                    _vdf["yield_act"] = _vdf.apply(
                        lambda r: float(r["fair_odds"]) - 1 if r["hit"] == 1 else -1.0, axis=1)
                    _vdf["fair_odds"] = _vdf["fair_odds"].astype(float)

                    # Biny kursowe
                    _kurs_bins = [
                        (1.10, 1.35, "1.10–1.35"),
                        (1.35, 1.50, "1.35–1.50"),
                        (1.50, 1.65, "1.50–1.65"),
                        (1.65, 1.85, "1.65–1.85"),
                        (1.85, 2.10, "1.85–2.10"),
                        (2.10, 2.50, "2.10–2.50"),
                        (2.50, 9.99, "2.50+"),
                    ]
                    _vrows = []
                    for lo, hi, label in _kurs_bins:
                        _sub = _vdf[(_vdf["fair_odds"] >= lo) & (_vdf["fair_odds"] < hi)]
                        if len(_sub) < 3: continue
                        _vrows.append({
                            "Kurs bin":   label,
                            "N":          len(_sub),
                            "Hit%":       _sub["hit"].mean(),
                            "Yield":      _sub["yield_act"].mean(),
                            "Avg kurs":   _sub["fair_odds"].mean(),
                        })
                    _vbin_df = _pdv.DataFrame(_vrows)

                    if len(_vbin_df) >= 2:
                        import altair as _altv

                        _vc1, _vc2 = st.columns([3, 2])

                        with _vc1:
                            # Wykres: dwie serie – Hit% i Yield – grouped bars
                            _vcd = _pdv.DataFrame({
                                "Kurs bin": _vbin_df["Kurs bin"].tolist() * 2,
                                "Wartość":  _vbin_df["Hit%"].tolist() + _vbin_df["Yield"].tolist(),
                                "Seria":    ["Hit Rate"] * len(_vbin_df) + ["Yield (fair)"] * len(_vbin_df),
                                "N":        _vbin_df["N"].tolist() * 2,
                            })
                            _vcolor = _altv.Scale(
                                domain=["Hit Rate", "Yield (fair)"],
                                range=["#4A90D9", "#E8602C"])
                            _vbars = (
                                _altv.Chart(_vcd)
                                .mark_bar(opacity=0.85)
                                .encode(
                                    x=_altv.X("Kurs bin:N",
                                              sort=[r[2] for r in _kurs_bins],
                                              axis=_altv.Axis(labelAngle=0)),
                                    y=_altv.Y("Wartość:Q",
                                              axis=_altv.Axis(format=".0%"),
                                              title="Hit Rate / Yield"),
                                    color=_altv.Color("Seria:N", scale=_vcolor,
                                                      legend=_altv.Legend(orient="top")),
                                    xOffset="Seria:N",
                                    tooltip=["Kurs bin", "Seria",
                                             _altv.Tooltip("Wartość:Q", format=".1%"),
                                             _altv.Tooltip("N:Q", title="Liczba typów")],
                                )
                                .properties(height=260,
                                            title="Hit Rate i Yield per przedział kursowy")
                            )
                            _vzero = (_altv.Chart(_pdv.DataFrame({"y": [0]}))
                                      .mark_rule(color="#888", strokeDash=[4, 4])
                                      .encode(y="y:Q"))
                            st.altair_chart(_vbars + _vzero, use_container_width=True)

                        with _vc2:
                            st.markdown("**Tabela per przedział**")
                            _best_yield_bin = _vbin_df.loc[_vbin_df["Yield"].idxmax()]
                            for _, _vr in _vbin_df.iterrows():
                                _yc = "#4CAF50" if _vr["Yield"] >= 0.02 else ("#FF9800" if _vr["Yield"] >= -0.05 else "#F44336")
                                _star = " ⭐" if _vr["Kurs bin"] == _best_yield_bin["Kurs bin"] else ""
                                st.markdown(
                                    f"<div style='background:var(--bg-card2);border-radius:8px;padding:7px 12px;"
                                    f"margin-bottom:5px;border-left:3px solid {_yc}'>"
                                    f"<b>{_vr['Kurs bin']}</b>{_star} · N={int(_vr['N'])} · "
                                    f"Hit={_vr['Hit%']:.0%} · "
                                    f"Yield=<span style='color:{_yc}'>{_vr['Yield']:+.1%}</span>"
                                    f"</div>",
                                    unsafe_allow_html=True)

                        # Automatyczny wniosek
                        _pos_bins = _vbin_df[_vbin_df["Yield"] >= 0.02]
                        _neg_bins  = _vbin_df[_vbin_df["Yield"] < -0.10]
                        if len(_pos_bins) > 0 and len(_neg_bins) > 0:
                            _pos_labels = ", ".join(_pos_bins["Kurs bin"].tolist())
                            _neg_labels = ", ".join(_neg_bins["Kurs bin"].tolist())
                            st.success(
                                f"💡 **Edge w kursach {_pos_labels}** "
                                f"(yield {_pos_bins['Yield'].max():+.1%}). "
                                f"Unikaj kursów {_neg_labels} — yield {_neg_bins['Yield'].min():+.1%} na fair odds.")
                        elif len(_pos_bins) > 0:
                            _pos_labels = ", ".join(_pos_bins["Kurs bin"].tolist())
                            st.success(f"💡 Dodatni yield w zakresach: **{_pos_labels}**. "
                                       f"Pozostałe przedziały neutralne.")
                        else:
                            st.warning("⚠️ Brak przedziału kursowego z trwale dodatnim yieldem. "
                                       "Model generuje straty na fair odds we wszystkich zakresach — "
                                       "sprawdź kalibrację lub poczekaj na więcej danych.")
                    else:
                        st.info("Za mało różnych przedziałów kursowych w danych.")
            except Exception as _e_vd:
                st.caption(f"Value Distribution: {_e_vd}")

    # =========================================================================
    # =========================================================================
    # TAB 5 – MULTI-LIGA GLOBAL DASHBOARD
    # =========================================================================
    with tab4:
        st.markdown("<div class='section-header'>🌍 Multi-Liga · Globalny przegląd strategii</div>",
                    unsafe_allow_html=True)
        st.caption("Agregacja predykcji i wyników ze wszystkich lig. Źródło: baza SQLite predykcje.db.")

        # ── Pobierz dane ze wszystkich lig ──────────────────────────────────
        try:
            _g_con = sqlite3.connect(DB_FILE)
            _g_df  = pd.read_sql_query(
                """SELECT liga, rynek, typ, p_model, fair_odds, trafione, data
                   FROM zdarzenia
                   WHERE trafione IS NOT NULL AND rynek = '1X2'""",
                _g_con)
            _g_con.close()
        except Exception:
            _g_df = pd.DataFrame()

        if _g_df.empty:
            st.info("📭 Brak danych. Zacznij śledzić predykcje w zakładce ⚽ Mecze.")
        else:
            _g_df["trafione"] = _g_df["trafione"].astype(int)
            _g_df["fair_odds"] = pd.to_numeric(_g_df["fair_odds"], errors="coerce")
            _g_df["p_model"]   = pd.to_numeric(_g_df["p_model"],   errors="coerce")
            _g_df = _g_df.dropna(subset=["fair_odds","p_model"])

            # ── KPI bar ──────────────────────────────────────────────────────
            _g_n    = len(_g_df)
            _g_traf = int(_g_df["trafione"].sum())
            _g_hr   = _g_traf / _g_n if _g_n > 0 else 0
            _g_roi_s = sum(
                (r["fair_odds"] - 1) if r["trafione"] == 1 else -1
                for _, r in _g_df.iterrows()
            )
            _g_roi_pct = _g_roi_s / _g_n * 100 if _g_n > 0 else 0
            _g_brier   = float(((1 - _g_df["p_model"][_g_df["trafione"]==1]).pow(2).sum()
                               + _g_df["p_model"][_g_df["trafione"]==0].pow(2).sum()) / _g_n)
            _g_ligi    = _g_df["liga"].nunique()

            with st.container(border=True):
                _gk1, _gk2, _gk3, _gk4, _gk5 = st.columns(5)
                _gk1.metric("🌍 Lig",         _g_ligi)
                _gk2.metric("📋 Typów",        _g_n)
                _gk3.metric("🎯 Global Hit Rate", f"{_g_hr:.1%}",
                            delta=f"{_g_hr - 0.50:+.1%} vs 50%",
                            delta_color="normal" if _g_hr > 0.50 else "inverse")
                _gk4.metric("💹 Global ROI",   f"{_g_roi_pct:+.1f}%",
                            delta_color="normal" if _g_roi_pct > 0 else "inverse")
                _gk5.metric("📐 Brier Score",  f"{_g_brier:.4f}",
                            help="0=idealny · 0.333=losowy")

            st.divider()

            # ── League Power Ranking ─────────────────────────────────────────
            st.markdown("### 🏆 League Power Ranking")
            _lig_rows = []
            for _lg_name, _lg_df in _g_df.groupby("liga"):
                _ln  = len(_lg_df)
                _lt  = int(_lg_df["trafione"].sum())
                _lhr = _lt / _ln if _ln > 0 else 0
                _lroi_s = sum(
                    (r["fair_odds"] - 1) if r["trafione"] == 1 else -1
                    for _, r in _lg_df.iterrows()
                )
                _lroi = _lroi_s / _ln * 100
                _lb   = float(((1 - _lg_df["p_model"][_lg_df["trafione"]==1]).pow(2).sum()
                               + _lg_df["p_model"][_lg_df["trafione"]==0].pow(2).sum()) / _ln)
                # Status
                if _lhr >= 0.62 and _lroi > 2:
                    _lst = "✅ Stabilna"
                elif _lhr >= 0.55 or _lroi > 0:
                    _lst = "🟡 OK"
                else:
                    _lst = "⚠️ Słaba"
                _roi_col = "#4CAF50" if _lroi > 2 else ("#FF9800" if _lroi > -2 else "#F44336")
                _hr_col  = "#4CAF50" if _lhr >= 0.62 else ("#FF9800" if _lhr >= 0.55 else "#F44336")
                _lig_rows.append({
                    "liga": _lg_name, "n": _ln, "traf": _lt,
                    "hr": _lhr, "roi": _lroi, "brier": _lb,
                    "status": _lst, "roi_col": _roi_col, "hr_col": _hr_col,
                })
            _lig_rows.sort(key=lambda x: -x["hr"])

            _lr_html_rows = []
            for _lr in _lig_rows:
                _bar_w = int(_lr["hr"] * 100)
                _lr_logo_cell = _liga_img(_lr["liga"], size="26px",
                    style="display:block;margin:0 auto 3px auto")
                # Skrócona nazwa ligi pod logo
                _lr_short = {"Premier League":"Premier L.","La Liga":"La Liga",
                             "Bundesliga":"Bundesliga","Serie A":"Serie A","Ligue 1":"Ligue 1"}.get(_lr["liga"], _lr["liga"])
                _lr_html_rows.append(
                    f"<tr style='border-bottom:1px solid #1e1e2e'>"
                    # Kolumna: logo + nazwa (pionowo)
                    f"<td style='padding:10px 12px;min-width:90px'>"
                    f"<div style='display:flex;flex-direction:column;align-items:center;gap:3px'>"
                    f"{_lr_logo_cell}"
                    f"<span style='font-size:0.72em;font-weight:600;color:#ccc;"
                    f"text-align:center;white-space:nowrap'>{_lr_short}</span>"
                    f"</div></td>"
                    # Typów
                    f"<td style='padding:10px 8px;text-align:center;color:#666;font-size:0.85em'>{_lr['n']}</td>"
                    # Hit Rate (pasek)
                    f"<td style='padding:10px 8px;min-width:100px'>"
                    f"<div style='margin-bottom:3px'>"
                    f"<span style='color:{_lr['hr_col']};font-weight:800;font-size:0.92em'>{_lr['hr']:.1%}</span>"
                    f"</div>"
                    f"<div style='background:var(--bg-card2);border-radius:3px;height:4px'>"
                    f"<div style='background:{_lr['hr_col']};width:{_bar_w}%;height:4px;border-radius:3px'></div>"
                    f"</div></td>"
                    # ROI
                    f"<td style='padding:10px 8px;text-align:right'>"
                    f"<span style='font-weight:800;font-size:0.92em;color:{_lr['roi_col']}'>{_lr['roi']:+.1f}%</span>"
                    f"</td>"
                    # Brier
                    f"<td style='padding:10px 8px;text-align:center;color:#555;font-size:0.80em'>{_lr['brier']:.4f}</td>"
                    f"</tr>"
                )
            st.markdown(
                "<div style='overflow-x:auto;border-radius:8px;border:1px solid var(--border)'>"
                "<table style='width:100%;border-collapse:collapse'>"
                "<thead><tr style='background:#1e1e2e;color:#444;font-size:0.68em;"
                "text-transform:uppercase;letter-spacing:.06em'>"
                "<th style='padding:7px 12px;text-align:center'>Liga</th>"
                "<th style='padding:7px 8px;text-align:center'>Typów</th>"
                "<th style='padding:7px 8px;text-align:left'>Hit Rate</th>"
                "<th style='padding:7px 8px;text-align:right'>ROI</th>"
                "<th style='padding:7px 8px;text-align:center'>Brier</th>"
                f"</tr></thead><tbody>{''.join(_lr_html_rows)}</tbody></table></div>",
                unsafe_allow_html=True
            )

            st.divider()

            # ── Bankroll Kelly per liga ──────────────────────────────────────────
            st.markdown("### Bankroll Kelly per liga i globalnie")
            _ML_KS = float(st.session_state.get("bankroll", 1000.0))
            _ML_KF_base = float(st.session_state.get("kelly_frac", 0.125))
            _ML_KF = _ML_KF_base * KELLY_FRAC_SCALE   # Half-Kelly identycznie jak tab4
            _ML_KMAX_EXP=0.05;_ML_KEV=0.05;_ML_KMIN_ODD=1.35;_ML_KMAX_ODD=3.50
            st.caption(
                f"Identyczna logika co zakładka Skuteczność: Conservative Kelly "
                f"(p×{KELLY_PROB_SCALE}, KF={_ML_KF:.4f}), "
                f"top 3 typy/kolejkę · Pinnacle/B365 1.35–3.50 · start {_ML_KS:.0f} zł/ligę")
            _ml_colors={"Premier League":"#00d4ff","La Liga":"#ff6b35","Bundesliga":"#ffd700","Serie A":"#4CAF50","Ligue 1":"#b44aff"}
            _ml_results={}

            # Sprawdź ile lig ma dane z tab4 w session_state
            _ml_from_ss = {liga: st.session_state[f"ml_bk_{liga}"]
                           for liga in LIGI.keys() if f"ml_bk_{liga}" in st.session_state}
            _ml_to_compute = [liga for liga in LIGI.keys() if liga not in _ml_from_ss]

            if _ml_from_ss:
                _ss_info = ", ".join(_ml_from_ss.keys())
                st.caption(f"⚡ Wyniki z zakładki Skuteczność (live): {_ss_info}"
                           + (f" · obliczam osobno: {', '.join(_ml_to_compute)}" if _ml_to_compute else ""))
                for _liga_ss, _data_ss in _ml_from_ss.items():
                    _ml_results[_liga_ss] = {
                        "bk":       _data_ss["bk"],
                        "bk_flat":  _data_ss["bk_flat"],
                        "roi":      _data_ss["roi"],
                        "roi_flat": _data_ss["roi_flat"],
                        "dd":       _data_ss["dd"],
                        "n":        _data_ss["n"],
                        "hr":       0,  # brak w cache – nie wyświetlamy
                        "bk_vals":  _data_ss["bk_vals"],
                        "flat_vals":[],
                        "kol_nums": _data_ss["kol_nums"],
                    }

            _ml_prog=st.progress(0,text="Laduje dane lig...") if _ml_to_compute else None
            for _ml_i,_ml_liga in enumerate(_ml_to_compute):
                if _ml_prog: _ml_prog.progress(_ml_i/max(len(_ml_to_compute),1),text=f"Obliczam Kelly - {_ml_liga}...")
                try:
                    _ml_csv=LIGI[_ml_liga]["csv_code"]
                    _ml_con2=sqlite3.connect(DB_FILE)
                    _ml_eq=pd.read_sql_query(
                        "SELECT kolejnosc,home,away,typ,trafione,p_model,fair_odds FROM zdarzenia WHERE liga=? AND sezon=? AND rynek='1X2' AND trafione IS NOT NULL ORDER BY kolejnosc,id",
                        _ml_con2,params=(_ml_liga, BIEZACY_SEZON))
                    _ml_con2.close()
                    if len(_ml_eq)<3: continue
                    _ml_hist=load_historical(_ml_csv)
                    if _ml_hist.empty: continue
                    _ml_hc = _ml_hist[_ml_hist["_sezon"]=="biezacy"] if "_sezon" in _ml_hist.columns else _ml_hist
                    _ml_ho=_ml_hc[[c for c in ["HomeTeam","AwayTeam","MaxCH","MaxCD","MaxCA","PSCH","PSCD","PSCA","B365H","B365D","B365A"] if c in _ml_hc.columns]].copy()
                    def _ml_kb(row,ho=_ml_ho):
                        m=ho[(ho["HomeTeam"]==row["home"])&(ho["AwayTeam"]==row["away"])]
                        if m.empty: return None
                        r=m.iloc[-1]
                        def _c(oh,od,oa):
                            try:
                                oh,od,oa=float(oh or 0),float(od or 0),float(oa or 0)
                                if min(oh,od,oa)<=1.01: return None
                                s=1/oh+1/od+1/oa;ih=(1/oh)/s;id_=(1/od)/s;ia=(1/oa)/s
                                t=str(row["typ"])
                                if t=="1": return oh
                                if t=="X": return od
                                if t=="2": return oa
                                if t=="1X": return round(1/(ih+id_),3)
                                if t=="X2": return round(1/(id_+ia),3)
                            except: return None
                        k=_c(r.get("MaxCH"),r.get("MaxCD"),r.get("MaxCA"))
                        if not k: k=_c(r.get("PSCH"),r.get("PSCD"),r.get("PSCA"))
                        if not k: k=_c(r.get("B365H"),r.get("B365D"),r.get("B365A"))
                        return k
                    _ml_eq["kurs_buk"]=_ml_eq.apply(_ml_kb,axis=1)
                    _ml_eq["ev"]=_ml_eq.apply(
                        lambda r: float(r["p_model"])*float(r["kurs_buk"])-1.0
                        if r["kurs_buk"] and not pd.isna(r["kurs_buk"]) and _ML_KMIN_ODD<=float(r["kurs_buk"])<=_ML_KMAX_ODD
                        else None,axis=1)
                    _ml_bk=_ML_KS;_ml_bk_flat=_ML_KS;_ml_bk_vals=[];_ml_flat_vals=[];_ml_kol_nums=[]
                    _ml_peak=_ML_KS;_ml_dd=0.0;_ml_n=0;_ml_traf=0
                    for _kol,_grp in _ml_eq.groupby("kolejnosc"):
                        for _,_r in _grp.iterrows():
                            _ml_bk_flat+=float(_r["fair_odds"])-1 if _r["trafione"]==1 else -1
                        _cand=_grp[_grp["ev"].notna()&(_grp["ev"]>=_ML_KEV)].nlargest(3,"ev")
                        for _,_r in _cand.iterrows():
                            _pt=0.5+(float(_r["p_model"])-0.5)*KELLY_PROB_SCALE  # Conservative -15%
                            _k=float(_r["kurs_buk"]);_b=_k-1
                            _f=min(max((_pt*_b-(1-_pt))/_b,0.0)*_ML_KF,_ML_KMAX_EXP)
                            _st=round(_ml_bk*_f,2)
                            if _st>0:
                                _wyg=int(_r["trafione"])==1
                                _ml_bk+=_st*_b if _wyg else -_st
                                _ml_n+=1;_ml_traf+=int(_wyg)
                            _ml_bk=max(_ml_bk,0.01)
                            if _ml_bk>_ml_peak: _ml_peak=_ml_bk
                            _d=(_ml_peak-_ml_bk)/_ml_peak*100
                            if _d>_ml_dd: _ml_dd=_d
                        _ml_bk_vals.append(round(_ml_bk,2));_ml_flat_vals.append(round(_ml_bk_flat,2));_ml_kol_nums.append(_kol)
                    _ml_results[_ml_liga]={"bk":round(_ml_bk,0),"bk_flat":round(_ml_bk_flat,0),
                        "roi":(_ml_bk-_ML_KS)/_ML_KS*100,"roi_flat":(_ml_bk_flat-_ML_KS)/_ML_KS*100,
                        "dd":_ml_dd,"n":_ml_n,"hr":_ml_traf/_ml_n if _ml_n else 0,
                        "bk_vals":_ml_bk_vals,"flat_vals":_ml_flat_vals,"kol_nums":_ml_kol_nums}
                except Exception: continue
            if _ml_prog: _ml_prog.empty()
            if not _ml_results:
                st.info("Brak danych Kelly per liga.")
            else:
                try:
                    import plotly.graph_objects as _go_ml
                    _fig_ml=_go_ml.Figure()
                    for _lname,_ld in _ml_results.items():
                        _col_ml=_ml_colors.get(_lname,"#888")
                        _lc_ml=_col_ml if _ld["bk"]>=_ML_KS else "#F44336"
                        _fig_ml.add_trace(_go_ml.Scatter(x=_ld["kol_nums"],y=_ld["bk_vals"],mode="lines",name=_lname,
                            line=dict(color=_lc_ml,width=2.5),
                            hovertemplate=f"<b>{_lname}</b><br>Kolejka %{{x}}<br>Bankroll: %{{y:.0f}} zl<extra></extra>"))
                    _fig_ml.add_hline(y=_ML_KS,line_dash="dot",line_color="#444",line_width=1.5,
                        annotation_text="Start 1000 zl",annotation_font_color="#555",annotation_position="bottom right")
                    _fig_ml.update_layout(paper_bgcolor="#0d0f14",plot_bgcolor="#0d0f14",height=300,
                        margin=dict(l=40,r=20,t=10,b=40),
                        xaxis=dict(title="Kolejka",color="#555",gridcolor="#1a1c24",tickmode="linear",dtick=5),
                        yaxis=dict(title="Bankroll (zl)",color="#555",gridcolor="#1a1c24"),
                        legend=dict(bgcolor="#0d0f14",bordercolor="#2a2a3a",borderwidth=1,
                            font=dict(color="#aaa",size=11),orientation="h",yanchor="bottom",y=1.01,xanchor="left",x=0),
                        font=dict(color="#888"),hovermode="x unified")
                    st.plotly_chart(_fig_ml,use_container_width=True,config={"displayModeBar":False})
                except ImportError:
                    _ml_cd={}
                    for _ln,_ld in _ml_results.items():
                        if _ld["kol_nums"]: _ml_cd[_ln]=pd.Series(_ld["bk_vals"],index=_ld["kol_nums"],name=_ln)
                    if _ml_cd: st.line_chart(pd.DataFrame(_ml_cd),height=240)
                # KPI globalne
                _ml_total_pnl=sum(_ld["bk"]-_ML_KS for _ld in _ml_results.values())
                _ml_global_bk=_ML_KS*len(_ml_results)+_ml_total_pnl
                _ml_global_roi=_ml_total_pnl/(_ML_KS*len(_ml_results))*100
                _ml_g1,_ml_g2,_ml_g3=st.columns(3)
                _ml_g1.metric("Globalny bankroll",f"{_ml_global_bk:.0f} zl",delta=f"{_ml_global_roi:+.1f}%",delta_color="normal")
                _ml_g2.metric("Lig na plusie",f"{sum(1 for _ld in _ml_results.values() if _ld['bk']>=_ML_KS)}/{len(_ml_results)}")
                _ml_best=max(_ml_results,key=lambda x:_ml_results[x]["roi"]) if _ml_results else "-"
                _ml_g3.metric("Najlepsza liga",_ml_best,delta=f"{_ml_results[_ml_best]['roi']:+.1f}%" if _ml_best in _ml_results else None)
                # Tabela
                _ml_rows=[]
                for _lname,_ld in sorted(_ml_results.items(),key=lambda x:-x[1]["bk"]):
                    _lc   = _ml_colors.get(_lname,"#888")
                    _pnl  = _ld["bk"] - _ML_KS
                    _bkc  = "#4CAF50" if _ld["bk"] >= _ML_KS else "#F44336"
                    _roic = "#4CAF50" if _ld["roi"] >= 0     else "#F44336"
                    _is_ok = _ld["bk"] >= _ML_KS
                    _status_dot = (
                        f"<span style='display:inline-block;width:7px;height:7px;"
                        f"border-radius:50%;background:{'#4CAF50' if _is_ok else '#F44336'};"
                        f"margin-right:5px;vertical-align:middle'></span>")
                    _ml_logo = _liga_img(_lname, size="22px",
                        style="display:block;margin:0 auto 2px auto")
                    _ml_short = {"Premier League":"PL","La Liga":"LaLiga",
                                 "Bundesliga":"BL","Serie A":"SA","Ligue 1":"L1"}.get(_lname, _lname[:5])
                    _ml_rows.append(
                        f"<tr style='border-bottom:1px solid #13141c'>"
                        # Liga: logo + skrót
                        f"<td style='padding:10px 12px;min-width:70px'>"
                        f"<div style='display:flex;flex-direction:column;align-items:center;gap:2px'>"
                        f"{_ml_logo}"
                        f"<span style='font-size:0.70em;color:#aaa;font-weight:600;white-space:nowrap'>{_ml_short}</span>"
                        f"</div></td>"
                        # Bankroll Kelly – główna liczba
                        f"<td style='padding:10px 12px;text-align:center'>"
                        f"<div style='font-size:1.05em;font-weight:800;color:{_bkc}'>{_ld['bk']:.0f} zł</div>"
                        f"<div style='font-size:0.72em;color:#555;margin-top:1px'>start {_ML_KS:.0f}</div>"
                        f"</td>"
                        # PnL + ROI badge
                        f"<td style='padding:10px 10px;text-align:center'>"
                        f"<div style='font-weight:700;color:{_roic}'>{_pnl:+.0f} zł</div>"
                        f"<div style='display:inline-block;margin-top:3px;padding:1px 7px;"
                        f"border-radius:10px;font-size:0.72em;font-weight:700;"
                        f"background:{'#0d2a0d' if _is_ok else '#2a0d0d'};"
                        f"color:{_roic}'>{_ld['roi']:+.1f}%</div>"
                        f"</td>"
                        # Typów / Hit
                        f"<td style='padding:10px 10px;text-align:center'>"
                        f"<div style='color:#aaa;font-size:0.85em'>{_ld['n']}</div>"
                        f"<div style='color:#666;font-size:0.76em'>{_ld['hr']:.0%} hit</div>"
                        f"</td>"
                        # Max DD
                        f"<td style='padding:10px 10px;text-align:center'>"
                        f"<span style='color:#F44336;font-size:0.82em;font-weight:600'>"
                        f"↓{_ld['dd']:.1f}%</span></td>"
                        # Flat (ref)
                        f"<td style='padding:10px 10px;text-align:center;color:#444;font-size:0.80em'>"
                        f"{_ld['bk_flat']:.0f}</td>"
                        f"</tr>"
                    )
                st.markdown(
                    "<div style='overflow-x:auto;border-radius:8px;border:1px solid var(--border);margin-top:8px'>"
                    "<table style='width:100%;border-collapse:collapse'>"
                    "<thead><tr style='background:var(--bg-card);color:#444;font-size:0.68em;"
                    "text-transform:uppercase;letter-spacing:.05em'>"
                    "<th style='padding:7px 12px;text-align:center'>Liga</th>"
                    "<th style='padding:7px 12px;text-align:center'>Bankroll</th>"
                    "<th style='padding:7px 12px;text-align:center'>PnL / ROI</th>"
                    "<th style='padding:7px 12px;text-align:center'>Typów / Hit</th>"
                    "<th style='padding:7px 12px;text-align:center'>Max DD</th>"
                    "<th style='padding:7px 12px;text-align:center'>Flat</th>"
                    f"</tr></thead><tbody>{''.join(_ml_rows)}</tbody></table></div>",
                    unsafe_allow_html=True)
                st.caption(f"Globalnie: start {_ML_KS*len(_ml_results):.0f} zl - koncowy {_ml_global_bk:.0f} zl - PnL {_ml_total_pnl:+.0f} zl - ROI {_ml_global_roi:+.1f}%")

            st.divider()

            # ── Hit Rate per typ (1/X/2) globalnie ───────────────────────────
            st.markdown("### 🎲 Rozkład typów globalnie")
            _g_typ_df = (_g_df.groupby("typ")
                         .agg(n=("trafione","count"), traf=("trafione","sum"),
                              p_avg=("p_model","mean"), fo_avg=("fair_odds","mean"))
                         .reset_index())
            _g_typ_df["hr"] = _g_typ_df["traf"] / _g_typ_df["n"]
            _g_typ_df["roi"] = _g_typ_df.apply(
                lambda r_: (r_["traf"] * (r_["fo_avg"] - 1) - (r_["n"] - r_["traf"])) / r_["n"] * 100, axis=1)
            _typ_cols = st.columns(len(_g_typ_df))
            for _ti, (_, _tr) in enumerate(_g_typ_df.sort_values("n", ascending=False).iterrows()):
                _tc = "#4CAF50" if _tr["hr"] >= _tr["p_avg"] - 0.03 else "#FF9800"
                _roic = "#4CAF50" if _tr["roi"] > 0 else "#F44336"
                _typ_cols[_ti].markdown(
                    f"<div style='text-align:center;background:var(--stat-bg);border:1px solid #1e2433;"
                    f"border-radius:8px;padding:10px 6px'>"
                    f"<div style='font-size:1.4em;font-weight:800;color:var(--text-primary)'>{_tr['typ']}</div>"
                    f"<div style='font-size:1.1em;font-weight:700;color:{_tc}'>{_tr['hr']:.0%}</div>"
                    f"<div style='font-size:0.72em;color:#555'>HR · p̄={_tr['p_avg']:.0%}</div>"
                    f"<div style='font-size:0.8em;color:{_roic};margin-top:3px'>ROI {_tr['roi']:+.1f}%</div>"
                    f"<div style='font-size:0.68em;color:#444'>n={int(_tr['n'])}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )


        # TAB 6 – BACKTEST
    # =========================================================================
    with tab5:
        st.subheader("🧪 Backtest – Symulacja walk-forward")
        st.caption(
            "Model trenowany kolejka po kolejce – dla K-tej kolejki widzi tylko dane do K-1. "
            "Zero look-ahead bias. Źródło: football-data.co.uk (jeden plik CSV per sezon)."
        )

        if not _BT_OK:
            st.error("❌ Plik `backtest.py` nie został znaleziony w folderze aplikacji.")
        else:
            BT_DB   = "backtest_results.db"
            BT_LIGA = LIGI[wybrana_liga]["csv_code"]

            SEZONY = {
                "E0":  [("2324","2223"),("2425","2324")],
                "SP1": [("2324","2223"),("2425","2324")],
                "D1":  [("2324","2223"),("2425","2324")],
                "I1":  [("2324","2223"),("2425","2324")],
                "F1":  [("2324","2223"),("2425","2324")],
            }
            LABELS = {"2324":"2023/24","2425":"2024/25","2223":"2022/23"}
            dostepne = SEZONY.get(BT_LIGA, [("2324","2223")])

            cfg1, cfg2, cfg3 = st.columns([2,2,3])
            with cfg1:
                opcje = [f"{LABELS.get(t,'?')} (test)" for t,_ in dostepne]
                sel   = st.selectbox("Sezon testowy", opcje, index=len(opcje)-1, key="bt_sezon")
                idx   = opcje.index(sel)
                BT_SEZON_TEST, BT_SEZON_PREV = dostepne[idx]
            with cfg2:
                st.metric("Sezon bazowy", LABELS.get(BT_SEZON_PREV, BT_SEZON_PREV))
                st.caption(f"Liga: {wybrana_liga} ({BT_LIGA})")
            with cfg3:
                bt_df = _bt.load_results(BT_LIGA, BT_SEZON_TEST, BT_DB)
                has_r = not bt_df.empty
                if has_r:
                    st.success(f"✅ Wyniki: **{len(bt_df)}** meczów ({bt_df['kolejka'].nunique()} kolejek)")
                else:
                    st.info("Brak wyników – uruchom backtest.")

            st.divider()
            run_col, info_col = st.columns([2,3])
            with run_col:
                run_bt = st.button(
                    "▶️ Uruchom backtest" if not has_r else "🔄 Uruchom ponownie",
                    type="primary",
                    help="Pobiera dane z football-data.co.uk i symuluje pełny sezon (~30-60s)."
                )
            with info_col:
                st.markdown(
                    "Dla każdej kolejki K model trenowany na kolejkach 1..K−1 "
                    "+ blend poprzedniego sezonu. Predykcja na K, weryfikacja vs prawdziwy wynik."
                )

            if run_bt:
                prog = st.progress(0.0)
                ptxt = st.empty()
                def _cb(f,m): prog.progress(f); ptxt.caption(f"⏳ {m}")
                with st.spinner("Symulacja w toku..."):
                    res = _bt.run_backtest(BT_LIGA, BT_SEZON_TEST, BT_SEZON_PREV, BT_DB, _cb)
                prog.empty(); ptxt.empty()
                if "error" in res:
                    st.error(f"❌ {res['error']}")
                else:
                    st.success(
                        f"✅ Zakończono · **{res['n']}** meczów · "
                        f"hit **{res['hit_rate']:.1%}** · "
                        f"Brier **{res['brier']:.4f}** · "
                        f"ROI **{res['roi_pct']:+.1f}%**"
                    )
                    if "df" in res:
                        st.download_button(
                            "⬇️ Pobierz CSV",
                            data=res["df"].to_csv(index=False, decimal=","),
                            file_name=f"backtest_{BT_LIGA}_{BT_SEZON_TEST}.csv",
                            mime="text/csv", key="bt_fresh"
                        )
                    bt_df = _bt.load_results(BT_LIGA, BT_SEZON_TEST, BT_DB)
                    has_r = not bt_df.empty

            if has_r:
                summ = _bt.summary(BT_LIGA, BT_SEZON_TEST, BT_DB)
                if summ:
                    st.divider()
                    st.markdown("### 📊 Wyniki out-of-sample")
                    m1,m2,m3,m4,m5 = st.columns(5)
                    m1.metric("Meczów",        summ["n"])
                    m2.metric("Hit Rate",      f"{summ['hit_rate']:.1%}")
                    m3.metric("Brier ↓",       f"{summ['brier']:.4f}",
                              delta=f"BSS {summ['bss']:+.3f}",
                              delta_color="normal" if summ["bss"]>0 else "inverse")
                    m4.metric("ROI (fair)",    f"{summ['roi_pct']:+.1f}%",
                              delta_color="normal" if summ["roi_pct"]>0 else "inverse")
                    # ROI po realnych kursach (jeśli dostępne w DB)
                    _roi_ps   = summ.get("roi_ps_pct")
                    _roi_b365 = summ.get("roi_b365_pct")
                    if _roi_ps is not None or _roi_b365 is not None:
                        _rc1, _rc2 = st.columns(2)
                        if _roi_ps is not None:
                            _rc1.metric("ROI Pinnacle", f"{_roi_ps:+.1f}%",
                                        help="ROI po kursach Pinnacle (DC-corrected, z marżą ~2%)",
                                        delta_color="normal" if _roi_ps>0 else "inverse")
                        if _roi_b365 is not None:
                            _rc2.metric("ROI Bet365", f"{_roi_b365:+.1f}%",
                                        help="ROI po kursach Bet365 (DC-corrected, z marżą ~5%)",
                                        delta_color="normal" if _roi_b365>0 else "inverse")

                # ── Porównanie sezonów side-by-side ────────────────────────
                st.divider()
                st.markdown("### 📊 Porównanie sezonów")
                _all_seasons = SEZONY.get(BT_LIGA, [])
                _comp_data = []
                for _cs_test, _cs_prev in _all_seasons:
                    _cs_df = _bt.load_results(BT_LIGA, _cs_test, BT_DB)
                    if _cs_df.empty: continue
                    _cs_summ = _bt.summary(BT_LIGA, _cs_test, BT_DB)
                    _cs_n    = _cs_summ.get("n", 0)
                    if _cs_n == 0: continue
                    _comp_data.append({
                        "Sezon": LABELS.get(_cs_test, _cs_test),
                        "Meczów": _cs_n,
                        "Hit Rate": f"{_cs_summ.get('hit_rate', 0):.1%}",
                        "Brier ↓": f"{_cs_summ.get('brier', 0):.4f}",
                        "BSS": f"{_cs_summ.get('bss', 0):+.3f}",
                        "ROI fair": f"{_cs_summ.get('roi_pct', 0):+.1f}%",
                        "ROI PS": f"{_cs_summ.get('roi_ps_pct', 0):+.1f}%" if _cs_summ.get('roi_ps_pct') is not None else "–",
                    })
                if len(_comp_data) >= 2:
                    _cdf = pd.DataFrame(_comp_data)
                    # Highlight best season per metric
                    st.dataframe(
                        _cdf.set_index("Sezon"),
                        use_container_width=True,
                    )
                    # Equity curves overlay
                    _eq_overlay = {}
                    for _cs_test, _ in _all_seasons:
                        _cs_df2 = _bt.load_results(BT_LIGA, _cs_test, BT_DB)
                        if _cs_df2.empty: continue
                        _cs_df2 = _cs_df2.sort_values(["kolejka","id"] if "id" in _cs_df2.columns else ["kolejka"])
                        _fo_col = [round(1/r["p_typ"],2) if r["p_typ"]>0 else 1.0
                                   for _,r in _cs_df2.iterrows()]
                        _pnl = [(fo-1) if tr==1 else -1
                                for fo, tr in zip(_fo_col, _cs_df2["trafiony"])]
                        _kap = 1000.0
                        _kap_vals = []
                        for _pv in _pnl:
                            _kap += _pv; _kap_vals.append(round(_kap,2))
                        _eq_overlay[LABELS.get(_cs_test,_cs_test)] = _kap_vals[:len(_cs_df2)]
                    if _eq_overlay:
                        st.markdown("**📈 Equity Curve – porównanie sezonów (flat 1j)**")
                        _max_len = max(len(v) for v in _eq_overlay.values())
                        _eq_comp_df = pd.DataFrame({
                            k: pd.Series(v) for k, v in _eq_overlay.items()
                        })
                        st.line_chart(_eq_comp_df, height=220)
                elif len(_comp_data) == 1:
                    st.info("Uruchom backtest dla drugiego sezonu aby zobaczyć porównanie.")
                else:
                    st.info("Brak danych backtestów. Uruchom backtest dla co najmniej jednego sezonu.")
                    m5.metric("Kolejek",       summ["per_kolejka"]["kolejka"].max()
                                               if not summ["per_kolejka"].empty else "–")

                    # ── Alt markets results ───────────────────────────────
                    if hasattr(_bt, "load_alt_results"):
                        _alt_df_bt = _bt.load_alt_results(BT_LIGA, BT_SEZON_TEST, BT_DB)
                        if not _alt_df_bt.empty:
                            st.divider()
                            st.markdown("### ⚡ Rynki Alternatywne – wyniki backtestów")
                            st.caption("Kelly 1/4 na fair odds. ⚠️ Brak marży – real ROI będzie niższy o ~3-8%.")
                            _alt_rynki = _alt_df_bt["rynek"].unique()
                            _alt_cols_bt = st.columns(min(len(_alt_rynki), 4))
                            for _ai, _ark in enumerate(_alt_rynki):
                                _ardf = _alt_df_bt[_alt_df_bt["rynek"] == _ark]
                                _arn  = len(_ardf)
                                _arhit = int(_ardf["trafiony"].sum())
                                _arroi_flat = float(sum(
                                    (r["fair_odds"]-1) if r["trafiony"] else -1
                                    for _, r in _ardf.iterrows())) / _arn * 100
                                _arroi_kelly = float(_ardf["kelly_pnl"].sum())
                                _arc = _alt_cols_bt[_ai % len(_alt_cols_bt)]
                                _kc = "#4CAF50" if _arroi_kelly > 0 else "#F44336"
                                _arc.markdown(
                                    f"<div style='background:var(--stat-bg);border:1px solid {_kc}33;"
                                    f"border-radius:8px;padding:10px;text-align:center'>"
                                    f"<div style='font-size:0.82em;color:#888;margin-bottom:4px'>{_ark}</div>"
                                    f"<div style='font-size:0.88em;color:#ccc'>N={_arn} · HR {_arhit/_arn:.0%}</div>"
                                    f"<div style='font-size:0.82em;color:#888;margin-top:3px'>"
                                    f"ROI flat: <b style='color:{'#4CAF50' if _arroi_flat>0 else '#F44336'}'>"
                                    f"{_arroi_flat:+.1f}%</b></div>"
                                    f"<div style='font-size:1.0em;font-weight:bold;margin-top:4px;color:{_kc}'>"
                                    f"Kelly PnL: {_arroi_kelly:+.3f} j</div></div>",
                                    unsafe_allow_html=True)
                            with st.expander("📋 Szczegóły per linia", expanded=False):
                                _agg = _alt_df_bt.groupby(["rynek","typ"]).agg(
                                    n=("trafiony","count"), hit=("trafiony","sum"),
                                    p_avg=("p_model","mean"),
                                    kelly_pnl=("kelly_pnl","sum"),
                                    brier=("brier_bin","mean"),
                                ).reset_index()
                                _agg["HR"]       = (_agg["hit"] / _agg["n"]).map("{:.0%}".format)
                                _agg["P model"]  = _agg["p_avg"].map("{:.0%}".format)
                                _agg["Kelly PnL"]= _agg["kelly_pnl"].map("{:+.3f}j".format)
                                _agg["Brier"]    = _agg["brier"].map("{:.4f}".format)
                                st.dataframe(
                                    _agg[["rynek","typ","n","hit","HR","P model","Brier","Kelly PnL"]]
                                    .rename(columns={"rynek":"Rynek","typ":"Typ","n":"N","hit":"Traf."}),
                                    use_container_width=True, hide_index=True)
                                st.caption(
                                "Kelly na fair odds (bez marży bukmachera – real ~3-8% niżej). "
                                "Frakcje po analizie backtestów: Gole/BTTS f=0.075 · Kartki/Rożne f=0.05. "
                                "Kalibracja liniowa p_model zastosowana (slope=0.88, intercept=0.06). "
                                "EV<5% odfiltrowane.")

                    st.divider()

                    st.divider()
                    st.markdown("### 🎯 Skuteczność per typ")
                    per_typ = summ["per_typ"]
                    if not per_typ.empty:
                        tc = {"1":"#2196F3","X":"#FF9800","2":"#E91E63","1X":"#9C27B0","X2":"#00BCD4"}
                        rows_t=[]
                        for _,tr in per_typ.sort_values("n",ascending=False).iterrows():
                            c=tc.get(tr["typ"],"#888"); bw=int(tr["hit"]*100)
                            bc="#4CAF50" if tr["brier"]<0.22 else("#FF9800" if tr["brier"]<0.27 else "#F44336")
                            grp=bt_df[bt_df["typ"]==tr["typ"]]
                            rv=0.0
                            if not grp.empty:
                                rs=sum((round(1/r["p_typ"],2)-1) if r["trafiony"]==1 else -1
                                       for _,r in grp.iterrows())
                                rv=rs/len(grp)*100
                            rc="#4CAF50" if rv>0 else("#F44336" if rv<-5 else "#888")
                            rows_t.append(
                                f"<tr>"
                                f"<td style='padding:6px 10px;text-align:center'>"
                                f"<span style='background:{c};color:white;padding:2px 10px;"
                                f"border-radius:10px;font-weight:bold'>{tr['typ']}</span></td>"
                                f"<td style='padding:6px 10px;text-align:center;color:#888'>{int(tr['n'])}</td>"
                                f"<td style='padding:6px 10px;width:120px'>"
                                f"<div style='display:flex;align-items:center;gap:5px'>"
                                f"<div style='flex:1;background:var(--bg-card2);border-radius:3px;height:5px'>"
                                f"<div style='background:{c};width:{bw}%;height:5px;border-radius:3px'></div></div>"
                                f"<span style='color:{c};font-size:0.82em'>{tr['hit']:.0%}</span>"
                                f"</div></td>"
                                f"<td style='padding:6px 10px;text-align:center;color:{bc};font-weight:bold'>{tr['brier']:.4f}</td>"
                                f"<td style='padding:6px 10px;text-align:center;color:{rc};font-weight:bold'>{rv:+.1f}%</td>"
                                f"</tr>"
                            )
                        st.markdown(
                            f"<div style='overflow-x:auto;border-radius:8px;border:1px solid #2a2a2a'>"
                            f"<table style='width:100%;border-collapse:collapse;font-size:0.87em'>"
                            f"<thead><tr style='background:var(--bg-card2);color:#666;font-size:0.73em;text-transform:uppercase'>"
                            f"<th style='padding:7px 10px'>Typ</th><th style='padding:7px 10px;text-align:center'>N</th>"
                            f"<th style='padding:7px 10px;text-align:left'>Hit Rate</th>"
                            f"<th style='padding:7px 10px;text-align:center'>Brier ↓</th>"
                            f"<th style='padding:7px 10px;text-align:center'>ROI</th>"
                            f"</tr></thead><tbody>{''.join(rows_t)}</tbody></table></div>",
                            unsafe_allow_html=True,
                        )

                    st.divider()
                    st.markdown("### 🎯 Kalibracja out-of-sample")
                    kal = summ["kalibracja"]
                    if not kal.empty:
                        kal_rows=[]
                        for _,rk in kal.iterrows():
                            dc="#4CAF50" if abs(rk["rozb"])<0.05 else("#FF9800" if abs(rk["rozb"])<0.12 else "#F44336")
                            ic="✅" if abs(rk["rozb"])<0.05 else("⚠️" if abs(rk["rozb"])<0.12 else "❌")
                            kal_rows.append(
                                f"<tr><td style='padding:5px 8px;font-weight:bold'>{rk['przedzial']}</td>"
                                f"<td style='padding:5px 8px;text-align:center;color:#888'>{int(rk['n'])}</td>"
                                f"<td style='padding:5px 8px;text-align:center;color:#2196F3'>{rk['p_mean']:.1%}</td>"
                                f"<td style='padding:5px 8px;text-align:center;color:#4CAF50'>{rk['hit']:.1%}</td>"
                                f"<td style='padding:5px 8px;text-align:center;color:{dc};font-weight:bold'>"
                                f"{ic} {rk['rozb']:+.1%}</td></tr>"
                            )
                        st.markdown(
                            f"<div style='overflow-x:auto;border-radius:8px;border:1px solid #2a2a2a'>"
                            f"<table style='width:100%;border-collapse:collapse;font-size:0.85em'>"
                            f"<thead><tr style='background:var(--bg-card2);color:#666;font-size:0.72em;text-transform:uppercase'>"
                            f"<th style='padding:6px 8px'>Przedział p</th>"
                            f"<th style='padding:6px 8px;text-align:center'>N</th>"
                            f"<th style='padding:6px 8px;text-align:center'>P model</th>"
                            f"<th style='padding:6px 8px;text-align:center'>Hit rzecz.</th>"
                            f"<th style='padding:6px 8px;text-align:center'>Δ</th>"
                            f"</tr></thead><tbody>{''.join(kal_rows)}</tbody></table></div>",
                            unsafe_allow_html=True,
                        )
                        avg_bias=float(kal["rozb"].mean())
                        if abs(avg_bias)<0.03: st.success(f"✅ Dobrze skalibrowany (bias {avg_bias:+.1%})")
                        elif avg_bias>0: st.warning(f"⚠️ Zbyt ostrożny (bias {avg_bias:+.1%})")
                        else: st.warning(f"⚠️ Zbyt pewny siebie (bias {avg_bias:+.1%})")

                    st.divider()
                    dl_col,_ = st.columns([2,3])
                    with dl_col:
                        st.download_button(
                            "⬇️ Pobierz CSV z predykcjami",
                            data=bt_df.to_csv(index=False, decimal=","),
                            file_name=f"backtest_{BT_LIGA}_{BT_SEZON_TEST}.csv",
                            mime="text/csv", key="bt_saved"
                        )

        # ═══════════════════════════════════════════════════════════════════
        # SEKCJA: DIAGNOSTYKA MODELU
        # ═══════════════════════════════════════════════════════════════════
        st.divider()
        st.markdown(
            "<div class='section-header'>🔍 Diagnostyka modelu"
            "<span style='font-size:.65em;color:#555;font-weight:400;margin-left:10px'>"
            "root cause analysis · kalibracja · bias per druzyna"
            "</span></div>",
            unsafe_allow_html=True)

        _diag_src = "backtest"
        try:
            _diag_df = bt_df.copy() if (not bt_df.empty and "trafiony" in bt_df.columns) else pd.DataFrame()
        except Exception:
            _diag_df = pd.DataFrame()

        if _diag_df.empty:
            try:
                _d_con = sqlite3.connect(DB_FILE)
                _diag_df = pd.read_sql_query(
                    """SELECT home as dom, away as gosc, typ, p_model as p_typ,
                              fair_odds, trafione as trafiony
                       FROM zdarzenia
                       WHERE liga=? AND rynek='1X2' AND trafione IS NOT NULL""",
                    _d_con, params=(wybrana_liga,))
                _d_con.close()
                _diag_df["trafiony"] = _diag_df["trafiony"].map({1:"TAK",0:"NIE",True:"TAK",False:"NIE"})
                _diag_src = "live"
                _diag_df["p_home"] = _diag_df["p_typ"]
                _diag_df["p_draw"] = 0.25
                _diag_df["p_away"] = 1 - _diag_df["p_typ"] - 0.25
            except Exception:
                _diag_df = pd.DataFrame()

        if _diag_df.empty:
            st.info("Uruchom backtest powyzej aby zobaczyc diagnostyke modelu.")
        else:
            # Normalizuj kolumny nazw (DB: home/away, CSV: dom/gosc)
            if "home" in _diag_df.columns and "dom" not in _diag_df.columns:
                _diag_df = _diag_df.rename(columns={"home": "dom", "away": "gosc"})
            for _c in ["p_typ","p_home","p_draw","p_away"]:
                if _c in _diag_df.columns:
                    _diag_df[_c] = pd.to_numeric(
                        _diag_df[_c].astype(str).str.replace(",","."), errors="coerce")
            # trafiony może być int (0/1 z DB) lub string ("TAK"/"NIE" z CSV)
            _diag_df["hit"] = _diag_df["trafiony"].apply(
                lambda x: 1 if x in (1, True, "TAK", "1", 1.0) else 0)
            _diag_df = _diag_df.dropna(subset=["p_typ"])
            _n_total  = len(_diag_df)
            _n_hit    = int(_diag_df["hit"].sum())
            _n_miss   = _n_total - _n_hit
            _hr_total = _n_hit / _n_total if _n_total else 0
            _src_label = "backtest walk-forward" if _diag_src=="backtest" else "live - biezacy sezon"

            st.markdown(f"**Analiza {_n_total} predykcji** - zrodlo: {_src_label}")
            _dc1, _dc2, _dc3, _dc4 = st.columns(4)
            _dc1.metric("Predykcji", _n_total)
            _dc2.metric("Trafione", _n_hit, delta=f"{_hr_total:.1%}")
            _dc3.metric("Chybione", _n_miss, delta=f"{_n_miss/_n_total:.1%}", delta_color="inverse")
            _brier_all = float(((_diag_df["p_typ"] - _diag_df["hit"])**2).mean())
            _dc4.metric("Brier Score", f"{_brier_all:.4f}", help="Im nizszy tym lepiej. Losowy=0.2500.")

            # ROOT CAUSE
            st.markdown("#### Root cause - skad pochodza bledy?")

            def _classify_err(row):
                if row["hit"] == 1: return "correct"
                ph  = float(row.get("p_home", row["p_typ"]))
                pd_ = float(row.get("p_draw", 0.25))
                pa  = float(row.get("p_away", max(1-ph-0.25, 0)))
                typ = str(row["typ"]); p = float(row["p_typ"])
                if typ == "1":
                    if ph >= 0.80: return "kalibracja"
                    if pa > 0.28:  return "lambda_error"
                    return "kalibracja"
                if typ == "2":
                    if pa >= 0.80: return "kalibracja"
                    if ph > 0.28:  return "lambda_error"
                    return "kalibracja"
                if typ == "1X":
                    if pa > 0.30: return "lambda_error"
                    return "wariancja"
                if typ == "X2":
                    if ph > 0.30: return "lambda_error"
                    return "wariancja"
                return "wariancja"

            _diag_df["err_cat"] = _diag_df.apply(_classify_err, axis=1)
            _miss_df = _diag_df[_diag_df["hit"]==0]
            _var_cnt  = int((_miss_df["err_cat"]=="wariancja").sum())
            _cal_cnt  = int((_miss_df["err_cat"]=="kalibracja").sum())
            _lam_cnt  = int((_miss_df["err_cat"]=="lambda_error").sum())

            _rc1, _rc2, _rc3 = st.columns(3)
            for _col_rc, _cnt_rc, _label_rc, _col_hex, _desc_rc in [
                (_rc1, _var_cnt,  "Wariancja (pech)",      "#888888",
                 "Model mial racje stat. Nienaprawialne."),
                (_rc2, _cal_cnt,  "Kalibracja (za pewny)", "#FF9800",
                 "Model przypisal zbyt wysokie p. Shrinkage moze pomoc."),
                (_rc3, _lam_cnt,  "Lambda-blad (sila)",    "#F44336",
                 "Niedoszacowano sile rywala. Wiecej danych moze pomoc."),
            ]:
                _pct_rc = _cnt_rc / _n_miss * 100 if _n_miss else 0
                _col_rc.markdown(
                    f"<div style='background:var(--bg-card);border:1px solid var(--border);"
                    f"border-top:3px solid {_col_hex};border-radius:8px;padding:12px 14px'>"
                    f"<div style='font-size:0.82em;font-weight:700;color:{_col_hex}'>{_label_rc}</div>"
                    f"<div style='font-size:1.9em;font-weight:800;color:#eee;margin:4px 0'>{_cnt_rc}</div>"
                    f"<div style='font-size:0.7em;color:#555'>{_pct_rc:.0f}% bledow - {_desc_rc}</div>"
                    f"</div>", unsafe_allow_html=True)

            # KALIBRACJA SZCZEGOLOWA
            st.markdown("#### Kalibracja - expected vs actual hit rate")
            import numpy as _npd
            _bins_c = _npd.arange(0.40, 1.01, 0.07)
            _diag_df["p_bin"] = pd.cut(_diag_df["p_typ"], bins=_bins_c)
            _cal_g = _diag_df.groupby("p_bin", observed=True)["hit"].agg(["mean","count"]).reset_index()
            _cal_g.columns = ["p_bin","actual","n"]
            _cal_g["expected"] = [float(b.mid) for b in _cal_g["p_bin"]]
            _cal_g["err"] = _cal_g["actual"] - _cal_g["expected"]
            _cal_g = _cal_g[_cal_g["n"] >= 4]
            if not _cal_g.empty:
                _cal_rows = []
                for _, _cr in _cal_g.iterrows():
                    _e = float(_cr["err"])
                    _ec = "#4CAF50" if abs(_e)<0.05 else ("#FF9800" if abs(_e)<0.10 else "#F44336")
                    _bw_a = int(float(_cr["actual"])*100); _bw_e = int(float(_cr["expected"])*100)
                    _cal_rows.append(
                        f"<tr>"
                        f"<td style='padding:7px 10px;color:#888;font-size:0.82em'>"
                        f"{float(_cr['expected']):.0%}</td>"
                        f"<td style='padding:7px 10px;width:42%'>"
                        f"<div style='position:relative;background:var(--bg-card2);border-radius:3px;height:8px'>"
                        f"<div style='background:#444;width:{_bw_e}%;height:8px;border-radius:3px;position:absolute'></div>"
                        f"<div style='background:#00d4ff;width:{_bw_a}%;height:8px;border-radius:3px;position:absolute;opacity:.85'></div>"
                        f"</div></td>"
                        f"<td style='padding:7px 10px;text-align:center;color:#00d4ff;font-size:0.84em'>{float(_cr['actual']):.1%}</td>"
                        f"<td style='padding:7px 10px;text-align:center;color:{_ec};font-weight:700;font-size:0.84em'>{_e:+.1%}</td>"
                        f"<td style='padding:7px 10px;text-align:center;color:#555;font-size:0.8em'>n={int(_cr['n'])}</td>"
                        f"</tr>"
                    )
                st.markdown(
                    "<div style='overflow-x:auto;border-radius:8px;border:1px solid var(--border)'>"
                    "<table style='width:100%;border-collapse:collapse'>"
                    "<thead><tr style='background:var(--bg-card);color:#444;font-size:0.7em;text-transform:uppercase'>"
                    "<th style='padding:7px 10px'>Model p</th>"
                    "<th style='padding:7px 10px'>Actual (niebieski) vs Expected (szary)</th>"
                    "<th style='padding:7px 10px;text-align:center'>Actual HR</th>"
                    "<th style='padding:7px 10px;text-align:center'>Blad</th>"
                    "<th style='padding:7px 10px;text-align:center'>N</th>"
                    f"</tr></thead><tbody>{''.join(_cal_rows)}</tbody></table></div>",
                    unsafe_allow_html=True)
                _avg_ce = float(_cal_g["err"].mean())
                if abs(_avg_ce)<0.03: st.success(f"Model dobrze skalibrowany (avg blad {_avg_ce:+.1%})")
                elif _avg_ce<-0.03:   st.warning(f"Model zbyt pewny siebie (avg {_avg_ce:+.1%}) - rozwaz shrinkage")
                else:                 st.info(f"Model zbyt ostrozny (avg {_avg_ce:+.1%})")

            # BIAS PER TYP
            st.markdown("#### Kalibracja per typ zakladu")
            _typ_rows = []
            _typ_hex = {"1":"#2196F3","X":"#FF9800","2":"#E91E63","1X":"#9C27B0","X2":"#00BCD4"}
            for _typ in ["1","X","2","1X","X2"]:
                _td = _diag_df[_diag_df["typ"]==_typ]
                if len(_td) < 4: continue
                _tavg = float(_td["p_typ"].mean()); _thr = float(_td["hit"].mean())
                _terr = _tavg - _thr
                _tbrier = float(((_td["p_typ"]-_td["hit"])**2).mean())
                _tc = _typ_hex.get(_typ,"#888")
                _errc = "#4CAF50" if abs(_terr)<0.05 else ("#FF9800" if abs(_terr)<0.10 else "#F44336")
                _verdict = "OK" if abs(_terr)<0.05 else ("Zbyt pewny" if _terr>0 else "Zbyt ostrozny")
                _bw_p = int(_tavg*100); _bw_h = int(_thr*100)
                _typ_rows.append(
                    f"<tr><td style='padding:8px 12px;text-align:center'>"
                    f"<span style='background:{_tc};color:#fff;padding:2px 10px;border-radius:10px;"
                    f"font-weight:700'>{_typ}</span></td>"
                    f"<td style='padding:8px 10px;text-align:center;color:#888'>{len(_td)}</td>"
                    f"<td style='padding:8px 10px;width:120px'>"
                    f"<div style='display:flex;align-items:center;gap:5px'>"
                    f"<div style='flex:1;background:var(--bg-card2);border-radius:3px;height:6px;position:relative'>"
                    f"<div style='background:#444;width:{_bw_p}%;height:6px;border-radius:3px;position:absolute'></div>"
                    f"<div style='background:{_tc};width:{_bw_h}%;height:6px;border-radius:3px;position:absolute;opacity:.9'></div>"
                    f"</div><span style='font-size:0.8em;color:{_tc}'>{_thr:.0%}</span></div></td>"
                    f"<td style='padding:8px 10px;text-align:center;color:#aaa;font-size:0.85em'>{_tavg:.0%}</td>"
                    f"<td style='padding:8px 10px;text-align:center;color:{_errc};font-weight:700'>{_terr:+.1%}</td>"
                    f"<td style='padding:8px 10px;text-align:center;color:#666;font-size:0.82em'>{_tbrier:.4f}</td>"
                    f"<td style='padding:8px 10px;text-align:center;color:{_errc};font-size:0.8em'>{_verdict}</td>"
                    f"</tr>"
                )
            st.markdown(
                "<div style='overflow-x:auto;border-radius:8px;border:1px solid var(--border)'>"
                "<table style='width:100%;border-collapse:collapse'>"
                "<thead><tr style='background:var(--bg-card);color:#444;font-size:0.7em;text-transform:uppercase'>"
                "<th style='padding:7px 12px'>Typ</th><th style='padding:7px 10px;text-align:center'>N</th>"
                "<th style='padding:7px 10px;text-align:left'>Actual (kolor) vs Model p (szary)</th>"
                "<th style='padding:7px 10px;text-align:center'>Avg p</th>"
                "<th style='padding:7px 10px;text-align:center'>Blad kal.</th>"
                "<th style='padding:7px 10px;text-align:center'>Brier</th>"
                "<th style='padding:7px 10px;text-align:center'>Ocena</th>"
                f"</tr></thead><tbody>{''.join(_typ_rows)}</tbody></table></div>",
                unsafe_allow_html=True)

            # BIAS PER DRUZYNA
            st.markdown("#### Bias per druzyna")
            _team_col = "dom" if "dom" in _diag_df.columns else "home"
            if _team_col in _diag_df.columns:
                _tb = _diag_df.groupby(_team_col).agg(
                    n=("hit","count"), hr=("hit","mean"), avg_p=("p_typ","mean")
                ).reset_index()
                _tb = _tb[_tb["n"] >= 4].copy()
                _tb["bias"] = _tb["avg_p"] - _tb["hr"]
                _tb = _tb.sort_values("bias", ascending=False)
                if not _tb.empty:
                    _hc1, _hc2 = st.columns(2)
                    with _hc1:
                        st.markdown("**Model zbyt pewny** (przeszacowuje):")
                        for _, _tr in _tb[_tb["bias"]>0.05].head(5).iterrows():
                            st.markdown(
                                f"<div style='background:var(--bg-card2);border-left:3px solid #F44336;"
                                f"padding:6px 10px;margin:3px 0;border-radius:4px;font-size:0.85em'>"
                                f"<b>{_tr[_team_col]}</b> "
                                f"model {float(_tr['avg_p']):.0%} - actual {float(_tr['hr']):.0%} "
                                f"<span style='color:#F44336'>({float(_tr['bias']):+.0%})</span></div>",
                                unsafe_allow_html=True)
                    with _hc2:
                        st.markdown("**Model zbyt ostrozny** (niedoszacowuje):")
                        for _, _tr in _tb[_tb["bias"]<-0.05].tail(5).iterrows():
                            st.markdown(
                                f"<div style='background:var(--bg-card2);border-left:3px solid #4CAF50;"
                                f"padding:6px 10px;margin:3px 0;border-radius:4px;font-size:0.85em'>"
                                f"<b>{_tr[_team_col]}</b> "
                                f"model {float(_tr['avg_p']):.0%} - actual {float(_tr['hr']):.0%} "
                                f"<span style='color:#4CAF50'>({float(_tr['bias']):+.0%})</span></div>",
                                unsafe_allow_html=True)
                    with st.expander(f"Wszystkie druzyny ({len(_tb)})"):
                        _tb_rows = []
                        for _, _tr in _tb.iterrows():
                            _bias = float(_tr["bias"])
                            _bc = "#F44336" if _bias>0.10 else ("#FF9800" if _bias>0.04 else ("#4CAF50" if _bias<-0.04 else "#888"))
                            _bw = min(abs(_bias)*400, 100)
                            _tb_rows.append(
                                f"<tr>"
                                f"<td style='padding:6px 10px;font-size:0.87em;color:#ccc'>{_tr[_team_col]}</td>"
                                f"<td style='padding:6px 10px;text-align:center;color:#888;font-size:0.82em'>{int(_tr['n'])}</td>"
                                f"<td style='padding:6px 10px;text-align:center;color:#888;font-size:0.82em'>{float(_tr['avg_p']):.0%}</td>"
                                f"<td style='padding:6px 10px;text-align:center;color:{_bc};font-size:0.82em'>{float(_tr['hr']):.0%}</td>"
                                f"<td style='padding:6px 10px;width:80px'>"
                                f"<div style='background:{_bc};width:{_bw:.0f}px;height:4px;border-radius:2px'></div>"
                                f"<span style='color:{_bc};font-size:0.78em;font-weight:700'> {_bias:+.1%}</span>"
                                f"</td></tr>"
                            )
                        st.markdown(
                            "<div style='overflow-x:auto;max-height:320px;overflow-y:auto'>"
                            "<table style='width:100%;border-collapse:collapse'>"
                            "<thead><tr style='background:var(--bg-card);color:#444;font-size:0.7em;"
                            "text-transform:uppercase;position:sticky;top:0'>"
                            "<th style='padding:6px 10px'>Druzyna</th>"
                            "<th style='padding:6px 10px;text-align:center'>N</th>"
                            "<th style='padding:6px 10px;text-align:center'>Avg p</th>"
                            "<th style='padding:6px 10px;text-align:center'>Actual HR</th>"
                            "<th style='padding:6px 10px'>Bias</th>"
                            f"</tr></thead><tbody>{''.join(_tb_rows)}</tbody></table></div>",
                            unsafe_allow_html=True)

            # DRIFT przez sezon
            if "kolejka" in _diag_df.columns:
                st.markdown("#### Drift modelu przez sezon")
                _drift = _diag_df.groupby("kolejka").agg(
                    hr=("hit","mean"), avg_p=("p_typ","mean"),
                    brier=("brier_pkt","mean") if "brier_pkt" in _diag_df.columns else ("hit","mean")
                ).reset_index()
                if len(_drift) >= 3:
                    _drift_chart = _drift.set_index("kolejka")[["hr","avg_p"]].rename(
                        columns={"hr":"Actual Hit Rate","avg_p":"Avg Model p"})
                    st.line_chart(_drift_chart, height=160)
                    st.caption("Zbieznosc linii = dobra kalibracja. Niebieska nad pomaranczowa = model niedoszacowuje.")

                # ── Alert: Brier Score w ostatnich 10 meczach ─────────────────
                if "brier_pkt" in _diag_df.columns and len(_diag_df) >= 10:
                    _brier_recent = float(_diag_df.tail(10)["brier_pkt"].mean())
                    _brier_all    = float(_diag_df["brier_pkt"].mean())
                    _brier_delta  = _brier_recent - _brier_all
                    _brier_thresh = 0.04  # +4pp powyżej średniej = niepokojący wzrost

                    if _brier_delta > _brier_thresh:
                        _drift_sev = "🔴 Krytyczny" if _brier_delta > 0.08 else "🟡 Umiarkowany"
                        st.markdown(
                            f"<div style='background:var(--bg-card2);border:1px solid var(--warn);"
                            f"border-radius:8px;padding:10px 14px;margin:8px 0'>"
                            f"<div style='font-size:0.82em;font-weight:700;color:var(--warn)'>"
                            f"⚠️ {_drift_sev} drift Brier Score</div>"
                            f"<div style='font-size:0.78em;color:var(--text-muted);margin-top:4px'>"
                            f"Ostatnie 10 meczów: <b style='color:var(--warn)'>{_brier_recent:.4f}</b> "
                            f"vs średnia sezonu: <b>{_brier_all:.4f}</b> "
                            f"(Δ <b style='color:var(--warn)'>+{_brier_delta:.4f}</b>)"
                            f"</div>"
                            f"<div style='font-size:0.76em;color:var(--text-dim);margin-top:6px;"
                            f"border-top:1px solid var(--border);padding-top:6px'>"
                            f"💡 <b>Sugestia modelu:</b> Rozważ zwiększenie shrinkage — podwyż próg pewności "
                            f"<code>PROG_PEWNY</code> o 2–4pp lub zmniejsz <code>KF</code> do trybu konserwatywnego. "
                            f"Możliwa zmiana siły drużyn (kontuzje, forma) nieuwzględniona w danych."
                            f"</div></div>",
                            unsafe_allow_html=True)
                    elif _brier_delta < -_brier_thresh:
                        st.markdown(
                            f"<div style='background:var(--bg-card2);border:1px solid var(--accent);"
                            f"border-radius:8px;padding:10px 14px;margin:8px 0'>"
                            f"<div style='font-size:0.82em;font-weight:700;color:var(--accent)'>"
                            f"✅ Model poprawia się</div>"
                            f"<div style='font-size:0.78em;color:var(--text-muted);margin-top:4px'>"
                            f"Ostatnie 10 meczów: Brier <b style='color:var(--accent)'>{_brier_recent:.4f}</b> "
                            f"vs średnia: <b>{_brier_all:.4f}</b> — kalibracja lepsza niż sezonowa.</div>"
                            f"</div>",
                            unsafe_allow_html=True)
                    else:
                        st.caption(
                            f"📊 Brier (ostatnie 10): {_brier_recent:.4f} · "
                            f"Sezon: {_brier_all:.4f} · Δ {_brier_delta:+.4f} — stabilny")

else:
    st.error("Nie udało się pobrać danych. Sprawdź połączenie z internetem.")