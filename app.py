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
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Any
import hashlib

# ===========================================================================
# KONFIGURACJA I INICJALIZACJA
# ===========================================================================
st.set_page_config(
    page_title="Football Analytics System", 
    layout="wide", 
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': 'Football Analytics System v2.0 - Dixon-Coles Model'
    }
)

# Inicjalizacja stanu sesji dla konfiguracji modelu
def init_session_state():
    defaults = {
        'sot_blend_w': 0.30,
        'tau_days': 30.0,
        'prog_pewny': 0.55,
        'prog_podwojna': 0.55,
        'shrink_alpha': 0.20,
        'debug_mode': False,
        'show_advanced': False
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# ===========================================================================
# KONFIGURACJA LIG
# ===========================================================================
LIGI = {
    "Premier League": {"csv_code": "E0",  "file": "terminarz_premier_2025.csv"},
    "La Liga":        {"csv_code": "SP1", "file": "terminarz_la_liga_2025.csv"},
    "Bundesliga":     {"csv_code": "D1",  "file": "terminarz_bundesliga_2025.csv"},
    "Serie A":        {"csv_code": "I1",  "file": "terminarz_serie_a_2025.csv"},
    "Ligue 1":        {"csv_code": "F1",  "file": "terminarz_ligue_1_2025.csv"},
}

DB_FILE = "predykcje.db"

# ===========================================================================
# KOMPONENTY UI - CSS I STYLE
# ===========================================================================
def load_css():
    """Załadowanie zoptymalizowanych styli CSS"""
    st.markdown("""
    <style>
    /* Główne kontenery */
    .main-container {
        background: linear-gradient(135deg, #0e1117 0%, #1a1a2e 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    
    /* Karty meczów */
    .match-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 16px;
        margin: 8px 0;
        border-left: 4px solid #4CAF50;
        transition: all 0.3s ease;
    }
    .match-card:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    /* Badge typów */
    .badge-1 { background: linear-gradient(90deg, #2196F3, #1976D2); }
    .badge-X { background: linear-gradient(90deg, #FF9800, #F57C00); }
    .badge-2 { background: linear-gradient(90deg, #E91E63, #C2185B); }
    .badge-1X { background: linear-gradient(90deg, #2196F3, #FF9800); }
    .badge-X2 { background: linear-gradient(90deg, #FF9800, #E91E63); }
    
    .type-badge {
        color: white;
        padding: 4px 14px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.9em;
        display: inline-block;
    }
    
    /* Paski postępu */
    .progress-bar {
        background: #333;
        border-radius: 4px;
        height: 6px;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    /* Metryki */
    .metric-container {
        text-align: center;
        padding: 12px;
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
    }
    .metric-value {
        font-size: 1.8em;
        font-weight: bold;
    }
    .metric-delta {
        font-size: 0.85em;
        margin-top: 4px;
    }
    
    /* Statusy */
    .status-high { color: #4CAF50; }
    .status-medium { color: #FF9800; }
    .status-low { color: #F44336; }
    
    /* Tabele */
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88em;
    }
    .custom-table th {
        background: #1e1e2e;
        color: #aaa;
        padding: 10px;
        text-transform: uppercase;
        font-size: 0.75em;
        text-align: left;
    }
    .custom-table td {
        padding: 8px 10px;
        border-bottom: 1px solid #2a2a3e;
    }
    .custom-table tr:hover {
        background: rgba(255,255,255,0.02);
    }
    
    /* Tooltip */
    .tooltip {
        position: relative;
        cursor: help;
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# ===========================================================================
# KOMPONENTY UI - FUNKCJE POMOCNICZE
# ===========================================================================
class UIComponents:
    """Klasa z komponentami UI do reużywalności"""
    
    @staticmethod
    def type_badge(typ: str) -> str:
        """Generuje badge dla typu zakładu"""
        badge_class = f"badge-{typ}" if typ in ["1", "X", "2", "1X", "X2"] else "badge-1"
        return f'<span class="type-badge {badge_class}">{typ}</span>'
    
    @staticmethod
    def progress_bar(value: float, color: str, width: int = 100) -> str:
        """Generuje HTML paska postępu"""
        pct = int(value * 100)
        return f"""
        <div class="progress-bar" style="width: {width}px; display: inline-block; vertical-align: middle; margin-right: 8px;">
            <div class="progress-fill" style="width: {pct}%; background: {color};"></div>
        </div>
        <span style="color: {color}; font-size: 0.85em;">{value:.0%}</span>
        """
    
    @staticmethod
    def metric_card(label: str, value: str, delta: str = None, status: str = "neutral"):
        """Generuje kartę metryki"""
        colors = {"good": "#4CAF50", "bad": "#F44336", "neutral": "#888", "warning": "#FF9800"}
        color = colors.get(status, "#888")
        delta_html = f'<div class="metric-delta" style="color: {color}">{delta}</div>' if delta else ""
        return f"""
        <div class="metric-container">
            <div style="color: #888; font-size: 0.8em; margin-bottom: 4px;">{label}</div>
            <div class="metric-value" style="color: {color};">{value}</div>
            {delta_html}
        </div>
        """
    
    @staticmethod
    def match_header(home: str, away: str, date_str: str, confidence: str) -> str:
        """Generuje nagłówek meczu"""
        conf_icons = {"High": "🟢", "Medium": "🟡", "Coinflip": "🔴"}
        conf_icon = conf_icons.get(confidence, "⚪")
        return f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <div style="font-weight: bold; font-size: 1.1em;">{home}</div>
            <div style="text-align: center; color: #888; font-size: 0.85em;">
                {conf_icon}<br>{date_str}
            </div>
            <div style="font-weight: bold; font-size: 1.1em; text-align: right;">{away}</div>
        </div>
        """
    
    @staticmethod
    def chaos_indicator(label: str, emoji: str, pct: float) -> str:
        """Generuje wskaźnik chaosu"""
        colors = {"Pewny": "#4CAF50", "Klarowny": "#8BC34A", "Otwarty": "#FF9800", "Chaos": "#F44336"}
        color = colors.get(label, "#888")
        bar_width = int(pct * 100)
        return f"""
        <div style="margin: 8px 0;">
            <div style="display: flex; justify-content: space-between; font-size: 0.8em; color: #888; margin-bottom: 4px;">
                <span>{emoji} {label}</span>
                <span>{pct:.0%} chaos</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {bar_width}%; background: {color};"></div>
            </div>
        </div>
        """

ui = UIComponents()

# ===========================================================================
# MAPOWANIA I FUNKCJE POMOCNICZE
# ===========================================================================
def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return str(name)
    name = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode("ASCII")
    for suffix in [" FC", " CF", " UD", " CD", " RCD", " AS", " Stade", " 1."]:
        name = name.replace(suffix, "")
    return name.strip()

NAZWY_MAP = {
    # PREMIER LEAGUE
    "Brighton & Hove Albion": "Brighton",
    "West Ham United": "West Ham",
    "Newcastle United": "Newcastle",
    "Tottenham Hotspur": "Tottenham",
    "Leeds United": "Leeds",
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "Wolverhampton": "Wolves",
    "Leicester City": "Leicester",
    "Sheffield United": "Sheffield Utd",
    # LA LIGA
    "Girona FC": "Girona",
    "Rayo Vallecano": "Vallecano",
    "RCD Mallorca": "Mallorca",
    "FC Barcelona": "Barcelona",
    "Deportivo Alavés": "Alaves",
    "Deportivo Alaves": "Alaves",
    "Levante UD": "Levante",
    "Valencia CF": "Valencia",
    "Real Sociedad": "Sociedad",
    "Celta Vigo": "Celta",
    "Getafe CF": "Getafe",
    "Athletic Club": "Ath Bilbao",
    "Athletic Bilbao": "Ath Bilbao",
    "Sevilla FC": "Sevilla",
    "RCD Espanyol": "Espanol",
    "Espanyol": "Espanol",
    "Atlético Madrid": "Ath Madrid",
    "Atletico Madrid": "Ath Madrid",
    "Elche CF": "Elche",
    "Real Betis": "Betis",
    "CA Osasuna": "Osasuna",
    "Real Valladolid": "Valladolid",
    "UD Las Palmas": "Las Palmas",
    "Cadiz CF": "Cadiz",
    "Granada CF": "Granada",
    "UD Almeria": "Almeria",
    # BUNDESLIGA
    "FC Bayern München": "Bayern Munich",
    "Bayern": "Bayern Munich",
    "Borussia Dortmund": "Dortmund",
    "Leipzig": "RB Leipzig",
    "Bayer 04 Leverkusen": "Leverkusen",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "VfB Stuttgart": "Stuttgart",
    "VfL Wolfsburg": "Wolfsburg",
    "Borussia Mönchengladbach": "M'gladbach",
    "Borussia Monchengladbach": "M'gladbach",
    "1. FC Union Berlin": "Union Berlin",
    "SC Freiburg": "Freiburg",
    "1. FC Köln": "FC Koln",
    "FC Köln": "FC Koln",
    "1. FSV Mainz 05": "Mainz",
    "FSV Mainz 05": "Mainz",
    "TSG Hoffenheim": "Hoffenheim",
    "SV Werder Bremen": "Werder Bremen",
    "VfL Bochum": "Bochum",
    "FC Augsburg": "Augsburg",
    "FC St. Pauli": "St Pauli",
    "St. Pauli": "St Pauli",
    "1. FC Heidenheim": "Heidenheim",
    "Hamburger SV": "Hamburg",
    # SERIE A
    "AC Milan": "Milan",
    "Internazionale": "Inter",
    "AS Roma": "Roma",
    "Hellas Verona": "Verona",
    # LIGUE 1
    "Paris Saint-Germain": "Paris SG",
    "PSG": "Paris SG",
    "Olympique de Marseille": "Marseille",
    "AS Monaco": "Monaco",
    "Olympique Lyonnais": "Lyon",
    "LOSC Lille": "Lille",
    "Stade Rennais": "Rennes",
    "OGC Nice": "Nice",
    "RC Lens": "Lens",
    "RC Strasbourg": "Strasbourg",
    "FC Nantes": "Nantes",
    "Montpellier HSC": "Montpellier",
    "FC Toulouse": "Toulouse",
    "Stade de Reims": "Reims",
    "Stade Brestois 29": "Brest",
    "Stade Brestois": "Brest",
    "Clermont Foot": "Clermont",
    "AJ Auxerre": "Auxerre",
    "Le Havre AC": "Le Havre",
    "FC Metz": "Metz",
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

def waga_poprzedniego(n_biezacy: int) -> float:
    return float(np.clip(0.8 - (n_biezacy / 30) * 0.6, 0.2, 0.8))

# ===========================================================================
# BAZA DANYCH
# ===========================================================================
def init_db():
    """Inicjalizuje bazę danych z nową strukturą analityczną"""
    con = sqlite3.connect(DB_FILE)
    
    # Tabela predykcje
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
    
    # Tabela zdarzenia
    con.execute("""
        CREATE TABLE IF NOT EXISTS zdarzenia (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            liga        TEXT,
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
            UNIQUE(liga, kolejnosc, mecz, rynek, typ, linia)
        )
    """)
    
    # Tabela kupony
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
    
    con.commit()
    con.close()

# ===========================================================================
# ŁADOWANIE DANYCH - Z CACHE'OWANIEM
# ===========================================================================
@st.cache_data(ttl=900)
def _pobierz_csv(league_code: str, sezon: str) -> pd.DataFrame:
    """Pobiera CSV z football-data.co.uk z obsługą błędów"""
    try:
        url = f"https://www.football-data.co.uk/mmz4281/{sezon}/{league_code}.csv"
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        
        # Uzupełnienie brakujących kolumn
        for col in ["HC", "AC", "HY", "AY", "HR", "AR", "HST", "AST"]:
            if col not in df.columns:
                df[col] = np.nan
        
        df["total_gole"] = df["FTHG"] + df["FTAG"]
        df["total_kartki"] = df["HY"] + df["AY"] + (df["HR"] + df["AR"]) * 2
        df["total_rozne"] = df["HC"] + df["AC"]
        df["HST"] = pd.to_numeric(df["HST"], errors="coerce")
        df["AST"] = pd.to_numeric(df["AST"], errors="coerce")
        
        return df.sort_values("Date")
    except Exception as e:
        st.error(f"Błąd pobierania {sezon}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=900)
def load_historical(league_code: str) -> pd.DataFrame:
    """Ładuje dane historyczne z blendem sezonów"""
    df_now = _pobierz_csv(league_code, "2526")
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

@st.cache_data(ttl=86400)
def load_schedule(filename: str) -> pd.DataFrame:
    """Ładuje terminarz z pliku CSV"""
    try:
        df = pd.read_csv(filename)
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
        
        if "round" in df.columns:
            df["round"] = pd.to_numeric(df["round"], errors="coerce").fillna(0).astype(int)
        else:
            df = df.sort_values("date")
            df["date_only"] = df["date"].dt.date
            unique_dates = sorted(df["date_only"].unique())
            date_to_round = {date: i+1 for i, date in enumerate(unique_dates)}
            df["round"] = df["date_only"].map(date_to_round)
            df = df.drop("date_only", axis=1)
        
        return df.sort_values("date").reset_index(drop=True).dropna(subset=["date"])
    except Exception as e:
        st.error(f"Problem z plikiem {filename}: {e}")
        return pd.DataFrame()

# ===========================================================================
# FUNKCJE POMOCNICZE DLA KOLEJEK
# ===========================================================================
def get_current_round(schedule: pd.DataFrame) -> int:
    """Zwraca numer aktualnej kolejki"""
    if schedule.empty:
        return 0
    
    dzisiaj = datetime.now().date()
    future_matches = schedule[schedule["date"].dt.date >= dzisiaj]
    
    if not future_matches.empty:
        return int(future_matches.iloc[0]["round"])
    else:
        return int(schedule.iloc[-1]["round"])

def get_round_status(schedule: pd.DataFrame, round_num: int) -> str:
    """Zwraca status kolejki"""
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
# STATYSTYKI I MODEL
# ===========================================================================
def weighted_mean(values: pd.Series, dates: pd.Series = None, tau_days: float = None) -> float:
    """Ważona średnia z wykładniczym decay"""
    if tau_days is None:
        tau_days = st.session_state.tau_days
    
    if len(values) == 0:
        return 0.0
    
    if dates is not None and len(dates) == len(values):
        try:
            dates_dt = pd.to_datetime(dates)
            ref = dates_dt.max()
            days_ago = (ref - dates_dt).dt.total_seconds() / 86400
            weights = np.exp(-days_ago.values / tau_days)
            weights = np.clip(weights, 0.01, None)
        except Exception:
            weights = np.linspace(1, 2, len(values))
    else:
        weights = np.linspace(1, 2, len(values))
    
    return float(np.average(values, weights=weights))

@st.cache_data
def oblicz_wszystkie_statystyki(df_json: str, tau_days: float = 30.0) -> pd.DataFrame:
    """Oblicza statystyki dla wszystkich drużyn"""
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
        
        h_dates = home["Date"] if "Date" in home.columns else None
        a_dates = away["Date"] if "Date" in away.columns else None
        
        home_sot = home["HST"].dropna() if "HST" in home.columns else pd.Series([], dtype=float)
        away_sot = away["AST"].dropna() if "AST" in away.columns else pd.Series([], dtype=float)
        
        h_sot_dates = home.loc[home["HST"].notna(), "Date"] if "HST" in home.columns and "Date" in home.columns else None
        a_sot_dates = away.loc[away["AST"].notna(), "Date"] if "AST" in away.columns and "Date" in away.columns else None
        
        dane[d] = {
            "Gole strzelone (dom)": weighted_mean(home["FTHG"], h_dates, tau_days),
            "Gole stracone (dom)": weighted_mean(home["FTAG"], h_dates, tau_days),
            "Gole strzelone (wyjazd)": weighted_mean(away["FTAG"], a_dates, tau_days),
            "Gole stracone (wyjazd)": weighted_mean(away["FTHG"], a_dates, tau_days),
            "Różne (dom)": weighted_mean(home["total_rozne"], h_dates, tau_days),
            "Różne (wyjazd)": weighted_mean(away["total_rozne"], a_dates, tau_days),
            "Kartki (dom)": weighted_mean(home["total_kartki"], h_dates, tau_days),
            "Kartki (wyjazd)": weighted_mean(away["total_kartki"], a_dates, tau_days),
            "SOT (dom)": weighted_mean(home.loc[home["HST"].notna(), "HST"], h_sot_dates, tau_days) if len(home_sot) >= 2 else None,
            "SOT (wyjazd)": weighted_mean(away.loc[away["AST"].notna(), "AST"], a_sot_dates, tau_days) if len(away_sot) >= 2 else None,
            "Konwersja (dom)": float(home["FTHG"].sum() / home_sot.sum()) if home_sot.sum() > 0 else None,
            "Konwersja (wyjazd)": float(away["FTAG"].sum() / away_sot.sum()) if away_sot.sum() > 0 else None,
        }
    
    return pd.DataFrame(dane).T.round(2)

@st.cache_data
def oblicz_srednie_ligowe(df_json: str) -> dict:
    """Oblicza średnie ligowe"""
    df = pd.read_json(df_json)
    if df.empty:
        return {"avg_home": 1.5, "avg_away": 1.2, "rho": -0.13, "n_biezacy": 0}
    
    n_biezacy = int((df.get("_sezon", pd.Series()) == "biezacy").sum()) if "_sezon" in df.columns else len(df)
    avg_h = float(df["FTHG"].mean())
    avg_a = float(df["FTAG"].mean())
    n = len(df)
    
    obs_00 = len(df[(df["FTHG"] == 0) & (df["FTAG"] == 0)]) / n
    obs_11 = len(df[(df["FTHG"] == 1) & (df["FTAG"] == 1)]) / n
    
    exp_00 = poisson.pmf(0, avg_h) * poisson.pmf(0, avg_a)
    exp_11 = poisson.pmf(1, avg_h) * poisson.pmf(1, avg_a)
    
    rho_00 = (obs_00 / exp_00 - 1) / (avg_h * avg_a) if exp_00 > 0 else -0.13
    rho_11 = -(obs_11 / exp_11 - 1) if exp_11 > 0 else -0.13
    rho = float(np.clip(np.mean([rho_00, rho_11]), -0.25, 0.0))
    
    avg_sot_h = float(df["HST"].dropna().mean()) if "HST" in df.columns and df["HST"].notna().sum() > 10 else None
    avg_sot_a = float(df["AST"].dropna().mean()) if "AST" in df.columns and df["AST"].notna().sum() > 10 else None
    
    return {
        "avg_home": avg_h, 
        "avg_away": avg_a, 
        "rho": rho,
        "n_biezacy": n_biezacy,
        "avg_sot_home": avg_sot_h, 
        "avg_sot_away": avg_sot_a
    }

def oblicz_lambdy(h: str, a: str, srednie_df: pd.DataFrame, srednie_lig: dict, 
                  forma_dict: dict, sot_w: float = None) -> tuple:
    """Oblicza parametry lambda dla meczu"""
    if sot_w is None:
        sot_w = st.session_state.sot_blend_w
    
    avg_h = max(srednie_lig["avg_home"], 0.5)
    avg_a = max(srednie_lig["avg_away"], 0.5)
    
    atak_h = srednie_df.loc[h, "Gole strzelone (dom)"] / avg_h
    obrona_a = srednie_df.loc[a, "Gole stracone (wyjazd)"] / avg_a
    atak_a = srednie_df.loc[a, "Gole strzelone (wyjazd)"] / avg_a
    obrona_h = srednie_df.loc[h, "Gole stracone (dom)"] / avg_h

    def form_weight(team: str) -> float:
        f = forma_dict.get(team, "")
        w = f.count("W")
        l = f.count("L")
        return float(np.clip(1.0 + (w - l) * 0.03, 0.85, 1.15))

    lam_h_goals = avg_h * atak_h * obrona_a * form_weight(h)
    lam_a_goals = avg_a * atak_a * obrona_h * form_weight(a)

    sot_aktywny = False
    lam_h = lam_h_goals
    lam_a = lam_a_goals

    if sot_w > 0:
        sot_h = srednie_df.loc[h, "SOT (dom)"] if "SOT (dom)" in srednie_df.columns else None
        sot_a = srednie_df.loc[a, "SOT (wyjazd)"] if "SOT (wyjazd)" in srednie_df.columns else None
        avg_sot_h = srednie_lig.get("avg_sot_home", None)
        avg_sot_a = srednie_lig.get("avg_sot_away", None)
        
        if (sot_h is not None and sot_a is not None and avg_sot_h and avg_sot_a and
                not np.isnan(sot_h) and not np.isnan(sot_a)):
            conv_h = avg_h / avg_sot_h
            conv_a = avg_a / avg_sot_a
            lam_sot_h = sot_h * conv_h * obrona_a * form_weight(h)
            lam_sot_a = sot_a * conv_a * obrona_h * form_weight(a)
            lam_h = (1 - sot_w) * lam_h_goals + sot_w * lam_sot_h
            lam_a = (1 - sot_w) * lam_a_goals + sot_w * lam_sot_a
            sot_aktywny = True

    lam_r = (srednie_df.loc[h, "Różne (dom)"] + srednie_df.loc[a, "Różne (wyjazd)"]) / 2
    lam_k = (srednie_df.loc[h, "Kartki (dom)"] + srednie_df.loc[a, "Kartki (wyjazd)"]) / 2
    
    sot_h_raw = srednie_df.loc[h, "SOT (dom)"] if "SOT (dom)" in srednie_df.columns else None
    sot_a_raw = srednie_df.loc[a, "SOT (wyjazd)"] if "SOT (wyjazd)" in srednie_df.columns else None
    lam_sot_total = None
    
    if sot_h_raw is not None and sot_a_raw is not None:
        try:
            sh = float(sot_h_raw)
            sa = float(sot_a_raw)
            if not (np.isnan(sh) or np.isnan(sa)):
                lam_sot_total = sh + sa
        except (TypeError, ValueError):
            pass
    
    return (
        float(np.clip(lam_h, 0.3, 4.5)),
        float(np.clip(lam_a, 0.3, 4.5)),
        lam_r, 
        lam_k, 
        sot_aktywny, 
        lam_sot_total
    )

def oblicz_forme(df: pd.DataFrame) -> dict:
    """Oblicza formę drużyn"""
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
    """Generuje tabelę ligową"""
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
        
        table[home]["GZ"] += hg
        table[home]["GS"] += ag
        table[home]["M"] += 1
        table[away]["GZ"] += ag
        table[away]["GS"] += hg
        table[away]["M"] += 1
        
        if hg > ag:
            table[home]["Pkt"] += 3
        elif hg < ag:
            table[away]["Pkt"] += 3
        else:
            table[home]["Pkt"] += 1
            table[away]["Pkt"] += 1
    
    res = pd.DataFrame(table).T
    res["RG"] = res["GZ"] - res["GS"]
    return res.sort_values(["Pkt", "RG", "GZ"], ascending=False)

# ===========================================================================
# MODEL DIXON-COLES
# ===========================================================================
def dixon_coles_adj(M: np.ndarray, lam_h: float, lam_a: float, rho: float = -0.13) -> np.ndarray:
    """Korekta Dixon-Coles dla niskich wyników"""
    M = M.copy()
    tau = {
        (0, 0): 1 - lam_h * lam_a * rho,
        (1, 0): 1 + lam_a * rho,
        (0, 1): 1 + lam_h * rho,
        (1, 1): 1 - rho
    }
    
    for (i, j), t in tau.items():
        if i < M.shape[0] and j < M.shape[1]:
            M[i, j] *= max(t, 0.001)
    
    M /= M.sum()
    return M

def kalibruj_prawdopodobienstwa(p_home: float, p_draw: float, p_away: float) -> tuple:
    """Kalibracja prawdopodobieństw z shrinkage"""
    a = st.session_state.shrink_alpha
    p_h = (1 - a) * p_home + a / 3
    p_d = (1 - a) * p_draw + a / 3
    p_a = (1 - a) * p_away + a / 3
    s = p_h + p_d + p_a
    return p_h / s, p_d / s, p_a / s

def wybierz_typ(p_home: float, p_draw: float, p_away: float) -> tuple:
    """Wybiera najlepszy typ zakładu"""
    p_home, p_draw, p_away = kalibruj_prawdopodobienstwa(p_home, p_draw, p_away)
    
    prog_pewny = st.session_state.prog_pewny
    prog_podwojna = st.session_state.prog_podwojna
    
    p_1x = p_home + p_draw
    p_x2 = p_away + p_draw
    
    if p_home >= prog_pewny:
        return "1", p_home
    if p_away >= prog_pewny:
        return "2", p_away
    if p_1x >= prog_podwojna or p_x2 >= prog_podwojna:
        return ("1X", p_1x) if p_1x >= p_x2 else ("X2", p_x2)
    
    probs = {"1": p_home, "X": p_draw, "2": p_away}
    t = max(probs, key=probs.get)
    return t, probs[t]

def wybierz_wynik(M: np.ndarray, lam_h: float, lam_a: float) -> tuple:
    """Wybiera najbardziej prawdopodobny wynik"""
    idx_max = np.unravel_index(M.argmax(), M.shape)
    p_max = float(M[idx_max])
    top3 = np.sort(M.flatten())[::-1][:3]
    
    if float(top3[0] - top3[2]) < 0.04:
        g_h = min(int(round(lam_h)), M.shape[0] - 1)
        g_a = min(int(round(lam_a)), M.shape[1] - 1)
        return g_h, g_a, float(M[g_h, g_a])
    
    return int(idx_max[0]), int(idx_max[1]), p_max

def entropy_meczu(p_home: float, p_draw: float, p_away: float) -> float:
    """Oblicza entropię meczu"""
    ps = [p for p in [p_home, p_draw, p_away] if p > 0]
    return float(-sum(p * np.log2(p) for p in ps))

def chaos_label(entropy: float) -> tuple:
    """Etykieta poziomu chaosu"""
    pct = entropy / np.log2(3)
    if pct < 0.55:
        return "Pewny", "🔒", pct
    elif pct < 0.75:
        return "Klarowny", "🎯", pct
    elif pct < 0.90:
        return "Otwarty", "⚡", pct
    else:
        return "Chaos", "🌀", pct

def confidence_score(p_home: float, p_draw: float, p_away: float) -> tuple:
    """Ocena pewności predykcji"""
    vals = sorted([p_home, p_draw, p_away], reverse=True)
    spread = vals[0] - vals[2]
    edge = vals[0] - vals[1]
    
    if edge > 0.18:
        return "High", "🟢", f"Wyraźny faworyt (+{edge:.0%} nad 2. opcją)"
    elif edge > 0.08:
        return "Medium", "🟡", f"Umiarkowana przewaga (+{edge:.0%})"
    else:
        return "Coinflip", "🔴", f"Mecz bardzo wyrównany (spread {spread:.0%})"

def fair_odds(p: float) -> float:
    """Oblicza fair odds"""
    return round(1 / p, 2) if 0 < p <= 1 else 999.0

# ===========================================================================
# CACHE'OWANE PREDYKCJE
# ===========================================================================
@st.cache_data(ttl=300)
def predykcja_meczu_cached(lam_h: float, lam_a: float, rho: float, 
                           config_hash: str) -> dict:
    """Cache'owana predykcja meczu"""
    max_gole = int(np.clip(np.ceil(max(lam_h, lam_a) + 4), 6, 10))
    
    M = dixon_coles_adj(
        np.outer(poisson.pmf(range(max_gole), lam_h),
                 poisson.pmf(range(max_gole), lam_a)),
        lam_h, lam_a, rho=rho
    )
    
    p_home = float(np.tril(M, -1).sum())
    p_draw = float(np.trace(M))
    p_away = float(np.triu(M, 1).sum())
    
    wynik_h, wynik_a, p_exact = wybierz_wynik(M, lam_h, lam_a)
    p_home_cal, p_draw_cal, p_away_cal = kalibruj_prawdopodobienstwa(p_home, p_draw, p_away)
    typ, p_typ = wybierz_typ(p_home, p_draw, p_away)
    
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
    }

def predykcja_meczu(lam_h: float, lam_a: float, rho: float = -0.13) -> dict:
    """Wrapper dla predykcji z hashowaniem configu"""
    config_str = f"{st.session_state.shrink_alpha}_{st.session_state.prog_pewny}_{st.session_state.prog_podwojna}"
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    return predykcja_meczu_cached(lam_h, lam_a, rho, config_hash)

# ===========================================================================
# ALTERNATYWNE ZDARZENIA
# ===========================================================================
@st.cache_data(ttl=300)
def alternatywne_zdarzenia_cached(lam_h: float, lam_a: float, lam_r: float,
                                   lam_k: float, rho: float, lam_sot: float = None,
                                   prog_min: float = 0.55) -> list:
    """Cache'owane alternatywne zdarzenia"""
    zdarzenia = []
    mg = int(np.clip(np.ceil(max(lam_h, lam_a) + 4), 6, 10))
    
    M = dixon_coles_adj(
        np.outer(poisson.pmf(range(mg), lam_h), poisson.pmf(range(mg), lam_a)),
        lam_h, lam_a, rho=rho
    )

    # Gole
    for linia in [1.5, 2.5, 3.5]:
        p_over = float(sum(M[i, j] for i in range(mg) for j in range(mg) if i + j > linia))
        p_under = 1 - p_over
        if p_over >= prog_min:
            zdarzenia.append(("⚽", f"Over {linia} goli", p_over, fair_odds(p_over), "Gole", linia))
        if p_under >= prog_min:
            zdarzenia.append(("⚽", f"Under {linia} goli", p_under, fair_odds(p_under), "Gole", linia))

    # BTTS
    p_btts = float(1 - M[0, :].sum() - M[:, 0].sum() + M[0, 0])
    p_nobtts = 1 - p_btts
    if p_btts >= prog_min:
        zdarzenia.append(("⚽", "BTTS – Tak", p_btts, fair_odds(p_btts), "BTTS", 0))
    if p_nobtts >= prog_min:
        zdarzenia.append(("⚽", "BTTS – Nie", p_nobtts, fair_odds(p_nobtts), "BTTS", 0))

    # Rożne
    for linia in [7.5, 8.5, 9.5, 10.5]:
        p_over = float(1 - poisson.cdf(int(linia), lam_r))
        if p_over >= prog_min:
            zdarzenia.append(("🚩", f"Over {linia} rożnych", p_over, fair_odds(p_over), "Rożne", linia))

    # Kartki
    for linia in [2.5, 3.5, 4.5]:
        p_over = float(1 - poisson.cdf(int(linia), lam_k))
        if p_over >= prog_min:
            zdarzenia.append(("🟨", f"Over {linia} kartek", p_over, fair_odds(p_over), "Kartki", linia))

    # SOT
    SOT_MIN_ODDS = 1.30
    if lam_sot is not None and lam_sot > 0:
        for linia in [3.5, 4.5, 5.5, 6.5]:
            p_over = float(1 - poisson.cdf(int(linia), lam_sot))
            fo_sot = fair_odds(p_over)
            if p_over >= prog_min and fo_sot >= SOT_MIN_ODDS:
                zdarzenia.append(("🎯", f"Over {linia} celnych", p_over, fo_sot, "SOT", linia))

    return sorted(zdarzenia, key=lambda x: -x[2])

def alternatywne_zdarzenia(lam_h: float, lam_a: float, lam_r: float,
                           lam_k: float, rho: float, prog_min: float = 0.55,
                           lam_sot: float = None) -> list:
    """Wrapper dla alternatywnych zdarzeń"""
    return alternatywne_zdarzenia_cached(lam_h, lam_a, lam_r, lam_k, rho, lam_sot, prog_min)

# ===========================================================================
# ZAPIS ZDARZEŃ
# ===========================================================================
def zapisz_zdarzenia(liga: str, kolejnosc: int, mecz: str, home: str, away: str,
                     rynek: str, typ: str, linia: float, p_model: float, fair_odds: float):
    """Zapisuje pojedyncze zdarzenie do bazy"""
    init_db()
    con = sqlite3.connect(DB_FILE)
    try:
        con.execute(
            """INSERT OR IGNORE INTO zdarzenia 
               (liga, kolejnosc, mecz, home, away, rynek, linia, typ, p_model, fair_odds, data)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (liga, kolejnosc, mecz, home, away, rynek, linia, typ, p_model, fair_odds,
             datetime.now().strftime("%Y-%m-%d"))
        )
        con.commit()
    except Exception as e:
        st.warning(f"Błąd zapisu zdarzenia: {e}")
    finally:
        con.close()

def aktualizuj_wynik_zdarzenia(home: str, away: str, hist: pd.DataFrame):
    """Aktualizuje wyniki zdarzeń po meczu"""
    if "_sezon" in hist.columns:
        hist = hist[hist["_sezon"] == "biezacy"]
    
    match = hist[(hist["HomeTeam"] == home) & (hist["AwayTeam"] == away)]
    if match.empty:
        return
    
    row = match.iloc[-1]
    hg = int(row["FTHG"])
    ag = int(row["FTAG"])
    wynik_1x2 = "1" if hg > ag else ("2" if ag > hg else "X")
    rzuty = int(row.get("HC", 0)) + int(row.get("AC", 0))
    kartki = int(row.get("HY", 0)) + int(row.get("AY", 0)) + \
             (int(row.get("HR", 0)) + int(row.get("AR", 0))) * 2
    
    init_db()
    con = sqlite3.connect(DB_FILE)
    
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
            trafione = (hg > 0 and ag > 0) if typ == "BTTS – Tak" else (hg == 0 or ag == 0)
        elif rynek == "Rożne":
            trafione = rzuty > linia
        elif rynek == "Kartki":
            trafione = kartki > linia
        elif rynek == "1X2":
            if typ == "1": 
                trafione = (wynik_1x2 == "1")
            elif typ == "X": 
                trafione = (wynik_1x2 == "X")
            elif typ == "2": 
                trafione = (wynik_1x2 == "2")
            elif typ == "1X": 
                trafione = (wynik_1x2 in ("1", "X"))
            elif typ == "X2": 
                trafione = (wynik_1x2 in ("X", "2"))
        elif rynek == "SOT":
            if "Over" in typ:
                hst = int(row.get("HST", 0)) if not pd.isna(row.get("HST", 0)) else 0
                ast = int(row.get("AST", 0)) if not pd.isna(row.get("AST", 0)) else 0
                trafione = (hst + ast) > linia
        
        con.execute(
            "UPDATE zdarzenia SET wynik=?, trafione=? WHERE id=?",
            (f"{hg}:{ag}", trafione, zid)
        )
    
    con.commit()
    con.close()

# ===========================================================================
# STATYSTYKI SKUTECZNOŚCI
# ===========================================================================
def statystyki_skutecznosci(liga: str = None) -> pd.DataFrame:
    """Zwraca statystyki skuteczności per rynek"""
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
    
    stats = []
    for (rynek, typ, linia), group in df.groupby(['rynek', 'typ', 'linia']):
        nazwa = f"{rynek}"
        if linia and linia > 0:
            nazwa += f" {linia}"
        if typ and typ not in ["Tak", "Nie"] and "Over" not in typ and "Under" not in typ:
            nazwa += f" {typ}"
        
        trafione = group['trafione'].sum()
        wszystkie = len(group)
        skutecznosc = trafione / wszystkie if wszystkie > 0 else 0
        sr_p_model = group['p_model'].mean()
        sr_fair = group['fair_odds'].mean()
        
        roi = (trafione * (sr_fair - 1) - (wszystkie - trafione)) / wszystkie if wszystkie > 0 else 0
        brier = float(((group["p_model"] - group["trafione"].astype(float)) ** 2).mean())
        brier_ref = sr_p_model * (1 - skutecznosc) ** 2 + (1 - sr_p_model) * skutecznosc ** 2
        brier_skill = 1 - brier / brier_ref if brier_ref > 0 else 0
        
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
            "_roi_v": roi,
            "_brier_v": brier,
            "_skut_v": skutecznosc,
            "Kolor": "🟢" if roi > 0.03 else ("🟡" if roi > -0.03 else "🔴")
        })
    
    return pd.DataFrame(stats).sort_values("Typów", ascending=False)

# ===========================================================================
# SIDEBAR I KONFIGURACJA
# ===========================================================================
def render_sidebar():
    """Renderuje sidebar z konfiguracją"""
    st.sidebar.header("🌍 Wybór Rozgrywek")
    wybrana_liga = st.sidebar.selectbox("Wybierz ligę", list(LIGI.keys()))
    
    # Zaawansowane ustawienia modelu
    with st.sidebar.expander("⚙️ Zaawansowane ustawienia modelu", expanded=False):
        st.session_state.sot_blend_w = st.slider(
            "Waga SOT w blendzie", 0.0, 0.5, st.session_state.sot_blend_w, 0.05,
            help="0 = tylko gole, 0.5 = równomierny blend"
        )
        st.session_state.tau_days = st.slider(
            "Tau (dni decay)", 10.0, 90.0, st.session_state.tau_days, 5.0,
            help="Okres półtrwania wag w ważonej średniej"
        )
        st.session_state.shrink_alpha = st.slider(
            "Shrinkage alpha", 0.0, 0.5, st.session_state.shrink_alpha, 0.05,
            help="Siła regresji do średniej (0 = brak, 0.5 = maksymalna)"
        )
        st.session_state.prog_pewny = st.slider(
            "Próg typu pewnego", 0.50, 0.70, st.session_state.prog_pewny, 0.01
        )
        st.session_state.prog_podwojna = st.slider(
            "Próg typu podwójnego", 0.50, 0.70, st.session_state.prog_podwojna, 0.01
        )
    
    st.session_state.debug_mode = st.sidebar.checkbox(
        "🔧 Debug – niezmapowane nazwy", 
        value=st.session_state.debug_mode
    )
    
    return wybrana_liga

# ===========================================================================
# GŁÓWNA APLIKACJA
# ===========================================================================
def main():
    """Główna funkcja aplikacji"""
    wybrana_liga = render_sidebar()
    
    # Ładowanie danych
    with st.spinner("Ładowanie danych..."):
        historical = load_historical(LIGI[wybrana_liga]["csv_code"])
        schedule = load_schedule(LIGI[wybrana_liga]["file"])
    
    # Nagłówek
    col_title, col_info = st.columns([3, 1])
    with col_title:
        st.title(f"⚽ {wybrana_liga}")
        st.caption("Dixon-Coles · SOT blend · Brier tracking · Kalibracja modelu")
    
    with col_info:
        if not schedule.empty:
            aktualna_kolejka = get_current_round(schedule)
            liczba_meczy = len(schedule[schedule["round"] == aktualna_kolejka])
            status = get_round_status(schedule, aktualna_kolejka)
            st.metric(f"{status} kolejka", f"#{aktualna_kolejka}", f"{liczba_meczy} meczów")
    
    if historical.empty:
        st.error("Nie udało się pobrać danych historycznych. Sprawdź połączenie.")
        return
    
    # Obliczenia
    with st.spinner("Obliczanie statystyk..."):
        srednie_df = oblicz_wszystkie_statystyki(
            historical.to_json(), 
            st.session_state.tau_days
        )
        srednie_lig = oblicz_srednie_ligowe(historical.to_json())
        forma_dict = oblicz_forme(historical)
        rho = srednie_lig["rho"]
        n_biezacy = srednie_lig["n_biezacy"]
        w_prev = waga_poprzedniego(n_biezacy)
    
    # Info w sidebarze
    st.sidebar.divider()
    st.sidebar.caption(f"📅 Sezon 2025/26 · {n_biezacy} meczów w bazie")
    st.sidebar.caption(f"ρ Dixon-Coles: `{rho:.4f}` · w_prev: `{w_prev:.2f}`")
    
    if not schedule.empty:
        aktualna_kolejka = get_current_round(schedule)
        wszystkie_kolejki = sorted(schedule["round"].unique())
        pozycja = wszystkie_kolejki.index(aktualna_kolejka) + 1 if aktualna_kolejka in wszystkie_kolejki else 0
        st.sidebar.progress(pozycja / len(wszystkie_kolejki), 
                           text=f"Kolejka {pozycja}/{len(wszystkie_kolejki)}")
        st.sidebar.info(f"⚽ Aktualna kolejka: **#{aktualna_kolejka}**")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Ranking Zdarzeń",
        "⚽ Analiza Meczu", 
        "📈 Skuteczność"
    ])
    
    # =========================================================================
    # TAB 1 - RANKING ZDARZEŃ
    # =========================================================================
    with tab1:
        st.subheader("📊 Ranking zdarzeń kolejki")
        st.caption("Value Bets, Safe Haven i Shot Kings dla najbliższej kolejki.")
        
        if schedule.empty or srednie_df.empty:
            st.warning("Brak danych do wygenerowania rankingu")
        else:
            aktualna_kolejka = get_current_round(schedule)
            mecze = schedule[schedule["round"] == aktualna_kolejka]
            status = get_round_status(schedule, aktualna_kolejka)
            
            st.info(f"📅 Kolejka #{aktualna_kolejka} ({status})")
            
            with st.spinner("Generowanie rankingu..."):
                wszystkie_zd = []
                shot_kings = []
                
                for _, mecz in mecze.iterrows():
                    h = map_nazwa(mecz["home_team"])
                    a = map_nazwa(mecz["away_team"])
                    
                    if h not in srednie_df.index or a not in srednie_df.index:
                        continue
                    
                    lam_h, lam_a, lam_r, lam_k, sot_ok, lam_sot = oblicz_lambdy(
                        h, a, srednie_df, srednie_lig, forma_dict
                    )
                    pred = predykcja_meczu(lam_h, lam_a, rho=rho)
                    mecz_str = f"{h} – {a}"
                    
                    def _ev(p_val, fo_val):
                        return round(p_val * fo_val - 1.0, 3)
                    
                    # Typ główny
                    if pred["p_typ"] >= 0.58 and pred["fo_typ"] >= 1.30:
                        ev = _ev(pred["p_typ"], pred["fo_typ"])
                        wszystkie_zd.append({
                            "Mecz": mecz_str,
                            "Rynek": "1X2",
                            "Typ": pred["typ"],
                            "P": pred["p_typ"],
                            "Fair": pred["fo_typ"],
                            "EV": ev,
                            "Kategoria": "1X2"
                        })
                    
                    # Alternatywne
                    alt = alternatywne_zdarzenia(lam_h, lam_a, lam_r, lam_k, rho, lam_sot=lam_sot)
                    for emoji, nazwa, p, fo, kat, linia in alt:
                        if fo >= 1.30:
                            ev = _ev(p, fo)
                            wszystkie_zd.append({
                                "Mecz": mecz_str,
                                "Rynek": kat,
                                "Typ": nazwa,
                                "P": p,
                                "Fair": fo,
                                "EV": ev,
                                "Kategoria": kat
                            })
                    
                    # Shot Kings
                    if lam_sot and lam_sot > 0:
                        p_over_55 = 1 - poisson.cdf(5, lam_sot)
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
                
                # Value Bets
                st.markdown("### 🔥 Value Bets (EV > 0)")
                value_bets = df_rank[df_rank["EV"] > 0].sort_values("EV", ascending=False)
                
                if not value_bets.empty:
                    for _, row in value_bets.head(10).iterrows():
                        ev_color = "#4CAF50" if row["EV"] > 0.05 else "#FF9800"
                        cols = st.columns([3, 1, 1, 1, 1])
                        cols[0].markdown(f"**{row['Mecz']}**<br><small>{row['Typ']}</small>", unsafe_allow_html=True)
                        cols[1].markdown(f"`{row['Rynek']}`")
                        cols[2].markdown(f"🎯 {row['P']:.0%}")
                        cols[3].markdown(f"💰 {row['Fair']:.2f}")
                        cols[4].markdown(f"<span style='color:{ev_color};font-weight:bold'>+{row['EV']:.3f}</span>", unsafe_allow_html=True)
                        st.divider()
                else:
                    st.info("Brak value bets w tej kolejce")
                
                # Safe Haven
                st.markdown("### 🛡️ Safe Haven (p > 70%)")
                safe_havens = df_rank[df_rank["P"] > 0.70].sort_values("P", ascending=False)
                
                if not safe_havens.empty:
                    for _, row in safe_havens.head(10).iterrows():
                        cols = st.columns([3, 1, 1, 2])
                        cols[0].markdown(f"**{row['Mecz']}**<br><small>{row['Typ']}</small>", unsafe_allow_html=True)
                        cols[1].markdown(f"`{row['Rynek']}`")
                        cols[2].markdown(f"🎯 {row['P']:.0%}")
                        cols[3].markdown(f"💹 fair {row['Fair']:.2f}")
                        st.divider()
                else:
                    st.info("Brak zdarzeń z p > 70%")
                
                # Shot Kings
                if shot_kings:
                    st.markdown("### 🎯 Shot Kings")
                    shot_df = pd.DataFrame(shot_kings).sort_values("Oczekiwane SOT", ascending=False)
                    for _, row in shot_df.head(10).iterrows():
                        cols = st.columns([3, 1, 2])
                        cols[0].markdown(f"**{row['Mecz']}**")
                        cols[1].markdown(f"🎯 {row['Oczekiwane SOT']} SOT")
                        if row["Over 5.5 SOT"] > 0:
                            cols[2].markdown(f"Over 5.5: {row['Over 5.5 SOT']:.0%} (fair {row['Fair']:.2f})")
                        st.divider()
            else:
                st.info("Brak zdarzeń spełniających kryteria")
    
    # =========================================================================
    # TAB 2 - ANALIZA MECZU
    # =========================================================================
    with tab2:
        st.subheader("⚽ Analiza meczu – aktualna kolejka")
        
        if schedule.empty or srednie_df.empty:
            st.warning("Brak danych")
        else:
            aktualna_kolejka = get_current_round(schedule)
            mecze = schedule[schedule["round"] == aktualna_kolejka]
            
            if mecze.empty:
                st.info("Brak meczów w tej kolejce")
            else:
                st.caption(f"Kolejka #{aktualna_kolejka} – {len(mecze)} meczów")
                
                pokaz_komentarz = st.toggle("💬 Komentarz", value=True)
                
                # Grid meczów
                cols = st.columns(2)
                for idx, (_, mecz) in enumerate(mecze.iterrows()):
                    h = map_nazwa(mecz["home_team"])
                    a = map_nazwa(mecz["away_team"])
                    
                    if h not in srednie_df.index or a not in srednie_df.index:
                        continue
                    
                    lam_h, lam_a, lam_r, lam_k, sot_ok, lam_sot = oblicz_lambdy(
                        h, a, srednie_df, srednie_lig, forma_dict
                    )
                    pred = predykcja_meczu(lam_h, lam_a, rho=rho)
                    data_meczu = mecz["date"].strftime("%d.%m %H:%M") if pd.notna(mecz["date"]) else ""
                    
                    with cols[idx % 2]:
                        with st.container(border=True):
                            # Nagłówek
                            st.markdown(ui.match_header(h, a, data_meczu, pred["conf_level"]), unsafe_allow_html=True)
                            
                            # Wynik przewidywany
                            st.markdown(
                                f"<div style='text-align: center; font-size: 2em; font-weight: bold; margin: 10px 0;'>"
                                f"⚽ {pred['wynik_h']}:{pred['wynik_a']} "
                                f"<span style='font-size: 0.5em; color: #888;'>({pred['p_exact']:.1%})</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                            
                            # Typ i fair odds
                            st.markdown(
                                f"<div style='text-align: center; margin: 10px 0;'>"
                                f"Typ: {ui.type_badge(pred['typ'])} "
                                f"<span style='color: #888; margin-left: 10px;'>Fair: <b>{pred['fo_typ']:.2f}</b> ({pred['p_typ']:.1%})</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                            
                            # Chaos indicator
                            st.markdown(
                                ui.chaos_indicator(pred["chaos_label"], pred["chaos_emoji"], pred["chaos_pct"]),
                                unsafe_allow_html=True
                            )
                            
                            # Prawdopodobieństwa
                            pcols = st.columns(3)
                            pcols[0].metric("1", f"{pred['p_home']:.0%}", f"fair {pred['fo_home']}")
                            pcols[1].metric("X", f"{pred['p_draw']:.0%}", f"fair {pred['fo_draw']}")
                            pcols[2].metric("2", f"{pred['p_away']:.0%}", f"fair {pred['fo_away']}")
                            
                            # Info o modelu
                            sot_info = " 🎯 SOT blend" if sot_ok else " Gole only"
                            st.caption(f"λ {h[:8]}: {lam_h:.2f} | λ {a[:8]}: {lam_a:.2f} | Σ: {lam_h+lam_a:.2f}{sot_info}")
                            
                            # Alternatywne rynki
                            with st.expander("📊 Alternatywne rynki"):
                                alt = alternatywne_zdarzenia(lam_h, lam_a, lam_r, lam_k, rho, lam_sot=lam_sot)
                                if alt:
                                    alt_df = pd.DataFrame(
                                        alt, 
                                        columns=["Emoji", "Nazwa", "P", "Fair", "Kategoria", "Linia"]
                                    )
                                    alt_df = alt_df[alt_df["P"] >= 0.55].head(8)
                                    
                                    for _, arow in alt_df.iterrows():
                                        cat_colors = {
                                            "Gole": "#2196F3", "BTTS": "#9C27B0", 
                                            "Rożne": "#FF9800", "Kartki": "#F44336", "SOT": "#00BCD4"
                                        }
                                        color = cat_colors.get(arow["Kategoria"], "#888")
                                        st.markdown(
                                            f"{arow['Emoji']} **{arow['Nazwa']}** "
                                            f"<span style='color: {color};'>{arow['P']:.0%}</span> "
                                            f"<span style='color: #888;'>(fair {arow['Fair']:.2f})</span>",
                                            unsafe_allow_html=True
                                        )
                                else:
                                    st.caption("Brak zdarzeń powyżej progu 55%")
                            
                            # Komentarz
                            if pokaz_komentarz:
                                fh = forma_dict.get(h, "?")
                                fa = forma_dict.get(a, "?")
                                roznica = pred["p_home"] - pred["p_away"]
                                
                                if roznica > 0.20:
                                    faw = f"{h} jest wyraźnym faworytem ({pred['p_home']:.0%})."
                                elif roznica > 0.08:
                                    faw = f"{h} jest lekkim faworytem."
                                elif roznica < -0.20:
                                    faw = f"{a} jest wyraźnym faworytem ({pred['p_away']:.0%})."
                                elif roznica < -0.08:
                                    faw = f"{a} jest lekkim faworytem."
                                else:
                                    faw = f"Mecz wyrównany – remis ma {pred['p_draw']:.0%} szans."
                                
                                gole = lam_h + lam_a
                                gole_opis = f"Spodziewane gole: {gole:.1f}"
                                
                                st.info(f"{faw} Forma: {h} [{fh}] vs {a} [{fa}]. {gole_opis}.")
    
    # =========================================================================
    # TAB 3 - SKUTECZNOŚĆ
    # =========================================================================
    with tab3:
        st.subheader("📈 Skuteczność modelu")
        
        stats_df = statystyki_skutecznosci(wybrana_liga)
        
        if stats_df.empty:
            st.info("Brak danych do analizy skuteczności")
            st.markdown("""
            **Jak zacząć zbierać dane?**
            1. Przejdź do zakładki ⚽ Analiza Meczu
            2. Włącz przełącznik 💾 Zapisz zdarzenia (wkrótce dostępny)
            3. Poczekaj na wyniki meczów
            4. System automatycznie wyliczy skuteczność
            """)
        else:
            # Metryki globalne
            total_typow = int(stats_df["Typów"].sum())
            total_trafion = int(stats_df["Trafione"].sum())
            avg_skut = total_trafion / total_typow if total_typow > 0 else 0
            
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("📋 Typów", total_typow)
            mcol2.metric("✅ Trafione", total_trafion)
            mcol3.metric("🎯 Hit Rate", f"{avg_skut:.1%}")
            
            roi_vals = stats_df["ROI"].str.replace("+", "").str.replace("%", "").astype(float)
            w_roi = (roi_vals * stats_df["Typów"]).sum() / total_typow if total_typow > 0 else 0
            mcol4.metric("💹 Ważony ROI", f"{w_roi:+.1f}%")
            
            # Tabela per rynek
            st.divider()
            st.markdown("### Szczegóły per rynek")
            
            # Sortowanie
            sort_opt = st.radio("Sortuj po", ["ROI", "Brier", "Typów"], horizontal=True)
            if sort_opt == "ROI":
                stats_sorted = stats_df.sort_values("_roi_v", ascending=False)
            elif sort_opt == "Brier":
                stats_sorted = stats_df.sort_values("_brier_v", ascending=True)
            else:
                stats_sorted = stats_df.sort_values("Typów", ascending=False)
            
            # Wyświetlenie tabeli
            for _, row in stats_sorted.iterrows():
                with st.container(border=True):
                    rcols = st.columns([2, 1, 1, 1, 1, 1])
                    rcols[0].markdown(f"**{row['Rynek']}**")
                    rcols[1].markdown(f"`{row['Typów']} typów`")
                    rcols[2].markdown(f"🎯 {row['Skuteczność']}")
                    rcols[3].markdown(f"Brier: {row['Brier ↓']}")
                    rcols[4].markdown(f"Skill: {row['Skill']}")
                    
                    roi_color = "#4CAF50" if row['_roi_v'] > 0.03 else ("#F44336" if row['_roi_v'] < -0.03 else "#FF9800")
                    rcols[5].markdown(f"<span style='color: {roi_color}; font-weight: bold;'>{row['ROI']}</span>", unsafe_allow_html=True)
    
    # Debug
    if st.session_state.debug_mode and _niezmapowane:
        st.sidebar.divider()
        st.sidebar.write("**🔧 Niezmapowane nazwy**")
        for n in sorted(_niezmapowane):
            st.sidebar.warning(f"`{n}`")

if __name__ == "__main__":
    main()
