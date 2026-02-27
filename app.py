import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import requests
from io import StringIO
import sqlite3
import unicodedata
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import poisson

# ===========================================================================
# KONFIGURACJA
# ===========================================================================
st.set_page_config(
    page_title="Smart Bet Builder Pro | Advanced Football Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicjalizacja stanu sesji
def init_session_state():
    defaults = {
        'sot_blend_w': 0.35,
        'tau_days': 25.0,
        'shrink_alpha': 0.15,
        'rho_dc': -0.13,
        'combo_edge_threshold': 0.05,
        'max_combo_legs': 4,
        'portfolio': [],
        'debug_mode': False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# ===========================================================================
# LIGI I KONFIGURACJA
# ===========================================================================
LIGI = {
    "Premier League": {"csv_code": "E0", "file": "terminarz_premier_2025.csv"},
    "La Liga": {"csv_code": "SP1", "file": "terminarz_la_liga_2025.csv"},
    "Bundesliga": {"csv_code": "D1", "file": "terminarz_bundesliga_2025.csv"},
    "Serie A": {"csv_code": "I1", "file": "terminarz_serie_a_2025.csv"},
    "Ligue 1": {"csv_code": "F1", "file": "terminarz_ligue_1_2025.csv"},
}

LIGA_PROFILES = {
    "Premier League": {
        "dampening_factor": 0.5,
        "avg_goals": 2.7,
        "style": "balanced_intensity"
    },
    "Bundesliga": {
        "dampening_factor": 0.6,
        "avg_goals": 3.2,
        "style": "high": "high_intensity"
    },
    "Serie A": {
        "dampening_factor": 0.4,
        "avg_goals": 2.3,
        "style": "tactical_defensive"
    },
    "La Liga": {
        "dampening_factor": 0.45,
        "avg_goals": 2.5,
        "style": "technical"
    },
    "Ligue 1": {
        "dampening_factor": 0.45,
        "avg_goals": 2.4,
        "style": "unbalanced"
    }
}

DB_FILE = "predykcje.db"

# ===========================================================================
# MAPOWANIA
# ===========================================================================
def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return str(name)
    name = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode("ASCII")
    for suffix in [" FC", " CF", " UD", " CD", " RCD", " AS", " Stade", " 1."]:
        name = name.replace(suffix, "")
    return name.strip()

NAZWY_MAP = {
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
    "AC Milan": "Milan",
    "Internazionale": "Inter",
    "AS Roma": "Roma",
    "Hellas Verona": "Verona",
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

# ===========================================================================
# BAZA DANYCH
# ===========================================================================
def init_db():
    con = sqlite3.connect(DB_FILE)
    
    con.execute("""
        CREATE TABLE IF NOT EXISTS predykcje (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            liga TEXT, home TEXT, away TEXT, round INTEGER,
            typ TEXT, fo_typ REAL, p_home REAL, p_draw REAL, p_away REAL,
            data TEXT, UNIQUE(liga, home, away, round)
        )
    """)
    
    con.execute("""
        CREATE TABLE IF NOT EXISTS zdarzenia (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            liga TEXT, kolejnosc INTEGER, mecz TEXT,
            home TEXT, away TEXT, rynek TEXT, linia REAL,
            typ TEXT, p_model REAL, fair_odds REAL,
            wynik TEXT, trafione BOOLEAN, data TEXT,
            UNIQUE(liga, kolejnosc, mecz, rynek, typ, linia)
        )
    """)
    
    con.commit()
    con.close()

# ===========================================================================
# ŁADOWANIE DANYCH
# ===========================================================================
@st.cache_data(ttl=900)
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
        
        df["total_gole"] = df["FTHG"] + df["FTAG"]
        df["total_kartki"] = df["HY"] + df["AY"] + (df["HR"] + df["AR"]) * 2
        df["total_rozne"] = df["HC"] + df["AC"]
        df["HST"] = pd.to_numeric(df["HST"], errors="coerce")
        df["AST"] = pd.to_numeric(df["AST"], errors="coerce")
        
        return df.sort_values("Date")
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=900)
def load_historical(league_code: str) -> pd.DataFrame:
    df_now = _pobierz_csv(league_code, "2526")
    df_prev = _pobierz_csv(league_code, "2425")
    
    if df_now.empty and df_prev.empty:
        return pd.DataFrame()
    if df_now.empty:
        return df_prev
    if df_prev.empty:
        return df_now
    
    n_now = len(df_now)
    w_prev = np.clip(0.8 - (n_now / 30) * 0.6, 0.2, 0.8)
    n_prev_target = min(int(n_now * w_prev / (1 - w_prev)), len(df_prev))
    
    df_prev_s = df_prev.tail(n_prev_target).copy()
    df_prev_s["_sezon"] = "poprzedni"
    df_now = df_now.copy()
    df_now["_sezon"] = "biezacy"
    
    return pd.concat([df_prev_s, df_now], ignore_index=True).sort_values("Date")

@st.cache_data(ttl=86400)
def load_schedule(filename: str) -> pd.DataFrame:
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
# STATYSTYKI I MODEL
# ===========================================================================
def weighted_mean(values: pd.Series, dates: pd.Series = None, tau_days: float = None) -> float:
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
def oblicz_wszystkie_statystyki(df_json: str, tau_days: float = 25.0) -> pd.DataFrame:
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

def oblicz_lambdy(h: str, a: str, srednie_df: pd.DataFrame, srednie_lig: dict, forma_dict: dict) -> tuple:
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

def get_current_round(schedule: pd.DataFrame) -> int:
    if schedule.empty:
        return 0
    
    dzisiaj = datetime.now().date()
    future_matches = schedule[schedule["date"].dt.date >= dzisiaj]
    
    if not future_matches.empty:
        return int(future_matches.iloc[0]["round"])
    else:
        return int(schedule.iloc[-1]["round"])

def get_round_status(schedule: pd.DataFrame, round_num: int) -> str:
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
# MODEL DIXON-COLES
# ===========================================================================
def dixon_coles_adj(M: np.ndarray, lam_h: float, lam_a: float, rho: float = -0.13) -> np.ndarray:
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
    a = st.session_state.shrink_alpha
    p_h = (1 - a) * p_home + a / 3
    p_d = (1 - a) * p_draw + a / 3
    p_a = (1 - a) * p_away + a / 3
    s = p_h + p_d + p_a
    return p_h / s, p_d / s, p_a / s

def wybierz_typ(p_home: float, p_draw: float, p_away: float) -> tuple:
    p_home, p_draw, p_away = kalibruj_prawdopodobienstwa(p_home, p_draw, p_away)
    
    p_1x = p_home + p_draw
    p_x2 = p_away + p_draw
    
    if p_home >= 0.55:
        return "1", p_home
    if p_away >= 0.55:
        return "2", p_away
    if p_1x >= 0.55 or p_x2 >= 0.55:
        return ("1X", p_1x) if p_1x >= p_x2 else ("X2", p_x2)
    
    probs = {"1": p_home, "X": p_draw, "2": p_away}
    t = max(probs, key=probs.get)
    return t, probs[t]

def fair_odds(p: float) -> float:
    return round(1 / p, 2) if 0 < p <= 1 else 999.0

@st.cache_data(ttl=300)
def predykcja_meczu_cached(lam_h: float, lam_a: float, rho: float, config_hash: str) -> dict:
    max_gole = int(np.clip(np.ceil(max(lam_h, lam_a) + 4), 6, 10))
    
    M = dixon_coles_adj(
        np.outer(poisson.pmf(range(max_gole), lam_h),
                 poisson.pmf(range(max_gole), lam_a)),
        lam_h, lam_a, rho=rho
    )
    
    p_home = float(np.tril(M, -1).sum())
    p_draw = float(np.trace(M))
    p_away = float(np.triu(M, 1).sum())
    
    idx_max = np.unravel_index(M.argmax(), M.shape)
    p_max = float(M[idx_max])
    
    top3 = np.sort(M.flatten())[::-1][:3]
    if float(top3[0] - top3[2]) < 0.04:
        wynik_h = min(int(round(lam_h)), M.shape[0] - 1)
        wynik_a = min(int(round(lam_a)), M.shape[1] - 1)
        p_exact = float(M[wynik_h, wynik_a])
    else:
        wynik_h, wynik_a = int(idx_max[0]), int(idx_max[1])
        p_exact = p_max
    
    p_home_cal, p_draw_cal, p_away_cal = kalibruj_prawdopodobienstwa(p_home, p_draw, p_away)
    typ, p_typ = wybierz_typ(p_home, p_draw, p_away)
    
    vals = sorted([p_home_cal, p_draw_cal, p_away_cal], reverse=True)
    spread = vals[0] - vals[2]
    edge = vals[0] - vals[1]
    
    if edge > 0.18:
        conf_level, conf_emoji, conf_opis = "High", "🟢", f"Wyraźny faworyt (+{edge:.0%} nad 2. opcją)"
    elif edge > 0.08:
        conf_level, conf_emoji, conf_opis = "Medium", "🟡", f"Umiarkowana przewaga (+{edge:.0%})"
    else:
        conf_level, conf_emoji, conf_opis = "Coinflip", "🔴", f"Mecz bardzo wyrównany (spread {spread:.0%})"
    
    ent = -sum(p * np.log2(p) for p in [p_home_cal, p_draw_cal, p_away_cal] if p > 0)
    pct = ent / np.log2(3)
    
    if pct < 0.55:
        ch_label, ch_emoji = "Pewny", "🔒"
    elif pct < 0.75:
        ch_label, ch_emoji = "Klarowny", "🎯"
    elif pct < 0.90:
        ch_label, ch_emoji = "Otwarty", "⚡"
    else:
        ch_label, ch_emoji = "Chaos", "🌀"
    
    return {
        "lam_h": lam_h, "lam_a": lam_a,
        "p_home": p_home_cal, "p_draw": p_draw_cal, "p_away": p_away_cal,
        "p_home_raw": p_home, "p_draw_raw": p_draw, "p_away_raw": p_away,
        "wynik_h": wynik_h, "wynik_a": wynik_a, "p_exact": p_exact,
        "fo_home": fair_odds(p_home_cal), "fo_draw": fair_odds(p_draw_cal), "fo_away": fair_odds(p_away_cal),
        "typ": typ, "p_typ": p_typ, "fo_typ": fair_odds(p_typ),
        "conf_level": conf_level, "conf_emoji": conf_emoji, "conf_opis": conf_opis,
        "entropy": ent, "chaos_label": ch_label, "chaos_emoji": ch_emoji, "chaos_pct": pct,
        "macierz": M,
    }

def predykcja_meczu(lam_h: float, lam_a: float, rho: float = -0.13) -> dict:
    config_str = f"{st.session_state.shrink_alpha}_0.55_0.55"
    config_hash = str(hash(config_str))
    return predykcja_meczu_cached(lam_h, lam_a, rho, config_hash)

@st.cache_data(ttl=300)
def alternatywne_zdarzenia_cached(lam_h: float, lam_a: float, lam_r: float,
                                   lam_k: float, rho: float, lam_sot: float = None,
                                   prog_min: float = 0.55) -> list:
    zdarzenia = []
    mg = int(np.clip(np.ceil(max(lam_h, lam_a) + 4), 6, 10))
    
    M = dixon_coles_adj(
        np.outer(poisson.pmf(range(mg), lam_h), poisson.pmf(range(mg), lam_a)),
        lam_h, lam_a, rho=rho
    )

    for linia in [1.5, 2.5, 3.5]:
        p_over = float(sum(M[i,j] for i in range(mg) for j in range(mg) if i+j > linia))
        p_under = 1 - p_over
        if p_over >= prog_min:
            zdarzenia.append(("⚽", f"Over {linia} goli", p_over, fair_odds(p_over), "Gole", linia))
        if p_under >= prog_min:
            zdarzenia.append(("⚽", f"Under {linia} goli", p_under, fair_odds(p_under), "Gole", linia))

    p_btts = float(1 - M[0,:].sum() - M[:,0].sum() + M[0,0])
    p_nobtts = 1 - p_btts
    if p_btts >= prog_min:
        zdarzenia.append(("⚽", "BTTS – Tak", p_btts, fair_odds(p_btts), "BTTS", 0))
    if p_nobtts >= prog_min:
        zdarzenia.append(("⚽", "BTTS – Nie", p_nobtts, fair_odds(p_nobtts), "BTTS", 0))

    for linia in [7.5, 8.5, 9.5, 10.5]:
        p_over = float(1 - poisson.cdf(int(linia), lam_r))
        if p_over >= prog_min:
            zdarzenia.append(("🚩", f"Over {linia} rożnych", p_over, fair_odds(p_over), "Rożne", linia))

    for linia in [2.5, 3.5, 4.5]:
        p_over = float(1 - poisson.cdf(int(linia), lam_k))
        if p_over >= prog_min:
            zdarzenia.append(("🟨", f"Over {linia} kartek", p_over, fair_odds(p_over), "Kartki", linia))

    if lam_sot is not None and lam_sot > 0:
        for linia in [3.5, 4.5, 5.5, 6.5]:
            p_over = float(1 - poisson.cdf(int(linia), lam_sot))
            fo_sot = fair_odds(p_over)
            if p_over >= prog_min and fo_sot >= 1.30:
                zdarzenia.append(("🎯", f"Over {linia} celnych", p_over, fo_sot, "SOT", linia))

    return sorted(zdarzenia, key=lambda x: -x[2])

def alternatywne_zdarzenia(lam_h: float, lam_a: float, lam_r: float,
                           lam_k: float, rho: float, prog_min: float = 0.55,
                           lam_sot: float = None) -> list:
    return alternatywne_zdarzenia_cached(lam_h, lam_a, lam_r, lam_k, rho, lam_sot, prog_min)

# ===========================================================================
# KORELACJE I SMART BET BUILDER
# ===========================================================================
class CorrelationEngine:
    def __init__(self):
        self._typical_correlations = {
            ('goals_over', 'btts_yes'): 0.65,
            ('goals_over', 'corners_over'): 0.45,
            ('goals_over', 'shots_over'): 0.70,
            ('btts_yes', 'cards_over'): 0.25,
            ('home_win', 'corners_over'): 0.15,
            ('away_win', 'cards_over'): 0.20,
        }
    
    def estimate_combo_probability(self, probs: List[float], markets: List[str], liga_profile: dict = None) -> float:
        if len(probs) == 0:
            return 0.0
        if len(probs) == 1:
            return probs[0]
        
        base_prob = np.prod(probs)
        
        dampening = liga_profile.get('dampening_factor', 0.5) if liga_profile else 0.5
        adjustment = 1.0
        
        for i, m1 in enumerate(markets):
            for j, m2 in enumerate(markets[i+1:], i+1):
                corr = self._get_correlation(m1, m2)
                adjustment *= (1 + corr * dampening)
        
        adjusted_prob = np.clip(base_prob * adjustment, base_prob * 0.5, min(probs) * 0.95)
        return adjusted_prob
    
    def _get_correlation(self, m1: str, m2: str) -> float:
        key = tuple(sorted([m1, m2]))
        return self._typical_correlations.get(key, 0.0)

correlation_engine = CorrelationEngine()

# ===========================================================================
# EDGE VISUALIZATION (KOMERCYJNE)
# ===========================================================================
class EdgeVisualizer:
    def render_money_on_table(self, combo: Dict, stake: float = 100.0):
        ev = combo.get('ev', 0)
        prob = combo.get('combined_prob', 0)
        odds = combo.get('combined_odds', 2.0)
        
        expected_profit = stake * ev
        break_even_rate = 1 / odds
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            profit_color = "#4CAF50" if expected_profit > 0 else "#F44336"
            fig = go.Figure(go.Indicator(
                mode="number+delta",
                value=expected_profit,
                number={'prefix': "€", 'font': {'size': 48, 'color': profit_color}},
                delta={'reference': 0, 'relative': False, 'valueformat': '.0f'},
                title={'text': "Expected Profit<br><span style='font-size:0.5em'>na 100€ stawki</span>"},
            ))
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[prob * 100],
                y=['Model'],
                orientation='h',
                name='Model P%',
                marker_color='#4CAF50',
                text=f"{prob:.1%}",
                textposition='inside'
            ))
            fig.add_trace(go.Scatter(
                x=[break_even_rate * 100],
                y=['Model'],
                mode='markers+text',
                name='Break-even',
                marker=dict(size=20, color='#F44336', symbol='line-ns'),
                text=['Break-even'],
                textposition='top center'
            ))
            fig.update_layout(
                title="Edge Visualization",
                xaxis_title="Prawdopodobieństwo (%)",
                showlegend=False,
                height=200,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            simulations = np.random.binomial(1, prob, 1000)
            cumulative = np.cumsum(simulations * odds * stake - stake)
            
            fig = px.line(
                x=range(1, 1001),
                y=cumulative,
                title="Symulacja: 1000 zakładów",
                labels={'x': 'Numer zakładu', 'y': 'Zysk/Strata (€)'}
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
    
    def render_confidence_interval(self, combo: Dict):
        prob = combo.get('combined_prob', 0.5)
        n_samples = combo.get('sample_size', 100)
        
        z = 1.96
        denominator = 1 + z**2/n_samples
        centre_adjusted_probability = prob + z*z / (2*n_samples)
        adjusted_standard_deviation = np.sqrt((prob*(1-prob) + z*z / (4*n_samples)) / n_samples)
        
        lower_bound = (centre_adjusted_probability - z*adjusted_standard_deviation) / denominator
        upper_bound = (centre_adjusted_probability + z*adjusted_standard_deviation) / denominator
        
        be = 1 / combo.get('combined_odds', 2.0)
        
        fig = go.Figure()
        fig.add_vrect(
            x0=lower_bound, x1=upper_bound,
            fillcolor="LightSalmon", opacity=0.3,
            layer="below", line_width=0,
        )
        fig.add_trace(go.Scatter(
            x=[prob],
            y=[1],
            mode='markers',
            marker=dict(size=30, color='#2196F3'),
            name='Model Estimate'
        ))
        fig.add_vline(x=be, line_dash="dash", line_color="red", annotation_text="Break-even")
        
        fig.update_layout(
            title=f"Niepewność modelu (95% CI)<br>Prawdopodobieństwo: {lower_bound:.1%} - {upper_bound:.1%}",
            xaxis_title="Prawdopodobieństwo sukcesu",
            yaxis_visible=False,
            showlegend=False,
            height=250
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if lower_bound > be:
            st.success("✅ **High Confidence Edge** - Nawet pesymistyczny scenariusz daje zysk")
        elif upper_bound < be:
            st.error("❌ **No Edge** - Nawet optymistyczny scenariusz poniżej break-even")
        else:
            st.warning("⚠️ **Marginal Edge** - Zależy od wariancji, wymaga większej próby")

class SyndicateTools:
    def render_staking_calculator(self, edge: float, bankroll: float, kelly_fraction: float = 0.25):
        b = edge
        p = 0.5 + edge/2
        q = 1 - p
        
        kelly_pct = (b * p - q) / b if b > 0 else 0
        kelly_pct = max(0, min(kelly_pct, 0.5))
        
        fractional_kelly = kelly_pct * kelly_fraction
        recommended_stake = bankroll * fractional_kelly
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Full Kelly", f"{kelly_pct:.1%}",
                   help="Teoretycznie optymalna stawka")
        col2.metric(f"Fractional ({kelly_fraction:.0%})", f"{fractional_kelly:.1%}",
                   help="Konserwatywna wersja Kelly")
        col3.metric("Rekomendowana stawka", f"€{recommended_stake:.2f}",
                   help=f""Przy bankrollu €{bankroll:.0f}")
        
        fractions = np.linspace(0, kelly_pct * 1.5, 50)
        growth_rates = [p * np.log(1 + f * b) + q * np.log(1 - f) for f in fractions]
        
        fig = px.line(
            x=fractions * 100,
            y=growth_rates,
            title="Expected Growth Rate vs Stake Size",
            labels={'x': 'Stawka (% bankrollu)', 'y': 'Expected log growth'}
        )
        fig.add_vline(x=kelly_pct*100, line_dash="dash", line_color="green", annotation_text="Full Kelly")
        fig.add_vline(x=fractional_kelly*100, line_dash="dash", line_color="orange", annotation_text="Your setting")
        st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# SMART COMBO GENERATOR
# ===========================================================================
class SmartComboGenerator:
    def __init__(self, correlation_engine: CorrelationEngine):
        self.ce = correlation_engine
        self.min_edge = st.session_state.combo_edge_threshold
        self.max_legs = st.session_state.max_combo_legs
    
    def generate_combos(self, match_data: Dict, available_markets: List[Dict], liga_profile: dict = None) -> List[Dict]:
        valid_markets = [m for m in available_markets if m.get('probability', 0) > 0.5]
        combos = []
        
        # Single legs
        for market in valid_markets:
            ev = market['probability'] * market.get('odds', 2.0) - 1.0
            if ev > self.min_edge:
                combos.append({
                    'type': 'single',
                    'legs': [market],
                    'combined_prob': market['probability'],
                    'combined_odds': market.get('odds', 2.0),
                    'ev': ev,
                    'rationale': f"Value single: {market['name']}"
                })
        
        # Doubles with correlation
        for i, m1 in enumerate(valid_markets):
            for m2 in valid_markets[i+1:]:
                if len(combos) > 50:
                    break
                
                combined_prob = self.ce.estimate_combo_probability(
                    [m1['probability'], m2['probability']],
                    [m1['market_type'], m2['market_type']],
                    liga_profile
                )
                
                market_odds = m1.get('odds', 2.0) * m2.get('odds', 2.0)
                implied_prob = 1 / market_odds
                edge = combined_prob - implied_prob
                
                if edge > self.min_edge:
                    combos.append({
                        'type': 'double',
                        'legs': [m1, m2],
                        'combined_prob': combined_prob,
                        'combined_odds': market_odds,
                        'naive_prob': m1['probability'] * m2['probability'],
                        'correlation_effect': combined_prob / (m1['probability'] * m2['probability']) - 1,
                        'ev': edge,
                        'rationale': f"Correlation play: {m1['name']} + {m2['name']}"
                    })
        
        return sorted(combos, key=lambda x: x['ev'], reverse=True)

# ===========================================================================
# UI KOMPONENTY
# ===========================================================================
def render_header():
    st.markdown("""
    <style>
    .main-header { font-size: 2.5em; font-weight: bold; margin-bottom: 0; }
    .sub-header { color: #666; font-size: 1.1em; margin-top: 0; }
    </style>
    """, unsafe_allow_html=True)

# ===========================================================================
# GŁÓWNA APLIKACJA
# ===========================================================================
def main():
    render_header()
    st.markdown('<p class="main-header">🧠 Smart Bet Builder Pro</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Football Analytics with Correlation-Aware Combo Engine</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Konfiguracja")
    wybrana_liga = st.sidebar.selectbox("Wybierz ligę", list(LIGI.keys()))
    
    with st.sidebar.expander("Parametry modelu", expanded=False):
        st.session_state.sot_blend_w = st.slider("Waga SOT", 0.0, 0.5, 0.35, 0.05)
        st.session_state.tau_days = st.slider("Tau (dni)", 10.0, 90.0, 25.0, 5.0)
        st.session_state.shrink_alpha = st.slider("Shrinkage", 0.0, 0.5, 0.15, 0.05)
    
    with st.sidebar.expander("Bet Builder", expanded=False):
        st.session_state.combo_edge_threshold = st.slider("Min Edge", 0.0, 0.20, 0.05, 0.01)
        st.session_state.max_combo_legs = st.slider("Max nóg", 2, 5, 4)
    
    # Ładowanie danych
    with st.spinner("Ładowanie danych..."):
        historical = load_historical(LIGI[wybrana_liga]["csv_code"])
        schedule = load_schedule(LIGI[wybrana_liga]["file"])
    
    if historical.empty:
        st.error("Brak danych historycznych")
        return
    
    # Obliczenia
    srednie_df = oblicz_wszystkie_statystyki(historical.to_json(), st.session_state.tau_days)
    srednie_lig = oblicz_srednie_ligowe(historical.to_json())
    forma_dict = oblicz_forme(historical)
    rho = srednie_lig["rho"]
    liga_profile = LIGA_PROFILES.get(wybrana_liga, LIGA_PROFILES["Premier League"])
    
    # Info
    aktualna_kolejka = get_current_round(schedule) if not schedule.empty else 0
    st.sidebar.metric("Aktualna kolejka", f"#{aktualna_kolejka}")
    st.sidebar.caption(f"ρ: {rho:.3f} | Style: {liga_profile['style']}")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Smart Bet Builder", "📊 Analiza Meczu", "📈 Skuteczność"])
    
    # TAB 1: Smart Bet Builder
    with tab1:
        st.subheader("Korelacyjny Silnik Bet Buildera")
        st.caption("Wykrywamy prawdziwe zależności między rynkami, nie zakładamy niezależności")
        
        if schedule.empty:
            st.warning("Brak terminarza")
        else:
            mecze = schedule[schedule["round"] == aktualna_kolejka]
            match_options = []
            for _, m in mecze.iterrows():
                h = map_nazwa(m["home_team"])
                a = map_nazwa(m["away_team"])
                if h in srednie_df.index and a in srednie_df.index:
                    match_options.append(f"{h} vs {a}")
            
            if match_options:
                selected = st.selectbox("Wybierz mecz", match_options)
                home, away = selected.split(" vs ")
                
                lam_h, lam_a, lam_r, lam_k, sot_ok, lam_sot = oblicz_lambdy(
                    home, away, srednie_df, srednie_lig, forma_dict
                )
                
                # Przygotowanie rynków
                markets = []
                pred = predykcja_meczu(lam_h, lam_a, rho)
                
                # 1X2
                for typ, prob, fo in [('1', pred['p_home'], pred['fo_home']),
                                      ('X', pred['p_draw'], pred['fo_draw']),
                                      ('2', pred['p_away'], pred['fo_away'])]:
                    markets.append({
                        'name': f'1X2: {typ}',
                        'market_type': f'1x2_{typ}',
                        'probability': prob,
                        'odds': fo * 0.95
                    })
                
                # Totals
                for line in [2.5, 3.5]:
                    mg = int(np.clip(np.ceil(max(lam_h, lam_a) + 4), 6, 10))
                    M = dixon_coles_adj(
                        np.outer(poisson.pmf(range(mg), lam_h),
                                 poisson.pmf(range(mg), lam_a)),
                        lam_h, lam_a, rho=rho
                    )
                    p_over = sum(M[i,j] for i in range(mg) for j in range(mg) if i+j > line)
                    markets.append({
                        'name': f'Over {line} goals',
                        'market_type': 'goals_over',
                        'probability': p_over,
                        'odds': fair_odds(p_over) * 0.95,
                        'line': line
                    })
                
                # BTTS
                p_btts = 1 - M[0,:].sum() - M[:,0].sum() + M[0,0]
                markets.append({
                    'name': 'BTTS Yes',
                    'market_type': 'btts_yes',
                    'probability': p_btts,
                    'odds': fair_odds(p_btts) * 0.95
                })
                
                # Generowanie combo
                generator = SmartComboGenerator(correlation_engine)
                combos = generator.generate_combos(
                    {'home': home, 'away': away},
                    markets,
                    liga_profile
                )
                
                if combos:
                    st.markdown("### 🎯 Rekomendowane kombinacje")
                    
                    for i, combo in enumerate(combos[:10]):
                        with st.container(border=True):
                            cols = st.columns([3, 1, 1])
                            
                            ev_color = "#4CAF50" if combo['ev'] > 0.1 else "#FF9800" if combo['ev'] > 0 else "#F44336"
                            
                            with cols[0]:
                                st.markdown(f"**{combo['type'].upper()}** | {combo['rationale'][:50]}...")
                                for leg in combo['legs']:
                                    st.caption(f"• {leg['name']} ({leg['probability']:.0%})")
                            
                            with cols[1]:
                                st.markdown(f"<span style='color: {ev_color}; font-weight: bold; font-size: 1.3em;'>EV: +{combo['ev']:.2f}</span>", unsafe_allow_html=True)
                            
                            with cols[2]:
                                st.markdown(f"P: {combo['combined_prob']:.1%}")
                                st.markdown(f"Odds: {combo['combined_odds']:.2f}")
                            
                            if 'correlation_effect' in combo:
                                ce = combo['correlation_effect']
                                ce_color = "#4CAF50" if ce > 0 else "#F44336"
                                st.caption(f"Efekt korelacji: <span style='color: {ce_color};'>{ce:+.1%}</span> vs niezależności", unsafe_allow_html=True)
                            
                            # Komercyjna analiza
                            if st.button(f"💰 Analiza komercyjna #{i}", key=f"comm_{i}"):
                                visualizer = EdgeVisualizer()
                                syndicate = SyndicateTools()
                                
                                visualizer.render_money_on_table(combo)
                                
                                with st.expander("Szczegółowa analiza"):
                                    visualizer.render_confidence_interval(combo)
                                
                                with st.expander("Kelly Calculator"):
                                    kelly_frac = st.slider("Kelly %", 0.05, 0.5, 0.25, 0.05, key=f"kelly_{i}")
                                    syndicate.render_staking_calculator(combo['ev'], 1000.0, kelly_frac)
                else:
                    st.info("Brak combo spełniających kryteria edge. Spróbuj obniżyć próg.")
            else:
                st.warning("Brak dostępnych meczów")
    
    # TAB 2: Analiza Meczu
    with tab2:
        st.subheader("Szczegółowa analiza meczów")
        
        if not schedule.empty:
            mecze = schedule[schedule["round"] == aktualna_kolejka]
            
            for _, mecz in mecze.iterrows():
                h = map_nazwa(mecz["home_team"])
                a = map_nazwa(mecz["away_team"])
                
                if h not in srednie_df.index or a not in srednie_df.index:
                    continue
                
                lam_h, lam_a, lam_r, lam_k, sot_ok, lam_sot = oblicz_lambdy(
                    h, a, srednie_df, srednie_lig, forma_dict
                )
                pred = predykcja_meczu(lam_h, lam_a, rho)
                
                with st.expander(f"{h} vs {a} - {pred['typ']} @ {pred['fo_typ']:.2f}"):
                    cols = st.columns(3)
                    cols[0].metric("1", f"{pred['p_home']:.0%}", f"fair {pred['fo_home']:.2f}")
                    cols[1].metric("X", f"{pred['p_draw']:.0%}", f"fair {pred['fo_draw']:.2f}")
                    cols[2].metric("2", f"{pred['p_away']:.0%}", f"fair {pred['fo_away']:.2f}")
                    
                    st.caption(f"Przewidywany wynik: {pred['wynik_h']}:{pred['wynik_a']} | Chaos: {pred['chaos_label']} {pred['chaos_emoji']}")
                    
                    if sot_ok:
                        st.success("🎯 SOT blend aktywny")
    
    # TAB 3: Skuteczność
    with tab3:
        st.subheader("Tracking skuteczności")
        st.info("Zapisuj zakłady przyciskiem 'Analiza komercyjna' -> 'Dodaj do portfela' aby śledzić wyniki.")
        
        if st.session_state.get('portfolio'):
            st.markdown(f"**Zapisane zakłady:** {len(st.session_state['portfolio'])}")
            for item in st.session_state['portfolio'][-5:]:
                st.caption(f"• {item.get('rationale', 'Unknown')[:50]}... (EV: {item.get('ev', 0):.2f})")

if __name__ == "__main__":
    main()
