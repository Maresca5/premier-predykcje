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

# ===========================================================================
# KONFIGURACJA
# ===========================================================================
LIGI = {
    "Premier League": {"csv_code": "E0",  "file": "terminarz_premier_2025.csv"},
    "La Liga":        {"csv_code": "SP1", "file": "terminarz_la_liga_2025.csv"},
    "Bundesliga":     {"csv_code": "D1",  "file": "terminarz_bundesliga_2025.csv"},
    "Serie A":        {"csv_code": "I1",  "file": "terminarz_serie_a_2025.csv"},
    "Ligue 1":        {"csv_code": "F1",  "file": "terminarz_ligue_1_2025.csv"},
}

DB_FILE    = "predykcje.db"

def waga_poprzedniego(n_biezacy: int) -> float:
    return float(np.clip(0.8 - (n_biezacy / 30) * 0.6, 0.2, 0.8))

st.set_page_config(page_title="Football Analytics System", layout="wide", initial_sidebar_state="expanded")

# ===========================================================================
# MAPOWANIA NAZW
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
    "Brighton & Hove Albion":  "Brighton",
    "West Ham United":         "West Ham",
    "Newcastle United":        "Newcastle",
    "Tottenham Hotspur":       "Tottenham",
    "Leeds United":            "Leeds",
    "Manchester United":       "Man United",
    "Manchester City":         "Man City",
    "Nottingham Forest":       "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "Wolverhampton":           "Wolves",
    "Leicester City":          "Leicester",
    "Sheffield United":        "Sheffield Utd",
    # LA LIGA
    "Girona FC":               "Girona",
    "Rayo Vallecano":          "Vallecano",
    "RCD Mallorca":            "Mallorca",
    "FC Barcelona":            "Barcelona",
    "Deportivo Alavés":        "Alaves",
    "Deportivo Alaves":        "Alaves",
    "Levante UD":              "Levante",
    "Valencia CF":             "Valencia",
    "Real Sociedad":           "Sociedad",
    "Celta Vigo":              "Celta",
    "Getafe CF":               "Getafe",
    "Athletic Club":           "Ath Bilbao",
    "Athletic Bilbao":         "Ath Bilbao",
    "Sevilla FC":              "Sevilla",
    "RCD Espanyol":            "Espanol",
    "Espanyol":                "Espanol",
    "Atlético Madrid":         "Ath Madrid",
    "Atletico Madrid":         "Ath Madrid",
    "Elche CF":                "Elche",
    "Real Betis":              "Betis",
    "CA Osasuna":              "Osasuna",
    "Real Valladolid":         "Valladolid",
    "UD Las Palmas":           "Las Palmas",
    "Cadiz CF":                "Cadiz",
    "Granada CF":              "Granada",
    "UD Almeria":              "Almeria",
    # BUNDESLIGA
    "FC Bayern München":              "Bayern Munich",
    "Bayern":                         "Bayern Munich",
    "Borussia Dortmund":              "Dortmund",
    "Leipzig":                        "RB Leipzig",
    "Bayer 04 Leverkusen":            "Leverkusen",
    "Eintracht Frankfurt":            "Ein Frankfurt",
    "VfB Stuttgart":                  "Stuttgart",
    "VfL Wolfsburg":                  "Wolfsburg",
    "Borussia Mönchengladbach":       "M'gladbach",
    "Borussia Monchengladbach":       "M'gladbach",
    "1. FC Union Berlin":             "Union Berlin",
    "SC Freiburg":                    "Freiburg",
    "1. FC Köln":                     "FC Koln",
    "FC Köln":                        "FC Koln",
    "1. FSV Mainz 05":                "Mainz",
    "FSV Mainz 05":                   "Mainz",
    "TSG Hoffenheim":                 "Hoffenheim",
    "SV Werder Bremen":               "Werder Bremen",
    "VfL Bochum":                     "Bochum",
    "FC Augsburg":                    "Augsburg",
    "FC St. Pauli":                   "St Pauli",
    "St. Pauli":                      "St Pauli",
    "1. FC Heidenheim":               "Heidenheim",
    "Hamburger SV":                   "Hamburg",
    # SERIE A
    "AC Milan":               "Milan",
    "Internazionale":         "Inter",
    "AS Roma":                "Roma",
    "Hellas Verona":          "Verona",
    # LIGUE 1
    "Paris Saint-Germain":    "Paris SG",
    "PSG":                    "Paris SG",
    "Olympique de Marseille": "Marseille",
    "AS Monaco":              "Monaco",
    "Olympique Lyonnais":     "Lyon",
    "LOSC Lille":             "Lille",
    "Stade Rennais":          "Rennes",
    "OGC Nice":               "Nice",
    "RC Lens":                "Lens",
    "RC Strasbourg":          "Strasbourg",
    "FC Nantes":              "Nantes",
    "Montpellier HSC":        "Montpellier",
    "FC Toulouse":            "Toulouse",
    "Stade de Reims":         "Reims",
    "Stade Brestois 29":      "Brest",
    "Stade Brestois":         "Brest",
    "Clermont Foot":          "Clermont",
    "AJ Auxerre":             "Auxerre",
    "Le Havre AC":            "Le Havre",
    "FC Metz":                "Metz",
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
        # SOT dostępne tylko gdy obie kolumny mają dane
        df["HST"] = pd.to_numeric(df["HST"], errors="coerce")
        df["AST"] = pd.to_numeric(df["AST"], errors="coerce")
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

# ===========================================================================
# POPRAWIONA FUNKCJA ŁADOWANIA TERMINARZA – GRUPOWANIE PO TYGODNIACH
# ===========================================================================
@st.cache_data(ttl=86400)
def load_schedule(filename: str) -> pd.DataFrame:
    """Wczytuje terminarz i poprawnie numeruje kolejki według tygodni ISO"""
    try:
        df = pd.read_csv(filename)
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
        
        # Dodaj kolumnę z datą bez czasu dla łatwiejszego porównania
        df["date_only"] = df["date"].dt.date
        
        # Jeśli brak kolumny 'round', utwórz ją na podstawie tygodni
        if "round" not in df.columns:
            df = df.sort_values("date")
            
            # Pobierz tydzień i rok dla każdej daty (ISO week)
            df["week"] = df["date"].dt.isocalendar().week
            df["year"] = df["date"].dt.isocalendar().year
            
            # Znajdź unikalne kombinacje (rok, tydzień)
            unique_weeks = df[["year", "week"]].drop_duplicates().sort_values(["year", "week"])
            
            # Stwórz mapowanie na numery kolejek (1,2,3...)
            week_to_round = {}
            for i, (_, row) in enumerate(unique_weeks.iterrows(), 1):
                week_to_round[(row["year"], row["week"])] = i
            
            # Przypisz numery kolejek
            df["round"] = df.apply(lambda x: week_to_round[(x["year"], x["week"])], axis=1)
            
            # Usuń kolumny pomocnicze
            df = df.drop(columns=["week", "year"])
        
        return df.dropna(subset=["date"]).sort_values("date")
        
    except Exception as e:
        st.error(f"Problem z plikiem terminarza {filename}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def load_schedule(filename: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filename)
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
        if "round" not in df.columns:
            df = df.sort_values("date")
            unique_dates = df["date"].dt.date.unique()
            df["round"] = df["date"].dt.date.map(
                {d: i + 1 for i, d in enumerate(unique_dates)}
            )
        return df.dropna(subset=["date"]).sort_values("date")
    except Exception as e:
        st.error(f"Problem z plikiem terminarza {filename}: {e}")
        return pd.DataFrame()

# ===========================================================================
# STATYSTYKI
# ===========================================================================
def weighted_mean(values: pd.Series, dates: pd.Series = None,
                  tau_days: float = 30.0) -> float:
    """Ważona średnia z wykładniczym decay względem czasu (tau=30 dni).
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
                  sot_w: float = SOT_BLEND_W) -> tuple:
    """
    Zwraca (lam_h, lam_a, lam_r, lam_k, sot_aktywny, lam_sot).
    lam_h/lam_a to blend: (1-sot_w)*lam_goals + sot_w*lam_sot gdy SOT dostępne.
    sot_aktywny=True gdy blend był użyty.
    """
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
            # λ_sot = oczekiwane gole wyliczone ze strzałów (konwersja SOT→gole)
            # Średnia konwersja SOT→gol w lidze to avg_goals/avg_sot
            conv_h = avg_h / avg_sot_h  # np. 1.5 gola / 5.0 SOT = 0.30
            conv_a = avg_a / avg_sot_a
            lam_sot_h = sot_h * conv_h * obrona_a * form_weight(h)
            lam_sot_a = sot_a * conv_a * obrona_h * form_weight(a)
            lam_h = (1 - sot_w) * lam_h_goals + sot_w * lam_sot_h
            lam_a = (1 - sot_w) * lam_a_goals + sot_w * lam_sot_a
            sot_aktywny = True

    lam_r = (srednie_df.loc[h, "Różne (dom)"] + srednie_df.loc[a, "Różne (wyjazd)"]) / 2
    lam_k = (srednie_df.loc[h, "Kartki (dom)"] + srednie_df.loc[a, "Kartki (wyjazd)"]) / 2
    # lam_sot_total = oczekiwana suma celnych strzałów (do rynku SOT)
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

def wybierz_typ(p_home: float, p_draw: float, p_away: float) -> tuple:
    PROG_PEWNY = 0.42; PROG_PODWOJNA = 0.62
    if p_home >= PROG_PEWNY: return "1",  p_home
    if p_away >= PROG_PEWNY: return "2",  p_away
    if p_draw >= PROG_PEWNY: return "X",  p_draw
    p_1x = p_home + p_draw; p_x2 = p_away + p_draw
    if p_1x >= PROG_PODWOJNA or p_x2 >= PROG_PODWOJNA:
        return ("1X", p_1x) if p_1x >= p_x2 else ("X2", p_x2)
    probs = {"1": p_home, "X": p_draw, "2": p_away}
    t = max(probs, key=probs.get)
    return t, probs[t]

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

def confidence_score(p_home: float, p_draw: float, p_away: float) -> tuple:
    vals  = sorted([p_home, p_draw, p_away], reverse=True)
    spread = vals[0] - vals[2]
    edge   = vals[0] - vals[1]
    if edge > 0.18:   return "High",     "🟢", f"Wyraźny faworyt (+{edge:.0%} nad 2. opcją)"
    elif edge > 0.08: return "Medium",   "🟡", f"Umiarkowana przewaga (+{edge:.0%})"
    else:             return "Coinflip", "🔴", f"Mecz bardzo wyrównany (spread {spread:.0%})"

def fair_odds(p: float) -> float:
    return round(1 / p, 2) if 0 < p <= 1 else 999.0

def predykcja_meczu(lam_h: float, lam_a: float, rho: float = -0.13) -> dict:
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
    typ, p_typ = wybierz_typ(p_home, p_draw, p_away)
    conf_level, conf_emoji, conf_opis = confidence_score(p_home, p_draw, p_away)
    ent = entropy_meczu(p_home, p_draw, p_away)
    ch_label, ch_emoji, ch_pct = chaos_label(ent)
    return {
        "lam_h": lam_h, "lam_a": lam_a,
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "wynik_h": wynik_h, "wynik_a": wynik_a, "p_exact": p_exact,
        "fo_home": fair_odds(p_home), "fo_draw": fair_odds(p_draw), "fo_away": fair_odds(p_away),
        "typ": typ, "p_typ": p_typ, "fo_typ": fair_odds(p_typ),
        "conf_level": conf_level, "conf_emoji": conf_emoji, "conf_opis": conf_opis,
        "entropy": ent, "chaos_label": ch_label, "chaos_emoji": ch_emoji, "chaos_pct": ch_pct,
        "macierz": M,
    }

# ===========================================================================
# ALTERNATYWNE ZDARZENIA
# ===========================================================================
def alternatywne_zdarzenia(lam_h: float, lam_a: float, lam_r: float,
                            lam_k: float, rho: float,
                            prog_min: float = 0.55,
                            lam_sot: float = None) -> list:
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
    if lam_sot is not None and lam_sot > 0:
        for linia in [3.5, 4.5, 5.5, 6.5]:
            p_over = float(1 - poisson.cdf(int(linia), lam_sot))
            if p_over >= prog_min:
                zdarzenia.append(("🎯", f"Over {linia} celnych", p_over,
                                  fair_odds(p_over), "SOT", linia))

    return sorted(zdarzenia, key=lambda x: -x[2])

# ===========================================================================
# ZAPIS ZDARZEŃ DO BAZY (tracking skuteczności)
# ===========================================================================
def zapisz_zdarzenia(liga: str, kolejnosc: int, mecz: str, home: str, away: str,
                     rynek: str, typ: str, linia: float, p_model: float, fair_odds: float):
    """Zapisuje pojedyncze zdarzenie do bazy (do późniejszej weryfikacji)"""
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
# KOMENTARZ
# ===========================================================================
def _ocen_forme(f: str) -> str:
    if not f or f == "?": return "forma nieznana"
    w = f.count("W"); l = f.count("L")
    if w >= 4: return "doskonała forma"
    if w >= 3: return "dobra forma"
    if l >= 3: return "słaba forma"
    return "nieregularna forma"

def generuj_komentarz(home: str, away: str, pred: dict, forma_dict: dict) -> str:
    fh = forma_dict.get(home, "?")
    fa = forma_dict.get(away, "?")
    try:
        import anthropic
        key = st.secrets.get("ANTHROPIC_API_KEY", None)
        if key:
            client = anthropic.Anthropic(api_key=key)
            roznica_sil = abs(pred["lam_h"] - pred["lam_a"])
            forma_vs_mod = (
                "forma kłóci się z modelem"
                if (pred["p_home"] > 0.5 and fh.count("L") >= 2)
                   or (pred["p_away"] > 0.5 and fa.count("L") >= 2)
                else "forma spójna z modelem"
            )
            upset_risk = pred["p_draw"] > 0.28 and roznica_sil > 0.4
            trap_game  = pred["conf_level"] == "High" and (fh.count("W") <= 1 or fa.count("W") >= 3)
            prompt = (
                f"Jesteś analitykiem piłkarskim piszącym w stylu 'Narrative Mode'.\n"
                f"Mecz: {home} vs {away}\n"
                f"λ gosp: {pred['lam_h']:.2f} | λ gości: {pred['lam_a']:.2f}\n"
                f"Szanse 1X2: {pred['p_home']:.1%}/{pred['p_draw']:.1%}/{pred['p_away']:.1%}\n"
                f"Typ: {pred['typ']} | Pewność: {pred['conf_level']}\n"
                f"Forma {home}: {fh} | {away}: {fa}\n"
                f"Sygnały: {forma_vs_mod}"
                f"{', ⚠️ ryzyko niespodzianki' if upset_risk else ''}"
                f"{', 🪤 trap game?' if trap_game else ''}\n\n"
                f"Napisz 2-3 zdania po polsku. Narracyjny styl, konkretny i analityczny."
            )
            msg = client.messages.create(
                model="claude-3-5-sonnet-20241022", max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
    except Exception:
        pass
    roznica = pred["p_home"] - pred["p_away"]
    if   roznica >  0.20: faw = f"{home} jest wyraźnym faworytem ({pred['p_home']:.0%})."
    elif roznica >  0.08: faw = f"{home} jest lekkim faworytem ({pred['p_home']:.0%} vs {pred['p_away']:.0%})."
    elif roznica < -0.20: faw = f"{away} jest wyraźnym faworytem ({pred['p_away']:.0%})."
    elif roznica < -0.08: faw = f"{away} jest lekkim faworytem ({pred['p_away']:.0%} vs {pred['p_home']:.0%})."
    else:                 faw = f"Mecz wyrównany – remis ma {pred['p_draw']:.0%} szans."
    gole = pred["lam_h"] + pred["lam_a"]
    gole_opis = (f"Model spodziewa się bramkostrzelnego meczu (śr. {gole:.1f} goli)."
                 if gole >= 2.8 else f"Model przewiduje defensywny mecz (śr. {gole:.1f} goli).")
    return f"{faw} Forma: {home} [{_ocen_forme(fh)}: {fh}], {away} [{_ocen_forme(fa)}: {fa}]. {gole_opis}"

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
    return (f"<span style='{styl};color:white;padding:2px 14px;"
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
    mg = int(np.clip(np.ceil(max(lam_h, lam_a) + 4), 6, 10))
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
    """Zwraca ostrzeżenie dotyczące sędziego"""
    if sedziowie_df.empty or sedzia not in sedziowie_df["Sędzia"].values:
        return "⚪ Brak danych"
    
    sedz = sedziowie_df[sedziowie_df["Sędzia"] == sedzia].iloc[0]
    avg_cards = sedz["Total Kart/M ↓"]
    
    if avg_cards >= prog_kartki:
        return f"🚩 Wysokie ryzyko kartek! Śr. {avg_cards:.1f}/mecz"
    elif avg_cards >= 3.5:
        return f"🟡 Umiarkowane ryzyko kartek ({avg_cards:.1f}/mecz)"
    else:
        return f"🟢 Niskie ryzyko kartek ({avg_cards:.1f}/mecz)"

# ===========================================================================
# DEEP DATA – Power Rankings + Sędziowie
# ===========================================================================
@st.cache_data
def deep_data_stats(df_json: str) -> tuple:
    """Zwraca (power_df, sedziowie_df) dla tab Deep Data."""
    df = pd.read_json(df_json)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # ── Power Rankings ────────────────────────────────────────────────────
    druzyny = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
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
        gol_str  = all_m["_gole_str"].mean()
        gol_strac = all_m["_gole_strac"].mean()
        sot_sr   = all_m["_sot"].dropna().mean() if all_m["_sot"].notna().any() else None
        kart_sr  = all_m["_kartki"].mean()
        rozne_sr = all_m["_rozne"].mean()
        konv     = (gol_str / sot_sr) if (sot_sr and sot_sr > 0) else None
        # xG-proxy: SOT × liga_średnia_konwersji (szacunkowa)
        xg_proxy = (sot_sr * 0.11) if sot_sr else None  # ~11% konwersja PL
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
st.sidebar.header("🌍 Wybór Rozgrywek")
wybrana_liga = st.sidebar.selectbox("Wybierz ligę", list(LIGI.keys()))
debug_mode   = st.sidebar.checkbox("🔧 Debug – niezmapowane nazwy", value=False)

historical = load_historical(LIGI[wybrana_liga]["csv_code"])
schedule   = load_schedule(LIGI[wybrana_liga]["file"])

# ── Nagłówek główny ────────────────────────────────────────────────────────
hcol1, hcol2 = st.columns([6, 2])
with hcol1:
    st.title(f"⚽ {wybrana_liga}")
    st.caption(f"Dixon-Coles · SOT blend · Brier tracking · Kalibracja modelu")
with hcol2:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    if not schedule.empty:
        dzisiaj_t = datetime.now().date()
        przyszle_t = schedule[schedule["date"].dt.date >= dzisiaj_t]
        if not przyszle_t.empty:
            # Znajdź najbliższą przyszłą kolejkę
            pierwsza_data = przyszle_t["date_only"].min()
            pierwsza_kolejka = przyszle_t[przyszle_t["date_only"] == pierwsza_data]["round"].iloc[0]
            n_m = len(schedule[schedule["round"] == pierwsza_kolejka])
            st.metric("Następna kolejka", f"#{pierwsza_kolejka}", f"{n_m} meczów")

if not historical.empty:
    srednie_df  = oblicz_wszystkie_statystyki(historical.to_json())
    srednie_lig = oblicz_srednie_ligowe(historical.to_json())
    forma_dict  = oblicz_forme(historical)
    tabela      = tabela_ligowa(historical)
    rho         = srednie_lig["rho"]
    n_biezacy   = srednie_lig["n_biezacy"]
    w_prev      = waga_poprzedniego(n_biezacy)

    # Pobierz dane do ostrzeżeń sędziowskich
    power_df, sedziowie_df = deep_data_stats(historical.to_json())

    # Sidebar: info o danych
    st.sidebar.divider()
    st.sidebar.caption(f"📅 Sezon 2025/26 · {n_biezacy} meczów w bazie")
    st.sidebar.caption(f"ρ Dixon-Coles: `{rho:.4f}` · w_prev: `{w_prev:.2f}`")
    if not schedule.empty:
        dzisiaj  = datetime.now().date()
        przyszle = schedule[schedule["date"].dt.date >= dzisiaj]
        if not przyszle.empty:
            pierwsza_data = przyszle["date_only"].min()
            pierwsza_kolejka = przyszle[przyszle["date_only"] == pierwsza_data]["round"].iloc[0]
            st.sidebar.info(f"⚽ Następna kolejka: **{int(pierwsza_kolejka)}**")

    # TABS – NOWA KOLEJNOŚĆ: Ranking → Analiza → Deep Data → Skuteczność → Kalibracja → Lab
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Ranking Zdarzeń",
        "⚽ Analiza Meczu",
        "🔬 Deep Data",
        "📈 Skuteczność + ROI",
        "📉 Kalibracja",
        "🎛️ Laboratorium",
    ])

    # =========================================================================
    # TAB 1 – RANKING ZDARZEŃ (główny widok)
    # =========================================================================
    with tab1:
        st.subheader("📊 Ranking zdarzeń kolejki")
        st.caption("Centrum systemu – Value Bets, Safe Haven i Shot Kings dla najbliższej kolejki.")
        
        if not schedule.empty and not srednie_df.empty:
            dzisiaj = datetime.now().date()
            przyszle = schedule[schedule["date"].dt.date >= dzisiaj]
            if not przyszle.empty:
                # Znajdź pierwszą przyszłą kolejkę (najbliższa data)
                pierwsza_data = przyszle["date_only"].min()
                pierwsza_kolejka = przyszle[przyszle["date_only"] == pierwsza_data]["round"].iloc[0]
                
                # Pobierz wszystkie mecze z tej kolejki
                mecze = schedule[schedule["round"] == pierwsza_kolejka]
                nb = pierwsza_kolejka

                with st.spinner("Generowanie rankingu..."):
                    wszystkie_zd = []
                    shot_kings = []
                    
                    for _, mecz in mecze.iterrows():
                        h = map_nazwa(mecz["home_team"])
                        a = map_nazwa(mecz["away_team"])
                        if h not in srednie_df.index or a not in srednie_df.index:
                            continue
                        
                        lam_h, lam_a, lam_r, lam_k, sot_ok, lam_sot = oblicz_lambdy(h, a, srednie_df, srednie_lig, forma_dict)
                        pred = predykcja_meczu(lam_h, lam_a, rho=rho)
                        mecz_str = f"{h} – {a}"

                        def _ev(p_val, fo_val):
                            return round(p_val * fo_val - 1.0, 3)

                        # Typ główny – tylko z fair odds ≥ 1.30
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

                        # Alternatywne zdarzenia – tylko z fair odds ≥ 1.30
                        alt = alternatywne_zdarzenia(lam_h, lam_a, lam_r, lam_k, rho, prog_min=0.55, lam_sot=lam_sot)
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
                    
                    # =================================================================
                    # SEKCJA 1: VALUE BETS (EV > 0, sortowane po EV)
                    # =================================================================
                    st.markdown("### 🔥 Value Bets (EV > 0)")
                    st.caption("Zdarzenia z dodatnim expected value – im wyższy EV, tym większa wartość")
                    
                    value_bets = df_rank[df_rank["EV"] > 0].sort_values("EV", ascending=False)
                    
                    if not value_bets.empty:
                        for _, row in value_bets.head(10).iterrows():
                            ev_color = "#4CAF50" if row["EV"] > 0.05 else "#FF9800"
                            pct = f"{row['P']:.0%}"
                            cols = st.columns([3, 1, 1, 1, 1])
                            with cols[0]:
                                st.markdown(f"**{row['Mecz']}**")
                                st.caption(f"{row['Typ']}")
                            with cols[1]:
                                st.markdown(f"`{row['Rynek']}`")
                            with cols[2]:
                                st.markdown(f"🎯 {pct}")
                            with cols[3]:
                                st.markdown(f"💰 {row['Fair']:.2f}")
                            with cols[4]:
                                st.markdown(f"<span style='color:{ev_color};font-weight:bold'>+{row['EV']:.3f}</span>", unsafe_allow_html=True)
                            st.divider()
                    else:
                        st.info("Brak value bets w tej kolejce")

                    # =================================================================
                    # SEKCJA 2: SAFE HAVEN (najwyższe prawdopodobieństwa)
                    # =================================================================
                    st.markdown("### 🛡️ Safe Haven (p > 70%)")
                    st.caption("Najpewniejsze typy – wysokie prawdopodobieństwo, niskie ryzyko")
                    
                    safe_havens = df_rank[df_rank["P"] > 0.70].sort_values("P", ascending=False)
                    
                    if not safe_havens.empty:
                        for _, row in safe_havens.head(10).iterrows():
                            cols = st.columns([3, 1, 1, 2])
                            with cols[0]:
                                st.markdown(f"**{row['Mecz']}**")
                                st.caption(f"{row['Typ']}")
                            with cols[1]:
                                st.markdown(f"`{row['Rynek']}`")
                            with cols[2]:
                                st.markdown(f"🎯 {row['P']:.0%}")
                            with cols[3]:
                                st.markdown(f"💹 fair {row['Fair']:.2f}")
                            st.divider()
                    else:
                        st.info("Brak zdarzeń z p > 70%")

                    # =================================================================
                    # SEKCJA 3: SHOT KINGS (najwięcej strzałów celnych)
                    # =================================================================
                    if shot_kings:
                        st.markdown("### 🎯 Shot Kings")
                        st.caption("Mecze z największą oczekiwaną liczbą strzałów celnych (fair odds ≥ 1.30)")
                        
                        shot_df = pd.DataFrame(shot_kings).sort_values("Oczekiwane SOT", ascending=False)
                        
                        for _, row in shot_df.head(10).iterrows():
                            cols = st.columns([3, 1, 2])
                            with cols[0]:
                                st.markdown(f"**{row['Mecz']}**")
                            with cols[1]:
                                st.markdown(f"🎯 {row['Oczekiwane SOT']} SOT")
                            with cols[2]:
                                p_over = row["Over 5.5 SOT"]
                                if p_over > 0:
                                    st.markdown(f"Over 5.5: {p_over:.0%} (fair {row['Fair']:.2f})")
                            st.divider()
                    else:
                        st.info("Brak zdarzeń Shot Kings z fair odds ≥ 1.30")

                    # =================================================================
                    # SEKCJA 4: PEŁNY RANKING (w expanderze)
                    # =================================================================
                    with st.expander("📋 Pełny ranking wszystkich zdarzeń", expanded=False):
                        # Filtry
                        col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
                        with col_f1:
                            filtr_rynek = st.selectbox(
                                "Rynek", 
                                ["Wszystkie", "1X2", "Gole", "BTTS", "Rożne", "Kartki", "SOT"],
                                key="full_rank_filter"
                            )
                        with col_f2:
                            prog_filtr = st.slider(
                                "Min. prawdopodobieństwo", 
                                0.55, 0.90, 0.60, 0.01,
                                format="%.2f",  # Zmiana z %.0f%% na %.2f
                                key="full_rank_slider"
                            )
                            # Dodatkowe wyświetlanie w procentach
                            st.caption(f"Wybrano: {prog_filtr:.0%}")
                        with col_f3:
                            only_value_full = st.checkbox("Tylko EV>0", key="full_rank_value")
                        
                        df_full = df_rank.copy()
                        if filtr_rynek != "Wszystkie":
                            df_full = df_full[df_full["Rynek"] == filtr_rynek]
                        df_full = df_full[df_full["P"] >= prog_filtr]
                        
                        if only_value_full:
                            df_full = df_full[df_full["EV"] > 0]
                        
                        df_full = df_full.sort_values("P", ascending=False)
                        
                        # Wyświetl jako tabelę
                        if not df_full.empty:
                            df_display = df_full.copy()
                            df_display["P"] = df_display["P"].apply(lambda x: f"{x:.0%}")
                            df_display["Fair"] = df_display["Fair"].apply(lambda x: f"{x:.2f}")
                            df_display["EV"] = df_display["EV"].apply(lambda x: f"{x:+.3f}")
                            
                            st.dataframe(
                                df_display[["Mecz", "Rynek", "Typ", "P", "Fair", "EV"]],
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            st.download_button(
                                "⬇️ Pobierz pełny ranking (CSV)",
                                data=df_full[["Mecz", "Rynek", "Typ", "P", "Fair", "EV"]].to_csv(index=False, decimal=","),
                                file_name=f"ranking_full_kolejka{int(nb)}.csv"
                            )
                        else:
                            st.info("Brak zdarzeń po zastosowaniu filtrów")
                else:
                    st.info("Brak zdarzeń spełniających kryteria")
            else:
                st.info("Brak nadchodzących meczów")
        else:
            st.warning("Brak danych")

    # =========================================================================
    # TAB 2 – ANALIZA MECZU
    # =========================================================================
    with tab2:
        st.subheader("⚽ Analiza meczu – najbliższa kolejka")
        st.caption("Szczegółowa analiza każdego meczu. Rozwiń mecz → sprawdź rynki → zapisz do trackingu.")

        # ── Onboarding tracking ───────────────────────────────────────────
        with st.expander("ℹ️ Jak działa tracking skuteczności?", expanded=False):
            st.markdown(
                """
**Workflow w 3 krokach:**

**Krok 1 →** Przed meczami: włącz przełącznik *💾 Zapisz zdarzenia* poniżej.  
Model zapisze wszystkie predykcje do bazy (1X2 + rynki alternatywne).

**Krok 2 →** Poczekaj na wyniki meczów.

**Krok 3 →** Po meczach: kliknij *🔄 Aktualizuj wyniki*.  
System automatycznie porówna predykcje z wynikami i wyliczy skuteczność.

Dane trafią do zakładki **📈 Skuteczność + ROI** i **📉 Kalibracja**.
                """
            )

        tgl1, tgl2 = st.columns(2)
        with tgl1: pokaz_komentarz = st.toggle("💬 Komentarz", value=True)
        with tgl2: pokaz_macierz  = st.toggle("🔢 Macierz wyników", value=False)

        if not schedule.empty and not srednie_df.empty:
            dzisiaj  = datetime.now().date()
            przyszle = schedule[schedule["date"].dt.date >= dzisiaj]
            if not przyszle.empty:
                # Znajdź pierwszą przyszłą kolejkę
                pierwsza_data = przyszle["date_only"].min()
                pierwsza_kolejka = przyszle[przyszle["date_only"] == pierwsza_data]["round"].iloc[0]
                
                # Pobierz wszystkie mecze z tej kolejki
                mecze = schedule[schedule["round"] == pierwsza_kolejka]
                nb = pierwsza_kolejka
                
                st.caption(f"Kolejka {int(nb)} – {len(mecze)} meczów")

                kol_a, kol_b = st.columns(2)
                mecze_list = list(mecze.iterrows())

                for idx, (_, mecz) in enumerate(mecze_list):
                    h = map_nazwa(mecz["home_team"])
                    a = map_nazwa(mecz["away_team"])
                    if h not in srednie_df.index or a not in srednie_df.index:
                        continue

                    lam_h, lam_a, lam_r, lam_k, sot_ok, lam_sot = oblicz_lambdy(h, a, srednie_df, srednie_lig, forma_dict)
                    pred = predykcja_meczu(lam_h, lam_a, rho=rho)
                    data_meczu = mecz["date"].strftime("%d.%m %H:%M") if pd.notna(mecz["date"]) else ""

                    kolumna = kol_a if idx % 2 == 0 else kol_b
                    ikony_t = {"1":"🔵","X":"🟠","2":"🔴","1X":"🟣","X2":"🟣"}
                    conf_i  = "🟢" if pred["conf_level"]=="High" else ("🟡" if pred["conf_level"]=="Medium" else "🔴")
                    sot_badge = " 🎯SOT" if sot_ok else ""
                    
                    # Dodaj ostrzeżenie sędziowskie do etykiety
                    sedzia = mecz.get("Referee", "Nieznany") if "Referee" in mecz else "Nieznany"
                    sedzia_ostr = ostrzezenie_sedziego(sedzia, sedziowie_df)
                    
                    label_t2 = (f"{conf_i} {h} vs {a}{sot_badge}"
                                f"  ·  {ikony_t.get(pred['typ'],'⚪')} {pred['typ']} @ {pred['fo_typ']:.2f}"
                                f"  ·  {data_meczu}")
                    with kolumna:
                        with st.expander(label_t2, expanded=False):
                            # Nagłówek
                            ch, cmid, ca = st.columns([5,2,5])
                            with ch: st.markdown(f"<div style='font-weight:bold'>{h}</div>", unsafe_allow_html=True)
                            with cmid: st.markdown(f"<div style='text-align:center;color:#888'>{data_meczu}</div>", unsafe_allow_html=True)
                            with ca: st.markdown(f"<div style='font-weight:bold;text-align:right'>{a}</div>", unsafe_allow_html=True)

                            # Przewidywany wynik
                            st.markdown(
                                f"<div style='text-align:center;font-size:1.7em;font-weight:bold;margin:4px 0'>"
                                f"⚽ {pred['wynik_h']}:{pred['wynik_a']}"
                                f"<span style='font-size:0.5em;color:#888;margin-left:8px'>({pred['p_exact']:.1%})</span></div>",
                                unsafe_allow_html=True,
                            )

                            # Typ + confidence + chaos index
                            conf_colors  = {"High":"#4CAF50","Medium":"#FF9800","Coinflip":"#F44336"}
                            chaos_colors = {"Pewny":"#4CAF50","Klarowny":"#8BC34A","Otwarty":"#FF9800","Chaos":"#F44336"}
                            conf_c = conf_colors.get(pred["conf_level"], "#888")
                            ch_c   = chaos_colors.get(pred["chaos_label"], "#888")
                            bar_w  = int(pred["chaos_pct"] * 100)
                            st.markdown(
                                f"<div style='text-align:center;margin-bottom:4px'>"
                                f"Typ: {badge_typ(pred['typ'])}&nbsp;&nbsp;"
                                f"<span style='font-size:0.88em;color:#888'>Fair Odds: <b>{pred['fo_typ']:.2f}</b> ({pred['p_typ']:.1%})</span>"
                                f"</div>"
                                f"<div style='text-align:center;font-size:0.80em;color:{conf_c};margin-bottom:6px'>"
                                f"{pred['conf_emoji']} <b>{pred['conf_level']}</b> · {pred['conf_opis']}"
                                f"</div>"
                                f"<div style='margin:0 8px 8px 8px'>"
                                f"<div style='font-size:0.78em;color:#888'>"
                                f"{pred['chaos_emoji']} Chaos Index: <b style='color:{ch_c}'>{pred['chaos_label']}</b>"
                                f" ({pred['entropy']:.2f} bits)</div>"
                                f"<div style='background:#333;border-radius:4px;height:4px;margin-top:4px'>"
                                f"<div style='background:{ch_c};width:{bar_w}%;height:4px;border-radius:4px'></div>"
                                f"</div></div>",
                                unsafe_allow_html=True,
                            )

                            # Metryki 1X2
                            mc1, mc2, mc3 = st.columns(3)
                            mc1.metric("1", f"{pred['p_home']:.0%}", f"fair {pred['fo_home']}")
                            mc2.metric("X", f"{pred['p_draw']:.0%}", f"fair {pred['fo_draw']}")
                            mc3.metric("2", f"{pred['p_away']:.0%}", f"fair {pred['fo_away']}")
                            
                            # Lambda info + SOT status
                            sot_info = " · 🎯 SOT blend aktywny" if sot_ok else " · gole only"
                            st.markdown(
                                f"<div style='text-align:center;font-size:0.78em;color:#555;margin-top:2px'>"
                                f"λ {h[:8]}: <b style='color:#aaa'>{lam_h:.2f}</b> &nbsp;|&nbsp; "
                                f"λ {a[:8]}: <b style='color:#aaa'>{lam_a:.2f}</b> &nbsp;|&nbsp; "
                                f"Σ: <b style='color:#aaa'>{lam_h+lam_a:.2f}</b>"
                                f"<span style='color:#4CAF50'>{sot_info}</span></div>",
                                unsafe_allow_html=True,
                            )
                            
                            # Ostrzeżenie sędziego
                            st.caption(f"🟨 **Sędzia:** {sedzia} – {sedzia_ostr}")

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
                                            f"<div style='flex:1;background:#333;border-radius:3px;height:5px'>"
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

                            if pokaz_komentarz:
                                st.info(generuj_komentarz(h, a, pred, forma_dict))

                            if pokaz_macierz:
                                st.markdown("**Macierz wyników**")
                                st.markdown(render_macierz_html(pred["macierz"], h, a), unsafe_allow_html=True)

                # ── Tracking i aktualizacja wyników ──────────────────────────
                st.divider()
                tc1, tc2 = st.columns(2)
                with tc1:
                    if st.toggle("💾 Zapisz zdarzenia tej kolejki do trackingu", key="save_zd",
                                 help="Zapisuje wszystkie zdarzenia (1X2 + alt. rynki) do bazy analitycznej."):
                        n_saved = 0
                        for _, mecz_s in mecze.iterrows():
                            h_s = map_nazwa(mecz_s["home_team"])
                            a_s = map_nazwa(mecz_s["away_team"])
                            if h_s not in srednie_df.index or a_s not in srednie_df.index:
                                continue
                            lhs, las, lrs, lks, _sot_sv, _lsot_sv = oblicz_lambdy(h_s, a_s, srednie_df, srednie_lig, forma_dict)
                            pred_s = predykcja_meczu(lhs, las, rho=rho)
                            mecz_str_s = f"{h_s} – {a_s}"
                            # Główny typ 1X2
                            zapisz_zdarzenia(wybrana_liga, int(nb), mecz_str_s, h_s, a_s,
                                             "1X2", pred_s["typ"], 0.0, pred_s["p_typ"], pred_s["fo_typ"])
                            n_saved += 1
                            # Alternatywne
                            for emo, nazwa_z, pz, foz, katz, liniz in alternatywne_zdarzenia(lhs, las, lrs, lks, rho, lam_sot=_lsot_sv):
                                zapisz_zdarzenia(wybrana_liga, int(nb), mecz_str_s, h_s, a_s,
                                                 katz, nazwa_z, liniz, pz, foz)
                        st.success(f"✅ Zapisano zdarzenia dla {n_saved} meczów.")
                with tc2:
                    if st.button("🔄 Aktualizuj wyniki (po meczach)", help="Sprawdza bazę historyczną i uzupełnia wyniki zapisanych zdarzeń."):
                        n_updated = 0
                        init_db()
                        con_u = sqlite3.connect(DB_FILE)
                        mecze_db = con_u.execute(
                            "SELECT DISTINCT home, away FROM zdarzenia WHERE liga=? AND trafione IS NULL",
                            (wybrana_liga,)
                        ).fetchall()
                        con_u.close()
                        for h_db, a_db in mecze_db:
                            aktualizuj_wynik_zdarzenia(h_db, a_db, historical)
                            n_updated += 1
                        st.success(f"✅ Zaktualizowano wyniki dla {n_updated} meczów.")

    # =========================================================================
    # TAB 3 – DEEP DATA (Power Rankings + Sędziowie)
    # =========================================================================
    with tab3:
        st.subheader("🔬 Deep Data – Power Rankings & Analiza")
        st.caption(
            "Poglądowe statystyki drużyn i sędziów. "
            "Konwersja SOT → sygnał czy skuteczność to talent czy szczęście. "
            "Dane historyczne (bieżący + poprzedni sezon)."
        )

        # power_df i sedziowie_df załadowane wyżej

        if not power_df.empty:
            # ── Filtry ───────────────────────────────────────────────────
            d6c1, d6c2 = st.columns([2, 6])
            with d6c1:
                sort_dd = st.radio("Sortuj po", ["Gole/M ↑", "Strac./M ↓", "Forma", "Kartki/M", "SOT/M"],
                                   key="sort_dd")
            with d6c2:
                search_dd = st.text_input("🔍 Filtruj drużynę", "", key="search_dd",
                                          placeholder="Wpisz nazwę...")

            sort_map = {
                "Gole/M ↑":   ("_gol_str",   False),
                "Strac./M ↓": ("_gol_strac", True),
                "Forma":      ("_forma",     False),
                "Kartki/M":   ("Kartki/M",   False),
                "SOT/M":      ("SOT/M",      False),
            }
            col_s, asc_s = sort_map.get(sort_dd, ("_gol_str", False))
            df_dd = power_df.copy()
            if search_dd:
                df_dd = df_dd[df_dd["Drużyna"].str.contains(search_dd, case=False, na=False)]
            try:
                df_dd = df_dd.sort_values(col_s, ascending=asc_s)
            except Exception:
                pass

            # ── HTML tabela Power Rankings ───────────────────────────────
            cat_dd = {"Gole/M ↑": "#4CAF50", "Strac./M ↓": "#F44336",
                      "SOT/M": "#2196F3", "Konwersja%": "#9C27B0",
                      "Kartki/M": "#FF9800", "Forma (pkt/5M)": "#4CAF50"}

            rows_dd = []
            max_gol = df_dd["_gol_str"].max() if not df_dd.empty else 1
            max_strac = df_dd["_gol_strac"].max() if not df_dd.empty else 1
            max_forma = 15  # max 5*3

            for _, row in df_dd.iterrows():
                gol_pct  = int(row["_gol_str"] / max_gol * 100) if max_gol > 0 else 0
                strac_pct = int(row["_gol_strac"] / max_strac * 100) if max_strac > 0 else 0
                form_pct  = int(row["_forma"] / max_forma * 100)
                form_col  = "#4CAF50" if row["_forma"] >= 9 else ("#FF9800" if row["_forma"] >= 5 else "#F44336")
                # xG-proxy vs gole_str – konwersja anomalia?
                try:
                    xg_v = float(row["xG-proxy"])
                    gol_v = float(row["_gol_str"])
                    xg_diff = gol_v - xg_v
                    xg_icon = "🍀" if xg_diff > 0.25 else ("😤" if xg_diff < -0.20 else "⚖️")
                    xg_str  = f"{xg_v:.2f} {xg_icon}"
                except (TypeError, ValueError):
                    xg_str = "–"
                rows_dd.append(
                    f"<tr>"
                    f"<td style='padding:6px 10px;font-weight:bold;font-size:0.88em'>{row['Drużyna']}</td>"
                    f"<td style='padding:6px 10px;text-align:center;color:#888'>{row['M']}</td>"
                    # Gole/M pasek
                    f"<td style='padding:6px 10px'>"
                    f"<div style='display:flex;align-items:center;gap:5px'>"
                    f"<div style='flex:1;background:#333;border-radius:3px;height:5px'>"
                    f"<div style='background:#4CAF50;width:{gol_pct}%;height:5px;border-radius:3px'></div></div>"
                    f"<span style='color:#4CAF50;font-size:0.85em;min-width:28px'>{row['Gole/M ↑']}</span>"
                    f"</div></td>"
                    # Stracone/M pasek
                    f"<td style='padding:6px 10px'>"
                    f"<div style='display:flex;align-items:center;gap:5px'>"
                    f"<div style='flex:1;background:#333;border-radius:3px;height:5px'>"
                    f"<div style='background:#F44336;width:{strac_pct}%;height:5px;border-radius:3px'></div></div>"
                    f"<span style='color:#F44336;font-size:0.85em;min-width:28px'>{row['Strac./M ↓']}</span>"
                    f"</div></td>"
                    f"<td style='padding:6px 10px;text-align:center;color:#2196F3'>{row['SOT/M']}</td>"
                    f"<td style='padding:6px 10px;text-align:center;color:#9C27B0'>{row['Konwersja%']}</td>"
                    f"<td style='padding:6px 10px;text-align:center;color:#888'>{xg_str}</td>"
                    f"<td style='padding:6px 10px;text-align:center;color:#FF9800'>{row['Kartki/M']}</td>"
                    f"<td style='padding:6px 10px;text-align:center;color:#aaa'>{row['Rożne/M']}</td>"
                    # Forma pasek
                    f"<td style='padding:6px 10px'>"
                    f"<div style='display:flex;align-items:center;gap:5px'>"
                    f"<div style='flex:1;background:#333;border-radius:3px;height:5px'>"
                    f"<div style='background:{form_col};width:{form_pct}%;height:5px;border-radius:3px'></div></div>"
                    f"<span style='color:{form_col};font-size:0.85em;min-width:18px'>{row['Forma (pkt/5M)']}</span>"
                    f"</div></td>"
                    f"</tr>"
                )

            st.markdown(
                f"<div style='overflow-x:auto;border-radius:8px;border:1px solid #333'>"
                f"<table style='width:100%;border-collapse:collapse;font-size:0.86em'>"
                f"<thead><tr style='background:#1e1e2e;color:#aaa;font-size:0.75em;text-transform:uppercase'>"
                f"<th style='padding:8px 10px;text-align:left'>Drużyna</th>"
                f"<th style='padding:8px 10px;text-align:center'>M</th>"
                f"<th style='padding:8px 10px;text-align:left'>Gole/M ↑</th>"
                f"<th style='padding:8px 10px;text-align:left'>Strac./M ↓</th>"
                f"<th style='padding:8px 10px;text-align:center'>SOT/M</th>"
                f"<th style='padding:8px 10px;text-align:center'>Konwersja</th>"
                f"<th style='padding:8px 10px;text-align:center'>xG-proxy</th>"
                f"<th style='padding:8px 10px;text-align:center'>Kartki/M</th>"
                f"<th style='padding:8px 10px;text-align:center'>Rożne/M</th>"
                f"<th style='padding:8px 10px;text-align:left'>Forma/5M</th>"
                f"</tr></thead><tbody>{''.join(rows_dd)}</tbody></table></div>",
                unsafe_allow_html=True,
            )
            st.caption(
                "🍀 Szczęściarz (gole > xG-proxy) · 😤 Pechowiec (gole < xG-proxy) · ⚖️ Normalny. "
                "Konwersja = gole/celne strzały (liga avg ~10-12%). "
                "xG-proxy = SOT × 0.11 (uproszczony szacunek)."
            )

            # Export Power Rankings
            export_cols = ["Drużyna","M","Gole/M ↑","Strac./M ↓","SOT/M",
                           "Konwersja%","xG-proxy","Kartki/M","Rożne/M","Forma (pkt/5M)"]
            avail_cols = [c for c in export_cols if c in df_dd.columns]
            st.download_button("⬇️ Pobierz Power Rankings (CSV)",
                               data=df_dd[avail_cols].to_csv(index=False, decimal=","),
                               file_name="power_rankings.csv", mime="text/csv")

            # ── Sędziowie ─────────────────────────────────────────────────
            st.divider()
            st.markdown("### 🟨 Profile Sędziów")
            if not sedziowie_df.empty:
                st.caption(
                    "Historyczny profil sędziów – średnia kartek i goli per mecz. "
                    "Nie jest połączony z predykcją (brak przypisania sędziego do przyszłych meczów), "
                    "ale przydatny przy analizie rynku kartek."
                )
                # SVG mini-bar dla kartek sędziów
                df_sed = sedziowie_df.sort_values("_tot_k", ascending=False).head(20)
                W_sed, H_sed, P_sed = 620, max(200, len(df_sed)*28+60), 160
                max_k = df_sed["_tot_k"].max() if not df_sed.empty else 1
                bars_sed = []
                for i, (_, sr) in enumerate(df_sed.iterrows()):
                    y_s = P_sed//3 + i*28
                    blen_s = sr["_tot_k"] / max_k * (W_sed - P_sed - 20)
                    k_col  = "#F44336" if sr["_tot_k"] > 5 else ("#FF9800" if sr["_tot_k"] > 3.5 else "#4CAF50")
                    bars_sed.append(
                        f"<rect x='{P_sed}' y='{y_s+6}' width='{blen_s:.0f}' height='14' "
                        f"fill='{k_col}' fill-opacity='0.8' rx='3'/>"
                        f"<text x='{P_sed-5}' y='{y_s+17}' text-anchor='end' "
                        f"font-size='9' fill='#aaa' font-family='sans-serif'>{str(sr['Sędzia'])[:22]}</text>"
                        f"<text x='{P_sed+blen_s+4:.0f}' y='{y_s+17}' "
                        f"font-size='9' fill='{k_col}' font-family='sans-serif' font-weight='bold'>"
                        f"{sr['Total Kart/M ↓']:.1f} ({sr['Meczów']}M)</text>"
                    )
                svg_sed = (
                    f'<svg width="{W_sed}" height="{H_sed}" '
                    f'style="background:#0e1117;border-radius:8px;display:block;margin:auto">'
                    f'<text x="{W_sed//2}" y="18" text-anchor="middle" '
                    f'font-size="11" fill="#888" font-family="sans-serif">Całkowite kartki/mecz (Y+2R)</text>'
                    f'{"".join(bars_sed)}</svg>'
                )
                st.markdown(svg_sed, unsafe_allow_html=True)

                # Tabela
                display_cols_sed = [c for c in ["Sędzia","Meczów","Kartki Y/M","Kartki R/M",
                                                  "Total Kart/M ↓","Gole/M"] if c in sedziowie_df.columns]
                st.dataframe(df_sed[display_cols_sed].reset_index(drop=True),
                             use_container_width=True, hide_index=True)
            else:
                st.info("Brak kolumny 'Referee' w danych – profil sędziów niedostępny dla tej ligi.")
        else:
            st.warning("Brak wystarczających danych do Power Rankings.")

    # =========================================================================
    # TAB 4 – SKUTECZNOŚĆ + ROI
    # =========================================================================
    with tab4:
        st.subheader("📈 Skuteczność modelu per rynek")

        stats_df = statystyki_skutecznosci(wybrana_liga)

        if not stats_df.empty:
            stats_df["ROI_value"] = stats_df["ROI"].str.replace("+","").str.replace("%","").astype(float)

            # ── Metryki globalne (Brier, ECE, BSS) ────────────────────────
            mg = metryki_globalne(wybrana_liga)
            total_typow   = int(stats_df["Typów"].sum())
            total_trafion = int(stats_df["Trafione"].sum())
            avg_skut      = total_trafion / total_typow if total_typow > 0 else 0
            w_roi_raw     = (stats_df["ROI_value"] * stats_df["Typów"]).sum() / total_typow if total_typow > 0 else 0

            with st.container(border=True):
                st.caption("📊 Metryki globalne modelu")
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("📋 Typów",       total_typow)
                m2.metric("✅ Trafione",     total_trafion)
                m3.metric("🎯 Hit Rate",    f"{avg_skut:.1%}")
                m4.metric("💹 Ważony ROI",  f"{w_roi_raw:+.1f}%",
                          delta_color="normal" if w_roi_raw > 0 else "inverse")
                if mg:
                    bss_delta = f"vs naive {mg['bss']:+.3f}"
                    m5.metric("📐 Brier Score ↓", f"{mg['brier']:.4f}",
                              delta=bss_delta,
                              delta_color="normal" if mg['bss'] > 0 else "inverse",
                              help="0=idealny, 0.25=losowy. BSS>0 = model bije naive predictor.")
                    ece_col  = "normal" if mg['ece'] < 0.05 else "inverse"
                    m6.metric("🎯 ECE ↓",    f"{mg['ece']:.4f}",
                              delta=f"sharpness {mg['sharpness']:.3f}",
                              delta_color=ece_col,
                              help="Expected Calibration Error. <0.05 = dobrze skalibrowany.")
                else:
                    m5.metric("📐 Brier", "–")
                    m6.metric("🎯 ECE",   "–")

            # ── HTML tabela per rynek ─────────────────────────────────────
            # Sortuj: wybór kryterium
            sort_by = st.radio("Sortuj po", ["ROI ↓", "Brier ↑", "Typów ↓"],
                               horizontal=True, key="sort_tab4")
            if sort_by == "ROI ↓":
                stats_sorted = stats_df.sort_values("_roi_v", ascending=False)
            elif sort_by == "Brier ↑":
                stats_sorted = stats_df.sort_values("_brier_v", ascending=True)
            else:
                stats_sorted = stats_df.sort_values("Typów", ascending=False)

            cat_col4 = {"1X2":"#4CAF50","Gole":"#2196F3","BTTS":"#9C27B0","Rożne":"#FF9800","Kartki":"#F44336"}

            rows_s4 = []
            for _, row in stats_sorted.iterrows():
                roi_v   = row["_roi_v"]
                brier_v = row["_brier_v"]
                skill_v = row["Skill"]
                roi_col = "#4CAF50" if roi_v > 0.03 else ("#F44336" if roi_v < -0.03 else "#FF9800")
                roi_bg  = "#1a2e1a" if roi_v > 0.03 else ("#2e1a1a" if roi_v < -0.03 else "transparent")
                skut_v  = row["_skut_v"]
                bw_s    = int(skut_v * 100)
                rynek_n = row["Rynek"]
                kat4    = next((k for k in cat_col4 if k in rynek_n), "Gole")
                kc4     = cat_col4.get(kat4, "#888")
                # Brier color: niższy = lepszy (<0.20 dobry, >0.25 zły)
                bc4 = "#4CAF50" if brier_v < 0.20 else ("#FF9800" if brier_v < 0.25 else "#F44336")
                # Skill score: >0 = lepszy od baseline
                sk_c = "#4CAF50" if skill_v > 0.05 else ("#888" if skill_v > -0.05 else "#F44336")
                rows_s4.append(
                    f"<tr style='background:{roi_bg}'>"
                    f"<td style='padding:6px 10px;font-weight:bold;font-size:0.88em'>{rynek_n}</td>"
                    f"<td style='padding:6px 10px;text-align:center;color:#888'>{row['Typów']}</td>"
                    f"<td style='padding:6px 10px;text-align:center;color:#888'>{row['Trafione']}</td>"
                    f"<td style='padding:6px 10px;width:110px'>"
                    f"<div style='display:flex;align-items:center;gap:5px'>"
                    f"<div style='flex:1;background:#333;border-radius:3px;height:5px'>"
                    f"<div style='background:{kc4};width:{bw_s}%;height:5px;border-radius:3px'></div></div>"
                    f"<span style='color:{kc4};font-size:0.82em;min-width:34px'>{row['Skuteczność']}</span></div></td>"
                    f"<td style='padding:6px 10px;text-align:center;color:{bc4};font-weight:bold'>{brier_v:.3f}</td>"
                    f"<td style='padding:6px 10px;text-align:center;color:{sk_c};font-size:0.85em'>{skill_v:+.2f}</td>"
                    f"<td style='padding:6px 10px;text-align:center;color:#888'>{row['Śr. P model']}</td>"
                    f"<td style='padding:6px 10px;text-align:center;color:#aaa'>{row['Śr. Fair']}</td>"
                    f"<td style='padding:6px 10px;text-align:right;font-weight:bold;color:{roi_col}'>{row['ROI']}</td>"
                    f"<td style='padding:6px 10px;text-align:center'>{row['Kolor']}</td>"
                    f"</tr>"
                )
            st.markdown(
                f"<div style='overflow-x:auto;border-radius:8px;border:1px solid #333'>"
                f"<table style='width:100%;border-collapse:collapse;font-size:0.85em'>"
                f"<thead><tr style='background:#1e1e2e;color:#aaa;font-size:0.75em;text-transform:uppercase'>"
                f"<th style='padding:8px 10px;text-align:left'>Rynek</th>"
                f"<th style='padding:8px 10px;text-align:center'>Typów</th>"
                f"<th style='padding:8px 10px;text-align:center'>Trafione</th>"
                f"<th style='padding:8px 10px;text-align:left'>Skuteczność</th>"
                f"<th style='padding:8px 10px;text-align:center'>Brier ↓</th>"
                f"<th style='padding:8px 10px;text-align:center'>Skill</th>"
                f"<th style='padding:8px 10px;text-align:center'>Śr. P</th>"
                f"<th style='padding:8px 10px;text-align:center'>Śr. Fair</th>"
                f"<th style='padding:8px 10px;text-align:right'>ROI</th>"
                f"<th style='padding:8px 10px;text-align:center'>Status</th>"
                f"</tr></thead><tbody>{''.join(rows_s4)}</tbody></table></div>"
                f"<p style='color:#444;font-size:0.73em;margin-top:4px'>"
                f"Brier ↓: niższy = lepszy (0=idealny, 0.25=losowy). "
                f"Skill: >0 = model bije baseline (naive predictor).</p>",
                unsafe_allow_html=True,
            )

            # ── SVG Bar chart ROI – zero zewnętrznych deps ────────────────
            st.divider()
            st.markdown("**📊 ROI per rynek** *(fair odds, bez marży)*")
            chart_sorted = stats_sorted.sort_values("ROI_value")
            W4, H4, pad4 = 600, max(200, len(chart_sorted) * 32 + 60), 50
            bar_max = max(abs(chart_sorted["ROI_value"].max()), abs(chart_sorted["ROI_value"].min()), 5)
            zero_x  = pad4 + (0 + bar_max) / (2 * bar_max) * (W4 - 2 * pad4)
            bars4   = []
            for i, (_, row4) in enumerate(chart_sorted.iterrows()):
                y4    = pad4 + i * 32
                rv    = row4["ROI_value"]
                blen  = abs(rv) / bar_max * (W4 - 2 * pad4) / 2
                bc4   = "#4CAF50" if rv >= 0 else "#F44336"
                bx4   = zero_x if rv >= 0 else zero_x - blen
                bars4.append(
                    f"<rect x='{bx4:.0f}' y='{y4+8:.0f}' width='{blen:.0f}' height='16' "
                    f"fill='{bc4}' fill-opacity='0.8' rx='3'/>"
                    f"<text x='{pad4-4:.0f}' y='{y4+20:.0f}' text-anchor='end' "
                    f"font-size='10' fill='#aaa' font-family='sans-serif'>{row4['Rynek'][:16]}</text>"
                    f"<text x='{(bx4+blen+4) if rv>=0 else (bx4-4):.0f}' y='{y4+20:.0f}' "
                    f"text-anchor='{'start' if rv>=0 else 'end'}' "
                    f"font-size='10' fill='{bc4}' font-family='sans-serif' font-weight='bold'>{rv:+.1f}%</text>"
                )
            svg4 = (
                f'<svg width="{W4}" height="{H4}" '
                f'style="background:#0e1117;border-radius:8px;display:block;margin:auto">'
                f'<line x1="{zero_x:.0f}" y1="{pad4-10}" x2="{zero_x:.0f}" y2="{H4-10}" '
                f'stroke="#444" stroke-width="1" stroke-dasharray="4,3"/>'
                f'<text x="{zero_x:.0f}" y="{pad4-14}" text-anchor="middle" '
                f'font-size="9" fill="#555" font-family="sans-serif">0%</text>'
                f'{"".join(bars4)}</svg>'
            )
            st.markdown(svg4, unsafe_allow_html=True)
            st.caption("ROI = (trafione × (fair−1) − chybione) / wszystkie. Symulacja na fair odds (bez marży bukmachera).")

            # Export
            internal_cols = [c for c in stats_df.columns if c.startswith("_") or c in ["ROI_value","Kolor"]]
            st.download_button("⬇️ Pobierz statystyki (CSV)",
                               data=stats_df.drop(columns=internal_cols, errors="ignore")
                                   .to_csv(index=False, decimal=","),
                               file_name="skutecznosc_modelu.csv", mime="text/csv")
        else:
            st.info("📭 Brak danych do analizy skuteczności.")
            with st.container(border=True):
                st.markdown("#### 🚀 Jak zacząć zbierać dane?")
                st.markdown("""
**1️⃣ Przejdź do zakładki ⚽ Analiza Meczu**  
Otwórz dowolny mecz z listy – zobaczysz predykcje modelu.

**2️⃣ Włącz przełącznik 💾 Zapisz zdarzenia**  
Model zapisze wszystkie predykcje do lokalnej bazy danych.

**3️⃣ Poczekaj na rozegranie meczów**  
Dane historyczne są pobierane automatycznie z football-data.co.uk.

**4️⃣ Kliknij 🔄 Aktualizuj wyniki**  
System dopasuje predykcje z wynikami i wyliczy skuteczność per rynek.

**5️⃣ Wróć tutaj** – zobaczysz Brier Score, ROI i tabelę per rynek.
                """)

    # =========================================================================
    # TAB 5 – KALIBRACJA + ROLLING PERFORMANCE
    # =========================================================================
    with tab5:
        st.subheader("📉 Kalibracja modelu & Rolling Performance")
        st.caption(
            "Kalibracja: czy model mówi 65% → trafia ~65%? "
            "Rolling: jak ewoluuje jakość modelu w czasie."
        )

        # ── ROLLING WINDOW ─────────────────────────────────────────────────
        st.markdown("### 📈 Rolling Performance (okno 50 zdarzeń)")
        okno_r = st.slider("Rozmiar okna rolling", 20, 100, 50, 10, key="roll_win")
        roll_df = rolling_stats(wybrana_liga, okno=okno_r)

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

        # ── BRIER PER KOLEJKA ─────────────────────────────────────────────
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
        st.markdown("### 🎯 Reliability Curve")
        kal_df = kalibracja_modelu(wybrana_liga if wybrana_liga != "Wszystkie" else None)
        
        if not kal_df.empty:
            kal_df = kal_df.dropna(subset=["skutecznosc"])

            # ── Tabela kalibracji ─────────────────────────────────────────
            rows_k = []
            for _, rk in kal_df.iterrows():
                diff = rk["rozbieznosc"]
                diff_col = "#4CAF50" if abs(diff) < 0.05 else ("#FF9800" if abs(diff) < 0.12 else "#F44336")
                diff_icon = "✅" if abs(diff) < 0.05 else ("⚠️" if abs(diff) < 0.12 else "❌")
                rows_k.append(
                    f"<tr>"
                    f"<td style='padding:6px 10px;font-weight:bold'>{rk['przedzial']}</td>"
                    f"<td style='padding:6px 10px;text-align:center;color:#888'>{int(rk['liczba'])}</td>"
                    f"<td style='padding:6px 10px;text-align:center;color:#2196F3'>{rk['p_srednia']:.1%}</td>"
                    f"<td style='padding:6px 10px;text-align:center;color:#4CAF50'>{rk['skutecznosc']:.1%}</td>"
                    f"<td style='padding:6px 10px;text-align:center;color:{diff_col};font-weight:bold'>"
                    f"{diff_icon} {diff:+.1%}</td>"
                    f"</tr>"
                )
            st.markdown(
                f"<div style='overflow-x:auto;border-radius:8px;border:1px solid #333'>"
                f"<table style='width:100%;border-collapse:collapse;font-size:0.88em'>"
                f"<thead><tr style='background:#1e1e2e;color:#aaa;font-size:0.78em;text-transform:uppercase'>"
                f"<th style='padding:8px 10px;text-align:left'>Przedział P</th>"
                f"<th style='padding:8px 10px;text-align:center'>Typów</th>"
                f"<th style='padding:8px 10px;text-align:center'>P model</th>"
                f"<th style='padding:8px 10px;text-align:center'>Hit Rate</th>"
                f"<th style='padding:8px 10px;text-align:center'>Rozbieżność</th>"
                f"</tr></thead><tbody>{''.join(rows_k)}</tbody></table></div>",
                unsafe_allow_html=True,
            )

            # ── SVG Reliability curve ─────────────────────────────────────
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
                f'style="background:#0e1117;border-radius:8px;display:block;margin:auto">'
                f'<polyline points="{diag5_line}" fill="none" stroke="#444" '
                f'stroke-width="1.5" stroke-dasharray="6,4"/>'
                f'<text x="{w_rc5 - pad_rc5 + 4}" y="{pad_rc5 - 4}" font-size="9" fill="#555" '
                f'font-family="sans-serif">idealny model</text>'
                f'{"".join(circles5)}'
                f'<text x="{w_rc5 // 2}" y="{h_rc5 - 6}" text-anchor="middle" '
                f'font-size="10" fill="#888" font-family="sans-serif">P modelu →</text>'
                f'<text x="12" y="{h_rc5 // 2}" text-anchor="middle" font-size="10" fill="#888" '
                f'font-family="sans-serif" transform="rotate(-90,12,{h_rc5 // 2})">Hit Rate →</text>'
                f'<!-- legenda -->'
                f'<circle cx="{w_rc5 - 130}" cy="20" r="6" fill="#4CAF50" stroke="white" stroke-width="1"/>'
                f'<text x="{w_rc5 - 120}" y="25" font-size="9" fill="#aaa" font-family="sans-serif">Dobrze skalibrowany (&lt;5%)</text>'
                f'<circle cx="{w_rc5 - 130}" cy="38" r="6" fill="#FF9800" stroke="white" stroke-width="1"/>'
                f'<text x="{w_rc5 - 120}" y="43" font-size="9" fill="#aaa" font-family="sans-serif">Umiarkowany (5–12%)</text>'
                f'<circle cx="{w_rc5 - 130}" cy="56" r="6" fill="#F44336" stroke="white" stroke-width="1"/>'
                f'<text x="{w_rc5 - 120}" y="61" font-size="9" fill="#aaa" font-family="sans-serif">Słabo skalibrowany (&gt;12%)</text>'
                f'</svg>'
            )
            st.markdown(svg5, unsafe_allow_html=True)

            # ── Interpretacja ─────────────────────────────────────────────
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
                st.markdown(
                    "Potrzebne są co najmniej **30 zdarzeń z wynikami** w każdym przedziale p.  \n"
                    "Wróć po kilku kolejkach trackingu – wtedy krzywa będzie statystycznie znacząca."
                )

    # =========================================================================
    # TAB 6 – LABORATORIUM (Bet Builder)
    # =========================================================================
    with tab6:
        st.subheader("🎛️ Bet Builder – laboratorium modelu")
        st.caption("Eksperymentuj z kombinacjami zdarzeń. Nie są to rekomendacje, tylko symulacje.")

        c1, c2, c3 = st.columns(3)
        with c1:
            linia_gole   = st.selectbox("Linia goli",    [1.5, 2.5, 3.5], index=1)
            typ_gole     = st.selectbox("Typ goli",      ["Over", "Under"])
        with c2:
            linia_rogi   = st.selectbox("Linia rożnych", [7.5, 8.5, 9.5, 10.5], index=1)
            typ_rogi     = st.selectbox("Typ rożnych",   ["Over", "Under"])
        with c3:
            linia_kartki = st.selectbox("Linia kartek",  [2.5, 3.5, 4.5, 5.5], index=1)
            typ_kartki   = st.selectbox("Typ kartek",    ["Over", "Under"])
        min_prob = st.slider("Min. prawdopodobieństwo", 0.0, 1.0, 0.40, 0.01)

        if not schedule.empty and not srednie_df.empty:
            dzisiaj  = datetime.now().date()
            przyszle = schedule[schedule["date"].dt.date >= dzisiaj]
            if not przyszle.empty:
                # Znajdź pierwszą przyszłą kolejkę
                pierwsza_data = przyszle["date_only"].min()
                pierwsza_kolejka = przyszle[przyszle["date_only"] == pierwsza_data]["round"].iloc[0]
                
                # Pobierz wszystkie mecze z tej kolejki
                mecze = schedule[schedule["round"] == pierwsza_kolejka]
                nb = pierwsza_kolejka
                
                st.caption(f"Kolejka {int(nb)} – {len(mecze)} meczów")

                for _, mecz in mecze.iterrows():
                    h = map_nazwa(mecz["home_team"])
                    a = map_nazwa(mecz["away_team"])
                    if h not in srednie_df.index or a not in srednie_df.index:
                        continue
                    
                    lam_h, lam_a, lam_r, lam_k, sot_ok, lam_sot = oblicz_lambdy(h, a, srednie_df, srednie_lig, forma_dict)
                    
                    p_g = macierz_goli_p(lam_h, lam_a, rho, int(linia_gole), typ_gole)
                    p_r = oblicz_p(typ_rogi, linia_rogi, lam_r)
                    p_k = oblicz_p(typ_kartki, linia_kartki, lam_k)
                    p_combo = p_g * p_r * p_k
                    # Pokaż ekspander tylko gdy combo >= min_prob
                    label_bb = (f"{'✅' if p_combo >= min_prob else '❌'} {h} vs {a}"
                                f"  ·  combo {p_combo:.0%}  ·  fair AKO {fair_odds(p_combo):.2f}")
                    with st.expander(label_bb, expanded=False):
                        bc1, bc2, bc3, bc4 = st.columns(4)
                        bc1.metric(f"{typ_gole} {linia_gole}",
                                   f"{p_g:.1%}", f"fair {fair_odds(p_g):.2f}",
                                   delta_color="normal" if p_g >= 0.55 else "off")
                        bc2.metric(f"{typ_rogi} {linia_rogi} rożnych",
                                   f"{p_r:.1%}", f"fair {fair_odds(p_r):.2f}",
                                   delta_color="normal" if p_r >= 0.55 else "off")
                        bc3.metric(f"{typ_kartki} {linia_kartki} kartek",
                                   f"{p_k:.1%}", f"fair {fair_odds(p_k):.2f}",
                                   delta_color="normal" if p_k >= 0.55 else "off")
                        if p_combo >= min_prob:
                            bc4.metric("COMBO", f"{p_combo:.1%}", f"fair AKO {fair_odds(p_combo):.2f}",
                                       delta_color="normal")
                        else:
                            bc4.metric("COMBO", f"{p_combo:.1%}", "↓ poniżej progu", delta_color="off")

    # Debug
    if debug_mode:
        st.sidebar.divider()
        st.sidebar.write("**🔧 Niezmapowane nazwy**")
        if _niezmapowane:
            for n in sorted(_niezmapowane): st.sidebar.warning(f"`{n}`")
        else:
            st.sidebar.success("Wszystkie zmapowane ✅")

else:
    st.error("Nie udało się pobrać danych. Sprawdź połączenie z internetem.")