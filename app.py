import streamlit as st
import pandas as pd
from scipy.stats import poisson
from datetime import datetime, timedelta
import requests
from io import StringIO
import numpy as np
import unicodedata
import sqlite3
import os
import json

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

DB_FILE = "predykcje.db"

# Waga poprzedniego sezonu – maleje w miarę gromadzenia danych bieżącego
def waga_poprzedniego(n_biezacy: int) -> float:
    return float(np.clip(0.8 - (n_biezacy / 30) * 0.6, 0.2, 0.8))

st.set_page_config(page_title="Predykcje Piłkarskie Top 5", layout="wide")

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
    "Brighton":                "Brighton",
    "West Ham United":         "West Ham",
    "West Ham":                "West Ham",
    "Newcastle United":        "Newcastle",
    "Newcastle":               "Newcastle",
    "Tottenham Hotspur":       "Tottenham",
    "Tottenham":               "Tottenham",
    "Leeds United":            "Leeds",
    "Leeds":                   "Leeds",
    "Manchester United":       "Man United",
    "Man United":              "Man United",
    "Manchester City":         "Man City",
    "Man City":                "Man City",
    "Nottingham Forest":       "Nott'm Forest",
    "Nott'm Forest":           "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "Wolverhampton":           "Wolves",
    "Wolves":                  "Wolves",
    "Leicester City":          "Leicester",
    "Leicester":               "Leicester",
    "Sheffield United":        "Sheffield Utd",
    "Sheffield Utd":           "Sheffield Utd",
    
    # LA LIGA
    "Girona FC":               "Girona",
    "Girona":                  "Girona",
    "Rayo Vallecano":          "Vallecano",
    "Vallecano":               "Rayo Vallecano",
    "Villarreal":              "Villarreal",
    "Real Oviedo":             "Oviedo",
    "Oviedo":                   "Real Oviedo",
    "Mallorca":                 "Mallorca",
    "RCD Mallorca":             "Mallorca",
    "Barcelona":                "Barcelona",
    "FC Barcelona":             "Barcelona",
    "Deportivo Alavés":         "Alaves",
    "Deportivo Alaves":         "Alaves",
    "Alaves":                   "Alaves",
    "Levante UD":               "Levante",
    "Levante":                  "Levante",
    "Valencia":                 "Valencia",
    "Valencia CF":              "Valencia",
    "Real Sociedad":            "Sociedad",
    "Sociedad":                 "Real Sociedad",
    "Celta Vigo":               "Celta",
    "Celta":                    "Celta",
    "Getafe":                   "Getafe",
    "Getafe CF":                "Getafe",
    "Athletic Club":            "Ath Bilbao",
    "Athletic Bilbao":          "Ath Bilbao",
    "Ath Bilbao":               "Ath Bilbao",
    "Sevilla":                  "Sevilla",
    "Sevilla FC":               "Sevilla",
    "Espanyol":                 "Espanol",
    "RCD Espanyol":             "Espanol",
    "Atlético Madrid":          "Ath Madrid",
    "Atletico Madrid":          "Ath Madrid",
    "Ath Madrid":               "Ath Madrid",
    "Elche":                    "Elche",
    "Elche CF":                 "Elche",
    "Real Betis":               "Betis",
    "Betis":                    "Betis",
    "Real Madrid":              "Real Madrid",
    "Osasuna":                  "Osasuna",
    "CA Osasuna":               "Osasuna",
    "Real Valladolid":          "Valladolid",
    "Valladolid":               "Valladolid",
    "UD Las Palmas":            "Las Palmas",
    "Las Palmas":               "Las Palmas",
    "Cadiz":                    "Cadiz",
    "Cadiz CF":                 "Cadiz",
    "Granada":                  "Granada",
    "Granada CF":               "Granada",
    "Almeria":                  "Almeria",
    "UD Almeria":               "Almeria",
    
    # BUNDESLIGA
    "FC Bayern München":              "Bayern Munich",
    "Bayern Munich":                  "Bayern Munich",
    "Bayern":                         "Bayern Munich",
    "Borussia Dortmund":              "Dortmund",
    "Dortmund":                       "Dortmund",
    "RB Leipzig":                     "RB Leipzig",
    "Leipzig":                        "RB Leipzig",
    "Bayer 04 Leverkusen":            "Leverkusen",
    "Leverkusen":                     "Leverkusen",
    "Eintracht Frankfurt":            "Eintracht",
    "Eintracht":                      "Eintracht",
    "VfB Stuttgart":                  "Stuttgart",
    "Stuttgart":                      "Stuttgart",
    "VfL Wolfsburg":                  "Wolfsburg",
    "Wolfsburg":                      "Wolfsburg",
    "Borussia Mönchengladbach":       "M'gladbach",
    "Borussia Monchengladbach":       "M'gladbach",
    "Borussia M'gladbach":            "M'gladbach",
    "M'gladbach":                     "M'gladbach",
    "1. FC Union Berlin":             "Union Berlin",
    "Union Berlin":                    "Union Berlin",
    "SC Freiburg":                     "Freiburg",
    "Freiburg":                        "Freiburg",
    "1. FC Köln":                      "FC Koln",
    "FC Köln":                         "FC Koln",
    "Köln":                            "FC Koln",
    "1. FSV Mainz 05":                 "Mainz",
    "FSV Mainz 05":                    "Mainz",
    "Mainz":                           "Mainz",
    "TSG Hoffenheim":                  "Hoffenheim",
    "Hoffenheim":                      "Hoffenheim",
    "Werder Bremen":                   "Werder Bremen",
    "SV Werder Bremen":                "Werder Bremen",
    "VfL Bochum":                      "Bochum",
    "Bochum":                          "Bochum",
    "FC Augsburg":                     "Augsburg",
    "Augsburg":                        "Augsburg",
    "FC St. Pauli":                    "St Pauli",
    "St. Pauli":                       "St Pauli",
    "St Pauli":                        "St Pauli",
    "1. FC Heidenheim":                "Heidenheim",
    "Heidenheim":                      "Heidenheim",
    "Hamburger SV":                    "Hamburg",
    "Hamburg":                         "Hamburg",
    
    # SERIE A
    "AC Milan":               "Milan",
    "Milan":                  "AC Milan",
    "Inter":                  "Inter",
    "Internazionale":         "Inter",
    "Juventus":               "Juventus",
    "AS Roma":                "Roma",
    "Roma":                   "Roma",
    "Lazio":                  "Lazio",
    "Napoli":                 "Napoli",
    "Atalanta":               "Atalanta",
    "Fiorentina":             "Fiorentina",
    "Torino":                 "Torino",
    "Bologna":                "Bologna",
    "Udinese":                "Udinese",
    "Sassuolo":               "Sassuolo",
    "Empoli":                 "Empoli",
    "Salernitana":            "Salernitana",
    "Hellas Verona":          "Verona",
    "Verona":                 "Verona",
    "Lecce":                  "Lecce",
    "Cagliari":               "Cagliari",
    "Genoa":                  "Genoa",
    "Monza":                  "Monza",
    "Frosinone":              "Frosinone",
    
    # LIGUE 1
    "Paris Saint-Germain":    "Paris SG",
    "PSG":                    "Paris SG",
    "Paris SG":               "Paris SG",
    "Olympique de Marseille": "Marseille",
    "Marseille":              "Marseille",
    "AS Monaco":              "Monaco",
    "Monaco":                 "Monaco",
    "Olympique Lyonnais":     "Lyon",
    "Lyon":                   "Lyon",
    "LOSC Lille":             "Lille",
    "Lille":                  "Lille",
    "Stade Rennais":          "Rennes",
    "Rennes":                 "Rennes",
    "OGC Nice":               "Nice",
    "Nice":                   "Nice",
    "RC Lens":                "Lens",
    "Lens":                   "Lens",
    "RC Strasbourg":          "Strasbourg",
    "Strasbourg":             "Strasbourg",
    "FC Nantes":              "Nantes",
    "Nantes":                 "Nantes",
    "Montpellier HSC":        "Montpellier",
    "Montpellier":            "Montpellier",
    "FC Toulouse":            "Toulouse",
    "Toulouse":               "Toulouse",
    "Stade de Reims":         "Reims",
    "Reims":                  "Reims",
    "Stade Brestois 29":      "Brest",
    "Stade Brestois":         "Brest",
    "Brest":                  "Brest",
    "Clermont Foot":          "Clermont",
    "Clermont":               "Clermont",
    "AJ Auxerre":             "Auxerre",
    "Auxerre":                "Auxerre",
    "Le Havre AC":            "Le Havre",
    "Le Havre":               "Le Havre",
    "FC Metz":                "Metz",
    "Metz":                   "Metz",
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
    # tabela predykcje
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
    # tabela kupony
    con.execute("""
        CREATE TABLE IF NOT EXISTS kupony (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            liga        TEXT,
            kolejnosc   INTEGER,
            typ_kuponu  TEXT,
            mecze_json  TEXT,
            ako         REAL,
            p_combo     REAL,
            data        TEXT,
            UNIQUE(liga, kolejnosc, typ_kuponu)
        )
    """)
    con.commit()
    con.close()

# ===========================================================================
# ŁADOWANIE DANYCH
# ===========================================================================
def _pobierz_csv(league_code: str, sezon: str) -> pd.DataFrame:
    try:
        url = f"https://www.football-data.co.uk/mmz4281/{sezon}/{league_code}.csv"
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        for col in ["HC", "AC", "HY", "AY", "HR", "AR"]:
            if col not in df.columns:
                df[col] = 0
        df["total_gole"]   = df["FTHG"] + df["FTAG"]
        df["total_kartki"] = df["HY"] + df["AY"] + (df["HR"] + df["AR"]) * 2
        df["total_rozne"]  = df["HC"] + df["AC"]
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

    n_prev_target = int(n_now * w_prev / (1 - w_prev))
    n_prev_target = min(n_prev_target, len(df_prev))

    df_prev_sample = df_prev.tail(n_prev_target).copy()
    df_prev_sample["_sezon"] = "poprzedni"
    df_now = df_now.copy()
    df_now["_sezon"] = "biezacy"

    df = pd.concat([df_prev_sample, df_now], ignore_index=True).sort_values("Date")
    return df

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
def weighted_mean(values: pd.Series) -> float:
    if len(values) == 0:
        return 0.0
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
        dane[d] = {
            "Gole strzelone (dom)":    weighted_mean(home["FTHG"]),
            "Gole stracone (dom)":     weighted_mean(home["FTAG"]),
            "Gole strzelone (wyjazd)": weighted_mean(away["FTAG"]),
            "Gole stracone (wyjazd)":  weighted_mean(away["FTHG"]),
            "Różne (dom)":             weighted_mean(home["total_rozne"]),
            "Różne (wyjazd)":          weighted_mean(away["total_rozne"]),
            "Kartki (dom)":            weighted_mean(home["total_kartki"]),
            "Kartki (wyjazd)":         weighted_mean(away["total_kartki"]),
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
    n     = len(df)

    obs_00 = len(df[(df["FTHG"] == 0) & (df["FTAG"] == 0)]) / n
    obs_11 = len(df[(df["FTHG"] == 1) & (df["FTAG"] == 1)]) / n

    from scipy.stats import poisson as _p
    exp_00 = _p.pmf(0, avg_h) * _p.pmf(0, avg_a)
    exp_11 = _p.pmf(1, avg_h) * _p.pmf(1, avg_a)

    rho_00 = (obs_00 / exp_00 - 1) / (avg_h * avg_a) if exp_00 > 0 else -0.13
    rho_11 = -(obs_11 / exp_11 - 1) if exp_11 > 0 else -0.13
    rho    = float(np.clip(np.mean([rho_00, rho_11]), -0.25, 0.0))

    return {"avg_home": avg_h, "avg_away": avg_a, "rho": rho, "n_biezacy": n_biezacy}

def oblicz_lambdy(h: str, a: str, srednie_df: pd.DataFrame,
                  srednie_lig: dict, forma_dict: dict) -> tuple:
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

    lam_h = avg_h * atak_h * obrona_a * form_weight(h)
    lam_a = avg_a * atak_a * obrona_h * form_weight(a)
    
    # dodatkowe statystyki dla alternatywnych rynków
    lam_r = (srednie_df.loc[h, "Różne (dom)"] + srednie_df.loc[a, "Różne (wyjazd)"]) / 2
    lam_k = (srednie_df.loc[h, "Kartki (dom)"] + srednie_df.loc[a, "Kartki (wyjazd)"]) / 2
    
    return float(np.clip(lam_h, 0.3, 4.5)), float(np.clip(lam_a, 0.3, 4.5)), lam_r, lam_k

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
# MODEL POISSONA / DIXON-COLES
# ===========================================================================
def dixon_coles_adj(M: np.ndarray, lam_h: float, lam_a: float,
                    rho: float = -0.13) -> np.ndarray:
    M = M.copy()
    tau = {
        (0, 0): 1 - lam_h * lam_a * rho,
        (1, 0): 1 + lam_a * rho,
        (0, 1): 1 + lam_h * rho,
        (1, 1): 1 - rho,
    }
    for (i, j), t in tau.items():
        if i < M.shape[0] and j < M.shape[1]:
            M[i, j] *= max(t, 0.001)
    M /= M.sum()
    return M

def wybierz_typ(p_home: float, p_draw: float, p_away: float) -> tuple:
    PROG_PEWNY    = 0.42
    PROG_PODWOJNA = 0.62
    if p_home >= PROG_PEWNY: return "1",  p_home
    if p_away >= PROG_PEWNY: return "2",  p_away
    if p_draw >= PROG_PEWNY: return "X",  p_draw
    p_1x = p_home + p_draw
    p_x2 = p_away + p_draw
    if p_1x >= PROG_PODWOJNA or p_x2 >= PROG_PODWOJNA:
        return ("1X", p_1x) if p_1x >= p_x2 else ("X2", p_x2)
    probs = {"1": p_home, "X": p_draw, "2": p_away}
    t = max(probs, key=probs.get)
    return t, probs[t]

def wybierz_wynik(M: np.ndarray, lam_h: float, lam_a: float) -> tuple:
    idx_max = np.unravel_index(M.argmax(), M.shape)
    p_max   = float(M[idx_max])
    flat    = M.flatten()
    top3    = np.sort(flat)[::-1][:3]
    spread  = float(top3[0] - top3[2])
    if spread < 0.04:
        g_h = min(int(round(lam_h)), M.shape[0] - 1)
        g_a = min(int(round(lam_a)), M.shape[1] - 1)
        return g_h, g_a, float(M[g_h, g_a])
    return int(idx_max[0]), int(idx_max[1]), p_max

def entropy_meczu(p_home: float, p_draw: float, p_away: float) -> float:
    ps = [p for p in [p_home, p_draw, p_away] if p > 0]
    return float(-sum(p * np.log2(p) for p in ps))

def chaos_label(entropy: float) -> tuple:
    pct = entropy / np.log2(3)
    if pct < 0.55:
        return "Pewny",    "🔒", pct
    elif pct < 0.75:
        return "Klarowny", "🎯", pct
    elif pct < 0.90:
        return "Otwarty",  "⚡", pct
    else:
        return "Chaos",    "🌀", pct

def confidence_score(p_home: float, p_draw: float, p_away: float) -> tuple:
    vals  = sorted([p_home, p_draw, p_away], reverse=True)
    spread = vals[0] - vals[2]
    edge   = vals[0] - vals[1]
    if edge > 0.18:
        return "High",     "🟢", f"Wyraźny faworyt (+{edge:.0%} nad 2. opcją)"
    elif edge > 0.08:
        return "Medium",   "🟡", f"Umiarkowana przewaga (+{edge:.0%})"
    else:
        return "Coinflip", "🔴", f"Mecz bardzo wyrównany (spread {spread:.0%})"

def predykcja_meczu(lam_h: float, lam_a: float, rho: float = -0.13) -> dict:
    max_gole = int(np.clip(np.ceil(max(lam_h, lam_a) + 4), 6, 10))
    M_raw = np.outer(
        poisson.pmf(range(max_gole), lam_h),
        poisson.pmf(range(max_gole), lam_a),
    )
    M = dixon_coles_adj(M_raw, lam_h, lam_a, rho=rho)

    p_home = float(np.tril(M, -1).sum())
    p_draw = float(np.trace(M))
    p_away = float(np.triu(M, 1).sum())

    wynik_h, wynik_a, p_exact = wybierz_wynik(M, lam_h, lam_a)

    def fo(p): return round(1 / p, 2) if p > 0.001 else 999.0

    typ, p_typ = wybierz_typ(p_home, p_draw, p_away)
    conf_level, conf_emoji, conf_opis = confidence_score(p_home, p_draw, p_away)
    ent   = entropy_meczu(p_home, p_draw, p_away)
    ch_label, ch_emoji, ch_pct = chaos_label(ent)

    return {
        "lam_h": lam_h, "lam_a": lam_a,
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "wynik_h": wynik_h, "wynik_a": wynik_a, "p_exact": p_exact,
        "fo_home": fo(p_home), "fo_draw": fo(p_draw), "fo_away": fo(p_away),
        "typ": typ, "p_typ": p_typ, "fo_typ": fo(p_typ),
        "conf_level": conf_level, "conf_emoji": conf_emoji, "conf_opis": conf_opis,
        "entropy": ent, "chaos_label": ch_label, "chaos_emoji": ch_emoji,
        "chaos_pct": ch_pct,
        "macierz": M,
    }

# ===========================================================================
# GENERATOR KUPONÓW
# ===========================================================================
def generuj_kupony(mecze_list, pred_dict, srednie_df, srednie_lig, forma_dict, rho):
    kupony = []

    # 1. PEWNIAKI – 2 lub 3 mecze z najwyższym confidence (High)
    pewne_mecze = []
    for m in mecze_list:
        key = (m['home'], m['away'])
        if key not in pred_dict:
            continue
        p = pred_dict[key]
        if p['conf_level'] == 'High' and p['p_typ'] > 0.58:
            pewne_mecze.append({
                'home': m['home'],
                'away': m['away'],
                'typ': p['typ'],
                'fo': p['fo_typ'],
                'p': p['p_typ']
            })
    if len(pewne_mecze) >= 2:
        selected = pewne_mecze[:3]
        ako = np.prod([m['fo'] for m in selected])
        p_combo = np.prod([m['p'] for m in selected])
        kupony.append({
            'typ': '💪 PEWNIAKI KOLEJKI',
            'mecze': [f"{m['home']} – {m['away']} ({m['typ']})" for m in selected],
            'mecze_json': [{'home': m['home'], 'away': m['away'], 'typ': m['typ']} for m in selected],
            'ako': round(ako, 2),
            'p_combo': round(p_combo, 3),
            'ryzyko': 'Niskie',
            'opis': f"{len(selected)} mecze z najwyższą pewnością modelu."
        })

    # 2. WSZYSTKIE BTTS
    btts_mecze = []
    for m in mecze_list:
        h, a = m['home'], m['away']
        if h not in srednie_df.index or a not in srednie_df.index:
            continue
        lh, la, _, _ = oblicz_lambdy(h, a, srednie_df, srednie_lig, forma_dict)
        mg = int(np.clip(np.ceil(max(lh, la) + 4), 6, 10))
        M = dixon_coles_adj(
            np.outer(poisson.pmf(range(mg), lh), poisson.pmf(range(mg), la)),
            lh, la, rho=rho
        )
        p_btts = float(1 - M[0, :].sum() - M[:, 0].sum() + M[0, 0])
        if p_btts >= 0.65:
            btts_mecze.append({
                'home': h, 'away': a,
                'p': p_btts,
                'fo': 1/p_btts
            })
    if len(btts_mecze) >= 3:
        selected = btts_mecze[:4]
        ako = np.prod([m['fo'] for m in selected])
        p_combo = np.prod([m['p'] for m in selected])
        kupony.append({
            'typ': '⚽ WSZYSTKIE BTTS',
            'mecze': [f"{m['home']} – {m['away']}" for m in selected],
            'mecze_json': [{'home': m['home'], 'away': m['away']} for m in selected],
            'ako': round(ako, 2),
            'p_combo': round(p_combo, 3),
            'ryzyko': 'Średnie',
            'opis': f"{len(selected)} meczów, w których obie drużyny powinny strzelić."
        })

    # 3. COMBO: Over 2.5 + BTTS
    combo_mecze = []
    for m in mecze_list:
        h, a = m['home'], m['away']
        if h not in srednie_df.index or a not in srednie_df.index:
            continue
        lh, la, _, _ = oblicz_lambdy(h, a, srednie_df, srednie_lig, forma_dict)
        mg = int(np.clip(np.ceil(max(lh, la) + 4), 6, 10))
        M = dixon_coles_adj(
            np.outer(poisson.pmf(range(mg), lh), poisson.pmf(range(mg), la)),
            lh, la, rho=rho
        )
        p_btts = float(1 - M[0, :].sum() - M[:, 0].sum() + M[0, 0])
        p_over25 = float(sum(M[i, j] for i in range(mg) for j in range(mg) if i + j > 2))
        if p_btts >= 0.6 and p_over25 >= 0.6:
            combo_mecze.append({
                'home': h, 'away': a,
                'p': p_btts * p_over25,
                'fo': 1/(p_btts * p_over25)
            })
    if len(combo_mecze) >= 2:
        selected = combo_mecze[:3]
        ako = np.prod([m['fo'] for m in selected])
        p_combo = np.prod([m['p'] for m in selected])
        kupony.append({
            'typ': '🎯 OVER 2.5 + BTTS',
            'mecze': [f"{m['home']} – {m['away']}" for m in selected],
            'mecze_json': [{'home': m['home'], 'away': m['away']} for m in selected],
            'ako': round(ako, 2),
            'p_combo': round(p_combo, 3),
            'ryzyko': 'Wysokie',
            'opis': f"{len(selected)} meczów, w których model widzi zarówno gole jak i strzelanie obu stron."
        })

    # 4. VALUE PICK
    value_candidates = []
    for m in mecze_list:
        key = (m['home'], m['away'])
        if key not in pred_dict:
            continue
        p = pred_dict[key]
        fair = p['fo_typ']
        market = fair * 0.92
        if market > 1.4 and (fair / market - 1) > 0.05:
            value_candidates.append({
                'home': m['home'], 'away': m['away'],
                'typ': p['typ'],
                'fair': fair,
                'market': round(market, 2),
                'value': (fair / market - 1)
            })
    if value_candidates:
        best = max(value_candidates, key=lambda x: x['value'])
        kupony.append({
            'typ': '💰 VALUE PICK',
            'mecze': [f"{best['home']} – {best['away']} ({best['typ']})"],
            'mecze_json': [{'home': best['home'], 'away': best['away'], 'typ': best['typ']}],
            'ako': best['fair'],
            'market': best['market'],
            'value_pct': round(best['value']*100, 1),
            'ryzyko': 'Średnie',
            'opis': f"Fair odds {best['fair']:.2f} vs rynek {best['market']:.2f} → +{best['value']*100:.1f}% value"
        })

    order = {'Niskie': 0, 'Średnie': 1, 'Wysokie': 2}
    kupony.sort(key=lambda x: order.get(x['ryzyko'], 99))
    return kupony

def zapisz_kupon_do_db(kupon, liga, kolejnosc):
    init_db()
    con = sqlite3.connect(DB_FILE)
    try:
        con.execute(
            """INSERT OR IGNORE INTO kupony
               (liga, kolejnosc, typ_kuponu, mecze_json, ako, p_combo, data)
               VALUES (?,?,?,?,?,?,?)""",
            (liga, kolejnosc, kupon['typ'],
             json.dumps(kupon['mecze_json'], ensure_ascii=False),
             kupon['ako'], kupon.get('p_combo'),
             datetime.now().strftime("%Y-%m-%d"))
        )
        con.commit()
    except Exception as e:
        st.warning(f"Nie udało się zapisać kuponu: {e}")
    finally:
        con.close()

def weryfikuj_kupony(liga, kolejnosc, df_history, pred_dict):
    init_db()
    con = sqlite3.connect(DB_FILE)
    rows = con.execute(
        "SELECT typ_kuponu, mecze_json, ako, p_combo FROM kupony WHERE liga=? AND kolejnosc=?",
        (liga, kolejnosc)
    ).fetchall()
    con.close()

    wyniki = []
    for typ, mecze_json, ako, p_combo in rows:
        mecze_list = json.loads(mecze_json)
        wszystkie_ok = True
        opis = ""
        for m in mecze_list:
            home, away = m['home'], m['away']
            match = df_history[(df_history['HomeTeam'] == home) & (df_history['AwayTeam'] == away)]
            if match.empty:
                wszystkie_ok = False
                opis = "oczekuje na wyniki"
                break
            if 'typ' in m:
                rzecz = "1" if match.iloc[0]['FTHG'] > match.iloc[0]['FTAG'] else \
                        ("2" if match.iloc[0]['FTHG'] < match.iloc[0]['FTAG'] else "X")
                typ_pred = m['typ']
                if typ_pred == "1X": traf = rzecz in ("1","X")
                elif typ_pred == "X2": traf = rzecz in ("X","2")
                else: traf = (rzecz == typ_pred)
                if not traf:
                    wszystkie_ok = False
                    opis = f"❌ {home}-{away}: typ {typ_pred}, wynik {match.iloc[0]['FTHG']}:{match.iloc[0]['FTAG']}"
                    break
            else:
                if typ == "⚽ WSZYSTKIE BTTS":
                    hg = match.iloc[0]['FTHG']; ag = match.iloc[0]['FTAG']
                    if not (hg > 0 and ag > 0):
                        wszystkie_ok = False
                        opis = f"❌ {home}-{away}: brak BTTS ({hg}:{ag})"
                        break
        if wszystkie_ok:
            wyniki.append({
                'typ': typ,
                'status': '✅ TRAFIONY',
                'ako': ako,
                'p_combo': p_combo,
                'opis': 'Kupon wygrany!'
            })
        else:
            wyniki.append({
                'typ': typ,
                'status': '❌ CHYBIONY' if '❌' in opis else '⏳ OCZEKUJE',
                'ako': ako,
                'p_combo': p_combo,
                'opis': opis if opis else 'Czeka na mecze'
            })
    return wyniki

# ===========================================================================
# KOMENTARZ ANALITYCZNY
# ===========================================================================
def _ocen_forme(f: str) -> str:
    if not f or f == "?":
        return "forma nieznana"
    w = f.count("W"); l = f.count("L")
    if w >= 4: return "doskonała forma"
    if w >= 3: return "dobra forma"
    if l >= 3: return "słaba forma"
    return "nieregularna forma"

def generuj_komentarz(home: str, away: str, pred: dict, forma_dict: dict) -> str:
    fh = forma_dict.get(home, "?")
    fa = forma_dict.get(away, "?")

    roznica = pred["p_home"] - pred["p_away"]
    if   roznica >  0.20: faw = f"{home} jest wyraźnym faworytem ({pred['p_home']:.0%})."
    elif roznica >  0.08: faw = f"{home} jest lekkim faworytem ({pred['p_home']:.0%} vs {pred['p_away']:.0%})."
    elif roznica < -0.20: faw = f"{away} jest wyraźnym faworytem ({pred['p_away']:.0%})."
    elif roznica < -0.08: faw = f"{away} jest lekkim faworytem ({pred['p_away']:.0%} vs {pred['p_home']:.0%})."
    else:                 faw = f"Mecz bardzo wyrównany – remis ma {pred['p_draw']:.0%} szans."
    
    gole = pred["lam_h"] + pred["lam_a"]
    gole_opis = (
        f"Model spodziewa się bramkostrzelnego meczu (śr. {gole:.1f} goli)."
        if gole >= 2.8 else
        f"Model przewiduje defensywny mecz (śr. {gole:.1f} goli)."
    )
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
    rows    = [
        "<table style='border-collapse:collapse;font-size:0.70em;width:100%;margin-top:6px'>",
        "<thead><tr>",
        f"<th style='padding:2px 5px;color:#999;font-size:0.85em'>{home[:7]}↓ {away[:7]}→</th>",
    ]
    for j in range(n):
        rows.append(f"<th style='padding:2px 5px;text-align:center;color:#aaa'>{j}</th>")
    rows.append("</tr></thead><tbody>")
    for i in range(n):
        rows.append("<tr>")
        rows.append(f"<td style='padding:2px 5px;font-weight:bold;color:#aaa;text-align:center'>{i}</td>")
        for j in range(n):
            v     = M[i, j]
            ratio = v / max_val
            g     = int(255 * (1 - ratio * 0.85))
            bold  = "font-weight:bold;" if (i, j) == best else ""
            bord  = "border:2px solid #E91E63;" if (i, j) == best else ""
            rows.append(
                f"<td style='padding:2px 5px;text-align:center;"
                f"background:rgb(255,{g},{g});{bold}{bord}'>{v:.1%}</td>"
            )
        rows.append("</tr>")
    rows.append("</tbody></table>")
    return "".join(rows)

def macierz_goli(lam_h, lam_a, rho, linia_int, typ_gole):
    mg  = int(np.clip(np.ceil(max(lam_h, lam_a) + 4), 6, 10))
    M   = dixon_coles_adj(
        np.outer(poisson.pmf(range(mg), lam_h), poisson.pmf(range(mg), lam_a)),
        lam_h, lam_a, rho=rho
    )
    p_over = float(sum(M[i, j] for i in range(mg) for j in range(mg) if i + j > linia_int))
    p_g    = p_over if typ_gole == "Over" else 1 - p_over
    return M, p_g

# ===========================================================================
# ŁADOWANIE DANYCH
# ===========================================================================
st.sidebar.header("🌍 Wybór Rozgrywek")
wybrana_liga = st.sidebar.selectbox("Wybierz ligę", list(LIGI.keys()))
debug_mode   = st.sidebar.checkbox("🔧 Debug – niezmapowane nazwy", value=False)

historical = load_historical(LIGI[wybrana_liga]["csv_code"])
schedule   = load_schedule(LIGI[wybrana_liga]["file"])

st.title(f"Predykcje {wybrana_liga} 2025/26")
st.markdown("Dixon-Coles • Blend sezonów • Forma • Fair Odds • SQLite weryfikacja")

if not historical.empty:
    srednie_df  = oblicz_wszystkie_statystyki(historical.to_json())
    srednie_lig = oblicz_srednie_ligowe(historical.to_json())
    forma_dict  = oblicz_forme(historical)
    tabela      = tabela_ligowa(historical)
    rho         = srednie_lig["rho"]
    n_biezacy   = srednie_lig["n_biezacy"]
    w_prev      = waga_poprzedniego(n_biezacy)

    # =========================================================================
    # TABS – nowy układ z 8 zakładkami
    # =========================================================================
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🎯 Bet Builder",
        "⚽ Przewidywane Wyniki",
        "🎲 Systemy / AKO",
        "✅ Weryfikacja",
        "📊 Tabela i Forma",
        "🏆 Power Rating",
        "📈 Model & Dane",
        "🔧 Debug"
    ])

    # =========================================================================
    # TAB 1 – BET BUILDER
    # =========================================================================
    with tab1:
        st.subheader("🎛️ Combo Builder")
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
        min_prob = st.slider("Min. prawdopodobieństwo combo", 0.0, 1.0, 0.40, 0.05)

        if not schedule.empty and not srednie_df.empty:
            dzisiaj  = datetime.now().date()
            przyszle = schedule[schedule["date"].dt.date >= dzisiaj]
            if not przyszle.empty:
                nb    = przyszle["round"].min()
                mecze = schedule[schedule["round"] == nb]
                st.caption(f"Kolejka {int(nb)} – {len(mecze)} meczów")
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Combo Builder**")
                    combo_count = 0
                    for _, mecz in mecze.iterrows():
                        h = map_nazwa(mecz["home_team"])
                        a = map_nazwa(mecz["away_team"])
                        if h not in srednie_df.index or a not in srednie_df.index:
                            continue
                        lam_h, lam_a, lam_r, lam_k = oblicz_lambdy(h, a, srednie_df, srednie_lig, forma_dict)
                        _, p_g = macierz_goli(lam_h, lam_a, rho, int(linia_gole), typ_gole)
                        p_r   = oblicz_p(typ_rogi,   linia_rogi,   lam_r)
                        p_k   = oblicz_p(typ_kartki, linia_kartki, lam_k)
                        p_combo = p_g * p_r * p_k
                        if p_combo >= min_prob:
                            combo_count += 1
                            with st.expander(f"{h} vs {a} ({p_combo:.1%})"):
                                st.write(f"{koloruj(p_g)} Gole {typ_gole} {linia_gole}: {p_g:.1%}")
                                st.write(f"{koloruj(p_r)} Rożne {typ_rogi} {linia_rogi}: {p_r:.1%}")
                                st.write(f"{koloruj(p_k)} Kartki {typ_kartki} {linia_kartki}: {p_k:.1%}")
                    if combo_count == 0:
                        st.info("Brak meczów spełniających kryteria. Zmniejsz próg.")

                with col2:
                    st.write("**BTTS Ranking**")
                    btts_data = []
                    for _, mecz in mecze.iterrows():
                        h = map_nazwa(mecz["home_team"])
                        a = map_nazwa(mecz["away_team"])
                        if h not in srednie_df.index or a not in srednie_df.index:
                            continue
                        lam_h, lam_a, _, _ = oblicz_lambdy(h, a, srednie_df, srednie_lig, forma_dict)
                        mg  = int(np.clip(np.ceil(max(lam_h, lam_a) + 4), 6, 10))
                        M_b = dixon_coles_adj(
                            np.outer(poisson.pmf(range(mg), lam_h), poisson.pmf(range(mg), lam_a)),
                            lam_h, lam_a, rho=rho
                        )
                        p_btts = float(1 - M_b[0,:].sum() - M_b[:,0].sum() + M_b[0,0])
                        btts_data.append((f"{h} - {a}", p_btts))
                    for ms, p in sorted(btts_data, key=lambda x: x[1], reverse=True):
                        st.write(f"{koloruj(p)} **{ms}**: {p:.1%}")
            else:
                st.info("Brak nadchodzących meczów w terminarzu.")

    # =========================================================================
    # TAB 2 – PRZEWIDYWANE WYNIKI
    # =========================================================================
    with tab2:
        st.subheader("⚽ Przewidywane wyniki – najbliższa kolejka")
        st.caption(
            "**Fair Odds** = kurs bez marży bukmachera. "
            "**Typ modelu** = optymalny typ wg modelu (1/X/2/1X/X2)."
        )

        tgl1, tgl2, tgl3 = st.columns(3)
        with tgl1: pokaz_komentarz = st.toggle("💬 Komentarz", value=True)
        with tgl2: pokaz_macierz  = st.toggle("🔢 Macierz", value=False)
        with tgl3:
            zapisuj_flag = st.toggle(
                "💾 Zapisz do weryfikacji", value=False,
                help="Zapisuje predykcje do SQLite – trwałe między sesjami."
            )

        if not schedule.empty and not srednie_df.empty:
            dzisiaj  = datetime.now().date()
            przyszle = schedule[schedule["date"].dt.date >= dzisiaj]
            if not przyszle.empty:
                nb    = przyszle["round"].min()
                mecze = schedule[schedule["round"] == nb]
                st.caption(f"Kolejka {int(nb)} – {len(mecze)} meczów")

                nowe_predykcje = []
                dopasowane     = 0

                kol_a, kol_b = st.columns(2)
                mecze_list   = list(mecze.iterrows())

                for idx, (_, mecz) in enumerate(mecze_list):
                    h = map_nazwa(mecz["home_team"])
                    a = map_nazwa(mecz["away_team"])
                    if h not in srednie_df.index or a not in srednie_df.index:
                        continue

                    lam_h, lam_a, lam_r, lam_k = oblicz_lambdy(h, a, srednie_df, srednie_lig, forma_dict)
                    pred         = predykcja_meczu(lam_h, lam_a, rho=rho)
                    dopasowane  += 1
                    data_meczu   = mecz["date"].strftime("%d.%m %H:%M") if pd.notna(mecz["date"]) else ""

                    nowe_predykcje.append({
                        "liga": wybrana_liga, "home": h, "away": a,
                        "round": int(nb), "typ": pred["typ"], "fo_typ": pred["fo_typ"],
                        "p_home": round(pred["p_home"], 4),
                        "p_draw": round(pred["p_draw"], 4),
                        "p_away": round(pred["p_away"], 4),
                        "data": datetime.now().strftime("%Y-%m-%d"),
                    })

                    kolumna = kol_a if idx % 2 == 0 else kol_b

                    ikony_typ = {"1": "🔵", "X": "🟠", "2": "🔴", "1X": "🟣", "X2": "🟣"}
                    ikona_typ = ikony_typ.get(pred["typ"], "⚪")
                    conf_ikona = "🟢" if pred["conf_level"] == "High" else ("🟡" if pred["conf_level"] == "Medium" else "🔴")
                    label_exp = (
                        f"{conf_ikona} {h} vs {a}"
                        f"  ·  {ikona_typ} {pred['typ']} @ {pred['fo_typ']:.2f}"
                        f"  ·  {data_meczu}"
                    )

                    with kolumna:
                        with st.expander(label_exp, expanded=False):
                            ch, cmid, ca = st.columns([5, 2, 5])
                            with ch:
                                st.markdown(f"<div style='font-size:1.05em;font-weight:bold'>{h}</div>",
                                            unsafe_allow_html=True)
                            with cmid:
                                st.markdown(f"<div style='text-align:center;color:#888;font-size:0.78em'>{data_meczu}</div>",
                                            unsafe_allow_html=True)
                            with ca:
                                st.markdown(f"<div style='font-size:1.05em;font-weight:bold;text-align:right'>{a}</div>",
                                            unsafe_allow_html=True)

                            st.markdown(
                                f"<div style='text-align:center;font-size:1.7em;font-weight:bold;margin:4px 0'>"
                                f"⚽ {pred['wynik_h']}:{pred['wynik_a']}"
                                f"<span style='font-size:0.5em;color:#888;font-weight:normal;margin-left:8px'>"
                                f"({pred['p_exact']:.1%})</span></div>",
                                unsafe_allow_html=True,
                            )

                            conf_colors = {"High": "#4CAF50", "Medium": "#FF9800", "Coinflip": "#F44336"}
                            conf_c = conf_colors.get(pred["conf_level"], "#888")
                            chaos_colors = {"Pewny": "#4CAF50", "Klarowny": "#8BC34A",
                                            "Otwarty": "#FF9800", "Chaos": "#F44336"}
                            ch_c      = chaos_colors.get(pred["chaos_label"], "#888")
                            bar_w     = int(pred["chaos_pct"] * 100)
                            bar_color = ch_c
                            st.markdown(
                                f"<div style='text-align:center;margin-bottom:4px'>"
                                f"Typ: {badge_typ(pred['typ'])}&nbsp;&nbsp;"
                                f"<span style='font-size:0.88em;color:#888'>"
                                f"Fair Odds: <b>{pred['fo_typ']:.2f}</b> ({pred['p_typ']:.1%})"
                                f"</span></div>"
                                f"<div style='text-align:center;font-size:0.80em;color:{conf_c};margin-bottom:6px'>"
                                f"{pred['conf_emoji']} <b>{pred['conf_level']}</b> · {pred['conf_opis']}"
                                f"</div>"
                                f"<div style='margin:0 8px 8px 8px'>"
                                f"<div style='display:flex;align-items:center;gap:8px;font-size:0.78em;color:#888'>"
                                f"<span>{pred['chaos_emoji']} Chaos Index: <b style='color:{ch_c}'>{pred['chaos_label']}</b>"
                                f" ({pred['entropy']:.2f} bits)</span>"
                                f"</div>"
                                f"<div style='background:#333;border-radius:4px;height:4px;margin-top:4px'>"
                                f"<div style='background:{bar_color};width:{bar_w}%;height:4px;border-radius:4px;transition:width 0.3s'></div>"
                                f"</div></div>",
                                unsafe_allow_html=True,
                            )

                            mc1, mc2, mc3 = st.columns(3)
                            with mc1:
                                st.metric(f"1 – {h[:12]}", f"{pred['p_home']:.1%}", f"@ {pred['fo_home']:.2f}")
                            with mc2:
                                st.metric("X – Remis", f"{pred['p_draw']:.1%}", f"@ {pred['fo_draw']:.2f}")
                            with mc3:
                                st.metric(f"2 – {a[:12]}", f"{pred['p_away']:.1%}", f"@ {pred['fo_away']:.2f}")

                            st.markdown(
                                f"<div style='text-align:center;font-size:0.82em;color:#888;margin-top:2px'>"
                                f"λ {h[:8]}: <b>{pred['lam_h']:.2f}</b> &nbsp;|&nbsp; "
                                f"λ {a[:8]}: <b>{pred['lam_a']:.2f}</b> &nbsp;|&nbsp; "
                                f"Łącznie: <b>{pred['lam_h']+pred['lam_a']:.2f}</b></div>",
                                unsafe_allow_html=True,
                            )

                            # =================================================
                            # ALTERNATYWNE RYNKI (NOWOŚĆ)
                            # =================================================
                            with st.expander("📊 Alternatywne rynki (60%+ pewność)", expanded=False):
                                alt_cols = st.columns(2)
                                with alt_cols[0]:
                                    st.markdown("**⚽ Gole**")
                                    for l in [1.5, 2.5, 3.5]:
                                        p_over = 1 - poisson.cdf(int(l), lam_h + lam_a)
                                        if p_over >= 0.6:
                                            fair = 1/p_over
                                            st.markdown(f"• Over {l}: {p_over:.1%} (fair {fair:.2f})")
                                    for l in [1.5, 2.5]:
                                        p_under = poisson.cdf(int(l), lam_h + lam_a)
                                        if p_under >= 0.6:
                                            fair = 1/p_under
                                            st.markdown(f"• Under {l}: {p_under:.1%} (fair {fair:.2f})")

                                with alt_cols[1]:
                                    st.markdown("**🚩 Rożne / 🟨 Kartki**")
                                    for l in [7.5, 8.5, 9.5, 10.5]:
                                        p_over = 1 - poisson.cdf(int(l), lam_r)
                                        if p_over >= 0.6:
                                            fair = 1/p_over
                                            st.markdown(f"• Over {l} rożnych: {p_over:.1%} (fair {fair:.2f})")
                                    for l in [3.5, 4.5, 5.5]:
                                        p_over = 1 - poisson.cdf(int(l), lam_k)
                                        if p_over >= 0.6:
                                            fair = 1/p_over
                                            st.markdown(f"• Over {l} kartek: {p_over:.1%} (fair {fair:.2f})")

                                # BTTS już liczone
                                mg_btts = int(np.clip(np.ceil(max(lam_h, lam_a) + 4), 6, 10))
                                M_btts = dixon_coles_adj(
                                    np.outer(poisson.pmf(range(mg_btts), lam_h), poisson.pmf(range(mg_btts), lam_a)),
                                    lam_h, lam_a, rho=rho
                                )
                                p_btts_local = float(1 - M_btts[0,:].sum() - M_btts[:,0].sum() + M_btts[0,0])
                                if p_btts_local >= 0.6:
                                    st.markdown(f"**⚽ Obie drużyny strzelą**: {p_btts_local:.1%} (fair {1/p_btts_local:.2f})")

                            if pokaz_komentarz:
                                st.info(f"💬 {generuj_komentarz(h, a, pred, forma_dict)}")

                            if pokaz_macierz:
                                st.markdown("**Macierz dokładnych wyników**")
                                st.markdown(render_macierz_html(pred["macierz"], h, a),
                                            unsafe_allow_html=True)

                if zapisuj_flag and nowe_predykcje:
                    dodane = 0
                    init_db()
                    con = sqlite3.connect(DB_FILE)
                    for p in nowe_predykcje:
                        try:
                            con.execute(
                                "INSERT OR IGNORE INTO predykcje (liga,home,away,round,typ,fo_typ,p_home,p_draw,p_away,data) VALUES (?,?,?,?,?,?,?,?,?,?)",
                                (p["liga"], p["home"], p["away"], p["round"], p["typ"], p["fo_typ"], p["p_home"], p["p_draw"], p["p_away"], p["data"])
                            )
                            dodane += con.execute("SELECT changes()").fetchone()[0]
                        except Exception:
                            pass
                    con.commit()
                    con.close()
                    st.success(f"✅ Zapisano {dodane} nowych predykcji do bazy." if dodane
                               else "Predykcje tej kolejki już są w bazie.")

                if dopasowane > 0:
                    df_export = pd.DataFrame(nowe_predykcje)[
                        ["liga","home","away","round","typ","fo_typ","p_home","p_draw","p_away","data"]
                    ]
                    st.download_button(
                        label="⬇️ Pobierz predykcje kolejki (CSV)",
                        data=df_export.to_csv(index=False, decimal=","),
                        file_name=f"predykcje_{wybrana_liga.replace(' ','_')}_kolejka{int(nb)}.csv",
                        mime="text/csv",
                    )

                if dopasowane == 0:
                    st.warning("Żaden mecz nie mógł być dopasowany – sprawdź Debug.")
            else:
                st.info("Brak nadchodzących meczów w terminarzu.")

    # =========================================================================
    # TAB 3 – SYSTEMY / AKO (NOWA)
    # =========================================================================
    with tab3:
        st.subheader("🎲 Systemy / AKO – propozycje na kolejkę")

        if not schedule.empty and not srednie_df.empty:
            dzisiaj = datetime.now().date()
            przyszle = schedule[schedule["date"].dt.date >= dzisiaj]
            if not przyszle.empty:
                nb = przyszle["round"].min()
                mecze_list = []
                pred_dict = {}
                for _, mecz in schedule[schedule["round"] == nb].iterrows():
                    h = map_nazwa(mecz["home_team"])
                    a = map_nazwa(mecz["away_team"])
                    if h not in srednie_df.index or a not in srednie_df.index:
                        continue
                    lam_h, lam_a, _, _ = oblicz_lambdy(h, a, srednie_df, srednie_lig, forma_dict)
                    pred = predykcja_meczu(lam_h, lam_a, rho=rho)
                    mecze_list.append({'home': h, 'away': a, 'date': mecz['date']})
                    pred_dict[(h, a)] = pred

                if mecze_list:
                    kupony = generuj_kupony(mecze_list, pred_dict, srednie_df, srednie_lig, forma_dict, rho)

                    col_save, _ = st.columns([1,3])
                    with col_save:
                        if st.button("💾 Zapisz te kupony do bazy", use_container_width=True):
                            for k in kupony:
                                zapisz_kupon_do_db(k, wybrana_liga, int(nb))
                            st.success("Kupony zapisane! Będą weryfikowane po meczach.")
                    st.divider()

                    for kupon in kupony:
                        with st.container(border=True):
                            cols = st.columns([3, 1, 1, 1])
                            with cols[0]:
                                st.markdown(f"**{kupon['typ']}**")
                                for m in kupon['mecze']:
                                    st.markdown(f"• {m}")
                                st.caption(kupon['opis'])
                            with cols[1]:
                                st.metric("AKO", f"{kupon['ako']:.2f}")
                                if 'p_combo' in kupon:
                                    st.caption(f"p = {kupon['p_combo']:.0%}")
                            with cols[2]:
                                ryzyko_color = {"Niskie": "🟢", "Średnie": "🟡", "Wysokie": "🔴"}.get(kupon['ryzyko'], "⚪")
                                st.markdown(f"**Ryzyko:** {ryzyko_color} {kupon['ryzyko']}")
                            with cols[3]:
                                if 'value_pct' in kupon:
                                    st.metric("Value", f"+{kupon['value_pct']}%")
                else:
                    st.info("Brak dopasowanych meczów dla tej kolejki.")
            else:
                st.info("Brak nadchodzących meczów.")
        else:
            st.warning("Brak danych.")

        # ------------------------------------------------------------
        # PODGLĄD WCZEŚNIEJSZYCH KUPONÓW
        # ------------------------------------------------------------
        with st.expander("📊 Historia i weryfikacja kuponów", expanded=False):
            init_db()
            con = sqlite3.connect(DB_FILE)
            wszystkie_kupony = con.execute(
                "SELECT liga, kolejnosc, typ_kuponu, mecze_json, ako, p_combo, data FROM kupony ORDER BY kolejnosc DESC"
            ).fetchall()
            con.close()

            if wszystkie_kupony:
                ligi_w_kuponach = sorted(set([r[0] for r in wszystkie_kupony]))
                wyb_liga_hist = st.selectbox("Filtruj ligę", ["Wszystkie"] + ligi_w_kuponach)

                dane_do_weryfikacji = []
                for r in wszystkie_kupony:
                    if wyb_liga_hist != "Wszystkie" and r[0] != wyb_liga_hist:
                        continue
                    mecze_json = json.loads(r[3])
                    wszystkie_rozegrane = True
                    for m in mecze_json:
                        match = historical[(historical['HomeTeam'] == m['home']) & (historical['AwayTeam'] == m['away'])]
                        if match.empty:
                            wszystkie_rozegrane = False
                            break
                    status = "✅ Rozegrany" if wszystkie_rozegrane else "⏳ Oczekuje"
                    dane_do_weryfikacji.append({
                        "Liga": r[0], "Kolejka": r[1], "Typ": r[2],
                        "AKO": round(r[4],2), "p combo": round(r[5],3) if r[5] else "-",
                        "Status": status, "Data zapisu": r[6][:10]
                    })
                if dane_do_weryfikacji:
                    st.dataframe(pd.DataFrame(dane_do_weryfikacji), use_container_width=True)
                else:
                    st.info("Brak kuponów dla wybranych kryteriów.")
            else:
                st.info("Brak zapisanych kuponów – użyj przycisku 'Zapisz te kupony do bazy'.")

    # =========================================================================
    # TAB 4 – WERYFIKACJA
    # =========================================================================
    with tab4:
        st.subheader("✅ Weryfikacja skuteczności modelu")
        st.caption("Predykcje zapisujesz w zakładce **⚽ Przewidywane Wyniki** (toggle 💾). Dane trzymane w SQLite.")

        init_db()
        con = sqlite3.connect(DB_FILE)
        predykcje_rows = con.execute(
            "SELECT liga, home, away, round, typ, fo_typ, p_home, p_draw, p_away, data FROM predykcje ORDER BY round DESC"
        ).fetchall()
        con.close()

        predykcje = []
        for r in predykcje_rows:
            predykcje.append({
                "liga": r[0], "home": r[1], "away": r[2], "round": r[3],
                "typ": r[4], "fo_typ": r[5], "p_home": r[6], "p_draw": r[7],
                "p_away": r[8], "data": r[9]
            })

        if not predykcje:
            st.info("Brak zapisanych predykcji. Zapisz je przed kolejką, wróć tu po meczach.")
        else:
            # Weryfikacja predykcji
            df_hist_biez = historical[historical["_sezon"] == "biezacy"] if "_sezon" in historical.columns else historical
            wyniki_rows = []
            for p in predykcje:
                if p["liga"] != wybrana_liga:
                    continue
                match = df_hist_biez[(df_hist_biez['HomeTeam'] == p['home']) & (df_hist_biez['AwayTeam'] == p['away'])]
                if match.empty:
                    status = "⏳ oczekuje"; wynik_r = "–"; trafiony = None; brier = None
                else:
                    row = match.iloc[-1]
                    hg, ag = int(row["FTHG"]), int(row["FTAG"])
                    wynik_r = f"{hg}:{ag}"
                    rzecz = "1" if hg > ag else ("2" if hg < ag else "X")
                    typ_pred = p["typ"]
                    if typ_pred == "1X": trafiony = rzecz in ("1","X")
                    elif typ_pred == "X2": trafiony = rzecz in ("X","2")
                    else: trafiony = (rzecz == typ_pred)
                    status = "✅ trafiony" if trafiony else "❌ chybiony"
                    r1 = 1.0 if rzecz == "1" else 0.0
                    rx = 1.0 if rzecz == "X" else 0.0
                    r2 = 1.0 if rzecz == "2" else 0.0
                    brier = round((p["p_home"]-r1)**2 + (p["p_draw"]-rx)**2 + (p["p_away"]-r2)**2, 3)
                wyniki_rows.append({
                    "Liga": p["liga"], "Mecz": f"{p['home']} vs {p['away']}",
                    "Kolejka": p["round"], "Typ": p["typ"], "Fair Odds": p["fo_typ"],
                    "Wynik": wynik_r, "Status": status, "Trafiony": trafiony,
                    "Brier": brier, "Data": p["data"]
                })

            df_wer = pd.DataFrame(wyniki_rows) if wyniki_rows else pd.DataFrame()
            zakonczone = df_wer[df_wer["Trafiony"].notna()]

            if not zakonczone.empty:
                trafione = int(zakonczone["Trafiony"].sum())
                wszystkie = len(zakonczone)
                skuteczn = trafione / wszystkie
                sr_odds_tr = zakonczone[zakonczone["Trafiony"] == True]["Fair Odds"].mean()
                brier_vals = zakonczone["Brier"].dropna()
                sr_brier = brier_vals.mean() if len(brier_vals) > 0 else float("nan")
                brier_delta = f"{0.667 - sr_brier:+.3f} vs random" if not np.isnan(sr_brier) else None

                with st.container(border=True):
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("⚽ Rozegrane",      wszystkie)
                    m2.metric("✅ Trafione",         trafione)
                    m3.metric("🎯 Skuteczność",     f"{skuteczn:.1%}")
                    m4.metric("💰 Śr. odds (hit)",  f"{sr_odds_tr:.2f}" if not np.isnan(sr_odds_tr) else "–")
                    m5.metric("📐 Brier Score ↓",   f"{sr_brier:.3f}" if not np.isnan(sr_brier) else "–",
                              delta=brier_delta, delta_color="normal")

            # Weryfikacja kuponów dla ostatniej kolejki
            st.divider()
            st.markdown("### 🎲 Weryfikacja kuponów")

            if not schedule.empty and not historical.empty:
                ostatnie_daty = historical['Date'].max()
                if not schedule.empty:
                    przeszle = schedule[schedule['date'].dt.date <= ostatnie_daty.date()]
                    if not przeszle.empty:
                        ostatnia_rozegrana = przeszle['round'].max()
                        wyniki_kuponow = weryfikuj_kupony(wybrana_liga, int(ostatnia_rozegrana), df_hist_biez, {})
                        if wyniki_kuponow:
                            st.caption(f"Kolejka {ostatnia_rozegrana}")
                            for wk in wyniki_kuponow:
                                cols = st.columns([2,1,2,3])
                                with cols[0]:
                                    st.markdown(f"**{wk['typ']}**")
                                with cols[1]:
                                    st.markdown(f"AKO {wk['ako']}")
                                with cols[2]:
                                    emoji = "✅" if wk['status'] == '✅ TRAFIONY' else ("❌" if '❌' in wk['status'] else "⏳")
                                    st.markdown(f"{emoji} {wk['status']}")
                                with cols[3]:
                                    st.caption(wk['opis'])
                        else:
                            st.info("Brak kuponów do weryfikacji dla ostatniej kolejki.")
            else:
                st.info("Brak rozegranych meczów – weryfikacja możliwa po kolejce.")

    # =========================================================================
    # TAB 5 – TABELA I FORMA
    # =========================================================================
    with tab5:
        st.subheader("📊 Aktualna Sytuacja")
        c_l, c_f = st.columns([2, 1])
        with c_l:
            st.write("**Tabela Ligowa** *(tylko bieżący sezon)*")
            st.dataframe(tabela, use_container_width=True)
        with c_f:
            st.write("**Forma (5 ostatnich)**")
            st.dataframe(
                pd.DataFrame.from_dict(forma_dict, orient="index", columns=["Forma"]),
                use_container_width=True,
            )

    # =========================================================================
    # TAB 6 – POWER RATING
    # =========================================================================
    with tab6:
        st.subheader("🏆 Power Rating drużyn")
        st.caption(
            "Siła ataku i obrony normalizowana do średniej ligowej (=1.0). "
            "Wielkość bąbelka = łączna siła ataku+obrony. Kolor = forma ostatnich 5 meczów."
        )

        if not srednie_df.empty:
            avg_h = srednie_lig["avg_home"]
            avg_a = srednie_lig["avg_away"]

            power_rows = []
            for team in srednie_df.index:
                row = srednie_df.loc[team]
                atak  = ((row["Gole strzelone (dom)"] / avg_h) +
                          (row["Gole strzelone (wyjazd)"] / avg_a)) / 2
                obrona = 2 - ((row["Gole stracone (dom)"] / avg_h) +
                               (row["Gole stracone (wyjazd)"] / avg_a)) / 2
                forma  = forma_dict.get(team, "")
                wins   = forma.count("W")
                losses = forma.count("L")
                forma_score = wins - losses
                power_rows.append({
                    "Drużyna":      team,
                    "Atak":         round(atak, 3),
                    "Obrona":       round(obrona, 3),
                    "Forma score":  forma_score,
                    "Forma":        forma if forma else "?",
                    "Power":        round((atak + obrona) / 2, 3),
                })

            df_power = pd.DataFrame(power_rows).sort_values("Power", ascending=False).reset_index(drop=True)

            st.markdown("**Ranking Power Rating**")
            df_display = df_power[["Drużyna","Power","Atak","Obrona","Forma"]].copy()
            df_display.index = range(1, len(df_display) + 1)
            st.dataframe(df_display, use_container_width=True)

    # =========================================================================
    # TAB 7 – MODEL & DANE
    # =========================================================================
    with tab7:
        st.subheader("📈 Parametry modelu i dane")

        with st.container(border=True):
            st.markdown("**🔀 Blend sezonów**")
            bc1, bc2, bc3 = st.columns(3)
            bc1.metric("Mecze bieżącego sezonu", n_biezacy)
            bc2.metric("Waga poprzedniego sezonu", f"{w_prev:.0%}",
                       help="Maleje automatycznie w miarę gromadzenia danych bieżącego sezonu")
            bc3.metric("Łącznie rekordów w modelu", len(historical))

        with st.container(border=True):
            st.markdown("**⚙️ Parametry Dixon-Coles**")
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("ρ (rho) kalibrowane",    f"{rho:.4f}")
            dc2.metric("Śr. gole (dom)",          f"{srednie_lig['avg_home']:.2f}")
            dc3.metric("Śr. gole (wyjazd)",       f"{srednie_lig['avg_away']:.2f}")

        st.divider()
        st.write("**Średnie ważone drużyn** *(ostatnie 10 meczów, wagi rosnące)*")
        st.dataframe(srednie_df.sort_index(), use_container_width=True)

        st.caption(f"📅 Ostatnia aktualizacja danych: {historical['Date'].max().strftime('%d.%m.%Y')}")
        if st.button("🔄 Odśwież dane"):
            st.cache_data.clear()
            st.rerun()

    # =========================================================================
    # TAB 8 – DEBUG
    # =========================================================================
    with tab8:
        if debug_mode:
            st.subheader("🔧 Niezmapowane nazwy")
            if _niezmapowane:
                for n in sorted(_niezmapowane):
                    st.warning(f"`{n}`")
            else:
                st.success("Wszystkie zmapowane ✅")
        else:
            st.info("Włącz opcję '🔧 Debug – niezmapowane nazwy' w sidebarze, aby zobaczyć szczegóły.")

else:
    st.error("Nie udało się pobrać danych. Sprawdź połączenie z internetem.")