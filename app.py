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

st.set_page_config(page_title="System Analityczny Piłkarski Top 5", layout="wide")

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
    n = len(df)
    obs_00 = len(df[(df["FTHG"] == 0) & (df["FTAG"] == 0)]) / n
    obs_11 = len(df[(df["FTHG"] == 1) & (df["FTAG"] == 1)]) / n
    from scipy.stats import poisson as _p
    exp_00 = _p.pmf(0, avg_h) * _p.pmf(0, avg_a)
    exp_11 = _p.pmf(1, avg_h) * _p.pmf(1, avg_a)
    rho_00 = (obs_00 / exp_00 - 1) / (avg_h * avg_a) if exp_00 > 0 else -0.13
    rho_11 = -(obs_11 / exp_11 - 1) if exp_11 > 0 else -0.13
    rho = float(np.clip(np.mean([rho_00, rho_11]), -0.25, 0.0))
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
        
        # ROI = (trafione * (śr_fair - 1) - (wszystkie - trafione)) / wszystkie
        roi = (trafione * (sr_fair - 1) - (wszystkie - trafione)) / wszystkie if wszystkie > 0 else 0
        
        stats.append({
            "Rynek": nazwa,
            "Typów": wszystkie,
            "Trafione": trafione,
            "Skuteczność": f"{skutecznosc:.1%}",
            "Śr. P model": f"{sr_p_model:.1%}",
            "Śr. Fair": f"{sr_fair:.2f}",
            "ROI": f"{roi:+.1%}",
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
# ŁADOWANIE DANYCH I SIDEBAR
# ===========================================================================
st.sidebar.header("🌍 Wybór Rozgrywek")
wybrana_liga = st.sidebar.selectbox("Wybierz ligę", list(LIGI.keys()))
debug_mode   = st.sidebar.checkbox("🔧 Debug – niezmapowane nazwy", value=False)

historical = load_historical(LIGI[wybrana_liga]["csv_code"])
schedule   = load_schedule(LIGI[wybrana_liga]["file"])

st.title(f"System Analityczny – {wybrana_liga} 2025/26")
st.markdown("Dixon-Coles • Tracking skuteczności • Kalibracja modelu")

if not historical.empty:
    srednie_df  = oblicz_wszystkie_statystyki(historical.to_json())
    srednie_lig = oblicz_srednie_ligowe(historical.to_json())
    forma_dict  = oblicz_forme(historical)
    tabela      = tabela_ligowa(historical)
    rho         = srednie_lig["rho"]
    n_biezacy   = srednie_lig["n_biezacy"]
    w_prev      = waga_poprzedniego(n_biezacy)

    # Sidebar: szybki podgląd kolejki
    if not schedule.empty and not srednie_df.empty:
        dzisiaj    = datetime.now().date()
        przyszle   = schedule[schedule["date"].dt.date >= dzisiaj]
        if not przyszle.empty:
            nb_side    = przyszle["round"].min()
            mecze_side = schedule[schedule["round"] == nb_side]
            st.sidebar.divider()
            st.sidebar.markdown(f"**⚽ Kolejka {int(nb_side)} – typy modelu**")
            for _, m in mecze_side.iterrows():
                h_s = map_nazwa(m["home_team"])
                a_s = map_nazwa(m["away_team"])
                if h_s not in srednie_df.index or a_s not in srednie_df.index:
                    continue
                lh_s, la_s, _, _ = oblicz_lambdy(h_s, a_s, srednie_df, srednie_lig, forma_dict)
                pr_s = predykcja_meczu(lh_s, la_s, rho=rho)
                ikona = {"1":"🔵","X":"🟠","2":"🔴","1X":"🟣","X2":"🟣"}.get(pr_s["typ"], "⚪")
                st.sidebar.markdown(
                    f"{ikona} **{h_s[:10]}** vs **{a_s[:10]}**  \n"
                    f"&nbsp;&nbsp;&nbsp;`{pr_s['typ']}` @ {pr_s['fo_typ']:.2f} "
                    f"· {pr_s['conf_emoji']} {pr_s['conf_level']}",
                    unsafe_allow_html=True,
                )

    # TABS – NOWA STRUKTURA
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Bet Builder (Laboratorium)",
        "⚽ Analiza Meczu",
        "📊 Ranking Zdarzeń",
        "📈 Skuteczność + ROI",
        "📉 Kalibracja Modelu"
    ])

    # =========================================================================
    # TAB 1 – BET BUILDER (Laboratorium)
    # =========================================================================
    with tab1:
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
        min_prob = st.slider("Min. prawdopodobieństwo", 0.0, 1.0, 0.40, 0.05)

        if not schedule.empty and not srednie_df.empty:
            dzisiaj  = datetime.now().date()
            przyszle = schedule[schedule["date"].dt.date >= dzisiaj]
            if not przyszle.empty:
                nb    = przyszle["round"].min()
                mecze = schedule[schedule["round"] == nb]
                st.caption(f"Kolejka {int(nb)} – {len(mecze)} meczów")

                for _, mecz in mecze.iterrows():
                    h = map_nazwa(mecz["home_team"])
                    a = map_nazwa(mecz["away_team"])
                    if h not in srednie_df.index or a not in srednie_df.index:
                        continue
                    
                    lam_h, lam_a, lam_r, lam_k = oblicz_lambdy(h, a, srednie_df, srednie_lig, forma_dict)
                    
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

    # =========================================================================
    # TAB 2 – ANALIZA MECZU
    # =========================================================================
    with tab2:
        st.subheader("⚽ Analiza meczu – najbliższa kolejka")
        st.caption("Szczegółowa analiza każdego meczu z macierzą wyników i alternatywnymi rynkami.")

        tgl1, tgl2 = st.columns(2)
        with tgl1: pokaz_komentarz = st.toggle("💬 Komentarz", value=True)
        with tgl2: pokaz_macierz  = st.toggle("🔢 Macierz wyników", value=False)

        if not schedule.empty and not srednie_df.empty:
            dzisiaj  = datetime.now().date()
            przyszle = schedule[schedule["date"].dt.date >= dzisiaj]
            if not przyszle.empty:
                nb    = przyszle["round"].min()
                mecze = schedule[schedule["round"] == nb]
                st.caption(f"Kolejka {int(nb)} – {len(mecze)} meczów")

                kol_a, kol_b = st.columns(2)
                mecze_list = list(mecze.iterrows())

                for idx, (_, mecz) in enumerate(mecze_list):
                    h = map_nazwa(mecz["home_team"])
                    a = map_nazwa(mecz["away_team"])
                    if h not in srednie_df.index or a not in srednie_df.index:
                        continue

                    lam_h, lam_a, lam_r, lam_k = oblicz_lambdy(h, a, srednie_df, srednie_lig, forma_dict)
                    pred = predykcja_meczu(lam_h, lam_a, rho=rho)
                    data_meczu = mecz["date"].strftime("%d.%m %H:%M") if pd.notna(mecz["date"]) else ""

                    kolumna = kol_a if idx % 2 == 0 else kol_b
                    ikony_t = {"1":"🔵","X":"🟠","2":"🔴","1X":"🟣","X2":"🟣"}
                    conf_i  = "🟢" if pred["conf_level"]=="High" else ("🟡" if pred["conf_level"]=="Medium" else "🔴")
                    label_t2 = (f"{conf_i} {h} vs {a}"
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

                            # Alternatywne rynki
                            with st.expander("📊 Alternatywne rynki (p ≥ 55%)", expanded=False):
                                alt = alternatywne_zdarzenia(lam_h, lam_a, lam_r, lam_k, rho)
                                if alt:
                                    cat_colors = {"Gole":"#2196F3","BTTS":"#9C27B0","Rożne":"#FF9800","Kartki":"#F44336","1X2":"#4CAF50"}
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
                            lhs, las, lrs, lks = oblicz_lambdy(h_s, a_s, srednie_df, srednie_lig, forma_dict)
                            pred_s = predykcja_meczu(lhs, las, rho=rho)
                            mecz_str_s = f"{h_s} – {a_s}"
                            # Główny typ 1X2
                            zapisz_zdarzenia(wybrana_liga, int(nb), mecz_str_s, h_s, a_s,
                                             "1X2", pred_s["typ"], 0.0, pred_s["p_typ"], pred_s["fo_typ"])
                            n_saved += 1
                            # Alternatywne
                            for _, nazwa_z, pz, foz, katz, liniz in alternatywne_zdarzenia(lhs, las, lrs, lks, rho):
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
    # TAB 3 – RANKING ZDARZEŃ
    # =========================================================================
    with tab3:
        st.subheader("📊 Ranking zdarzeń kolejki")
        st.caption("Wszystkie zdarzenia z p ≥ 60%, sortowane według pewności modelu.")

        if not schedule.empty and not srednie_df.empty:
            dzisiaj = datetime.now().date()
            przyszle = schedule[schedule["date"].dt.date >= dzisiaj]
            if not przyszle.empty:
                nb = przyszle["round"].min()
                mecze = schedule[schedule["round"] == nb]

                with st.spinner("Generowanie rankingu..."):
                    wszystkie_zd = []
                    for _, mecz in mecze.iterrows():
                        h = map_nazwa(mecz["home_team"])
                        a = map_nazwa(mecz["away_team"])
                        if h not in srednie_df.index or a not in srednie_df.index:
                            continue
                        
                        lam_h, lam_a, lam_r, lam_k = oblicz_lambdy(h, a, srednie_df, srednie_lig, forma_dict)
                        pred = predykcja_meczu(lam_h, lam_a, rho=rho)
                        mecz_str = f"{h} – {a}"

                        # Typ główny meczu (prog min 0.58 – slider decyduje o wyświetleniu)
                        if pred["p_typ"] >= 0.58:
                            wszystkie_zd.append({
                                "Mecz": mecz_str,
                                "Rynek": "1X2",
                                "Typ": pred["typ"],
                                "P": f"{pred['p_typ']:.0%}",
                                "P_val": pred["p_typ"],
                                "Fair": pred["fo_typ"],
                                "Kolor": "🟢" if pred["p_typ"] >= 0.65 else "🟡"
                            })

                        # Alternatywne zdarzenia (prog min 0.55 – slider decyduje o wyświetleniu)
                        alt = alternatywne_zdarzenia(lam_h, lam_a, lam_r, lam_k, rho, prog_min=0.55)
                        for emoji, nazwa, p, fo, kat, linia in alt:
                            wszystkie_zd.append({
                                "Mecz": mecz_str,
                                "Rynek": kat,
                                "Typ": nazwa,
                                "P": f"{p:.0%}",
                                "P_val": p,
                                "Fair": fo,
                                "Kolor": "🟢" if p >= 0.65 else "🟡"
                            })

                if wszystkie_zd:
                    df_rank = pd.DataFrame(wszystkie_zd)

                    # Filtry – trwałe (st.radio zachowuje stan)
                    rf1, rf2 = st.columns([2, 6])
                    with rf1:
                        filtr_rynek = st.radio(
                            "Filtruj rynek", ["Wszystkie", "1X2", "Gole", "BTTS", "Rożne", "Kartki"],
                            horizontal=False, label_visibility="collapsed"
                        )
                    with rf2:
                        prog_filtr = st.slider("Min. p modelu", 0.55, 0.90, 0.60, 0.05,
                                               format="%.0f%%",
                                               help="Pokaż tylko zdarzenia z p ≥ wartość")

                    df_show_r = df_rank.copy()
                    if filtr_rynek != "Wszystkie":
                        df_show_r = df_show_r[df_show_r["Rynek"] == filtr_rynek]
                    df_show_r = df_show_r[df_show_r["P_val"] >= prog_filtr].sort_values("P_val", ascending=False)

                    if df_show_r.empty:
                        st.info(f"Brak zdarzeń {filtr_rynek} z p ≥ {prog_filtr:.0%}.")
                    else:
                        cat_cols3 = {"1X2":"#4CAF50","Gole":"#2196F3","BTTS":"#9C27B0","Rożne":"#FF9800","Kartki":"#F44336"}
                        html_r3 = []
                        for _, row in df_show_r.iterrows():
                            kc3 = cat_cols3.get(row["Rynek"], "#888")
                            bw3 = int(row["P_val"] * 100)
                            fo3 = row["Fair"]
                            fc3 = "#4CAF50" if fo3 <= 1.60 else ("#FF9800" if fo3 <= 2.00 else "#aaa")
                            sila_opis = "🔥 Silny" if row["P_val"] >= 0.70 else ("🎯 Dobry" if row["P_val"] >= 0.62 else "💡 Uwaga")
                            html_r3.append(
                                f"<tr>"
                                f"<td style='padding:6px 10px;font-weight:bold;font-size:0.88em'>{row['Mecz']}</td>"
                                f"<td style='padding:6px 10px;text-align:center'>"
                                f"<span style='background:{kc3}22;color:{kc3};padding:2px 7px;"
                                f"border-radius:8px;font-size:0.82em;font-weight:bold'>{row['Rynek']}</span></td>"
                                f"<td style='padding:6px 10px;font-size:0.88em'>{row['Typ']}</td>"
                                f"<td style='padding:6px 10px;width:120px'>"
                                f"<div style='display:flex;align-items:center;gap:6px'>"
                                f"<div style='flex:1;background:#333;border-radius:3px;height:6px'>"
                                f"<div style='background:{kc3};width:{bw3}%;height:6px;border-radius:3px'></div></div>"
                                f"<span style='color:{kc3};font-weight:bold;font-size:0.85em;min-width:32px'>{row['P']}</span>"
                                f"</div></td>"
                                f"<td style='padding:6px 10px;text-align:right;color:{fc3};font-weight:bold'>{fo3:.2f}</td>"
                                f"<td style='padding:6px 10px;text-align:center;font-size:0.82em'>{sila_opis}</td>"
                                f"</tr>"
                            )
                        st.markdown(
                            f"<div style='overflow-x:auto;border-radius:8px;border:1px solid #333;margin-top:8px'>"
                            f"<table style='width:100%;border-collapse:collapse;font-size:0.88em'>"
                            f"<thead><tr style='background:#1e1e2e;color:#aaa;font-size:0.78em;text-transform:uppercase'>"
                            f"<th style='padding:8px 10px;text-align:left'>Mecz</th>"
                            f"<th style='padding:8px 10px;text-align:center'>Rynek</th>"
                            f"<th style='padding:8px 10px;text-align:left'>Zdarzenie</th>"
                            f"<th style='padding:8px 10px;text-align:left'>P modelu</th>"
                            f"<th style='padding:8px 10px;text-align:right'>Fair Odds</th>"
                            f"<th style='padding:8px 10px;text-align:center'>Siła</th>"
                            f"</tr></thead><tbody>{''.join(html_r3)}</tbody></table></div>"
                            f"<p style='color:#444;font-size:0.75em;margin-top:6px'>"
                            f"Pokazano {len(df_show_r)} zdarzeń · Fair Odds = bez marży bukmachera</p>",
                            unsafe_allow_html=True,
                        )
                        # Export rankingu
                        st.download_button("⬇️ Pobierz ranking (CSV)",
                                           data=df_show_r.drop(columns=["P_val","Kolor"]).to_csv(index=False, decimal=","),
                                           file_name=f"ranking_kolejka{int(nb)}.csv", mime="text/csv")
                else:
                    st.info("Brak zdarzeń spełniających kryterium p ≥ 60%")
            else:
                st.info("Brak nadchodzących meczów")
        else:
            st.warning("Brak danych")

    # =========================================================================
    # TAB 4 – SKUTECZNOŚĆ + ROI
    # =========================================================================
    with tab4:
        st.subheader("📈 Skuteczność modelu per rynek")
        st.caption(
            "Analiza skuteczności i ROI po rozegraniu meczów. "
            "Aby zobaczyć dane: zapisz zdarzenia w Analizie Meczu, poczekaj na wyniki, "
            "potem kliknij 🔄 Aktualizuj wyniki w Tab 2."
        )

        stats_df = statystyki_skutecznosci(wybrana_liga)

        if not stats_df.empty:
            # Wyodrębnij ROI numerycznie (dla sortowania i wykresu)
            stats_df["ROI_value"] = stats_df["ROI"].str.replace("+","").str.replace("%","").astype(float)

            # ── Metryki ogólne ────────────────────────────────────────────
            total_typow   = int(stats_df["Typów"].sum())
            total_trafion = int(stats_df["Trafione"].sum())
            avg_skut      = total_trafion / total_typow if total_typow > 0 else 0
            w_roi         = (stats_df["ROI_value"] * stats_df["Typów"]).sum() / total_typow if total_typow > 0 else 0

            with st.container(border=True):
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("📋 Typów łącznie",   total_typow)
                m2.metric("✅ Trafione",          total_trafion)
                m3.metric("🎯 Skuteczność",      f"{avg_skut:.1%}")
                m4.metric("💹 Ważony ROI",       f"{w_roi:+.1f}%",
                          delta_color="normal" if w_roi > 0 else "inverse")

            # ── HTML tabela per rynek ─────────────────────────────────────
            st.markdown("**Szczegóły per rynek** *(sortowane po ROI malejąco)*")
            stats_sorted = stats_df.sort_values("ROI_value", ascending=False)
            cat_col4 = {"1X2":"#4CAF50","Gole":"#2196F3","BTTS":"#9C27B0","Rożne":"#FF9800","Kartki":"#F44336"}

            rows_s4 = []
            for _, row in stats_sorted.iterrows():
                roi_v = row["ROI_value"]
                roi_col = "#4CAF50" if roi_v > 3 else ("#F44336" if roi_v < -3 else "#FF9800")
                roi_bg  = "#1a2e1a" if roi_v > 3 else ("#2e1a1a" if roi_v < -3 else "transparent")
                # Pasek skuteczności
                skut_v  = float(row["Skuteczność"].replace("%","")) / 100
                bw_s    = int(skut_v * 100)
                # Kategoria (z nazwy rynku)
                rynek_n = row["Rynek"]
                kat4    = next((k for k in cat_col4 if k in rynek_n), "Gole")
                kc4     = cat_col4.get(kat4, "#888")
                rows_s4.append(
                    f"<tr style='background:{roi_bg}'>"
                    f"<td style='padding:6px 10px;font-weight:bold;font-size:0.88em'>{rynek_n}</td>"
                    f"<td style='padding:6px 10px;text-align:center;color:#888'>{row['Typów']}</td>"
                    f"<td style='padding:6px 10px;text-align:center;color:#888'>{row['Trafione']}</td>"
                    f"<td style='padding:6px 10px;width:120px'>"
                    f"<div style='display:flex;align-items:center;gap:5px'>"
                    f"<div style='flex:1;background:#333;border-radius:3px;height:5px'>"
                    f"<div style='background:{kc4};width:{bw_s}%;height:5px;border-radius:3px'></div></div>"
                    f"<span style='color:{kc4};font-size:0.82em;min-width:34px'>{row['Skuteczność']}</span></div></td>"
                    f"<td style='padding:6px 10px;text-align:center;color:#888'>{row['Śr. P model']}</td>"
                    f"<td style='padding:6px 10px;text-align:center;color:#aaa'>{row['Śr. Fair']}</td>"
                    f"<td style='padding:6px 10px;text-align:right;font-weight:bold;color:{roi_col}'>{row['ROI']}</td>"
                    f"<td style='padding:6px 10px;text-align:center'>{row['Kolor']}</td>"
                    f"</tr>"
                )
            st.markdown(
                f"<div style='overflow-x:auto;border-radius:8px;border:1px solid #333'>"
                f"<table style='width:100%;border-collapse:collapse;font-size:0.88em'>"
                f"<thead><tr style='background:#1e1e2e;color:#aaa;font-size:0.78em;text-transform:uppercase'>"
                f"<th style='padding:8px 10px;text-align:left'>Rynek</th>"
                f"<th style='padding:8px 10px;text-align:center'>Typów</th>"
                f"<th style='padding:8px 10px;text-align:center'>Trafione</th>"
                f"<th style='padding:8px 10px;text-align:left'>Skuteczność</th>"
                f"<th style='padding:8px 10px;text-align:center'>Śr. P</th>"
                f"<th style='padding:8px 10px;text-align:center'>Śr. Fair</th>"
                f"<th style='padding:8px 10px;text-align:right'>ROI</th>"
                f"<th style='padding:8px 10px;text-align:center'>Status</th>"
                f"</tr></thead><tbody>{''.join(rows_s4)}</tbody></table></div>",
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
            st.download_button("⬇️ Pobierz statystyki (CSV)",
                               data=stats_df.drop(columns=["ROI_value","Kolor"]).to_csv(index=False, decimal=","),
                               file_name="skutecznosc_modelu.csv", mime="text/csv")
        else:
            st.info("Brak danych do analizy. Zapisz predykcje w zakładce 'Analiza Meczu' i poczekaj na wyniki.")

    # =========================================================================
    # TAB 5 – KALIBRACJA MODELU
    # =========================================================================
    with tab5:
        st.subheader("📉 Kalibracja modelu")
        st.caption("Sprawdza, czy model jest dobrze skalibrowany – czy prawdopodobieństwa pokrywają się z rzeczywistością.")

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
                px = pad_rc5 + (xv - 0.45) / 0.55 * (w_rc5 - 2*pad_rc5)
                py = h_rc5 - pad_rc5 - (yv - 0.45) / 0.55 * (h_rc5 - 2*pad_rc5)
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
                    f"<text x='{px5+r5+4:.0f}' y='{py5+4:.0f}' font-size='9' fill='#ccc' "
                    f"font-family='sans-serif'>{rk['przedzial']} (n={n5})</text>"
                )
            svg5 = (
                f'<svg width="{w_rc5}" height="{h_rc5}" '
                f'style="background:#0e1117;border-radius:8px;display:block;margin:auto">'
                f'<polyline points="{diag5_line}" fill="none" stroke="#444" '
                f'stroke-width="1.5" stroke-dasharray="6,4"/>'
                f'<text x="{w_rc5-pad_rc5+4}" y="{pad_rc5-4}" font-size="9" fill="#555" '
                f'font-family="sans-serif">idealny model</text>'
                f'{"".join(circles5)}'
                f'<text x="{w_rc5//2}" y="{h_rc5-6}" text-anchor="middle" '
                f'font-size="10" fill="#888" font-family="sans-serif">P modelu →</text>'
                f'<text x="12" y="{h_rc5//2}" text-anchor="middle" font-size="10" fill="#888" '
                f'font-family="sans-serif" transform="rotate(-90,12,{h_rc5//2})">Hit Rate →</text>'
                f'<!-- legenda -->'
                f'<circle cx="{w_rc5-130}" cy="20" r="6" fill="#4CAF50" stroke="white" stroke-width="1"/>'
                f'<text x="{w_rc5-120}" y="25" font-size="9" fill="#aaa" font-family="sans-serif">Dobrze skalibrowany (&lt;5%)</text>'
                f'<circle cx="{w_rc5-130}" cy="38" r="6" fill="#FF9800" stroke="white" stroke-width="1"/>'
                f'<text x="{w_rc5-120}" y="43" font-size="9" fill="#aaa" font-family="sans-serif">Umiarkowany (5–12%)</text>'
                f'<circle cx="{w_rc5-130}" cy="56" r="6" fill="#F44336" stroke="white" stroke-width="1"/>'
                f'<text x="{w_rc5-120}" y="61" font-size="9" fill="#aaa" font-family="sans-serif">Słabo skalibrowany (&gt;12%)</text>'
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
            st.info("Brak danych do kalibracji. Potrzebne są zapisane predykcje z wynikami.")

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