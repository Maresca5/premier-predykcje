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
                    
                    with st.expander(f"{h} vs {a}"):
                        p_g = macierz_goli_p(lam_h, lam_a, rho, int(linia_gole), typ_gole)
                        p_r = oblicz_p(typ_rogi, linia_rogi, lam_r)
                        p_k = oblicz_p(typ_kartki, linia_kartki, lam_k)
                        p_combo = p_g * p_r * p_k
                        
                        col_left, col_right = st.columns(2)
                        
                        with col_left:
                            st.metric(f"{typ_gole} {linia_gole}", f"{p_g:.1%}", f"fair {fair_odds(p_g)}")
                            st.metric(f"{typ_rogi} {linia_rogi} rożnych", f"{p_r:.1%}", f"fair {fair_odds(p_r)}")
                        
                        with col_right:
                            st.metric(f"{typ_kartki} {linia_kartki} kartek", f"{p_k:.1%}", f"fair {fair_odds(p_k)}")
                            if p_combo >= min_prob:
                                st.metric("COMBO", f"{p_combo:.1%}", f"fair AKO {fair_odds(p_combo)}")
                            else:
                                st.caption(f"p_combo = {p_combo:.1%} (poniżej progu)")

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
                    with kolumna:
                        with st.expander(f"{h} vs {a} – {data_meczu}", expanded=False):
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

                            # Typ + confidence
                            st.markdown(
                                f"<div style='text-align:center;margin-bottom:8px'>"
                                f"Typ: {badge_typ(pred['typ'])} @ {pred['fo_typ']} · {pred['conf_emoji']} {pred['conf_level']}</div>",
                                unsafe_allow_html=True,
                            )

                            # Metryki 1X2
                            mc1, mc2, mc3 = st.columns(3)
                            mc1.metric("1", f"{pred['p_home']:.0%}", f"fair {pred['fo_home']}")
                            mc2.metric("X", f"{pred['p_draw']:.0%}", f"fair {pred['fo_draw']}")
                            mc3.metric("2", f"{pred['p_away']:.0%}", f"fair {pred['fo_away']}")

                            # Alternatywne rynki
                            with st.expander("📊 Alternatywne rynki", expanded=False):
                                alt = alternatywne_zdarzenia(lam_h, lam_a, lam_r, lam_k, rho)
                                for emoji, nazwa, p, fo, kat, linia in alt[:6]:
                                    st.markdown(f"{emoji} **{nazwa}** – {p:.0%} (fair {fo})")
                                    
                                    # Automatyczny zapis do bazy (tracking)
                                    zapisz_zdarzenia(
                                        liga=wybrana_liga,
                                        kolejnosc=int(nb),
                                        mecz=f"{h} – {a}",
                                        home=h,
                                        away=a,
                                        rynek=kat,
                                        typ=nazwa,
                                        linia=linia,
                                        p_model=p,
                                        fair_odds=fo
                                    )

                            if pokaz_komentarz:
                                st.info(generuj_komentarz(h, a, pred, forma_dict))

                            if pokaz_macierz:
                                st.markdown("**Macierz wyników**")
                                st.markdown(render_macierz_html(pred["macierz"], h, a), unsafe_allow_html=True)

    # =========================================================================
    # TAB 3 – RANKING ZDARZEŃ (POPRAWIONY)
    # =========================================================================
    with tab3:
        st.subheader("📊 Ranking zdarzeń kolejki")
        st.caption("Zdarzenia z p ≥ 60% i fair odds ≥ 1.30, sortowane według pewności modelu.")

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

                        # Typ główny meczu
                        if pred["p_typ"] >= 0.60 and pred["fo_typ"] >= 1.30:
                            wszystkie_zd.append({
                                "Mecz": mecz_str,
                                "Rynek": "1X2",
                                "Typ": pred["typ"],
                                "P": pred["p_typ"],
                                "Fair": pred["fo_typ"],
                                "Kategoria": "1X2"
                            })

                        # Alternatywne zdarzenia
                        alt = alternatywne_zdarzenia(lam_h, lam_a, lam_r, lam_k, rho, prog_min=0.60)
                        for emoji, nazwa, p, fo, kat, linia in alt:
                            if fo >= 1.30:
                                wszystkie_zd.append({
                                    "Mecz": mecz_str,
                                    "Rynek": kat,
                                    "Typ": nazwa,
                                    "P": p,
                                    "Fair": fo,
                                    "Kategoria": kat
                                })

                if wszystkie_zd:
                    # Sortuj po prawdopodobieństwie malejąco
                    wszystkie_zd.sort(key=lambda x: -x["P"])
                    
                    # Przygotuj DataFrame do wyświetlenia
                    df_rank = pd.DataFrame(wszystkie_zd)
                    
                    # Dodaj kolumny pomocnicze
                    df_rank["P%"] = (df_rank["P"] * 100).round(0).astype(int).astype(str) + "%"
                    df_rank["Kolor"] = df_rank["P"].apply(lambda x: 
                        "🟢" if x >= 0.70 else ("🟡" if x >= 0.65 else "🔵" if x >= 0.60 else "⚪"))
                    df_rank["Fair_display"] = df_rank["Fair"].apply(lambda x: f"{x:.2f}")
                    
                    # Filtry w sidebarze
                    with st.sidebar.expander("🔍 Filtry rankingu", expanded=False):
                        min_p = st.slider("Min. prawdopodobieństwo", 0.60, 0.90, 0.60, 0.05)
                        min_fair = st.slider("Min. fair odds", 1.30, 3.00, 1.30, 0.10)
                        rynki = ["Wszystkie"] + sorted(df_rank["Rynek"].unique().tolist())
                        wybrany_rynek = st.selectbox("Rynek", rynki)
                    
                    # Zastosuj filtry
                    df_filtered = df_rank[df_rank["P"] >= min_p]
                    df_filtered = df_filtered[df_filtered["Fair"] >= min_fair]
                    if wybrany_rynek != "Wszystkie":
                        df_filtered = df_filtered[df_filtered["Rynek"] == wybrany_rynek]
                    
                    # Wyświetl statystyki
                    col_s1, col_s2, col_s3 = st.columns(3)
                    with col_s1:
                        st.metric("Liczba zdarzeń", len(df_filtered))
                    with col_s2:
                        st.metric("Śr. prawdopodobieństwo", f"{df_filtered['P'].mean():.1%}" if not df_filtered.empty else "0%")
                    with col_s3:
                        st.metric("Śr. fair odds", f"{df_filtered['Fair'].mean():.2f}" if not df_filtered.empty else "0.00")
                    
                    if not df_filtered.empty:
                        # Wyświetl tabelę z kolorowaniem
                        st.dataframe(
                            df_filtered[["Kolor", "Mecz", "Rynek", "Typ", "P%", "Fair_display"]],
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Kolor": st.column_config.TextColumn("", width="small"),
                                "Mecz": st.column_config.TextColumn("Mecz", width="medium"),
                                "Rynek": st.column_config.TextColumn("Rynek", width="small"),
                                "Typ": st.column_config.TextColumn("Typ", width="medium"),
                                "P%": st.column_config.TextColumn("P", width="small"),
                                "Fair_display": st.column_config.TextColumn("Fair", width="small"),
                            }
                        )
                        
                        # Wykres słupkowy dla top 10
                        st.divider()
                        st.markdown("**📊 Top 10 najpewniejszych zdarzeń**")
                        
                        top10 = df_filtered.head(10).copy()
                        top10["label"] = top10["Mecz"].str[:20] + " - " + top10["Typ"].str[:15]
                        
                        chart_data = pd.DataFrame({
                            "Zdarzenie": top10["label"],
                            "Prawdopodobieństwo": top10["P"] * 100
                        }).set_index("Zdarzenie")
                        
                        st.bar_chart(chart_data, height=300)
                        
                        # Eksport do CSV
                        csv_data = df_filtered[["Mecz", "Rynek", "Typ", "P", "Fair"]].copy()
                        csv_data["P"] = csv_data["P"].apply(lambda x: f"{x:.1%}")
                        csv_data["Fair"] = csv_data["Fair"].apply(lambda x: f"{x:.2f}")
                        
                        st.download_button(
                            "⬇️ Pobierz ranking (CSV)",
                            data=csv_data.to_csv(index=False, decimal=","),
                            file_name=f"ranking_{wybrana_liga.replace(' ','_')}_kolejka{int(nb)}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("Brak zdarzeń po zastosowaniu filtrów")
                else:
                    st.info("Brak zdarzeń spełniających kryteria (p ≥ 60%, fair odds ≥ 1.30)")
            else:
                st.info("Brak nadchodzących meczów")
        else:
            st.warning("Brak danych")

    # =========================================================================
    # TAB 4 – SKUTECZNOŚĆ + ROI
    # =========================================================================
    with tab4:
        st.subheader("📈 Skuteczność modelu per rynek")
        st.caption("Analiza skuteczności i ROI dla każdego typu rynku. ROI liczone na fair odds (bez marży).")

        stats_df = statystyki_skutecznosci(wybrana_liga if wybrana_liga != "Wszystkie" else None)
        
        if not stats_df.empty:
            # Kolorowanie wierszy
            def koloruj_wiersz(row):
                if row['ROI'].strip('%+') > '3':
                    return ['background-color: #1a3a1a'] * len(row)
                elif row['ROI'].strip('%+') < '-3':
                    return ['background-color: #3a1a1a'] * len(row)
                return [''] * len(row)
            
            styled_df = stats_df.style.apply(koloruj_wiersz, axis=1)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Wykres ROI
            st.divider()
            st.markdown("**📊 ROI per rynek**")
            
            chart_data = stats_df.copy()
            chart_data['ROI_value'] = chart_data['ROI'].str.replace('%', '').str.replace('+', '').astype(float)
            chart_data = chart_data.sort_values('ROI_value', ascending=True)
            
            st.bar_chart(chart_data.set_index('Rynek')['ROI_value'])
            
            # Podsumowanie
            st.divider()
            st.markdown("**📋 Podsumowanie**")
            
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                total_typów = stats_df['Typów'].sum()
                st.metric("Łącznie typów", f"{total_typów}")
            with col_s2:
                total_trafione = stats_df['Trafione'].sum()
                st.metric("Łącznie trafione", f"{total_trafione}")
            with col_s3:
                avg_skutecznosc = total_trafione / total_typów if total_typów > 0 else 0
                st.metric("Śr. skuteczność", f"{avg_skutecznosc:.1%}")
            with col_s4:
                # Ważone ROI
                weighted_roi = (chart_data['ROI_value'] * stats_df['Typów']).sum() / stats_df['Typów'].sum()
                st.metric("Ważony ROI", f"{weighted_roi:+.1%}")
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
            # Tabela kalibracji
            kal_display = kal_df.copy()
            kal_display['p_srednia'] = kal_display['p_srednia'].apply(lambda x: f"{x:.1%}")
            kal_display['skutecznosc'] = kal_display['skutecznosc'].apply(lambda x: f"{x:.1%}")
            kal_display['rozbieznosc'] = kal_display['rozbieznosc'].apply(lambda x: f"{x:+.1%}")
            
            st.dataframe(
                kal_display[['przedzial', 'liczba', 'p_srednia', 'skutecznosc', 'rozbieznosc']],
                use_container_width=True,
                hide_index=True
            )
            
            # Wykres kalibracji
            st.divider()
            st.markdown("**📊 Wykres kalibracji**")
            
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            fig = make_subplots(rows=1, cols=2, 
                                subplot_titles=("Skuteczność vs P modelu", "Rozbieżność"),
                                specs=[[{"type": "bar"}, {"type": "bar"}]])
            
            fig.add_trace(
                go.Bar(name='P modelu', x=kal_df['przedzial'], y=kal_df['p_srednia'],
                       marker_color='#2196F3', text=kal_df['p_srednia'].apply(lambda x: f"{x:.1%}"),
                       textposition='outside'),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(name='Rzeczywista', x=kal_df['przedzial'], y=kal_df['skutecznosc'],
                       marker_color='#4CAF50', text=kal_df['skutecznosc'].apply(lambda x: f"{x:.1%}"),
                       textposition='outside'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(name='Rozbieżność', x=kal_df['przedzial'], y=kal_df['rozbieznosc'],
                       marker_color=['#4CAF50' if x > 0 else '#F44336' for x in kal_df['rozbieznosc']],
                       text=kal_df['rozbieznosc'].apply(lambda x: f"{x:+.1%}"),
                       textposition='outside'),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=True, margin=dict(l=50, r=50, t=50, b=50))
            fig.update_xaxes(title_text="Przedział prawdopodobieństwa", row=1, col=1)
            fig.update_xaxes(title_text="Przedział prawdopodobieństwa", row=1, col=2)
            fig.update_yaxes(title_text="Prawdopodobieństwo", row=1, col=1, tickformat='.0%')
            fig.update_yaxes(title_text="Rozbieżność", row=1, col=2, tickformat='.0%')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretacja
            st.divider()
            st.markdown("**🔍 Interpretacja**")
            
            avg_bias = kal_df['rozbieznosc'].mean()
            if abs(avg_bias) < 0.02:
                st.success(f"✅ Model jest dobrze skalibrowany (średni bias {avg_bias:+.1%})")
            elif avg_bias > 0:
                st.warning(f"⚠️ Model jest zbyt ostrożny – niedoszacowuje szans (bias {avg_bias:+.1%})")
            else:
                st.warning(f"⚠️ Model jest zbyt pewny siebie – przeszacowuje szanse (bias {avg_bias:+.1%})")
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