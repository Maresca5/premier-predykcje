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
TARGET_AKO = 5.0

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
    "AC Milan":               "Milan",
    "Internazionale":         "Inter",
    "AS Roma":                "Roma",
    "Hellas Verona":          "Verona",
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
# BAZA DANYCH
# ===========================================================================
def init_db():
    con = sqlite3.connect(DB_FILE)
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
    """Zwraca (lam_h, lam_a, lam_r, lam_k)"""
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
    """
    Zwraca listę zdarzeń z p >= prog_min.
    Format: (emoji, nazwa, p, fair_odds, kategoria)
    """
    zdarzenia = []
    mg = int(np.clip(np.ceil(max(lam_h, lam_a) + 4), 6, 10))
    M = dixon_coles_adj(
        np.outer(poisson.pmf(range(mg), lam_h), poisson.pmf(range(mg), lam_a)),
        lam_h, lam_a, rho=rho
    )

    # ── Gole (z macierzy D-C – spójne z modelem) ───────────────────────────
    for linia in [1.5, 2.5, 3.5]:
        p_over = float(sum(M[i,j] for i in range(mg) for j in range(mg) if i+j > linia))
        p_under = 1 - p_over
        if p_over >= prog_min:
            zdarzenia.append(("⚽", f"Over {linia} goli",  p_over,  fair_odds(p_over),  "Gole"))
        if p_under >= prog_min:
            zdarzenia.append(("⚽", f"Under {linia} goli", p_under, fair_odds(p_under), "Gole"))

    # ── BTTS ────────────────────────────────────────────────────────────────
    p_btts   = float(1 - M[0,:].sum() - M[:,0].sum() + M[0,0])
    p_nobtts = 1 - p_btts
    if p_btts   >= prog_min: zdarzenia.append(("⚽", "BTTS – Tak",  p_btts,   fair_odds(p_btts),   "BTTS"))
    if p_nobtts >= prog_min: zdarzenia.append(("⚽", "BTTS – Nie",  p_nobtts, fair_odds(p_nobtts), "BTTS"))

    # ── Rożne (Poisson – orientacyjne) ─────────────────────────────────────
    for linia in [7.5, 8.5, 9.5, 10.5]:
        p_over = float(1 - poisson.cdf(int(linia), lam_r))
        if p_over >= prog_min:
            zdarzenia.append(("🚩", f"Over {linia} rożnych", p_over, fair_odds(p_over), "Rożne"))

    # ── Kartki (Poisson – orientacyjne) ────────────────────────────────────
    for linia in [2.5, 3.5, 4.5]:
        p_over = float(1 - poisson.cdf(int(linia), lam_k))
        if p_over >= prog_min:
            zdarzenia.append(("🟨", f"Over {linia} kartek", p_over, fair_odds(p_over), "Kartki"))

    return sorted(zdarzenia, key=lambda x: -x[2])

# ===========================================================================
# GENERATOR KUPONU AKO ~5.0  (itertools.combinations – pełne przeszukiwanie)
# ===========================================================================
def generuj_kupon(mecze_data: list, target_ako: float = 5.0,
                  n_min: int = 3, n_max: int = 5) -> dict | None:
    """
    mecze_data: lista dict {mecz, typ, p, fair, kategoria}
    Przeszukuje WSZYSTKIE kombinacje długości n_min..n_max,
    wybiera tę z AKO najbliższym target_ako.
    Warunki na kandydatów:
      - fair odds 1.25–2.50 (wyklucza zbyt pewne i zbyt ryzykowne)
      - max 2 zdarzenia z tego samego meczu
      - p >= 0.55
    """
    # Filtruj kandydatów
    kandydaci = [z for z in mecze_data if 1.25 <= z["fair"] <= 2.50 and z["p"] >= 0.55]
    if len(kandydaci) < n_min:
        return None

    najlepszy     = None
    min_roznica   = float("inf")

    for n in range(n_min, min(n_max + 1, len(kandydaci) + 1)):
        for combo in itertools.combinations(kandydaci, n):
            # Sprawdź max 2 zdarzenia z tego samego meczu
            mecze_count: dict = {}
            for z in combo:
                mecze_count[z["mecz"]] = mecze_count.get(z["mecz"], 0) + 1
            if any(v > 2 for v in mecze_count.values()):
                continue

            ako     = float(np.prod([z["fair"] for z in combo]))
            p_combo = float(np.prod([z["p"]    for z in combo]))
            roznica = abs(ako - target_ako)
            if roznica < min_roznica:
                min_roznica = roznica
                najlepszy   = {
                    "zdarzenia": list(combo),
                    "ako":       round(ako, 2),
                    "p_combo":   round(p_combo, 3),
                }

    return najlepszy

def zapisz_kupon_db(kupon: dict, liga: str, kolejnosc: int, typ_kuponu: str = "AKO ~5.0"):
    init_db()
    con = sqlite3.connect(DB_FILE)
    try:
        con.execute(
            "INSERT INTO kupony (liga,kolejnosc,typ_kuponu,zdarzenia_json,ako,p_combo,data) "
            "VALUES (?,?,?,?,?,?,?)",
            (liga, kolejnosc, typ_kuponu,
             json.dumps(kupon["zdarzenia"], ensure_ascii=False),
             kupon["ako"], kupon["p_combo"],
             datetime.now().strftime("%Y-%m-%d"))
        )
        con.commit()
    except Exception as e:
        st.warning(f"Zapis kuponu: {e}")
    finally:
        con.close()

def wczytaj_kupony(liga: str) -> list:
    init_db()
    con = sqlite3.connect(DB_FILE)
    rows = con.execute(
        "SELECT id,kolejnosc,typ_kuponu,zdarzenia_json,ako,p_combo,data "
        "FROM kupony WHERE liga=? ORDER BY kolejnosc DESC, id DESC",
        (liga,)
    ).fetchall()
    con.close()
    return [{"id": r[0], "kolejnosc": r[1], "typ_kuponu": r[2],
             "zdarzenia": json.loads(r[3]), "ako": r[4],
             "p_combo": r[5], "data": r[6]} for r in rows]

def weryfikuj_kupon(kupon: dict, hist: pd.DataFrame) -> str:
    """
    Sprawdza każdą nogę kuponu. Zwraca: '✅ Trafiony' / '❌ Chybiony' / '⏳ Oczekuje'.
    Obsługuje typy: '1'/'X'/'2'/'1X'/'X2', 'Over N goli', 'Under N goli',
    'BTTS – Tak', 'BTTS – Nie', 'Over N rożnych', 'Over N kartek'.
    """
    if "_sezon" in hist.columns:
        hist = hist[hist["_sezon"] == "biezacy"]

    for z in kupon["zdarzenia"]:
        parts = z["mecz"].split(" – ")
        if len(parts) != 2:
            return "⏳ Oczekuje"
        h_z, a_z = parts[0].strip(), parts[1].strip()
        match = hist[(hist["HomeTeam"] == h_z) & (hist["AwayTeam"] == a_z)]
        if match.empty:
            return "⏳ Oczekuje"

        row  = match.iloc[-1]
        hg   = int(row["FTHG"]); ag = int(row["FTAG"])
        wynik_1x2 = "1" if hg > ag else ("2" if ag > hg else "X")
        typ  = z["typ"]

        if   typ == "1":           ok = wynik_1x2 == "1"
        elif typ == "X":           ok = wynik_1x2 == "X"
        elif typ == "2":           ok = wynik_1x2 == "2"
        elif typ == "1X":          ok = wynik_1x2 in ("1", "X")
        elif typ == "X2":          ok = wynik_1x2 in ("X", "2")
        elif typ.startswith("Over") and "goli" in typ:
            linia = float(typ.split()[1])
            ok = (hg + ag) > linia
        elif typ.startswith("Under") and "goli" in typ:
            linia = float(typ.split()[1])
            ok = (hg + ag) < linia
        elif typ == "BTTS – Tak":  ok = hg > 0 and ag > 0
        elif typ == "BTTS – Nie":  ok = hg == 0 or ag == 0
        elif typ.startswith("Over") and "rożnych" in typ:
            linia = float(typ.split()[1])
            rzuty = int(row.get("HC", 0)) + int(row.get("AC", 0))
            ok = rzuty > linia
        elif typ.startswith("Over") and "kartek" in typ:
            linia = float(typ.split()[1])
            kartki = int(row.get("HY", 0)) + int(row.get("AY", 0)) + \
                     (int(row.get("HR", 0)) + int(row.get("AR", 0))) * 2
            ok = kartki > linia
        else:
            return "⏳ Oczekuje"   # nieznany typ – nie werdyktuj

        if not ok:
            return "❌ Chybiony"

    return "✅ Trafiony"

# ===========================================================================
# WERYFIKACJA PREDYKCJI
# ===========================================================================
def wczytaj_predykcje() -> list:
    init_db()
    con = sqlite3.connect(DB_FILE)
    rows = con.execute(
        "SELECT liga,home,away,round,typ,fo_typ,p_home,p_draw,p_away,data "
        "FROM predykcje ORDER BY data,round"
    ).fetchall()
    con.close()
    keys = ["liga","home","away","round","typ","fo_typ","p_home","p_draw","p_away","data"]
    return [dict(zip(keys, r)) for r in rows]

def zapisz_predykcje_db(nowe: list) -> int:
    init_db()
    con = sqlite3.connect(DB_FILE)
    dodane = 0
    for p in nowe:
        try:
            con.execute(
                "INSERT OR IGNORE INTO predykcje "
                "(liga,home,away,round,typ,fo_typ,p_home,p_draw,p_away,data) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (p["liga"], p["home"], p["away"], p["round"],
                 p["typ"], p["fo_typ"], p["p_home"], p["p_draw"], p["p_away"], p["data"])
            )
            dodane += con.execute("SELECT changes()").fetchone()[0]
        except Exception:
            pass
    con.commit()
    con.close()
    return dodane

def usun_wszystkie_predykcje():
    init_db()
    con = sqlite3.connect(DB_FILE)
    con.execute("DELETE FROM predykcje")
    con.commit()
    con.close()

def weryfikuj_predykcje(predykcje: list, hist: pd.DataFrame) -> pd.DataFrame:
    if "_sezon" in hist.columns:
        hist = hist[hist["_sezon"] == "biezacy"]
    wyniki = []
    for p in predykcje:
        h, a = p["home"], p["away"]
        match = hist[(hist["HomeTeam"] == h) & (hist["AwayTeam"] == a)]
        brier = None
        if match.empty:
            status = "⏳ oczekuje"; wynik_r = "–"; trafiony = None
        else:
            row = match.iloc[-1]
            hg, ag   = int(row["FTHG"]), int(row["FTAG"])
            wynik_r  = f"{hg}:{ag}"
            rzecz    = "1" if hg > ag else ("2" if hg < ag else "X")
            typ_pred = p["typ"]
            if   typ_pred == "1X": trafiony = rzecz in ("1", "X")
            elif typ_pred == "X2": trafiony = rzecz in ("X", "2")
            else:                  trafiony = (rzecz == typ_pred)
            status = "✅ trafiony" if trafiony else "❌ chybiony"
            if all(p.get(k) is not None for k in ("p_home", "p_draw", "p_away")):
                r1 = 1.0 if rzecz=="1" else 0.0
                rx = 1.0 if rzecz=="X" else 0.0
                r2 = 1.0 if rzecz=="2" else 0.0
                brier = round((p["p_home"]-r1)**2 + (p["p_draw"]-rx)**2 + (p["p_away"]-r2)**2, 3)
        wyniki.append({
            "Liga": p.get("liga","–"), "Mecz": f"{h} vs {a}",
            "Kolejka": p.get("round","–"), "Typ": p["typ"],
            "Fair Odds": p["fo_typ"], "Wynik": wynik_r,
            "Status": status, "Trafiony": trafiony,
            "Brier": brier, "Data": p.get("data","–"),
        })
    return pd.DataFrame(wyniki) if wyniki else pd.DataFrame()

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
                model="claude-opus-4-6", max_tokens=200,
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

def render_alt_zdarzenia_html(zdarzenia: list) -> str:
    """Mini-tabelka HTML dla alternatywnych rynków – spójna wizualnie z kartą meczu."""
    if not zdarzenia:
        return "<p style='color:#888;font-size:0.85em'>Brak zdarzeń spełniających próg 55%.</p>"
    prog_kol = {"Gole": "#2196F3", "BTTS": "#9C27B0", "Rożne": "#FF9800", "Kartki": "#F44336"}
    rows = []
    for emoji, nazwa, p, fo, kat in zdarzenia[:8]:
        kol    = prog_kol.get(kat, "#888")
        bar_w  = int(p * 100)
        fo_col = "#4CAF50" if fo <= 1.60 else ("#FF9800" if fo <= 2.00 else "#aaa")
        rows.append(
            f"<tr>"
            f"<td style='padding:4px 6px;font-size:0.9em'>{emoji} {nazwa}</td>"
            f"<td style='padding:4px 6px;width:120px'>"
            f"  <div style='display:flex;align-items:center;gap:5px'>"
            f"    <div style='flex:1;background:#333;border-radius:3px;height:5px'>"
            f"      <div style='background:{kol};width:{bar_w}%;height:5px;border-radius:3px'></div>"
            f"    </div>"
            f"    <span style='color:{kol};font-size:0.82em;min-width:32px'>{p:.0%}</span>"
            f"  </div>"
            f"</td>"
            f"<td style='padding:4px 6px;text-align:right;color:{fo_col};font-weight:bold;font-size:0.88em'>"
            f"  {fo:.2f}</td>"
            f"</tr>"
        )
    return (
        "<table style='width:100%;border-collapse:collapse;margin-top:4px'>"
        "<thead><tr style='color:#666;font-size:0.75em;text-transform:uppercase'>"
        "<th style='padding:3px 6px;text-align:left'>Rynek</th>"
        "<th style='padding:3px 6px;text-align:left'>Prawdopodob.</th>"
        "<th style='padding:3px 6px;text-align:right'>Fair Odds</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        "<p style='color:#555;font-size:0.72em;margin-top:4px'>"
        "⚠️ Rożne i kartki – Poisson bez korelacji. Traktuj orientacyjnie.</p>"
    )

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

st.title(f"Predykcje {wybrana_liga} 2025/26")
st.markdown("Dixon-Coles • Blend sezonów • Forma • Fair Odds • Kupon AKO • SQLite")

if not historical.empty:
    srednie_df  = oblicz_wszystkie_statystyki(historical.to_json())
    srednie_lig = oblicz_srednie_ligowe(historical.to_json())
    forma_dict  = oblicz_forme(historical)
    tabela      = tabela_ligowa(historical)
    rho         = srednie_lig["rho"]
    n_biezacy   = srednie_lig["n_biezacy"]
    w_prev      = waga_poprzedniego(n_biezacy)

    # ── Sidebar: szybki podgląd kolejki ──────────────────────────────────────
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

    # ── TABS ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🎯 Bet Builder",
        "⚽ Przewidywane Wyniki",
        "🎲 Kupon AKO ~5.0",
        "✅ Weryfikacja",
        "📊 Tabela i Forma",
        "🏆 Power Rating",
        "📈 Model & Dane",
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
                    st.write("**Combo Builder** *(Over/Under przez macierz D-C)*")
                    combo_count = 0
                    for _, mecz in mecze.iterrows():
                        h = map_nazwa(mecz["home_team"])
                        a = map_nazwa(mecz["away_team"])
                        if h not in srednie_df.index or a not in srednie_df.index:
                            continue
                        lam_h, lam_a, lam_r, lam_k = oblicz_lambdy(h, a, srednie_df, srednie_lig, forma_dict)
                        p_g     = macierz_goli_p(lam_h, lam_a, rho, int(linia_gole), typ_gole)
                        p_r     = oblicz_p(typ_rogi,   linia_rogi,   lam_r)
                        p_k     = oblicz_p(typ_kartki, linia_kartki, lam_k)
                        p_combo = p_g * p_r * p_k
                        fo_g    = fair_odds(p_g); fo_r = fair_odds(p_r); fo_k = fair_odds(p_k)
                        fo_c    = fair_odds(p_combo)
                        if p_combo >= min_prob:
                            combo_count += 1
                            with st.expander(f"{h} vs {a}  ·  {p_combo:.1%}  ·  Fair AKO: {fo_c:.2f}"):
                                st.write(f"{koloruj(p_g)} Gole {typ_gole} {linia_gole}: {p_g:.1%}  *(fair {fo_g:.2f})*")
                                st.write(f"{koloruj(p_r)} Rożne {typ_rogi} {linia_rogi}: {p_r:.1%}  *(fair {fo_r:.2f})*")
                                st.write(f"{koloruj(p_k)} Kartki {typ_kartki} {linia_kartki}: {p_k:.1%}  *(fair {fo_k:.2f})*")
                                st.caption(f"Łączne fair AKO combo: **{fo_c:.2f}** | p_combo: {p_combo:.1%}")
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
                        btts_data.append((f"{h} - {a}", p_btts, fair_odds(p_btts)))
                    for ms, p, fo in sorted(btts_data, key=lambda x: x[1], reverse=True):
                        st.write(f"{koloruj(p)} **{ms}**: {p:.1%}  *(fair {fo:.2f})*")

                st.divider()
                vc1, vc2 = st.columns(2)
                with vc1:
                    st.write("**🔥 Power Index – największa różnica sił**")
                    power_data = []
                    for _, mecz in mecze.iterrows():
                        h2 = map_nazwa(mecz["home_team"]); a2 = map_nazwa(mecz["away_team"])
                        if h2 not in srednie_df.index or a2 not in srednie_df.index: continue
                        lh2, la2, _, _ = oblicz_lambdy(h2, a2, srednie_df, srednie_lig, forma_dict)
                        power_data.append((f"{h2} vs {a2}", abs(lh2-la2), lh2, la2))
                    for label, diff, lh2, la2 in sorted(power_data, key=lambda x: x[1], reverse=True)[:5]:
                        st.write(f"⚡ **{label}** – gap {diff:.2f} (λ {lh2:.2f} vs {la2:.2f})")
                with vc2:
                    st.write("**⚽ Najbardziej bramkowy mecz**")
                    gole_data = []
                    for _, mecz in mecze.iterrows():
                        h2 = map_nazwa(mecz["home_team"]); a2 = map_nazwa(mecz["away_team"])
                        if h2 not in srednie_df.index or a2 not in srednie_df.index: continue
                        lh2, la2, _, _ = oblicz_lambdy(h2, a2, srednie_df, srednie_lig, forma_dict)
                        gole_data.append((f"{h2} vs {a2}", lh2 + la2))
                    for label, total in sorted(gole_data, key=lambda x: x[1], reverse=True)[:5]:
                        st.write(f"🎯 **{label}** – śr. {total:.2f} goli")
            else:
                st.info("Brak nadchodzących meczów w terminarzu.")

    # =========================================================================
    # TAB 2 – PRZEWIDYWANE WYNIKI
    # =========================================================================
    with tab2:
        st.subheader("⚽ Przewidywane wyniki – najbliższa kolejka")
        st.caption("**Fair Odds** = kurs bez marży bukmachera. Kliknij mecz aby rozwinąć szczegóły.")

        tgl1, tgl2, tgl3 = st.columns(3)
        with tgl1: pokaz_komentarz = st.toggle("💬 Komentarz analityczny", value=True)
        with tgl2: pokaz_macierz  = st.toggle("🔢 Macierz wyników",        value=False)
        with tgl3:
            zapisuj_flag = st.toggle("💾 Zapisz do weryfikacji", value=False,
                                     help="Zapisuje predykcje do SQLite.")

        if not schedule.empty and not srednie_df.empty:
            dzisiaj  = datetime.now().date()
            przyszle = schedule[schedule["date"].dt.date >= dzisiaj]
            if not przyszle.empty:
                nb    = przyszle["round"].min()
                mecze = schedule[schedule["round"] == nb]
                st.caption(f"Kolejka {int(nb)} – {len(mecze)} meczów")

                nowe_predykcje = []
                dopasowane     = 0
                kol_a, kol_b   = st.columns(2)
                mecze_list     = list(mecze.iterrows())

                for idx, (_, mecz) in enumerate(mecze_list):
                    h = map_nazwa(mecz["home_team"])
                    a = map_nazwa(mecz["away_team"])
                    if h not in srednie_df.index or a not in srednie_df.index:
                        continue

                    lam_h, lam_a, lam_r, lam_k = oblicz_lambdy(h, a, srednie_df, srednie_lig, forma_dict)
                    pred       = predykcja_meczu(lam_h, lam_a, rho=rho)
                    dopasowane += 1
                    data_meczu = mecz["date"].strftime("%d.%m %H:%M") if pd.notna(mecz["date"]) else ""

                    nowe_predykcje.append({
                        "liga": wybrana_liga, "home": h, "away": a,
                        "round": int(nb), "typ": pred["typ"], "fo_typ": pred["fo_typ"],
                        "p_home": round(pred["p_home"], 4),
                        "p_draw": round(pred["p_draw"], 4),
                        "p_away": round(pred["p_away"], 4),
                        "data": datetime.now().strftime("%Y-%m-%d"),
                    })

                    ikony_typ  = {"1":"🔵","X":"🟠","2":"🔴","1X":"🟣","X2":"🟣"}
                    conf_ikona = "🟢" if pred["conf_level"]=="High" else ("🟡" if pred["conf_level"]=="Medium" else "🔴")
                    label_exp  = (f"{conf_ikona} {h} vs {a}"
                                  f"  ·  {ikony_typ.get(pred['typ'],'⚪')} {pred['typ']} @ {pred['fo_typ']:.2f}"
                                  f"  ·  {data_meczu}")

                    kolumna = kol_a if idx % 2 == 0 else kol_b
                    with kolumna:
                        with st.expander(label_exp, expanded=False):
                            # Nagłówek drużyn
                            ch, cmid, ca = st.columns([5,2,5])
                            with ch:
                                st.markdown(f"<div style='font-size:1.05em;font-weight:bold'>{h}</div>",
                                            unsafe_allow_html=True)
                            with cmid:
                                st.markdown(f"<div style='text-align:center;color:#888;font-size:0.78em'>{data_meczu}</div>",
                                            unsafe_allow_html=True)
                            with ca:
                                st.markdown(f"<div style='font-size:1.05em;font-weight:bold;text-align:right'>{a}</div>",
                                            unsafe_allow_html=True)

                            # Przewidywany wynik
                            st.markdown(
                                f"<div style='text-align:center;font-size:1.7em;font-weight:bold;margin:4px 0'>"
                                f"⚽ {pred['wynik_h']}:{pred['wynik_a']}"
                                f"<span style='font-size:0.5em;color:#888;font-weight:normal;margin-left:8px'>"
                                f"({pred['p_exact']:.1%})</span></div>",
                                unsafe_allow_html=True,
                            )

                            # Typ + confidence + chaos bar
                            conf_colors  = {"High":"#4CAF50","Medium":"#FF9800","Coinflip":"#F44336"}
                            chaos_colors = {"Pewny":"#4CAF50","Klarowny":"#8BC34A","Otwarty":"#FF9800","Chaos":"#F44336"}
                            conf_c = conf_colors.get(pred["conf_level"], "#888")
                            ch_c   = chaos_colors.get(pred["chaos_label"], "#888")
                            bar_w  = int(pred["chaos_pct"] * 100)
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
                                f"<div style='font-size:0.78em;color:#888'>"
                                f"{pred['chaos_emoji']} Chaos Index: "
                                f"<b style='color:{ch_c}'>{pred['chaos_label']}</b>"
                                f" ({pred['entropy']:.2f} bits)</div>"
                                f"<div style='background:#333;border-radius:4px;height:4px;margin-top:4px'>"
                                f"<div style='background:{ch_c};width:{bar_w}%;height:4px;border-radius:4px'></div>"
                                f"</div></div>",
                                unsafe_allow_html=True,
                            )

                            # Metryki 1X2
                            mc1, mc2, mc3 = st.columns(3)
                            with mc1:
                                st.metric(f"1 – {h[:12]}", f"{pred['p_home']:.1%}", f"@ {pred['fo_home']:.2f}",
                                          delta_color="normal" if pred["typ"] in ("1","1X") else "off")
                            with mc2:
                                st.metric("X – Remis", f"{pred['p_draw']:.1%}", f"@ {pred['fo_draw']:.2f}",
                                          delta_color="normal" if pred["typ"] in ("X","1X","X2") else "off")
                            with mc3:
                                st.metric(f"2 – {a[:12]}", f"{pred['p_away']:.1%}", f"@ {pred['fo_away']:.2f}",
                                          delta_color="normal" if pred["typ"] in ("2","X2") else "off")

                            # Lambda info
                            st.markdown(
                                f"<div style='text-align:center;font-size:0.82em;color:#888;margin-top:2px'>"
                                f"λ {h[:8]}: <b>{pred['lam_h']:.2f}</b> &nbsp;|&nbsp; "
                                f"λ {a[:8]}: <b>{pred['lam_a']:.2f}</b> &nbsp;|&nbsp; "
                                f"Łącznie: <b>{pred['lam_h']+pred['lam_a']:.2f}</b></div>",
                                unsafe_allow_html=True,
                            )

                            # ── ALTERNATYWNE RYNKI ────────────────────────────
                            with st.expander("📊 Alternatywne rynki", expanded=False):
                                alt = alternatywne_zdarzenia(lam_h, lam_a, lam_r, lam_k, rho)
                                st.markdown(render_alt_zdarzenia_html(alt), unsafe_allow_html=True)

                            if pokaz_komentarz:
                                st.info(f"💬 {generuj_komentarz(h, a, pred, forma_dict)}")

                            if pokaz_macierz:
                                st.markdown("**Macierz dokładnych wyników**")
                                st.markdown(render_macierz_html(pred["macierz"], h, a),
                                            unsafe_allow_html=True)

                # Zapis predykcji
                if zapisuj_flag and nowe_predykcje:
                    dodane = zapisz_predykcje_db(nowe_predykcje)
                    st.success(f"✅ Zapisano {dodane} nowych predykcji." if dodane
                               else "Predykcje tej kolejki już są w bazie.")

                # Export CSV
                if dopasowane > 0:
                    df_exp = pd.DataFrame(nowe_predykcje)[
                        ["liga","home","away","round","typ","fo_typ","p_home","p_draw","p_away","data"]
                    ]
                    st.download_button(
                        "⬇️ Pobierz predykcje kolejki (CSV)",
                        data=df_exp.to_csv(index=False, decimal=","),
                        file_name=f"predykcje_{wybrana_liga.replace(' ','_')}_kolejka{int(nb)}.csv",
                        mime="text/csv",
                    )
                if dopasowane == 0:
                    st.warning("Żaden mecz nie mógł być dopasowany – sprawdź Debug w sidebarze.")
            else:
                st.info("Brak nadchodzących meczów w terminarzu.")

    # =========================================================================
    # TAB 3 – KUPON AKO ~5.0
    # =========================================================================
    with tab3:
        st.subheader("🎲 Kupon kolejki – AKO ~5.0")
        st.caption(
            "Model przeszukuje wszystkie kombinacje 3–5 zdarzeń z bieżącej kolejki "
            "i wybiera zestaw z AKO najbliższym 5.0. "
            "⚠️ Kursy to **fair odds** (bez marży) – rzeczywiste kursy bukmacherskie będą niższe."
        )

        target_slider = st.slider("Docelowe AKO", 3.0, 10.0, 5.0, 0.5,
                                  help="Możesz zmienić cel – np. 3.0 to bezpieczniejszy 3-pak.")

        if not schedule.empty and not srednie_df.empty:
            dzisiaj  = datetime.now().date()
            przyszle = schedule[schedule["date"].dt.date >= dzisiaj]
            if not przyszle.empty:
                nb    = przyszle["round"].min()
                mecze = schedule[schedule["round"] == nb]

                # Zbierz wszystkie zdarzenia z całej kolejki
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
                    if pred["p_typ"] >= 0.58:
                        wszystkie_zd.append({
                            "mecz": mecz_str, "typ": pred["typ"],
                            "p": pred["p_typ"], "fair": pred["fo_typ"],
                            "kategoria": "1X2", "emoji": "🏟️"
                        })

                    # Alternatywne
                    for emoji, nazwa, p, fo, kat in alternatywne_zdarzenia(lam_h, lam_a, lam_r, lam_k, rho):
                        wszystkie_zd.append({
                            "mecz": mecz_str, "typ": nazwa,
                            "p": p, "fair": fo,
                            "kategoria": kat, "emoji": emoji
                        })

                kupon = generuj_kupon(wszystkie_zd, target_ako=target_slider)

                if kupon:
                    # ── Karta kuponu ──────────────────────────────────────────
                    ako_color = "#4CAF50" if abs(kupon["ako"] - target_slider) < 0.5 else "#FF9800"
                    st.markdown(
                        f"<div style='background:#1a1a2e;border:1px solid #333;border-radius:10px;"
                        f"padding:16px;margin-bottom:12px'>"
                        f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                        f"<div style='font-size:1.1em;font-weight:bold'>🎲 Kupon kolejki {int(nb)}</div>"
                        f"<div style='font-size:2em;font-weight:bold;color:{ako_color}'>"
                        f"AKO {kupon['ako']:.2f}</div></div>"
                        f"<div style='color:#888;font-size:0.85em;margin-top:4px'>"
                        f"Łączne p_combo: <b style='color:#aaa'>{kupon['p_combo']:.1%}</b> &nbsp;|&nbsp; "
                        f"{len(kupon['zdarzenia'])} nogi &nbsp;|&nbsp; "
                        f"fair odds (bez marży)</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    # Nogi kuponu jako HTML tabela
                    cat_colors = {"1X2":"#2196F3","Gole":"#4CAF50","BTTS":"#9C27B0",
                                  "Rożne":"#FF9800","Kartki":"#F44336"}
                    nogi_html = []
                    for i, z in enumerate(kupon["zdarzenia"], 1):
                        cc = cat_colors.get(z.get("kategoria",""), "#888")
                        nogi_html.append(
                            f"<tr>"
                            f"<td style='padding:6px 10px;color:#888;width:24px'>{i}</td>"
                            f"<td style='padding:6px 10px'>{z.get('emoji','⚽')} <b>{z['mecz']}</b></td>"
                            f"<td style='padding:6px 10px'>"
                            f"  <span style='background:{cc}22;color:{cc};padding:2px 8px;"
                            f"border-radius:10px;font-size:0.85em'>{z['typ']}</span>"
                            f"</td>"
                            f"<td style='padding:6px 10px;text-align:right;color:#aaa'>{z['p']:.0%}</td>"
                            f"<td style='padding:6px 10px;text-align:right;font-weight:bold;color:#4CAF50'>"
                            f"{z['fair']:.2f}</td>"
                            f"</tr>"
                        )
                    st.markdown(
                        f"<div style='border:1px solid #333;border-radius:8px;overflow:hidden'>"
                        f"<table style='width:100%;border-collapse:collapse;font-size:0.9em'>"
                        f"<thead><tr style='background:#1e1e2e;color:#666;font-size:0.78em;text-transform:uppercase'>"
                        f"<th style='padding:8px 10px'>#</th>"
                        f"<th style='padding:8px 10px;text-align:left'>Mecz</th>"
                        f"<th style='padding:8px 10px;text-align:left'>Zdarzenie</th>"
                        f"<th style='padding:8px 10px;text-align:right'>P</th>"
                        f"<th style='padding:8px 10px;text-align:right'>Fair Odds</th>"
                        f"</tr></thead>"
                        f"<tbody>{''.join(nogi_html)}</tbody>"
                        f"</table></div>",
                        unsafe_allow_html=True,
                    )

                    # Zapis
                    kc1, kc2 = st.columns([3, 1])
                    with kc1:
                        if st.button("💾 Zapisz kupon do bazy", use_container_width=True):
                            zapisz_kupon_db(kupon, wybrana_liga, int(nb),
                                            typ_kuponu=f"AKO ~{target_slider:.1f}")
                            st.success("✅ Zapisano. Weryfikacja automatyczna po meczach.")
                    with kc2:
                        kupon_csv = pd.DataFrame(kupon["zdarzenia"]).to_csv(index=False, decimal=",")
                        st.download_button("⬇️ CSV", data=kupon_csv,
                                           file_name=f"kupon_kolejka{int(nb)}.csv", mime="text/csv")
                else:
                    st.warning("Nie udało się wygenerować kuponu – za mało zdarzeń spełniających kryteria.")

                # ── Historia kuponów z weryfikacją ────────────────────────────
                st.divider()
                st.markdown("### 📋 Historia kuponów")
                kupony_hist = wczytaj_kupony(wybrana_liga)
                if kupony_hist:
                    for k in kupony_hist:
                        status_k = weryfikuj_kupon(k, historical)
                        bg_k = {"✅ Trafiony":"#1a2e1a","❌ Chybiony":"#2e1a1a"}.get(status_k, "transparent")
                        with st.expander(
                            f"{status_k}  ·  Kolejka {k['kolejnosc']}  "
                            f"·  AKO {k['ako']:.2f}  ·  {k['typ_kuponu']}  ·  {k['data']}",
                            expanded=False
                        ):
                            for z in k["zdarzenia"]:
                                cc = cat_colors.get(z.get("kategoria",""), "#888")
                                st.markdown(
                                    f"<div style='background:{bg_k};padding:4px 8px;border-radius:4px;margin-bottom:3px'>"
                                    f"{z.get('emoji','⚽')} **{z['mecz']}** – "
                                    f"<span style='color:{cc}'>{z['typ']}</span> "
                                    f"({z['p']:.0%} · fair {z['fair']:.2f})"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                            st.caption(f"p_combo: {k['p_combo']:.1%}")
                else:
                    st.info("Brak zapisanych kuponów dla tej ligi.")
            else:
                st.info("Brak nadchodzących meczów w terminarzu.")

    # =========================================================================
    # TAB 4 – WERYFIKACJA PREDYKCJI
    # =========================================================================
    with tab4:
        st.subheader("✅ Weryfikacja skuteczności modelu")
        st.caption("Predykcje zapisujesz w zakładce **⚽ Przewidywane Wyniki** (toggle 💾). Dane w SQLite.")

        predykcje = wczytaj_predykcje()

        if not predykcje:
            st.info("Brak zapisanych predykcji. Zapisz je przed kolejką, wróć tu po meczach.")
        else:
            df_wer     = weryfikuj_predykcje(predykcje, historical)
            zakonczone = df_wer[df_wer["Trafiony"].notna()]

            if not zakonczone.empty:
                trafione   = int(zakonczone["Trafiony"].sum())
                wszystkie  = len(zakonczone)
                skuteczn   = trafione / wszystkie
                sr_odds_tr = zakonczone[zakonczone["Trafiony"] == True]["Fair Odds"].mean()
                brier_vals = zakonczone["Brier"].dropna()
                sr_brier   = brier_vals.mean() if len(brier_vals) > 0 else float("nan")
                brier_delta = f"{0.667 - sr_brier:+.3f} vs random" if not np.isnan(sr_brier) else None

                with st.container(border=True):
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("⚽ Rozegrane",     wszystkie)
                    m2.metric("✅ Trafione",        trafione)
                    m3.metric("🎯 Skuteczność",    f"{skuteczn:.1%}")
                    m4.metric("💰 Śr. odds (hit)", f"{sr_odds_tr:.2f}" if not np.isnan(sr_odds_tr) else "–")
                    m5.metric("📐 Brier Score ↓",  f"{sr_brier:.3f}" if not np.isnan(sr_brier) else "–",
                              delta=brier_delta, delta_color="normal",
                              help="Niżej = lepiej. Losowy model ~0.667, dobry <0.50")

                wyk1, wyk2 = st.columns(2)
                with wyk1:
                    if len(brier_vals) >= 3:
                        st.markdown("**📈 Brier Score w czasie**")
                        bs_time = (zakonczone[zakonczone["Brier"].notna()]
                                   .groupby("Kolejka")["Brier"].mean()
                                   .reset_index().rename(columns={"Brier": "Brier Score"}))
                        bs_time["Kolejka"] = bs_time["Kolejka"].astype(str)
                        st.line_chart(bs_time.set_index("Kolejka"), height=220)
                    else:
                        st.markdown("**📈 Brier Score w czasie**")
                        st.caption("Dostępne po min. 3 kolejkach.")
                with wyk2:
                    st.markdown("**🎯 Skuteczność per typ**")
                    per_typ = (zakonczone.groupby("Typ")["Trafiony"]
                               .agg(["sum","count"])
                               .rename(columns={"sum":"Trafione","count":"Mecze"}))
                    per_typ["Hit %"] = (per_typ["Trafione"]/per_typ["Mecze"]*100).round(1)
                    st.dataframe(per_typ[["Mecze","Trafione","Hit %"]].sort_values("Hit %",ascending=False),
                                 use_container_width=True, height=220)

                # Reliability curve
                pred_z_prob = zakonczone[zakonczone["Brier"].notna()].copy()
                if len(pred_z_prob) >= 15:
                    st.markdown("**📐 Reliability Curve** *(kalibracja modelu)*")
                    bins     = [0, 0.40, 0.50, 0.60, 0.70, 0.80, 1.01]
                    labels_b = ["<40%","40-50%","50-60%","60-70%","70-80%",">80%"]
                    pred_z_prob["p_typ_est"] = pred_z_prob["Fair Odds"].apply(
                        lambda o: 1/o if o > 0 else 0.5)
                    pred_z_prob["koszyk"] = pd.cut(
                        pred_z_prob["p_typ_est"], bins=bins, labels=labels_b, right=False)
                    rel = (pred_z_prob.groupby("koszyk", observed=True)["Trafiony"]
                           .agg(["mean","count"])
                           .rename(columns={"mean":"Hit Rate","count":"N"})
                           .reset_index())
                    rel = rel[rel["N"] >= 3]
                    if len(rel) >= 2:
                        w_rc, h_rc, pad_rc = 500, 300, 50
                        def rc_px(xv, yv):
                            px = pad_rc + (xv - 0.3) / 0.6 * (w_rc - 2*pad_rc)
                            py = h_rc - pad_rc - yv * (h_rc - 2*pad_rc)
                            return px, py
                        diag      = [rc_px(t, t) for t in [0.3, 0.5, 0.7, 0.9]]
                        diag_line = " ".join(f"{p[0]:.0f},{p[1]:.0f}" for p in diag)
                        circles_rc = []
                        bin_mid_map = {"<40%":0.35,"40-50%":0.45,"50-60%":0.55,
                                       "60-70%":0.65,"70-80%":0.75,">80%":0.85}
                        for _, rrow in rel.iterrows():
                            xv = bin_mid_map.get(str(rrow["koszyk"]), 0.5)
                            yv = rrow["Hit Rate"]; n = rrow["N"]
                            px, py = rc_px(xv, yv)
                            r_size = min(max(int(n * 1.5), 5), 20)
                            diff   = yv - xv
                            col_rc = "#4CAF50" if abs(diff)<0.05 else ("#FF9800" if abs(diff)<0.12 else "#F44336")
                            circles_rc.append(
                                f"<circle cx='{px:.0f}' cy='{py:.0f}' r='{r_size}' "
                                f"fill='{col_rc}' fill-opacity='0.8' stroke='white' stroke-width='1.5'>"
                                f"<title>Model: {xv:.0%} | Hit: {yv:.0%} | N={n}</title></circle>"
                                f"<text x='{px+r_size+3:.0f}' y='{py+4:.0f}' "
                                f"font-size='9' fill='#aaa' font-family='sans-serif'>"
                                f"{yv:.0%} (n={n})</text>"
                            )
                        svg_rc = (
                            f'<svg width="{w_rc}" height="{h_rc}" '
                            f'style="background:#0e1117;border-radius:8px;display:block;margin:auto">'
                            f'<polyline points="{diag_line}" fill="none" stroke="#444" '
                            f'stroke-width="1.5" stroke-dasharray="6,4"/>'
                            f'<text x="{w_rc-pad_rc+4}" y="{pad_rc-4}" font-size="9" '
                            f'fill="#555" font-family="sans-serif">idealny model</text>'
                            f'{"".join(circles_rc)}'
                            f'<text x="{w_rc//2}" y="{h_rc-6}" text-anchor="middle" '
                            f'font-size="10" fill="#888" font-family="sans-serif">'
                            f'Pewność modelu (p_typ) →</text>'
                            f'<text x="12" y="{h_rc//2}" text-anchor="middle" '
                            f'font-size="10" fill="#888" font-family="sans-serif" '
                            f'transform="rotate(-90,12,{h_rc//2})">Rzeczywisty Hit Rate →</text>'
                            f'</svg>'
                        )
                        st.markdown(svg_rc, unsafe_allow_html=True)
                        st.caption("Punkty na przekątnej = model idealnie skalibrowany.")
                    else:
                        st.caption("Za mało danych w koszykach.")
                elif 5 <= len(pred_z_prob) < 15:
                    st.info(f"📐 Reliability curve po min. 15 predykcjach. Masz {len(pred_z_prob)}.")

                per_liga = (zakonczone.groupby("Liga")["Trafiony"]
                            .agg(["sum","count"])
                            .rename(columns={"sum":"Trafione","count":"Mecze"}))
                per_liga["Hit %"] = (per_liga["Trafione"]/per_liga["Mecze"]*100).round(1)
                if len(per_liga) > 1:
                    st.markdown("**🌍 Skuteczność per liga**")
                    st.bar_chart(per_liga["Hit %"], height=180)

            # Historia predykcji
            st.divider()
            st.markdown("### 📋 Historia predykcji")
            fc1, fc2, _ = st.columns([2, 2, 6])
            with fc1:
                filtr_s = st.selectbox("Status", ["Wszystkie","✅ Trafione","❌ Chybione","⏳ Oczekujące"],
                                       label_visibility="collapsed")
            with fc2:
                ligi_d = ["Wszystkie ligi"] + sorted(df_wer["Liga"].unique().tolist())
                filtr_l = st.selectbox("Liga", ligi_d, label_visibility="collapsed")

            df_show = df_wer.copy()
            if filtr_s == "✅ Trafione":    df_show = df_show[df_show["Status"]=="✅ trafiony"]
            elif filtr_s == "❌ Chybione":  df_show = df_show[df_show["Status"]=="❌ chybiony"]
            elif filtr_s == "⏳ Oczekujące": df_show = df_show[df_show["Status"]=="⏳ oczekuje"]
            if filtr_l != "Wszystkie ligi": df_show = df_show[df_show["Liga"]==filtr_l]

            ikony_t2 = {"1":"🔵","X":"🟠","2":"🔴","1X":"🟣","X2":"🟣"}
            html_rows = []
            for _, row in df_show[["Liga","Mecz","Kolejka","Typ","Fair Odds","Wynik","Status","Brier","Data"]].iterrows():
                bg = "#1a2e1a" if row["Status"]=="✅ trafiony" else ("#2e1a1a" if row["Status"]=="❌ chybiony" else "transparent")
                st_html = (f"<span style='color:#4CAF50;font-weight:bold'>✅ trafiony</span>"
                           if row["Status"]=="✅ trafiony" else
                           f"<span style='color:#F44336;font-weight:bold'>❌ chybiony</span>"
                           if row["Status"]=="❌ chybiony" else
                           "<span style='color:#888'>⏳ oczekuje</span>")
                br_str = f"{row['Brier']:.3f}" if pd.notna(row["Brier"]) else "–"
                html_rows.append(
                    f"<tr style='background:{bg}'>"
                    f"<td style='padding:5px 8px;color:#aaa;font-size:0.82em'>{row['Liga']}</td>"
                    f"<td style='padding:5px 8px;font-weight:bold'>{row['Mecz']}</td>"
                    f"<td style='padding:5px 8px;text-align:center;color:#888'>{row['Kolejka']}</td>"
                    f"<td style='padding:5px 8px;text-align:center'>"
                    f"{ikony_t2.get(row['Typ'],'⚪')} <b>{row['Typ']}</b></td>"
                    f"<td style='padding:5px 8px;text-align:center;color:#888'>{row['Fair Odds']:.2f}</td>"
                    f"<td style='padding:5px 8px;text-align:center;font-weight:bold'>{row['Wynik']}</td>"
                    f"<td style='padding:5px 8px'>{st_html}</td>"
                    f"<td style='padding:5px 8px;text-align:center;color:#888;font-size:0.85em'>{br_str}</td>"
                    f"<td style='padding:5px 8px;color:#666;font-size:0.80em'>{row['Data']}</td>"
                    f"</tr>"
                )
            st.markdown(
                f"<div style='overflow-x:auto;border-radius:8px;border:1px solid #333;margin-top:8px'>"
                f"<table style='width:100%;border-collapse:collapse;font-size:0.88em'>"
                f"<thead><tr style='background:#1e1e2e;color:#aaa;font-size:0.82em;"
                f"text-transform:uppercase;letter-spacing:0.05em'>"
                f"<th style='padding:8px;text-align:left'>Liga</th>"
                f"<th style='padding:8px;text-align:left'>Mecz</th>"
                f"<th style='padding:8px;text-align:center'>Kol.</th>"
                f"<th style='padding:8px;text-align:center'>Typ</th>"
                f"<th style='padding:8px;text-align:center'>Odds</th>"
                f"<th style='padding:8px;text-align:center'>Wynik</th>"
                f"<th style='padding:8px;text-align:left'>Status</th>"
                f"<th style='padding:8px;text-align:center'>Brier</th>"
                f"<th style='padding:8px;text-align:left'>Data</th>"
                f"</tr></thead><tbody>{''.join(html_rows)}</tbody></table></div>",
                unsafe_allow_html=True,
            )

            ec1, ec2 = st.columns([3,1])
            with ec1:
                st.download_button("⬇️ Pobierz historię (CSV)",
                                   data=df_wer.drop(columns=["Trafiony"]).to_csv(index=False, decimal=","),
                                   file_name="historia_predykcji.csv", mime="text/csv")
            with ec2:
                if st.button("🗑️ Wyczyść bazę", type="secondary"):
                    usun_wszystkie_predykcje()
                    st.success("Baza wyczyszczona.")
                    st.rerun()

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
            st.dataframe(pd.DataFrame.from_dict(forma_dict, orient="index", columns=["Forma"]),
                         use_container_width=True)

    # =========================================================================
    # TAB 6 – POWER RATING
    # =========================================================================
    with tab6:
        st.subheader("🏆 Power Rating drużyn")
        st.caption("Siła ataku i obrony normalizowana do średniej ligowej (=1.0). "
                   "Kolor punktu = forma ostatnich 5 meczów.")

        if not srednie_df.empty:
            avg_h_pr = srednie_lig["avg_home"]
            avg_a_pr = srednie_lig["avg_away"]
            power_rows = []
            for team in srednie_df.index:
                row = srednie_df.loc[team]
                atak   = ((row["Gole strzelone (dom)"] / avg_h_pr) +
                           (row["Gole strzelone (wyjazd)"] / avg_a_pr)) / 2
                obrona = 2 - ((row["Gole stracone (dom)"] / avg_h_pr) +
                               (row["Gole stracone (wyjazd)"] / avg_a_pr)) / 2
                forma  = forma_dict.get(team, "")
                fs     = forma.count("W") - forma.count("L")
                power_rows.append({"Drużyna": team, "Atak": round(atak,3),
                                   "Obrona": round(obrona,3), "Forma score": fs,
                                   "Forma": forma or "?",
                                   "Power": round((atak+obrona)/2, 3)})
            df_power = pd.DataFrame(power_rows).sort_values("Power",ascending=False).reset_index(drop=True)

            # SVG scatter
            x_vals = df_power["Atak"].tolist(); y_vals = df_power["Obrona"].tolist()
            names  = df_power["Drużyna"].tolist(); fs_vals = df_power["Forma score"].tolist()
            x_min, x_max = min(x_vals)-0.1, max(x_vals)+0.1
            y_min, y_max = min(y_vals)-0.1, max(y_vals)+0.1
            W, H, pad = 700, 480, 60
            def to_px(xv, yv):
                return (pad + (xv-x_min)/(x_max-x_min)*(W-2*pad),
                        H - pad - (yv-y_min)/(y_max-y_min)*(H-2*pad))
            def fc(fs): return "#4CAF50" if fs>=2 else ("#F44336" if fs<=-2 else "#FF9800")
            circles_pr = []; labels_pr = []
            for i, (nm, xv, yv, fs) in enumerate(zip(names, x_vals, y_vals, fs_vals)):
                px, py = to_px(xv, yv)
                col    = fc(fs)
                circles_pr.append(
                    f"<circle cx='{px:.1f}' cy='{py:.1f}' r='7' fill='{col}' "
                    f"fill-opacity='0.85' stroke='white' stroke-width='1.5'>"
                    f"<title>{nm}\nAtak: {xv:.2f} | Obrona: {yv:.2f} | Forma: {df_power.loc[i,'Forma']}</title>"
                    f"</circle>"
                )
                labels_pr.append(f"<text x='{px+9:.1f}' y='{py+4:.1f}' "
                                  f"font-size='9' fill='#ccc' font-family='sans-serif'>{nm[:8]}</text>")
            cx_pr, _  = to_px(1.0, y_min); _, cy_pr = to_px(x_min, 1.0)
            svg_pr = (
                f'<svg width="{W}" height="{H}" style="background:#0e1117;border-radius:8px;display:block;margin:auto">'
                f'<line x1="{pad}" y1="{H-pad}" x2="{W-pad}" y2="{H-pad}" stroke="#444" stroke-width="1"/>'
                f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{H-pad}" stroke="#444" stroke-width="1"/>'
                f'<line x1="{cx_pr:.1f}" y1="{pad}" x2="{cx_pr:.1f}" y2="{H-pad}" '
                f'stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>'
                f'<line x1="{pad}" y1="{cy_pr:.1f}" x2="{W-pad}" y2="{cy_pr:.1f}" '
                f'stroke="#555" stroke-width="1" stroke-dasharray="4,4"/>'
                f'<text x="{cx_pr+6:.1f}" y="{cy_pr-8:.1f}" font-size="9" fill="#555" font-family="sans-serif">'
                f'silny atak + silna obrona</text>'
                f'<text x="{W//2}" y="{H-8}" text-anchor="middle" font-size="11" fill="#888" font-family="sans-serif">Siła ataku →</text>'
                f'<text x="12" y="{H//2}" text-anchor="middle" font-size="11" fill="#888" '
                f'font-family="sans-serif" transform="rotate(-90,12,{H//2})">Siła obrony →</text>'
                f'{"".join(circles_pr)}{"".join(labels_pr)}'
                f'<circle cx="{W-120}" cy="20" r="6" fill="#4CAF50" stroke="white" stroke-width="1"/>'
                f'<text x="{W-110}" y="25" font-size="9" fill="#aaa" font-family="sans-serif">Dobra forma</text>'
                f'<circle cx="{W-120}" cy="38" r="6" fill="#FF9800" stroke="white" stroke-width="1"/>'
                f'<text x="{W-110}" y="43" font-size="9" fill="#aaa" font-family="sans-serif">Neutralna</text>'
                f'<circle cx="{W-120}" cy="56" r="6" fill="#F44336" stroke="white" stroke-width="1"/>'
                f'<text x="{W-110}" y="61" font-size="9" fill="#aaa" font-family="sans-serif">Słaba forma</text>'
                f'</svg>'
            )
            st.markdown(svg_pr, unsafe_allow_html=True)
            st.caption("Hover nad punktem = pełna nazwa i statystyki.")

            # Tabela rankingowa z paskami
            st.divider()
            st.markdown("**Ranking Power Rating**")
            max_pw = df_power["Power"].max()
            pr_rows = []
            for rank, row in df_power[["Drużyna","Power","Atak","Obrona","Forma"]].iterrows():
                bw = int(row["Power"] / max_pw * 100)
                pc = "#4CAF50" if row["Power"] >= 1.0 else "#F44336"
                pr_rows.append(
                    f"<tr><td style='padding:5px 8px;color:#888;width:30px'>{rank+1}</td>"
                    f"<td style='padding:5px 8px;font-weight:bold'>{row['Drużyna']}</td>"
                    f"<td style='padding:5px 8px;width:160px'>"
                    f"  <div style='display:flex;align-items:center;gap:6px'>"
                    f"    <div style='flex:1;background:#333;border-radius:3px;height:6px'>"
                    f"      <div style='background:{pc};width:{bw}%;height:6px;border-radius:3px'></div>"
                    f"    </div>"
                    f"    <span style='color:{pc};font-weight:bold;font-size:0.85em;min-width:36px'>{row['Power']:.3f}</span>"
                    f"  </div></td>"
                    f"<td style='padding:5px 8px;text-align:center;color:#4CAF50'>{row['Atak']:.3f}</td>"
                    f"<td style='padding:5px 8px;text-align:center;color:#2196F3'>{row['Obrona']:.3f}</td>"
                    f"<td style='padding:5px 8px;font-family:monospace;letter-spacing:2px'>{row['Forma']}</td>"
                    f"</tr>"
                )
            st.markdown(
                f"<div style='overflow-x:auto;border-radius:8px;border:1px solid #333'>"
                f"<table style='width:100%;border-collapse:collapse;font-size:0.88em'>"
                f"<thead><tr style='background:#1e1e2e;color:#aaa;font-size:0.80em;text-transform:uppercase'>"
                f"<th style='padding:8px'>#</th><th style='padding:8px;text-align:left'>Drużyna</th>"
                f"<th style='padding:8px;text-align:left'>Power Rating</th>"
                f"<th style='padding:8px;text-align:center'>Atak</th>"
                f"<th style='padding:8px;text-align:center'>Obrona</th>"
                f"<th style='padding:8px;text-align:left'>Forma</th></tr></thead>"
                f"<tbody>{''.join(pr_rows)}</tbody></table></div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Brak danych do Power Rating.")

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
                       help="Maleje automatycznie w miarę gromadzenia danych")
            bc3.metric("Łącznie rekordów w modelu", len(historical))
        with st.container(border=True):
            st.markdown("**⚙️ Parametry Dixon-Coles**")
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("ρ (rho) kalibrowane", f"{rho:.4f}",
                       help="Kalibrowane empirycznie z historii tej ligi.")
            dc2.metric("Śr. gole (dom)",    f"{srednie_lig['avg_home']:.2f}")
            dc3.metric("Śr. gole (wyjazd)", f"{srednie_lig['avg_away']:.2f}")
        st.divider()
        st.write("**Średnie ważone drużyn** *(ostatnie 10 meczów, wagi rosnące)*")
        st.dataframe(srednie_df.sort_index(), use_container_width=True)
        st.caption(f"📅 Ostatnia aktualizacja: {historical['Date'].max().strftime('%d.%m.%Y')}")
        if st.button("🔄 Odśwież dane"):
            st.cache_data.clear()
            st.rerun()

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
