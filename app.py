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
import math

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

# Stałe dla kuponów
TARGET_AKO_MEDIUM = 5.0      # docelowe AKO dla "Złotego Środka"
TARGET_AKO_TAPE = 20.0        # docelowe AKO dla "Taśmy"
MIN_SINGLE_ODDS = 1.4         # minimalny kurs pojedynczego zdarzenia
MAX_SINGLE_ODDS = 2.2         # maksymalny kurs pojedynczego zdarzenia (bezpieczeństwo)
MARKET_ODDS_BENCHMARK = 1.9   # referencyjny kurs rynkowy do liczenia value

# Waga poprzedniego sezonu
def waga_poprzedniego(n_biezacy: int) -> float:
    return float(np.clip(0.8 - (n_biezacy / 30) * 0.6, 0.2, 0.8))

st.set_page_config(page_title="Predykcje Piłkarskie Top 5", layout="wide")

# ===========================================================================
# MAPOWANIA NAZW (skrócone dla czytelności – zachowaj swoje pełne mapowanie)
# ===========================================================================
def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return str(name)
    name = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode("ASCII")
    for suffix in [" FC", " CF", " UD", " CD", " RCD", " AS", " Stade", " 1."]:
        name = name.replace(suffix, "")
    return name.strip()

# Tutaj wstaw swoje pełne mapowanie NAZWY_MAP z poprzedniej wersji
# (zachowuję miejsce, ale w praktyce wklej cały słownik)
NAZWY_MAP = {
    # ... twój pełny słownik z poprzedniej odpowiedzi ...
    # Dla oszczędności miejsca nie kopiuję go ponownie, ale w kodzie musi być
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
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            liga        TEXT,
            kolejnosc   INTEGER,
            typ_kuponu  TEXT,
            mecze_json  TEXT,
            zdarzenia_json TEXT,
            ako         REAL,
            p_combo     REAL,
            data        TEXT
        )
    """)
    con.commit()
    con.close()

# ===========================================================================
# ŁADOWANIE DANYCH (bez zmian)
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
# STATYSTYKI (oblicz_lambdy rozszerzone o lam_r, lam_k)
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
# MODEL POISSONA
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
# GENERATOR KUPONÓW – NOWA INTELIGENTNA WERSJA
# ===========================================================================
def generuj_wszystkie_zdarzenia(mecze_list, srednie_df, srednie_lig, forma_dict, rho):
    """
    Generuje wszystkie możliwe zdarzenia (typy, overy, undery) z ich fair odds.
    Zwraca listę słowników: {"mecz": str, "typ": str, "fair_odds": float, "p": float, "kategoria": str}
    """
    zdarzenia = []
    
    for m in mecze_list:
        h, a = m['home'], m['away']
        if h not in srednie_df.index or a not in srednie_df.index:
            continue
            
        lam_h, lam_a, lam_r, lam_k = oblicz_lambdy(h, a, srednie_df, srednie_lig, forma_dict)
        pred = predykcja_meczu(lam_h, lam_a, rho=rho)
        
        # 1. Typ meczu (1/X/2/1X/X2)
        for typ in ["1", "X", "2", "1X", "X2"]:
            if typ == "1": p = pred["p_home"]
            elif typ == "X": p = pred["p_draw"]
            elif typ == "2": p = pred["p_away"]
            elif typ == "1X": p = pred["p_home"] + pred["p_draw"]
            else: p = pred["p_away"] + pred["p_draw"]
            
            if p >= 0.45:  # tylko sensowne typy
                fair = 1/p
                if MIN_SINGLE_ODDS <= fair <= MAX_SINGLE_ODDS:
                    zdarzenia.append({
                        "mecz": f"{h} - {a}",
                        "home": h, "away": a,
                        "typ": typ,
                        "fair_odds": round(fair, 2),
                        "p": round(p, 3),
                        "kategoria": "mecz"
                    })
        
        # 2. Over/Under gole
        for linia in [1.5, 2.5, 3.5]:
            mg = int(np.clip(np.ceil(max(lam_h, lam_a) + 4), 6, 10))
            M = dixon_coles_adj(
                np.outer(poisson.pmf(range(mg), lam_h), poisson.pmf(range(mg), lam_a)),
                lam_h, lam_a, rho=rho
            )
            p_over = float(sum(M[i, j] for i in range(mg) for j in range(mg) if i + j > linia))
            p_under = 1 - p_over
            
            for typ, p_val in [("Over", p_over), ("Under", p_under)]:
                if p_val >= 0.55:
                    fair = 1/p_val
                    if MIN_SINGLE_ODDS <= fair <= MAX_SINGLE_ODDS:
                        zdarzenia.append({
                            "mecz": f"{h} - {a}",
                            "home": h, "away": a,
                            "typ": f"{typ} {linia}",
                            "fair_odds": round(fair, 2),
                            "p": round(p_val, 3),
                            "kategoria": "gole"
                        })
        
        # 3. Over/Under rożne
        for linia in [7.5, 8.5, 9.5]:
            p_over = 1 - poisson.cdf(int(linia), lam_r)
            if p_over >= 0.55:
                fair = 1/p_over
                if MIN_SINGLE_ODDS <= fair <= MAX_SINGLE_ODDS:
                    zdarzenia.append({
                        "mecz": f"{h} - {a}",
                        "home": h, "away": a,
                        "typ": f"Over {linia} rożnych",
                        "fair_odds": round(fair, 2),
                        "p": round(p_over, 3),
                        "kategoria": "rozne"
                    })
        
        # 4. Over/Under kartki
        for linia in [3.5, 4.5]:
            p_over = 1 - poisson.cdf(int(linia), lam_k)
            if p_over >= 0.55:
                fair = 1/p_over
                if MIN_SINGLE_ODDS <= fair <= MAX_SINGLE_ODDS:
                    zdarzenia.append({
                        "mecz": f"{h} - {a}",
                        "home": h, "away": a,
                        "typ": f"Over {linia} kartek",
                        "fair_odds": round(fair, 2),
                        "p": round(p_over, 3),
                        "kategoria": "kartki"
                    })
        
        # 5. BTTS
        mg = int(np.clip(np.ceil(max(lam_h, lam_a) + 4), 6, 10))
        M_btts = dixon_coles_adj(
            np.outer(poisson.pmf(range(mg), lam_h), poisson.pmf(range(mg), lam_a)),
            lam_h, lam_a, rho=rho
        )
        p_btts = float(1 - M_btts[0,:].sum() - M_btts[:,0].sum() + M_btts[0,0])
        if p_btts >= 0.55:
            fair = 1/p_btts
            if MIN_SINGLE_ODDS <= fair <= MAX_SINGLE_ODDS:
                zdarzenia.append({
                    "mecz": f"{h} - {a}",
                    "home": h, "away": a,
                    "typ": "BTTS",
                    "fair_odds": round(fair, 2),
                    "p": round(p_btts, 3),
                    "kategoria": "btts"
                })
    
    return zdarzenia

def generuj_kupon_zloty_srodek(zdarzenia, target_ako=TARGET_AKO_MEDIUM):
    """
    Tworzy kupon z AKO ~target_ako, każde zdarzenie z kursem ~1.5
    """
    if len(zdarzenia) < 3:
        return None
    
    # Sortuj według odległości od idealnego kursu 1.5
    zdarzenia = sorted(zdarzenia, key=lambda x: abs(x['fair_odds'] - 1.5))
    
    # Próbuj różne kombinacje
    best_combo = None
    best_ako_diff = float('inf')
    
    for n in range(3, min(6, len(zdarzenia))):
        for i in range(len(zdarzenia) - n + 1):
            combo = zdarzenia[i:i+n]
            ako = np.prod([z['fair_odds'] for z in combo])
            ako_diff = abs(ako - target_ako)
            
            if ako_diff < best_ako_diff:
                # Sprawdź czy każdy kurs w dopuszczalnym zakresie
                if all(MIN_SINGLE_ODDS <= z['fair_odds'] <= MAX_SINGLE_ODDS for z in combo):
                    best_ako_diff = ako_diff
                    best_combo = combo
    
    if best_combo:
        return {
            "typ": "🎯 ZŁOTY ŚRODEK",
            "zdarzenia": best_combo,
            "ako": round(np.prod([z['fair_odds'] for z in best_combo]), 2),
            "p_combo": round(np.prod([z['p'] for z in best_combo]), 3),
            "opis": f"{len(best_combo)} zdarzeń z kursami ~1.5, AKO ~5.0",
            "ryzyko": "Średnie"
        }
    return None

def generuj_kupon_tasma(zdarzenia, target_ako=TARGET_AKO_TAPE):
    """
    Tworzy kupon "taśmę" – dużo zdarzeń (6-8) z wysokim prawdopodobieństwem, AKO 15-30
    """
    if len(zdarzenia) < 6:
        return None
    
    # Weź zdarzenia z najwyższym prawdopodobieństwem (ale nie za niskie kursy)
    wysokie_p = [z for z in zdarzenia if z['p'] >= 0.65 and z['fair_odds'] <= 2.0]
    wysokie_p = sorted(wysokie_p, key=lambda x: -x['p'])
    
    if len(wysokie_p) < 6:
        return None
    
    # Wybierz 6-8 zdarzeń
    for n in [8, 7, 6]:
        if len(wysokie_p) >= n:
            combo = wysokie_p[:n]
            ako = np.prod([z['fair_odds'] for z in combo])
            
            if 10 <= ako <= 35:  # akceptowalny zakres
                return {
                    "typ": "📼 TAŚMA",
                    "zdarzenia": combo,
                    "ako": round(ako, 2),
                    "p_combo": round(np.prod([z['p'] for z in combo]), 5),
                    "opis": f"{n} zdarzeń z wysokim prawdopodobieństwem (śr. {np.mean([z['p'] for z in combo]):.0%})",
                    "ryzyko": "Niskie" if ako < 15 else "Średnie"
                }
    return None

def generuj_kupon_value(zdarzenia):
    """
    Wybiera 3 zdarzenia z największym value (fair_odds / MARKET_ODDS_BENCHMARK - 1)
    """
    for z in zdarzenia:
        z['value'] = (z['fair_odds'] / MARKET_ODDS_BENCHMARK) - 1
    
    value_bety = sorted([z for z in zdarzenia if z['value'] > 0.05], 
                        key=lambda x: -x['value'])
    
    if len(value_bety) >= 3:
        combo = value_bety[:3]
        return {
            "typ": "💰 VALUE PACK",
            "zdarzenia": combo,
            "ako": round(np.prod([z['fair_odds'] for z in combo]), 2),
            "p_combo": round(np.prod([z['p'] for z in combo]), 3),
            "opis": f"3 zdarzenia z najwyższym value (śr. +{(np.mean([z['value'] for z in combo])*100):.0f}%)",
            "ryzyko": "Średnie",
            "srednie_value": f"+{np.mean([z['value'] for z in combo])*100:.0f}%"
        }
    return None

def generuj_wszystkie_kupony(mecze_list, srednie_df, srednie_lig, forma_dict, rho):
    """Generuje wszystkie trzy typy kuponów"""
    zdarzenia = generuj_wszystkie_zdarzenia(mecze_list, srednie_df, srednie_lig, forma_dict, rho)
    
    kupony = []
    
    zloty = generuj_kupon_zloty_srodek(zdarzenia)
    if zloty:
        kupony.append(zloty)
    
    tasma = generuj_kupon_tasma(zdarzenia)
    if tasma:
        kupony.append(tasma)
    
    value = generuj_kupon_value(zdarzenia)
    if value:
        kupony.append(value)
    
    return kupony

def zapisz_kupon_do_db(kupon, liga, kolejnosc):
    init_db()
    con = sqlite3.connect(DB_FILE)
    try:
        con.execute(
            """INSERT INTO kupony
               (liga, kolejnosc, typ_kuponu, mecze_json, zdarzenia_json, ako, p_combo, data)
               VALUES (?,?,?,?,?,?,?,?)""",
            (liga, kolejnosc, kupon['typ'],
             json.dumps([f"{z['home']} - {z['away']}" for z in kupon['zdarzenia']], ensure_ascii=False),
             json.dumps(kupon['zdarzenia'], ensure_ascii=False, default=str),
             kupon['ako'], kupon.get('p_combo'),
             datetime.now().strftime("%Y-%m-%d"))
        )
        con.commit()
    except Exception as e:
        st.warning(f"Nie udało się zapisać kuponu: {e}")
    finally:
        con.close()

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
st.markdown("Dixon-Coles • Blend sezonów • Forma • Fair Odds • Systemy AKO")

if not historical.empty:
    srednie_df  = oblicz_wszystkie_statystyki(historical.to_json())
    srednie_lig = oblicz_srednie_ligowe(historical.to_json())
    forma_dict  = oblicz_forme(historical)
    tabela      = tabela_ligowa(historical)
    rho         = srednie_lig["rho"]
    n_biezacy   = srednie_lig["n_biezacy"]
    w_prev      = waga_poprzedniego(n_biezacy)

    # =========================================================================
    # TABS
    # =========================================================================
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🎯 Bet Builder",
        "⚽ Przewidywane Wyniki",
        "🎲 Systemy AKO",
        "✅ Weryfikacja",
        "📊 Tabela i Forma",
        "🏆 Power Rating",
        "📈 Model",
        "🔧 Debug"
    ])

    # =========================================================================
    # TAB 1 – BET BUILDER (z fair odds i value)
    # =========================================================================
    with tab1:
        st.subheader("🎛️ Bet Builder – znajdź value")
        st.caption("Dla każdej linii pokazujemy fair odds i porównanie z typowym kursem rynkowym (1.90)")

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
                
                # Dla każdego meczu pokaż fair odds dla wybranych linii
                for _, mecz in mecze.iterrows():
                    h = map_nazwa(mecz["home_team"])
                    a = map_nazwa(mecz["away_team"])
                    if h not in srednie_df.index or a not in srednie_df.index:
                        continue
                    
                    lam_h, lam_a, lam_r, lam_k = oblicz_lambdy(h, a, srednie_df, srednie_lig, forma_dict)
                    
                    with st.expander(f"{h} vs {a}"):
                        col_left, col_right = st.columns(2)
                        
                        with col_left:
                            # Gole
                            mg = int(np.clip(np.ceil(max(lam_h, lam_a) + 4), 6, 10))
                            M = dixon_coles_adj(
                                np.outer(poisson.pmf(range(mg), lam_h), poisson.pmf(range(mg), lam_a)),
                                lam_h, lam_a, rho=rho
                            )
                            p_g = float(sum(M[i, j] for i in range(mg) for j in range(mg) if i + j > linia_gole))
                            if typ_gole == "Under":
                                p_g = 1 - p_g
                            fair_g = 1/p_g if p_g > 0 else 99
                            value_g = (fair_g / MARKET_ODDS_BENCHMARK) - 1
                            
                            st.markdown(f"**⚽ {typ_gole} {linia_gole}**")
                            st.markdown(f"p = {p_g:.1%} | fair odds = {fair_g:.2f}")
                            if value_g > 0:
                                st.markdown(f"✅ Value: +{value_g*100:.0f}%")
                            else:
                                st.markdown(f"❌ Brak value")
                        
                        with col_right:
                            # Rożne
                            p_r = oblicz_p(typ_rogi, linia_rogi, lam_r)
                            fair_r = 1/p_r if p_r > 0 else 99
                            value_r = (fair_r / MARKET_ODDS_BENCHMARK) - 1
                            
                            st.markdown(f"**🚩 {typ_rogi} {linia_rogi} rożnych**")
                            st.markdown(f"p = {p_r:.1%} | fair odds = {fair_r:.2f}")
                            if value_r > 0:
                                st.markdown(f"✅ Value: +{value_r*100:.0f}%")
                            
                            # Kartki
                            p_k = oblicz_p(typ_kartki, linia_kartki, lam_k)
                            fair_k = 1/p_k if p_k > 0 else 99
                            value_k = (fair_k / MARKET_ODDS_BENCHMARK) - 1
                            
                            st.markdown(f"**🟨 {typ_kartki} {linia_kartki} kartek**")
                            st.markdown(f"p = {p_k:.1%} | fair odds = {fair_k:.2f}")
                            if value_k > 0:
                                st.markdown(f"✅ Value: +{value_k*100:.0f}%")
                        
                        # Combo
                        p_combo = p_g * p_r * p_k
                        if p_combo >= min_prob:
                            st.divider()
                            st.markdown(f"### 🎯 Combo AKO = {1/p_combo:.2f}")
                            st.markdown(f"p(combo) = {p_combo:.2%}")
            else:
                st.info("Brak nadchodzących meczów.")

    # =========================================================================
    # TAB 2 – PRZEWIDYWANE WYNIKI (skrócone dla miejsca)
    # =========================================================================
    with tab2:
        st.subheader("⚽ Przewidywane wyniki – najbliższa kolejka")
        # ... (zachowaj swoją istniejącą implementację z TAB2)
        # Dla oszczędności miejsca nie kopiuję całego kodu, ale w praktyce tu wklej swój kod z TAB2

    # =========================================================================
    # TAB 3 – SYSTEMY AKO (NOWA, ULEPSZONA WERSJA)
    # =========================================================================
    with tab3:
        st.subheader("🎲 Systemy AKO – gotowe kupony na kolejkę")
        st.caption("Automatycznie generowane kupony z optymalnym AKO i value")

        if not schedule.empty and not srednie_df.empty:
            dzisiaj = datetime.now().date()
            przyszle = schedule[schedule["date"].dt.date >= dzisiaj]
            if not przyszle.empty:
                nb = przyszle["round"].min()
                mecze_list = []
                for _, mecz in schedule[schedule["round"] == nb].iterrows():
                    h = map_nazwa(mecz["home_team"])
                    a = map_nazwa(mecz["away_team"])
                    if h in srednie_df.index and a in srednie_df.index:
                        mecze_list.append({'home': h, 'away': a})

                if mecze_list:
                    kupony = generuj_wszystkie_kupony(mecze_list, srednie_df, srednie_lig, forma_dict, rho)

                    col_save, _ = st.columns([1, 3])
                    with col_save:
                        if st.button("💾 Zapisz kupony do bazy", use_container_width=True):
                            for k in kupony:
                                zapisz_kupon_do_db(k, wybrana_liga, int(nb))
                            st.success("Kupony zapisane!")

                    for kupon in kupony:
                        with st.container(border=True):
                            cols = st.columns([3, 1, 1, 2])
                            with cols[0]:
                                st.markdown(f"**{kupon['typ']}**")
                                for z in kupon['zdarzenia']:
                                    st.markdown(f"• {z['mecz']}: **{z['typ']}** @ {z['fair_odds']}")
                                st.caption(kupon['opis'])
                            with cols[1]:
                                st.metric("AKO", f"{kupon['ako']:.2f}")
                                if 'p_combo' in kupon:
                                    st.caption(f"p = {kupon['p_combo']:.2%}")
                            with cols[2]:
                                ryzyko_color = {"Niskie": "🟢", "Średnie": "🟡", "Wysokie": "🔴"}.get(kupon['ryzyko'], "⚪")
                                st.markdown(f"**Ryzyko:** {ryzyko_color}")
                            with cols[3]:
                                if 'srednie_value' in kupon:
                                    st.metric("Śr. value", kupon['srednie_value'])
                else:
                    st.info("Brak meczów do analizy.")
            else:
                st.info("Brak nadchodzących meczów.")
        else:
            st.warning("Brak danych.")

        # Podgląd zapisanych kuponów
        with st.expander("📊 Historia kuponów", expanded=False):
            init_db()
            con = sqlite3.connect(DB_FILE)
            wszystkie = con.execute(
                "SELECT liga, kolejnosc, typ_kuponu, ako, p_combo, data FROM kupony ORDER BY kolejnosc DESC"
            ).fetchall()
            con.close()
            
            if wszystkie:
                df_kup = pd.DataFrame(wszystkie, columns=["Liga","Kolejka","Typ","AKO","p_combo","Data"])
                st.dataframe(df_kup, use_container_width=True)
            else:
                st.info("Brak zapisanych kuponów.")

    # =========================================================================
    # POZOSTAŁE ZAKŁADKI (TAB4-TAB8) – zachowaj swoje istniejące implementacje
    # =========================================================================
    # Dla oszczędności miejsca pomijam je, ale w praktyce wklej swój kod

else:
    st.error("Nie udało się pobrać danych historycznych.")