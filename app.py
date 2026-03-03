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
except ImportError:
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
    "Real Oviedo":             "Oviedo",
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
"Borussia M'gladbach": "M'gladbach",
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
        # Kursy bukmacherów – zachowaj B365 i Pinnacle do analizy ROI
        for col in ["B365H","B365D","B365A","PSH","PSD","PSA","BbAvH","BbAvD","BbAvA"]:
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

@st.cache_data(ttl=86400)
def load_schedule(filename: str) -> pd.DataFrame:
    """Ładuje terminarz z pliku CSV i używa istniejącej kolumny round"""
    try:
        df = pd.read_csv(filename)
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
        
        # Sprawdź czy kolumna 'round' istnieje w pliku
        if "round" in df.columns:
            # Użyj istniejącej kolumny round, upewnij się że to liczby całkowite
            df["round"] = pd.to_numeric(df["round"], errors="coerce").fillna(0).astype(int)
        else:
            # Jeśli nie ma kolumny 'round', wygeneruj ją (awaryjnie)
            st.warning(f"⚠️ Brak kolumny round w pliku {filename}, generuję automatycznie")
            df = df.sort_values("date")
            df["date_only"] = df["date"].dt.date
            unique_dates = sorted(df["date_only"].unique())
            date_to_round = {date: i+1 for i, date in enumerate(unique_dates)}
            df["round"] = df["date_only"].map(date_to_round)
            df = df.drop("date_only", axis=1)
        
        # Sortuj według daty dla pewności
        df = df.sort_values("date").reset_index(drop=True)
        
        return df.dropna(subset=["date"])
    except Exception as e:
        st.error(f"Problem z plikiem terminarza {filename}: {e}")
        return pd.DataFrame()

# ===========================================================================
# FUNKCJE POMOCNICZE DLA KOLEJEK
# ===========================================================================
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

    Asymetryczny shrink lambdy (potwierdzone backtestem na 3 datasetach):
      ALPHA_LAM_OFF = 0.10  – atak jest zmienny → shrinkujemy mocniej
      ALPHA_LAM_DEF = 0.20  – obrona jest stabilna → shrinkujemy słabiej
    Wynik: +1.5–2.6pp hit rate, +2–5pp ROI, Brier -0.002 na wszystkich 3 datasetach.
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
    if p_home >= PROG_PEWNY: return "1",  p_home
    if p_away >= PROG_PEWNY: return "2",  p_away
    if p_1x >= PROG_PODWOJNA or p_x2 >= PROG_PODWOJNA:
        return ("1X", p_1x) if p_1x >= p_x2 else ("X2", p_x2)
    probs = {"1": p_home, "X": p_draw, "2": p_away}
    t = max(probs, key=probs.get)
    return t, probs[t]

# ── Market Noise & Kelly helpers ──────────────────────────────────────
MARKET_NOISE_MAX = 0.25
KELLY_FRACTION   = 0.25
KELLY_BANKROLL_DEFAULT = 1000.0

def market_noise_check(p_model, p_impl):
    diff = abs(p_model - p_impl)
    noise = diff > MARKET_NOISE_MAX
    kierunek = "Model wyzej niz rynek" if p_model > p_impl else "Rynek wyzej niz model"
    kolor = "#F44336" if noise else ("#FF9800" if diff > 0.15 else "#4CAF50")
    zgodnosc = max(0.0, 1.0 - diff / MARKET_NOISE_MAX)
    return {"noise": noise, "diff": diff, "kierunek": kierunek,
            "kolor": kolor, "zgodnosc_pct": zgodnosc}

def kelly_stake(p_model, kurs_buk, bankroll=KELLY_BANKROLL_DEFAULT, fraction=KELLY_FRACTION):
    try:
        b = kurs_buk - 1.0
        q = 1.0 - p_model
        if b <= 0 or p_model <= 0 or p_model >= 1:
            return {"f_full":0,"f_frac":0,"stake_pln":0,"ev_per_unit":0,"safe":False}
        f_full = max(0.0, (p_model * b - q) / b)
        f_frac = f_full * fraction
        stake  = round(bankroll * f_frac, 2)
        ev_puu = p_model * b - q
        return {"f_full":round(f_full,4),"f_frac":round(f_frac,4),
                "stake_pln":stake,"ev_per_unit":round(ev_puu,4),
                "safe": f_frac > 0 and ev_puu > 0}
    except Exception:
        return {"f_full":0,"f_frac":0,"stake_pln":0,"ev_per_unit":0,"safe":False}

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

# ── Bankroll persistence helpers ─────────────────────────────────────────
def zapisz_wynik_bankroll(liga: str, kolejnosc: int, opis: str,
                           kwota: float, typ: str = "wynik") -> None:
    """Zapisuje zmianę bankrollu (wygrana/przegrana/korekta) do DB."""
    init_db()
    con = sqlite3.connect(DB_FILE)
    try:
        # Oblicz aktualny kapitał
        row = con.execute("SELECT kapital_po FROM bankroll ORDER BY id DESC LIMIT 1").fetchone()
        kapital_przed = float(row[0]) if row else 1000.0
        kapital_po    = kapital_przed + kwota
        con.execute(
            "INSERT OR IGNORE INTO bankroll (data,liga,kolejnosc,opis,kwota,typ,kapital_po) "
            "VALUES (?,?,?,?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M"), liga, kolejnosc, opis, kwota, typ, kapital_po)
        )
        con.commit()
    except Exception:
        pass
    finally:
        con.close()

def pobierz_bankroll_history(liga: str = None) -> pd.DataFrame:
    """Zwraca historię bankrollu z DB."""
    init_db()
    con = sqlite3.connect(DB_FILE)
    q = "SELECT data, liga, kolejnosc, opis, kwota, typ, kapital_po FROM bankroll"
    params = []
    if liga:
        q += " WHERE liga=?"; params.append(liga)
    q += " ORDER BY id ASC"
    df = pd.read_sql_query(q, con, params=params)
    con.close()
    return df

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

def predykcja_meczu(lam_h: float, lam_a: float, rho: float = -0.13, csv_code: str = "E0", n_train: int = 200) -> dict:
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
            lh, la, lr, lk, _, lsot = oblicz_lambdy(h, a, srednie_df, srednie_lig, forma_dict)
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

# ── Kursy live (The Odds API) ─────────────────────────────────────────────
_CSV_CODE = LIGI[wybrana_liga]["csv_code"]
_OA_DB    = "predykcje.db"
_oa_cached = {}

st.sidebar.divider()
st.sidebar.markdown("### 💰 Kursy bukmacherskie")

def _oa_get_key():
    try: return st.secrets["ODDS_API_KEY"]
    except Exception: return st.session_state.get("_odds_key")

_oa_key = _oa_get_key()
if not _oa_key:
    _mk = st.sidebar.text_input("Klucz The Odds API", type="password",
        placeholder="lub dodaj ODDS_API_KEY do Streamlit secrets")
    if _mk:
        st.session_state["_odds_key"] = _mk
        _oa_key = _mk

if _OA_OK and _oa_key:
    _stats = _oa.get_usage_stats(_OA_DB)
    _rem   = _stats.get("requests_remaining")
    _last  = _stats.get("last_per_liga", {}).get(_CSV_CODE)
    if _rem is not None:
        _bc = "#4CAF50" if _rem > 200 else ("#FF9800" if _rem > 50 else "#F44336")
        st.sidebar.markdown(
            f"<div style='font-size:0.78em;color:#888;margin-bottom:2px'>"
            f"Pozostało requestów: <b style='color:{_bc}'>{_rem}/500</b></div>"
            f"<div style='background:#333;border-radius:3px;height:4px'>"
            f"<div style='background:{_bc};width:{min(int(_rem/5),100)}%;height:4px;border-radius:3px'></div></div>",
            unsafe_allow_html=True)
    if _last:
        try:
            _ldt = datetime.fromisoformat(_last.replace("Z","+00:00"))
            _age = (datetime.now(_ldt.tzinfo) - _ldt).total_seconds()/3600
            st.sidebar.caption(f"Ostatnia aktualizacja: {_age:.0f}h temu")
        except Exception: pass
    _cr, _cf = st.sidebar.columns(2)
    if _cr.button("🔄 Odśwież kursy", use_container_width=True):
        with st.spinner("Pobieranie kursów..."):
            _res = _oa.fetch_odds(_CSV_CODE, _OA_DB, _oa_key)
        if _res["ok"] and not _res.get("from_cache"):
            st.sidebar.success(f"✅ {_res['n_events']} meczów | zostało: {_res['requests_remaining']}")
        elif _res.get("from_cache"):
            st.sidebar.info(f"Cache świeży ({_res['age_h']}h). Następne za {_res['next_refresh_h']}h.")
        else:
            st.sidebar.error(_res["error"])
    if _cf.button("⚡ Wymuś", use_container_width=True):
        with st.spinner("Pobieranie..."):
            _res = _oa.fetch_odds(_CSV_CODE, _OA_DB, _oa_key, force=True)
        if _res["ok"]:
            st.sidebar.success(f"✅ {_res['n_events']} meczów | zostało: {_res['requests_remaining']}")
        else:
            st.sidebar.error(_res["error"])
    _oa_cached = _oa.get_cached_odds(_CSV_CODE, _OA_DB)
elif not _OA_OK:
    st.sidebar.caption("ℹ️ Plik `odds_api.py` nie znaleziony.")

# ── Kelly Bankroll input ───────────────────────────────────────────
st.sidebar.divider()
st.sidebar.markdown("### 💼 Bankroll (Kelly)")
_br_input = st.sidebar.number_input(
    "Twój bankroll (zł)", min_value=10.0, max_value=1_000_000.0,
    value=float(st.session_state.get("bankroll", 1000.0)),
    step=100.0, key="_br_widget",
    help="Kwota używana do obliczenia stawki Kelly 1/4")
st.session_state["bankroll"] = _br_input
_kelly_info_val = _br_input * 0.005
st.sidebar.caption(f"Typowa stawka (0.5% Kelly): **{_kelly_info_val:.0f} zł**")


historical = load_historical(LIGI[wybrana_liga]["csv_code"])
schedule   = load_schedule(LIGI[wybrana_liga]["file"])

# Auto-aktualizacja wynikow: przy kazdym wczytaniu sprawdz nowe wyniki w CSV
# Uzywa session_state zeby nie robic pelnego skanu przy kazdym rerunie
_auto_update_key = f"autoupd_{wybrana_liga}_{len(historical)}"
if _auto_update_key not in st.session_state:
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
    st.session_state[_auto_update_key] = _n_updated
    if _n_updated > 0:
        st.toast(f"✅ Auto-zaktualizowano wyniki {_n_updated} meczów z football-data.co.uk", icon="⚽")

# ── Nagłówek główny ────────────────────────────────────────────────────────
hcol1, hcol2 = st.columns([6, 2])
with hcol1:
    st.title(f"⚽ {wybrana_liga}")
    st.caption(f"Dixon-Coles · SOT blend · Brier tracking · Kalibracja modelu")
with hcol2:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    if not schedule.empty:
        aktualna_kolejka = get_current_round(schedule)
        liczba_meczy = len(schedule[schedule["round"] == aktualna_kolejka])
        status = get_round_status(schedule, aktualna_kolejka)
        st.metric(f"{status} kolejka", f"#{aktualna_kolejka}", f"{liczba_meczy} meczów")

if not historical.empty:
    srednie_df  = oblicz_wszystkie_statystyki(historical.to_json())
    srednie_lig = oblicz_srednie_ligowe(historical.to_json())
    forma_dict  = oblicz_forme(historical)
    tabela      = tabela_ligowa(historical)
    rho         = srednie_lig["rho"]
    n_biezacy   = srednie_lig["n_biezacy"]
    w_prev      = waga_poprzedniego(n_biezacy)

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

    # Sidebar: info o danych
    st.sidebar.divider()
    st.sidebar.caption(f"📅 Sezon 2025/26 · {n_biezacy} meczów w bazie")
    _shrink_now = _get_shrink(LIGI[wybrana_liga]["csv_code"], n_biezacy)
    _shrink_base = KALIBRACJA_PER_LIGA.get(LIGI[wybrana_liga]["csv_code"], SHRINK_ALPHA)
    _shrink_bonus = _shrink_now - _shrink_base
    _shrink_info = f"shrink {_shrink_now:.2f}" + (f" (+{_shrink_bonus:.2f} sezón)" if _shrink_bonus > 0.01 else "")
    st.sidebar.caption(f"ρ Dixon-Coles: `{rho:.4f}` · w_prev: `{w_prev:.2f}` · {_shrink_info}")
    
    if not schedule.empty:
        aktualna_kolejka = get_current_round(schedule)
        wszystkie_kolejki = sorted(schedule["round"].unique())
        pozycja = wszystkie_kolejki.index(aktualna_kolejka) + 1 if aktualna_kolejka in wszystkie_kolejki else 0
        st.sidebar.progress(pozycja / len(wszystkie_kolejki), 
                           text=f"Kolejka {pozycja}/{len(wszystkie_kolejki)}")
        st.sidebar.info(f"⚽ Aktualna kolejka: **#{aktualna_kolejka}**")

    # ── Model Sharpness vs Rynek (sidebar) – tu historical i srednie_df są dostępne
    if _OA_OK and _oa_key and _oa_cached and not schedule.empty:
        _ms = model_sharpness_vs_rynek(
            wybrana_liga, _oa_cached, _oa,
            schedule, srednie_df, srednie_lig, forma_dict,
            rho, n_biezacy, LIGI[wybrana_liga]["csv_code"])
        if _ms:
            st.sidebar.divider()
            st.sidebar.markdown("### 🎯 Model Sharpness")
            _ad = _ms["avg_diff"]
            _sc = "#4CAF50" if 0.05 <= _ad <= 0.15 else ("#FF9800" if _ad < 0.05 else "#F44336")
            _label = "✅ Sweet Spot" if 0.05 <= _ad <= 0.15 else ("⚠️ Zbyt blisko rynku" if _ad < 0.05 else "⚠️ Zbyt daleko")
            st.sidebar.markdown(
                f"<div style='background:#0e1117;border:1px solid #2a2a3a;border-radius:8px;padding:10px'>"
                f"<div style='font-size:1.3em;font-weight:bold;color:{_sc};text-align:center'>"
                f"{_ad:.1%}</div>"
                f"<div style='font-size:0.78em;color:{_sc};text-align:center;margin-bottom:6px'>"
                f"{_label}</div>"
                f"<div style='display:flex;justify-content:space-around;font-size:0.72em;color:#666'>"
                f"<div>🔵 Zbieżne<br><b style='color:#888'>{_ms['aligned']}</b></div>"
                f"<div>🟢 Sweet<br><b style='color:#4CAF50'>{_ms['sweet']}</b></div>"
                f"<div>🔴 Noise<br><b style='color:#F44336'>{_ms['noise']}</b></div>"
                f"</div></div>",
                unsafe_allow_html=True)
            st.sidebar.caption("Sweet spot: różnica 5–15% model vs rynek")

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
                    _sh, _sa, srednie_df, srednie_lig, forma_dict)
                _sp = predykcja_meczu(_slh, _sla, rho=rho,
                                       csv_code=LIGI[wybrana_liga]["csv_code"], n_train=n_biezacy)
                _so = _oa.znajdz_kursy(_sh, _sa, _oa_cached) if _OA_OK and _oa_key else None
                _skdc = None
                if _so:
                    _skdc, _sidc = _kurs_dc_live(_sp["typ"], _so["odds_h"], _so["odds_d"], _so["odds_a"])
                _sev = _sp["p_typ"] * (_skdc or _sp["fo_typ"]) - 1
                _smn = market_noise_check(_sp["p_typ"], _sidc) if _so and _skdc else None
                _start_top.append({
                    "mecz": f"{_sh} – {_sa}",
                    "typ": _sp["typ"], "p": _sp["p_typ"],
                    "fo": _sp["fo_typ"], "kurs_buk": _skdc,
                    "ev": _sev, "is_val": _sev >= 0.04,
                    "noise": _smn["noise"] if _smn else False,
                    "data": _sm.get("date",""),
                })
            except Exception: continue
        _top_val = sorted([x for x in _start_top if x["is_val"] and not x["noise"]],
                          key=lambda x: -x["ev"])[:4]
        if _top_val:
            st.markdown("---")
            st.markdown("### 🏆 Najlepsze okazje tej kolejki")
            _tv_cols = st.columns(len(_top_val))
            for _tvi, (_tvc, _tv) in enumerate(zip(_tv_cols, _top_val)):
                _kb = st.session_state.get("bankroll", 1000.0)
                _kl = kelly_stake(_tv["p"], _tv["kurs_buk"] or _tv["fo"], bankroll=_kb)
                _kurs_str = f"{_tv['kurs_buk']:.2f}" if _tv["kurs_buk"] else f"{_tv['fo']:.2f}✦"
                _tvc.markdown(
                    f"<div style='background:#0a1628;border:2px solid #2a6b2a;"
                    f"border-radius:10px;padding:12px;text-align:center'>"
                    f"<div style='font-size:0.74em;color:#888;margin-bottom:4px'>"
                    f"{str(_tv['data'])[:10] if _tv['data'] else ''}</div>"
                    f"<div style='font-weight:bold;color:#fff;font-size:0.88em;margin-bottom:6px'>"
                    f"{_tv['mecz']}</div>"
                    f"<div style='font-size:1.1em;font-weight:bold;color:#4CAF50'>"
                    f"{_tv['typ']} @ {_kurs_str}</div>"
                    f"<div style='font-size:0.82em;color:#aaa;margin-top:4px'>"
                    f"P model: {_tv['p']:.0%} · EV: <b style='color:#4CAF50'>{_tv['ev']:+.1%}</b></div>"
                    f"{'<div style="font-size:0.78em;color:#4CAF50;margin-top:3px">💰 Kelly: ' + str(int(_kl['stake_pln'])) + ' zł</div>' if _kl['safe'] else ''}"
                    f"</div>",
                    unsafe_allow_html=True)
            st.markdown("---")

    # TABS
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "⚽ Analiza Meczu",
        "📊 Ranking Zdarzeń",
        "🔬 Deep Data",
        "📈 Skuteczność + ROI",
        "📉 Kalibracja",
        "🎛️ Laboratorium",
        "🧪 Backtest",
    ])

    # =========================================================================
    # TAB 2 – RANKING ZDARZEŃ (przeniesiony)
    # =========================================================================
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
                    
                    lam_h, lam_a, lam_r, lam_k, sot_ok, lam_sot = oblicz_lambdy(h, a, srednie_df, srednie_lig, forma_dict)
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

                    # Typ główny – dopisz kurs live i kelly
                    if pred["p_typ"] >= 0.58 and pred["fo_typ"] >= 1.30:
                        ev = _ev(pred["p_typ"], pred["fo_typ"])
                        # Pobierz kurs bukmachera jeśli dostępny
                        _kbuk = None
                        if _kurs_live_1x2:
                            _kbuk, _ = _kurs_dc_live(pred["typ"],
                                _kurs_live_1x2["odds_h"], _kurs_live_1x2["odds_d"], _kurs_live_1x2["odds_a"])
                        _br_t2 = st.session_state.get("bankroll", KELLY_BANKROLL_DEFAULT)
                        _kel = kelly_stake(pred["p_typ"], _kbuk if _kbuk else pred["fo_typ"],
                                           bankroll=_br_t2)
                        wszystkie_zd.append({
                            "Mecz": mecz_str,
                            "Rynek": "1X2",
                            "Typ": pred["typ"],
                            "P": pred["p_typ"],
                            "Fair": pred["fo_typ"],
                            "KursBuk": _kbuk,
                            "EV": ev,
                            "Kelly_stake": _kel["stake_pln"] if _kel["safe"] else None,
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
                                "KursBuk": None,
                                "EV": ev,
                                "Kelly_stake": None,
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

                # VALUE BETS
                st.markdown("### 🔥 Value Bets (EV > 0)")
                value_bets = df_rank[df_rank["EV"] > 0].sort_values("EV", ascending=False)
                if not value_bets.empty:
                    for _, row in value_bets.head(10).iterrows():
                        ev_color = "#4CAF50" if row["EV"] > 0.05 else "#FF9800"
                        _ks = row.get("Kelly_stake")
                        _kb = row.get("KursBuk")
                        cols = st.columns([3, 1, 1, 1, 1, 1])
                        with cols[0]:
                            st.markdown(f"**{row['Mecz']}**")
                            st.caption(f"{row['Typ']}")
                        with cols[1]:
                            st.markdown(f"`{row['Rynek']}`")
                        with cols[2]:
                            st.markdown(f"🎯 {row['P']:.0%}")
                        with cols[3]:
                            kurs_disp = f"{_kb:.2f}" if _kb else f"{row['Fair']:.2f}✦"
                            st.markdown(f"💰 {kurs_disp}")
                        with cols[4]:
                            st.markdown(f"<span style='color:{ev_color};font-weight:bold'>EV {row['EV']:+.3f}</span>",
                                        unsafe_allow_html=True)
                        with cols[5]:
                            if _ks:
                                st.markdown(f"<span style='color:#4CAF50;font-weight:bold'>🏦 {_ks:.0f} zł</span>",
                                            unsafe_allow_html=True)
                            else:
                                st.markdown("<span style='color:#444'>–</span>", unsafe_allow_html=True)
                        st.divider()
                else:
                    st.info("Brak value bets w tej kolejce")

                # SAFE HAVEN + SHOT KINGS → zwijana lista na dole
                with st.expander("🛡️ Safe Haven & 🎯 Shot Kings", expanded=False):
                    safe_havens = df_rank[df_rank["P"] > 0.70].sort_values("P", ascending=False)
                    st.markdown("**🛡️ Safe Haven** – zdarzenia z p > 70%")
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
                    st.divider()
                    st.markdown("**🎯 Shot Kings** – mecze z wysokim oczekiwanym SOT")
                    if shot_kings:
                        shot_df = pd.DataFrame(shot_kings).sort_values("Oczekiwane SOT", ascending=False)
                        for _, row in shot_df.head(10).iterrows():
                            cols = st.columns([3, 1, 2])
                            with cols[0]:
                                st.markdown(f"**{row['Mecz']}**")
                            with cols[1]:
                                st.markdown(f"🎯 {row['Oczekiwane SOT']} SOT")
                            with cols[2]:
                                if row["Over 5.5 SOT"] > 0:
                                    st.markdown(f"Over 5.5: {row['Over 5.5 SOT']:.0%} (fair {row['Fair']:.2f})")
                            st.divider()
                    else:
                        st.info("Brak zdarzeń Shot Kings z fair odds ≥ 1.30")

                # PEŁNY RANKING – na samym dole
                with st.expander("📋 Pełny ranking wszystkich zdarzeń", expanded=False):
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
                            format="%.2f",
                            key="full_rank_slider"
                        )
                    with col_f3:
                        only_value_full = st.checkbox("Tylko EV>0", key="full_rank_value")
                    
                    df_full = df_rank.copy()
                    if filtr_rynek != "Wszystkie":
                        df_full = df_full[df_full["Rynek"] == filtr_rynek]
                    df_full = df_full[df_full["P"] >= prog_filtr]
                    
                    if only_value_full:
                        df_full = df_full[df_full["EV"] > 0]
                    
                    df_full = df_full.sort_values("P", ascending=False)
                    
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
                            file_name=f"ranking_full_kolejka{aktualna_kolejka}.csv"
                        )
                    else:
                        st.info("Brak zdarzeń po zastosowaniu filtrów")
            else:
                st.info("Brak zdarzeń spełniających kryteria")
        else:
            st.warning("Brak danych")

    # =========================================================================
    # TAB 1 – ANALIZA MECZU (przeniesiona na pierwsze miejsce)
    # =========================================================================
    with tab1:
        st.subheader("⚽ Analiza meczu – aktualna kolejka")
        st.caption("Szczegółowa analiza każdego meczu. Rozwiń mecz → sprawdź rynki → zapisz do trackingu.")

        with st.expander("ℹ️ Jak działa tracking skuteczności?", expanded=False):
            st.markdown("""
**Workflow w 3 krokach:**

**Krok 1 →** Przed meczami: włącz przełącznik *💾 Zapisz zdarzenia* poniżej.  
Model zapisze wszystkie predykcje do bazy (1X2 + rynki alternatywne).

**Krok 2 →** Poczekaj na wyniki meczów.

**Krok 3 →** Po meczach: kliknij *🔄 Aktualizuj wyniki*.  
System automatycznie porówna predykcje z wynikami i wyliczy skuteczność.

Dane trafią do zakładki **📈 Skuteczność + ROI** i **📉 Kalibracja**.
            """)

        tgl1, tgl2 = st.columns(2)
        with tgl1: pokaz_komentarz = st.toggle("💬 Komentarz", value=True)
        with tgl2: pokaz_macierz  = st.toggle("🔢 Macierz wyników", value=False)

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
                        f"<div style='background:linear-gradient(90deg,#1a1a2e 0%,#16213e 100%);"
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

                        lam_h, lam_a, lam_r, lam_k, sot_ok, lam_sot = oblicz_lambdy(h, a, srednie_df, srednie_lig, forma_dict)
                        pred = predykcja_meczu(lam_h, lam_a, rho=rho, csv_code=LIGI[wybrana_liga]["csv_code"], n_train=n_biezacy)
                        data_meczu = mecz["date"].strftime("%d.%m %H:%M") if pd.notna(mecz["date"]) else ""

                        kolumna = kol_a if idx % 2 == 0 else kol_b
                        conf_i  = "🟢" if pred["conf_level"]=="High" else ("🟡" if pred["conf_level"]=="Medium" else "🔴")
                        sot_badge = " 🎯SOT" if sot_ok else ""

                        sedzia = mecz.get("Referee", "Nieznany") if "Referee" in mecz else "Nieznany"
                        sedzia_ostr = ostrzezenie_sedziego(sedzia, sedziowie_df)

                        label_t2 = (f"{conf_i} {h} vs {a}{sot_badge}"
                                    f"  ·  {ikony_t.get(pred['typ'],'⚪')} {pred['typ']} @ {pred['fo_typ']:.2f}"
                                    f"  ·  {data_meczu}")
                        with kolumna:
                            with st.expander(label_t2, expanded=False):
                                ch, cmid, ca = st.columns([5,2,5])
                                with ch: st.markdown(f"<div style='font-weight:bold'>{h}</div>", unsafe_allow_html=True)
                                with cmid: st.markdown(f"<div style='text-align:center;color:#888'>{data_meczu}</div>", unsafe_allow_html=True)
                                with ca: st.markdown(f"<div style='font-weight:bold;text-align:right'>{a}</div>", unsafe_allow_html=True)

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
                                st.markdown(
                                    f"<div style='text-align:center;margin-bottom:4px'>"
                                    f"Typ: {badge_typ(pred['typ'])}&nbsp;&nbsp;"
                                    f"<span style='font-size:0.88em;color:#888'>Fair Odds: <b>{pred['fo_typ']:.2f}</b> "
                                    f"({pred['p_typ']:.1%}"
                                    f"{'±' + f"{pred['ci_half']:.0%}" if pred.get('ci_half',0)>0.01 else ''})"
                                    f"</span>"
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

                                # 1/X/2 – kompaktowy HTML, mniejsza czcionka
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

                                # ── Oczekiwane statystyki – kompaktowy pasek ────
                                _sot_d = f"{lam_sot:.1f}" if (lam_sot and lam_sot > 0) else "–"
                                st.markdown(
                                    f"<div style='display:flex;justify-content:space-around;"
                                    f"background:#1a1a2e;border-radius:6px;padding:5px 4px;margin:4px 0'>"
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
                                    st.caption(f"🟨 **Sędzia:** {sedzia}")

                                # ── Head-to-Head ─────────────────────────
                                _h2h = historical[
                                    ((historical["HomeTeam"]==h) & (historical["AwayTeam"]==a)) |
                                    ((historical["HomeTeam"]==a) & (historical["AwayTeam"]==h))
                                ].sort_values("Date", ascending=False).head(5)
                                if not _h2h.empty:
                                    with st.expander(f"📜 H2H – ostatnie {len(_h2h)} spotkania", expanded=False):
                                        _h2h_rows = []
                                        for _, _hm in _h2h.iterrows():
                                            _hg = int(_hm["FTHG"]); _ag = int(_hm["FTAG"])
                                            _hw = _hm["HomeTeam"]; _aw = _hm["AwayTeam"]
                                            _dt = _hm["Date"].strftime("%d.%m.%Y") if pd.notna(_hm["Date"]) else "?"
                                            if _hg > _ag:
                                                _res_c, _res = "#4CAF50", f"{_hg}:{_ag}"
                                            elif _hg < _ag:
                                                _res_c, _res = "#F44336", f"{_hg}:{_ag}"
                                            else:
                                                _res_c, _res = "#FF9800", f"{_hg}:{_ag}"
                                            _bold_h = "font-weight:bold" if _hw == h else ""
                                            _bold_a = "font-weight:bold" if _aw == h else ""
                                            _h2h_rows.append(
                                                f"<tr style='border-bottom:1px solid #1a1a2e'>"
                                                f"<td style='padding:4px 8px;color:#666;font-size:0.78em'>{_dt}</td>"
                                                f"<td style='padding:4px 8px;color:#ccc;font-size:0.82em;{_bold_h}'>{_hw}</td>"
                                                f"<td style='padding:4px 8px;text-align:center;color:{_res_c};"
                                                f"font-weight:bold;font-size:0.88em'>{_res}</td>"
                                                f"<td style='padding:4px 8px;color:#ccc;font-size:0.82em;{_bold_a}'>{_aw}</td>"
                                                f"</tr>"
                                            )
                                        # Bilans
                                        _h_wins = sum(1 for _, m in _h2h.iterrows()
                                                      if (m["HomeTeam"]==h and m["FTHG"]>m["FTAG"]) or
                                                         (m["AwayTeam"]==h and m["FTAG"]>m["FTHG"]))
                                        _draws  = sum(1 for _, m in _h2h.iterrows() if m["FTHG"]==m["FTAG"])
                                        _a_wins = len(_h2h) - _h_wins - _draws
                                        st.markdown(
                                            f"<div style='font-size:0.76em;color:#888;margin-bottom:4px'>"
                                            f"Bilans: <b style='color:#4CAF50'>{h} {_h_wins}W</b> · "
                                            f"<b style='color:#FF9800'>{_draws}D</b> · "
                                            f"<b style='color:#F44336'>{_a_wins}W {a}</b></div>"
                                            f"<table style='width:100%;border-collapse:collapse'>"
                                            f"{''.join(_h2h_rows)}</table>",
                                            unsafe_allow_html=True)

                                # ── Kursy live z The Odds API ────────────
                                # Due to Score flag
                                for _dts_team_name, _dts_col in [(h,"dom"),(a,"wyjazd")]:
                                    _dts = due_to_score_flag(_dts_team_name, srednie_df, historical)
                                    if _dts and _dts.get("active"):
                                        st.markdown(
                                            f"<div style='background:#1a1200;border:1px solid #FF9800;"
                                            f"border-radius:6px;padding:6px 12px;margin:3px 0;"
                                            f"font-size:0.79em'>🎯 <b>Due to Score</b> – "
                                            f"<b>{_dts_team_name}</b>: {_dts['msg']}</div>",
                                            unsafe_allow_html=True)

                                if _OA_OK and _oa_key and _oa_cached:
                                    _o = _oa.znajdz_kursy(h, a, _oa_cached)
                                    if _o:
                                        _kdc, _idc = _kurs_dc_live(pred["typ"], _o["odds_h"], _o["odds_d"], _o["odds_a"])
                                        if _kdc:
                                            _ev_val  = pred["p_typ"] * _kdc - 1
                                            _edge    = pred["p_typ"] - _idc
                                            _is_val  = _ev_val >= 0.04
                                            _ev_c    = "#4CAF50" if _is_val else ("#FF9800" if _ev_val > -0.02 else "#888")
                                            _bk_lbl  = _o.get("bookmaker","").replace("_"," ").title()
                                            _vbadge  = "&nbsp;🎯 <b>VALUE BET</b>" if _is_val else ""

                                            # Market Noise
                                            _mn = market_noise_check(pred["p_typ"], _idc)
                                            if _mn["noise"]:
                                                st.markdown(
                                                    f"<div style='background:#1a0000;border:1px solid #F44336;"
                                                    f"border-radius:5px;padding:4px 10px;margin:3px 0;"
                                                    f"font-size:0.76em;color:#F44336'>"
                                                    f"⚠️ <b>Market Noise</b> – różnica modelu vs rynek: "
                                                    f"<b>{_mn['diff']:.0%}</b> · {_mn['kierunek']}. "
                                                    f"Sprawdź skład/kontuzje!</div>",
                                                    unsafe_allow_html=True)
                                            elif _mn["diff"] > 0.15:
                                                st.markdown(
                                                    f"<div style='font-size:0.74em;color:#FF9800;margin:2px 0'>"
                                                    f"⚠️ Różnica {_mn['diff']:.0%} – sprawdź aktualności</div>",
                                                    unsafe_allow_html=True)

                                            # Zgodnosc z rynkiem pasek
                                            _zgod_pct = int(_mn["zgodnosc_pct"] * 100)
                                            _zgod_c   = _mn["kolor"]

                                            # Kelly
                                            _bankroll = st.session_state.get("bankroll", KELLY_BANKROLL_DEFAULT)
                                            _kelly = kelly_stake(pred["p_typ"], _kdc, bankroll=_bankroll)

                                            st.markdown(
                                                f"<div style='background:#0a1628;border:1px solid "
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
                                                f"Edge: <span style='color:{_ev_c}'><b>{_edge:+.1%}</b></span> | "
                                                f"<span style='color:{_ev_c}'><b>EV {_ev_val:+.1%}</b></span>"
                                                f"{_vbadge}</div>"
                                                f"<div style='margin-top:5px'>"
                                                f"<div style='font-size:0.68em;color:#555;margin-bottom:2px'>Zgodność z rynkiem</div>"
                                                f"<div style='background:#222;border-radius:3px;height:4px'>"
                                                f"<div style='background:{_zgod_c};width:{_zgod_pct}%;"
                                                f"height:4px;border-radius:3px'></div></div></div>"
                                                f"</div>",
                                                unsafe_allow_html=True)

                                            if _is_val and _kelly["safe"]:
                                                st.markdown(
                                                    f"<div style='background:#001a0a;border:1px solid #2a6b2a;"
                                                    f"border-radius:5px;padding:5px 10px;margin:3px 0;"
                                                    f"font-size:0.82em'>"
                                                    f"💰 <b>Kelly 1/4</b>: postaw "
                                                    f"<b style='color:#4CAF50'>{_kelly['stake_pln']:.0f} zł</b>"
                                                    f" ({_kelly['f_frac']:.1%} bankrollu) · "
                                                    f"EV/jedn.: <b style='color:#4CAF50'>{_kelly['ev_per_unit']:+.3f}</b>"
                                                    f"</div>",
                                                    unsafe_allow_html=True)

                                elif _OA_OK and _oa_key and not _oa_cached:
                                    st.caption("📊 Brak kursów — kliknij 'Odśwież kursy' w sidebarze.")

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

                st.divider()
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
                        lhs, las, lrs, lks, _sot_sv, _lsot_sv = oblicz_lambdy(h_s, a_s, srednie_df, srednie_lig, forma_dict)
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
                tc1, tc2 = st.columns(2)
                with tc1:
                    if st.button("💾 Zapisz ponownie", key="save_zd",
                                 help="Nadpisuje predykcje dla tej kolejki (jesli zmienily sie dane)."):
                        n_saved = 0
                        for _, mecz_s in mecze.iterrows():
                            h_s = map_nazwa(mecz_s["home_team"])
                            a_s = map_nazwa(mecz_s["away_team"])
                            if h_s not in srednie_df.index or a_s not in srednie_df.index:
                                continue
                            lhs, las, lrs, lks, _sot_sv, _lsot_sv = oblicz_lambdy(h_s, a_s, srednie_df, srednie_lig, forma_dict)
                            pred_s = predykcja_meczu(lhs, las, rho=rho, csv_code=LIGI[wybrana_liga]["csv_code"], n_train=n_biezacy)
                            mecz_str_s = f"{h_s} – {a_s}"
                            zapisz_zdarzenia(wybrana_liga, int(aktualna_kolejka), mecz_str_s, h_s, a_s,
                                             "1X2", pred_s["typ"], 0.0, pred_s["p_typ"], pred_s["fo_typ"])
                            n_saved += 1
                            for emo, nazwa_z, pz, foz, katz, liniz in alternatywne_zdarzenia(lhs, las, lrs, lks, rho, lam_sot=_lsot_sv):
                                zapisz_zdarzenia(wybrana_liga, int(aktualna_kolejka), mecz_str_s, h_s, a_s,
                                                 katz, nazwa_z, liniz, pz, foz)
                        st.session_state[_save_key] = n_saved
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
            else:
                st.info("Brak meczów w tej kolejce")
        else:
            st.warning("Brak danych")

    # =========================================================================
    # TAB 3 – DEEP DATA
    # =========================================================================
    with tab3:
        st.subheader("🔬 Deep Data – Power Rankings & Analiza")
        st.caption("Poglądowe statystyki drużyn i sędziów. Dane historyczne (bieżący + poprzedni sezon).")

        if not power_df.empty:
            search_dd = st.text_input("🔍 Filtruj drużynę", "", key="search_dd",
                                      placeholder="Wpisz nazwę...")
            st.caption("💡 Kliknij nagłówek kolumny żeby posortować")

            df_dd = power_df.copy()
            if search_dd:
                df_dd = df_dd[df_dd["Drużyna"].str.contains(search_dd, case=False, na=False)]

            # Kolumny do wyświetlenia – natywny st.dataframe z sortowaniem przez klik nagłówka
            display_cols_dd = ["Drużyna","M","Gole/M ↑","Strac./M ↓","SOT/M",
                               "Konwersja%","xG-proxy","Kartki/M","Rożne/M","Forma (pkt/5M)"]
            avail_dd = [c for c in display_cols_dd if c in df_dd.columns]
            df_show = df_dd[avail_dd].copy()

            # Konwertuj kolumny numeryczne (mogą być stringiem np. "1.23")
            for col in ["Gole/M ↑","Strac./M ↓","SOT/M","Konwersja%","xG-proxy","Kartki/M","Rożne/M","Forma (pkt/5M)"]:
                if col in df_show.columns:
                    df_show[col] = pd.to_numeric(df_show[col], errors="coerce")

            st.dataframe(
                df_show,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Drużyna":        st.column_config.TextColumn("Drużyna", width="medium"),
                    "M":              st.column_config.NumberColumn("Mecze", format="%d", width="small",
                                                                    help="Łączna liczba meczów drużyny uwzględnionych w statystykach (bieżący + poprzedni sezon)"),
                    "Gole/M ↑":       st.column_config.NumberColumn("Gole/M ↑",  format="%.2f"),
                    "Strac./M ↓":     st.column_config.NumberColumn("Strac./M ↓", format="%.2f"),
                    "SOT/M":          st.column_config.NumberColumn("SOT/M",      format="%.1f"),
                    "Konwersja%":     st.column_config.NumberColumn("Konwersja%", format="%.1f%%"),
                    "xG-proxy":       st.column_config.NumberColumn("xG-proxy",   format="%.2f"),
                    "Kartki/M":       st.column_config.NumberColumn("Kartki/M",   format="%.2f"),
                    "Rożne/M":        st.column_config.NumberColumn("Rożne/M",    format="%.1f"),
                    "Forma (pkt/5M)": st.column_config.NumberColumn("Forma/5M",   format="%.0f"),
                },
            )
            st.caption("🍀 Szczęściarz (gole > xG-proxy) · 😤 Pechowiec · ⚖️ Normalny. Konwersja = gole/SOT. xG-proxy = SOT×0.11.")

            export_cols = ["Drużyna","M","Gole/M ↑","Strac./M ↓","SOT/M",
                           "Konwersja%","xG-proxy","Kartki/M","Rożne/M","Forma (pkt/5M)"]
            avail_cols = [c for c in export_cols if c in df_dd.columns]
            st.download_button("⬇️ Pobierz Power Rankings (CSV)",
                               data=df_dd[avail_cols].to_csv(index=False, decimal=","),
                               file_name="power_rankings.csv", mime="text/csv")

            st.divider()
            if not sedziowie_df.empty:
                st.markdown("### 🟨 Profile Sędziów")
                st.caption("Historyczny profil sędziów – średnia kartek i goli per mecz. Dane dostępne tylko dla Premier League.")
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
        # KPI bar – metryki globalne na samej górze
        _mg_top = metryki_globalne(wybrana_liga)
        if _mg_top:
            _kpi1, _kpi2, _kpi3, _kpi4, _kpi5 = st.columns(5)
            _con_kpi = sqlite3.connect(DB_FILE)
            _kpi_row = _con_kpi.execute(
                "SELECT COUNT(*), SUM(trafione) FROM zdarzenia WHERE liga=? AND trafione IS NOT NULL AND rynek='1X2'",
                (wybrana_liga,)).fetchone()
            _con_kpi.close()
            _kpi_n    = int(_kpi_row[0]) if _kpi_row and _kpi_row[0] else 0
            _kpi_traf = int(_kpi_row[1]) if _kpi_row and _kpi_row[1] else 0
            _kpi_hit  = _kpi_traf/_kpi_n if _kpi_n else 0
            _kpi1.metric("🏆 Typów 1X2", _kpi_n)
            _kpi2.metric("✅ Trafione",  _kpi_traf)
            _kpi3.metric("🎯 Hit Rate",  f"{_kpi_hit:.1%}",
                         delta_color="normal" if _kpi_hit>=0.60 else "inverse")
            _kpi4.metric("📐 Brier ↓",   f"{_mg_top['brier']:.4f}",
                         delta=f"BSS {_mg_top['bss']:+.3f}",
                         delta_color="normal" if _mg_top['bss']>0 else "inverse")
            _kpi5.metric("🎯 ECE ↓",     f"{_mg_top['ece']:.4f}",
                         delta="dobrze" if _mg_top['ece']<0.05 else "wymaga uwagi",
                         delta_color="normal" if _mg_top['ece']<0.05 else "inverse")
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
                    hr = rk["hit_rate"]
                    hr_c = "#4CAF50" if hr and hr >= 0.62 else ("#FF9800" if hr and hr >= 0.55 else ("#888" if hr is None else "#F44336"))
                    hr_str = f"{hr:.0%}" if hr is not None else "–"
                    wyn_str = f"{int(rk['n_traf'])}/{int(rk['n_typow'])}" if hr is not None else f"–/{int(rk['n_typow'])}"
                    status_ico = "✅" if hr and hr >= 0.62 else ("⚠️" if hr and hr >= 0.50 else ("⏳" if hr is None else "❌"))
                    _rows_hk.append(
                        f"<tr>"
                        f"<td style='padding:5px 10px;font-weight:bold'>#{int(rk['kolejnosc'])}</td>"
                        f"<td style='padding:5px 10px;text-align:center;color:#888'>{int(rk['n_typow'])}</td>"
                        f"<td style='padding:5px 10px;text-align:center'>{wyn_str}</td>"
                        f"<td style='padding:5px 10px;text-align:center;color:{hr_c};font-weight:bold'>{status_ico} {hr_str}</td>"
                        f"</tr>"
                    )
                st.markdown(
                    f"<div style='overflow-x:auto;border-radius:8px;border:1px solid #2a2a2a;margin-bottom:12px'>"
                    f"<table style='width:100%;border-collapse:collapse;font-size:0.85em'>"
                    f"<thead><tr style='background:#1a1a2e;color:#666;font-size:0.72em;text-transform:uppercase'>"
                    f"<th style='padding:5px 10px'>Kolejka</th>"
                    f"<th style='padding:5px 10px;text-align:center'>Typów</th>"
                    f"<th style='padding:5px 10px;text-align:center'>Trafione</th>"
                    f"<th style='padding:5px 10px;text-align:center'>Hit Rate</th>"
                    f"</tr></thead><tbody>{''.join(_rows_hk)}</tbody></table></div>",
                    unsafe_allow_html=True
                )
                st.caption("⏳ = predykcje zapisane, wyniki jeszcze nie dostępne w football-data.co.uk")

        # ── Bankroll Tracking – realny kapitał ────────────────────────────
        _br_hist = pobierz_bankroll_history(wybrana_liga)
        if not _br_hist.empty and "kapital_po" in _br_hist.columns:
            st.markdown("**💼 Realny Bankroll**")
            _br_cur = float(_br_hist["kapital_po"].iloc[-1])
            _br_start = 1000.0
            _br_roi = (_br_cur - _br_start) / _br_start * 100
            _brc1, _brc2, _brc3 = st.columns(3)
            _brc1.metric("💰 Aktualny kapital", f"{_br_cur:.0f} zł",
                         delta=f"{_br_roi:+.1f}%",
                         delta_color="normal" if _br_roi >= 0 else "inverse")
            _brc2.metric("📅 Wpisów", len(_br_hist))
            _brc3.metric("📊 Zmian", len(_br_hist[_br_hist["typ"]=="wynik"]))
            _br_chart = _br_hist[["kapital_po"]].rename(columns={"kapital_po":"Kapitał"})
            st.line_chart(_br_chart, height=160, color="#4CAF50" if _br_roi>=0 else "#F44336")

        # Formularz dodawania wpisu
        with st.expander("➕ Dodaj wpis bankrollu", expanded=False):
            _brf1, _brf2, _brf3 = st.columns(3)
            _br_kwota = _brf1.number_input("Kwota (+ zysk / - strata)", value=0.0, step=10.0, key="_br_kwota")
            _br_opis  = _brf2.text_input("Opis (np. 'Chelsea X2 kolejka 29')", key="_br_opis")
            _br_typ   = _brf3.selectbox("Typ", ["wynik","korekta","depozyt","wypłata"], key="_br_typ")
            if st.button("💾 Zapisz wpis", key="_br_save"):
                if _br_kwota != 0 or _br_typ in ("depozyt","wypłata"):
                    zapisz_wynik_bankroll(wybrana_liga,
                                         int(get_current_round(schedule)) if not schedule.empty else 0,
                                         _br_opis, _br_kwota, _br_typ)
                    st.success(f"Zapisano: {_br_kwota:+.0f} zł · {_br_opis}")
                    st.rerun()
                else:
                    st.warning("Podaj kwotę != 0")
        st.divider()

        # ── Equity Curve (teoretyczna z DB) ───────────────────────────────
        _con_eq = sqlite3.connect(DB_FILE)
        _eq_df  = pd.read_sql_query(
            """SELECT kolejnosc, trafione, p_model, fair_odds
               FROM zdarzenia
               WHERE liga=? AND rynek='1X2' AND trafione IS NOT NULL
               ORDER BY kolejnosc, id""",
            _con_eq, params=(wybrana_liga,))
        _con_eq.close()

        if len(_eq_df) >= 3:
            _eq_df = _eq_df.copy()
            # PnL per typ: trafiony → zysk = fair_odds - 1, chybiony → -1
            _eq_df["pnl"] = _eq_df.apply(
                lambda r: r["fair_odds"] - 1 if r["trafione"] == 1 else -1, axis=1)
            _eq_df["kapital"] = 1000 + _eq_df["pnl"].cumsum() * (1000 / len(_eq_df) * 0)
            # Znormalizowana equity: startujemy od 1000, każdy typ = 1 jednostka (flat)
            _eq_df["kapital"] = 1000.0
            _running = 1000.0
            _kap_vals = []
            for _, _r in _eq_df.iterrows():
                _running += _r["pnl"]
                _kap_vals.append(round(_running, 2))
            _eq_df["kapital"] = _kap_vals
            _final_kap = _kap_vals[-1]
            _roi_total = (_final_kap - 1000) / 1000 * 100
            _kap_color = "#4CAF50" if _final_kap >= 1000 else "#F44336"

            st.markdown("**📈 Equity Curve (flat 1 jednostka / typ)**")
            _ec1, _ec2 = st.columns([3, 1])
            with _ec1:
                _chart_eq = _eq_df[["kapital"]].reset_index(drop=True)
                _chart_eq.index.name = "Typ #"
                st.line_chart(_chart_eq, height=200, color=_kap_color)
            with _ec2:
                st.metric("Start", "1 000 zł")
                st.metric("Teraz", f"{_final_kap:.0f} zł",
                          delta=f"{_roi_total:+.1f}%",
                          delta_color="normal" if _roi_total >= 0 else "inverse")
                st.caption(f"Na podstawie {len(_eq_df)} typów 1X2 z wynikami")

        # ── Edge Distribution (EV histogram) ─────────────────────────────
        _con_edge = sqlite3.connect(DB_FILE)
        _edge_df  = pd.read_sql_query(
            """SELECT p_model, fair_odds, trafione, rynek
               FROM zdarzenia
               WHERE liga=? AND trafione IS NOT NULL AND fair_odds IS NOT NULL AND fair_odds > 1""",
            _con_edge, params=(wybrana_liga,))
        _con_edge.close()

        if len(_edge_df) >= 10:
            _edge_df = _edge_df.copy()
            # EV = p_model * fair_odds - 1
            _edge_df["ev"] = _edge_df["p_model"] * _edge_df["fair_odds"] - 1
            # Podziel na buckety
            _ev_bins  = [-1.0, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30, 2.0]
            _ev_labs  = ["< -10%", "-10–5%", "-5–0%", "0–5%", "5–10%", "10–20%", "20–30%", "> 30%"]
            _edge_df["bucket"] = pd.cut(_edge_df["ev"], bins=_ev_bins, labels=_ev_labs)
            _bkt_grp  = _edge_df.groupby("bucket", observed=True).agg(
                n=("ev", "count"),
                hit_rate=("trafione", "mean"),
                total_pnl=("ev", lambda x: (
                    sum((_edge_df.loc[x.index, "fair_odds"] - 1)
                        .where(_edge_df.loc[x.index, "trafione"] == 1, -1))
                ))
            ).reset_index()

            st.divider()
            st.markdown("**📊 Edge Distribution – skąd pochodzi zysk?**")
            st.caption("Jeśli większość PnL pochodzi z bucketów EV 5-20% → stabilny system. Jeśli z > 30% → uważaj na małą próbkę.")

            _ed_rows = []
            for _, bk in _bkt_grp.iterrows():
                if bk["n"] == 0: continue
                pnl = bk["total_pnl"]
                pnl_c = "#4CAF50" if pnl > 0 else "#F44336"
                bar_w = min(int(abs(pnl) / max(_bkt_grp["total_pnl"].abs().max(), 0.01) * 80), 80)
                bar_c = "#4CAF50" if pnl > 0 else "#F44336"
                hr = bk["hit_rate"]
                _ed_rows.append(
                    f"<tr style='border-bottom:1px solid #1a1a2e'>"
                    f"<td style='padding:5px 10px;color:#aaa;font-size:0.84em'>{bk['bucket']}</td>"
                    f"<td style='padding:5px 8px;text-align:center;color:#666'>{int(bk['n'])}</td>"
                    f"<td style='padding:5px 8px;text-align:center;color:#888'>{hr:.0%}</td>"
                    f"<td style='padding:5px 8px'>"
                    f"<div style='display:flex;align-items:center;gap:6px'>"
                    f"<div style='background:{bar_c};width:{bar_w}px;height:8px;border-radius:2px'></div>"
                    f"<span style='color:{pnl_c};font-weight:bold;font-size:0.84em'>{pnl:+.1f} j</span>"
                    f"</div></td>"
                    f"</tr>"
                )
            st.markdown(
                f"<div style='border-radius:8px;border:1px solid #2a2a3a;overflow:hidden'>"
                f"<table style='width:100%;border-collapse:collapse;font-size:0.85em'>"
                f"<thead><tr style='background:#1e1e2e;color:#555;font-size:0.72em;text-transform:uppercase'>"
                f"<th style='padding:6px 10px;text-align:left'>EV bucket</th>"
                f"<th style='padding:6px 8px;text-align:center'>N typów</th>"
                f"<th style='padding:6px 8px;text-align:center'>Hit Rate</th>"
                f"<th style='padding:6px 8px;text-align:left'>PnL (j)</th>"
                f"</tr></thead><tbody>{''.join(_ed_rows)}</tbody></table></div>",
                unsafe_allow_html=True)

        stats_df = statystyki_skutecznosci(wybrana_liga)

        if not stats_df.empty:
            stats_df["ROI_value"] = stats_df["ROI"].str.replace("+","").str.replace("%","").astype(float)

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
                bc4 = "#4CAF50" if brier_v < 0.20 else ("#FF9800" if brier_v < 0.25 else "#F44336")
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
                    f"<div style='border-radius:8px;border:1px solid #2a2a3a;overflow:hidden'>"
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
            f"<div style='background:#333;border-radius:3px;height:5px'>"
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
                    f"<div style='position:relative;background:#1a1a2e;border-radius:3px;height:12px;width:120px;overflow:hidden'>"
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
                f"<div style='overflow-x:auto;border-radius:8px;border:1px solid #2a2a3a'>"
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

    # =========================================================================
    # TAB 6 – LABORATORIUM (Bet Builder)
    # =========================================================================
    with tab6:
        st.subheader("🎛️ Laboratorium modelu")

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
                        _mh, _ma, srednie_df, srednie_lig, forma_dict)
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
                    f"style='background:#0e1117;border-radius:6px;display:block'>"
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

        st.divider()
        st.markdown("### 🔧 Bet Builder – kombinator zdarzeń")
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
            aktualna_kolejka = get_current_round(schedule)
            mecze = schedule[schedule["round"] == aktualna_kolejka]
            
            if not mecze.empty:
                st.caption(f"Kolejka {aktualna_kolejka} – {len(mecze)} meczów")

                # Zbierz wszystkie mecze z combo p – posortuj po p_combo malejąco
                lab_rows = []
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
                    lab_rows.append((p_combo, h, a, lam_h, lam_a, lam_r, lam_k, p_g, p_r, p_k))

                # Sortuj: powyżej progu pierwsze (malejąco), poniżej na końcu
                lab_rows.sort(key=lambda x: -x[0])

                for p_combo, h, a, lam_h, lam_a, lam_r, lam_k, p_g, p_r, p_k in lab_rows:
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
            else:
                st.info("Brak meczów w tej kolejce")
        else:
            st.warning("Brak danych")

    # =========================================================================
    # TAB 7 – BACKTEST
    # =========================================================================
    with tab7:
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
                sel   = st.selectbox("Sezon testowy", opcje, key="bt_sezon")
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

                    st.divider()
                    ec1, ec2 = st.columns(2)
                    with ec1:
                        st.markdown("**📈 Krzywa kapitału**")
                        eq = summ["equity_df"].rename(columns={"equity":"Kapitał"}).set_index("kolejka")
                        st.line_chart(eq["Kapitał"], height=220)
                    with ec2:
                        st.markdown("**📉 Brier & Hit per kolejka**")
                        per_k = summ["per_kolejka"]
                        if not per_k.empty:
                            chart_k = per_k[["kolejka","brier","hit_rate"]].rename(
                                columns={"kolejka":"Kolejka","brier":"Brier","hit_rate":"Hit"}
                            ).set_index("Kolejka")
                            st.line_chart(chart_k, height=220)

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
                                f"<div style='flex:1;background:#333;border-radius:3px;height:5px'>"
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
                            f"<thead><tr style='background:#1a1a2e;color:#666;font-size:0.73em;text-transform:uppercase'>"
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
                            f"<thead><tr style='background:#1a1a2e;color:#666;font-size:0.72em;text-transform:uppercase'>"
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
                    with st.expander("📋 Wszystkie predykcje", expanded=False):
                        disp=bt_df[["kolejka","data","home","away","fthg","ftag",
                                    "wynik","typ","p_typ","trafiony","brier"]].copy()
                        disp["p_typ"]=disp["p_typ"].apply(lambda x:f"{x:.0%}")
                        disp["brier"]=disp["brier"].apply(lambda x:f"{x:.4f}")
                        disp["trafiony"]=disp["trafiony"].map({1:"✅",0:"❌"})
                        disp.columns=["Kol.","Data","Dom","Gość","GH","GA","Wynik","Typ","P","✓","Brier"]
                        st.dataframe(disp, use_container_width=True, hide_index=True)

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