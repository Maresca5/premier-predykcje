"""
bot_runner.py – Samodzielny runner Telegram bota predykcji piłkarskich.

Uruchamiany jako OSOBNY WĄTEK przy starcie Streamlit (przez app.py).
Działa w tle niezależnie od tego czy użytkownik ma otwartą stronę.

Architektura:
  - NIE importuje streamlit (brak st.*, st.cache_data, st.session_state)
  - Pobiera dane bezpośrednio przez HTTP (football-data.co.uk CSV)
  - Odpowiada na komendy /value /status /help z opóźnieniem max 30s
  - Własna pętla polling co 15 sekund (niezależna od reruns Streamlit)

Uruchomienie z app.py:
    import bot_runner
    bot_runner.start()   # uruchamia wątek w tle, nie blokuje Streamlit
"""

import threading
import time
import sqlite3
import json
import os
import requests
import unicodedata
import itertools
import logging
from io import StringIO
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import poisson

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BOT] %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("bot_runner")

# ── Singleton guard ───────────────────────────────────────────────────────────
# Używamy pliku-locka zamiast zmiennej w pamięci.
# Streamlit może tworzyć nowe procesy/wątki przy rerunach — zmienna modułu
# nie jest wystarczającym guardem jeśli moduł jest importowany wielokrotnie.
_BOT_STARTED = False
_BOT_LOCK    = threading.Lock()
_BOT_LOCKFILE = "/tmp/zipybets_bot.lock"

# Klucze API — wstrzykiwane przez start() z kontekstu Streamlit
_TG_TOKEN   = None
_TG_CHAT    = None
_FD_API_KEY = None
OA_DB       = "predykcje.db"  # ten sam plik co app.py

# Import odds_api (opcjonalny)
try:
    import odds_api as _oa
    _OA_OK = True
except ImportError:
    _oa    = None
    _OA_OK = False

# ── Konfiguracja (skopiowana z app.py, bez importowania Streamlit) ────────────
LIGI = {
    "Premier League": {"csv_code": "E0",  "fd_org_id": 2021, "file": "terminarz_premier_2025.csv",  "tau": 30.0},
    "La Liga":        {"csv_code": "SP1", "fd_org_id": 2014, "file": "terminarz_la_liga_2025.csv",  "tau": 28.0},
    "Bundesliga":     {"csv_code": "D1",  "fd_org_id": 2002, "file": "terminarz_bundesliga_2025.csv","tau": 28.0},
    "Serie A":        {"csv_code": "I1",  "fd_org_id": 2019, "file": "terminarz_serie_a_2025.csv",  "tau": 28.0},
    "Ligue 1":        {"csv_code": "F1",  "fd_org_id": 2015, "file": "terminarz_ligue_1_2025.csv",  "tau": 21.0},
}

LIGA_EMOJI = {
    "Premier League": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "La Liga":        "🇪🇸",
    "Bundesliga":     "🇩🇪",
    "Serie A":        "🇮🇹",
    "Ligue 1":        "🇫🇷",
}

DB_FILE          = "predykcje.db"
TG_DB_FILE       = "telegram_alerts.db"
POLL_INTERVAL    = 15        # sekund między sprawdzeniami Telegrama
DATA_TTL         = 900       # sekundy ważności cache danych (15 min)
DIGEST_HOUR      = 9         # godzina porannego digestu (9:00)
DIGEST_DB_TABLE  = "bot_digest_sent"  # tabela w TG_DB_FILE
SOT_BLEND_W      = 0.30
KELLY_BANKROLL   = 1000.0
KELLY_PROB_SCALE = 0.85
KELLY_FRAC_SCALE = 0.50
PROG_PEWNY       = 0.55      # identyczny z app.py (nie 0.58!)
PROG_PODWOJNA    = 0.55      # próg dla 1X / X2
MIN_EV_ALERT     = 0.08

# Kalibracja shrinkage – identyczna z app.py
KALIBRACJA_PER_LIGA = {"E0": 0.25, "SP1": 0.38, "D1": 0.40, "I1": 0.28, "F1": 0.45}
SHRINK_ALPHA        = 0.25
TELEGRAM_API     = "https://api.telegram.org/bot{token}/{method}"

# ── Sezon ────────────────────────────────────────────────────────────────────
def _biezacy_sezon() -> str:
    y = datetime.now().year if datetime.now().month >= 7 else datetime.now().year - 1
    return f"{str(y)[2:]}{str(y+1)[2:]}"

BIEZACY_SEZON = _biezacy_sezon()

# ── Cache danych (prosty dict z TTL) ─────────────────────────────────────────
_data_cache: dict = {}   # key → (timestamp, value)

def _cache_get(key: str):
    entry = _data_cache.get(key)
    if entry and time.time() - entry[0] < DATA_TTL:
        return entry[1]
    return None

def _cache_set(key: str, value):
    _data_cache[key] = (time.time(), value)

# ─────────────────────────────────────────────────────────────────────────────
# TELEGRAM API
# ─────────────────────────────────────────────────────────────────────────────
def _get_credentials():
    """Zwraca (token, chat_id) — wstrzyknięte przez start() lub z env."""
    token   = _TG_TOKEN   or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = _TG_CHAT    or os.environ.get("TELEGRAM_CHAT_ID", "")
    return token, str(chat_id)

def _tg_api(token: str, method: str, **kwargs) -> dict:
    try:
        url = TELEGRAM_API.format(token=token, method=method)
        r = requests.post(url, json=kwargs, timeout=10)
        return r.json()
    except Exception as e:
        log.warning(f"Telegram API error ({method}): {e}")
        return {}

def send_message(token: str, chat_id: str, text: str) -> bool:
    if not token or not chat_id or not text:
        return False
    res = _tg_api(token, "sendMessage",
                  chat_id=chat_id, text=text[:4096],
                  parse_mode="HTML", disable_web_page_preview=True)
    ok = res.get("ok", False)
    if not ok:
        log.warning(f"sendMessage failed: {res.get('description','')}")
    return ok

def get_updates(token: str, offset: int) -> list:
    res = _tg_api(token, "getUpdates", offset=offset, timeout=5)
    if res.get("ok"):
        return res.get("result", [])
    return []

# ── Offset persistence ────────────────────────────────────────────────────────
def _get_offset() -> int:
    try:
        con = sqlite3.connect(TG_DB_FILE)
        con.execute("CREATE TABLE IF NOT EXISTS bot_offset (id INTEGER PRIMARY KEY, val INTEGER)")
        row = con.execute("SELECT val FROM bot_offset WHERE id=1").fetchone()
        con.close()
        return int(row[0]) if row else 0
    except Exception:
        return 0

def _set_offset(val: int):
    try:
        con = sqlite3.connect(TG_DB_FILE)
        con.execute("CREATE TABLE IF NOT EXISTS bot_offset (id INTEGER PRIMARY KEY, val INTEGER)")
        con.execute("INSERT OR REPLACE INTO bot_offset (id, val) VALUES (1, ?)", (val,))
        con.commit()
        con.close()
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# NAZWY DRUŻYN (uproszczona mapa — kluczowe mapowania z app.py)
# ─────────────────────────────────────────────────────────────────────────────
NAZWY_MAP = {
    # Premier League
    "Arsenal FC": "Arsenal", "Aston Villa FC": "Aston Villa",
    "AFC Bournemouth": "Bournemouth", "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton", "Brighton & Hove Albion": "Brighton",
    "Chelsea FC": "Chelsea", "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton", "Fulham FC": "Fulham",
    "Ipswich Town FC": "Ipswich", "Leicester City FC": "Leicester",
    "Liverpool FC": "Liverpool", "Manchester City FC": "Man City",
    "Manchester United FC": "Man United", "Newcastle United FC": "Newcastle",
    "Nottingham Forest FC": "Nott'm Forest", "Southampton FC": "Southampton",
    "Tottenham Hotspur FC": "Tottenham", "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves",
    # shortNames
    "Arsenal": "Arsenal", "Aston Villa": "Aston Villa", "Bournemouth": "Bournemouth",
    "Brentford": "Brentford", "Brighton": "Brighton", "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace", "Everton": "Everton", "Fulham": "Fulham",
    "Ipswich": "Ipswich", "Leicester": "Leicester", "Liverpool": "Liverpool",
    "Man City": "Man City", "Man United": "Man United", "Newcastle": "Newcastle",
    "Nott'm Forest": "Nott'm Forest", "Southampton": "Southampton",
    "Tottenham": "Tottenham", "West Ham": "West Ham", "Wolves": "Wolves",
    # La Liga
    "Club Atlético de Madrid": "Ath Madrid", "Athletic Club": "Ath Bilbao",
    "CA Osasuna": "Osasuna", "Deportivo Alavés": "Alaves",
    "Getafe CF": "Getafe", "Girona FC": "Girona",
    "Rayo Vallecano de Madrid": "Vallecano", "RC Celta de Vigo": "Celta",
    "RCD Espanyol de Barcelona": "Espanyol", "RCD Mallorca": "Mallorca",
    "Real Betis Balompié": "Betis", "Real Madrid CF": "Real Madrid",
    "Real Sociedad de Fútbol": "Sociedad", "Real Valladolid CF": "Valladolid",
    "Sevilla FC": "Sevilla", "UD Las Palmas": "Las Palmas",
    "Valencia CF": "Valencia", "Villarreal CF": "Villarreal",
    "FC Barcelona": "Barcelona", "Leganés": "Leganes",
    "Atlético Madrid": "Ath Madrid", "Ath Bilbao": "Ath Bilbao",
    # Bundesliga
    "1. FC Heidenheim 1846": "Heidenheim", "1. FC Köln": "Koln",
    "1. FC Union Berlin": "Union Berlin", "1. FSV Mainz 05": "Mainz",
    "Bayer 04 Leverkusen": "Leverkusen", "Borussia Dortmund": "Dortmund",
    "Borussia Mönchengladbach": "Ein Frankfurt", "Eintracht Frankfurt": "Ein Frankfurt",
    "FC Augsburg": "Augsburg", "FC Bayern München": "Bayern Munich",
    "FC St. Pauli 1910": "St Pauli", "Holstein Kiel": "Kiel",
    "RB Leipzig": "RB Leipzig", "SC Freiburg": "Freiburg",
    "TSG 1899 Hoffenheim": "Hoffenheim", "VfB Stuttgart": "Stuttgart",
    "VfL Bochum 1848": "Bochum", "VfL Wolfsburg": "Wolfsburg",
    "Werder Bremen": "Werder Bremen", "Hamburger SV": "Hamburg",
    "Bayern Munich": "Bayern Munich", "Dortmund": "Dortmund",
    # Serie A
    "AC Milan": "Milan", "ACF Fiorentina": "Fiorentina",
    "AS Roma": "Roma", "Atalanta BC": "Atalanta",
    "Bologna FC 1909": "Bologna", "Cagliari Calcio": "Cagliari",
    "Empoli FC": "Empoli", "FC Internazionale Milano": "Inter",
    "Frosinone Calcio": "Frosinone", "Genoa CFC": "Genoa",
    "Hellas Verona FC": "Verona", "Inter": "Inter", "Juventus FC": "Juventus",
    "Lazio": "Lazio", "Lecce": "Lecce", "Monza": "Monza",
    "Napoli": "Napoli", "Parma Calcio 1913": "Parma",
    "SS Lazio": "Lazio", "SSC Napoli": "Napoli",
    "Salernitana": "Salernitana", "Sassuolo": "Sassuolo",
    "Torino FC": "Torino", "UC Sampdoria": "Sampdoria",
    "Udinese Calcio": "Udinese", "US Lecce": "Lecce",
    "Venezia FC": "Venezia", "Como 1907": "Como",
    # Ligue 1
    "AJ Auxerre": "Auxerre", "AC Ajaccio": "Ajaccio",
    "Angers SCO": "Angers", "AS Monaco FC": "Monaco",
    "AS Saint-Étienne": "St Etienne", "Clermont Foot 63": "Clermont",
    "FC Lorient": "Lorient", "FC Metz": "Metz",
    "FC Nantes": "Nantes", "FC Toulouse": "Toulouse",
    "LOSC Lille": "Lille", "Le Havre AC": "Le Havre",
    "Montpellier HSC": "Montpellier", "OGC Nice": "Nice",
    "Olympique Lyonnais": "Lyon", "Olympique de Marseille": "Marseille",
    "Paris Saint-Germain FC": "Paris SG", "Stade Brestois 29": "Brest",
    "Stade Rennais FC 1901": "Rennes", "RC Lens": "Lens",
    "RC Strasbourg Alsace": "Strasbourg", "Stade de Reims": "Reims",
}

def _normalize(name: str) -> str:
    if not isinstance(name, str):
        return str(name)
    n = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode("ASCII")
    for suffix in [" FC", " CF", " UD", " CD", " RCD", " AS", " Stade", " 1."]:
        n = n.replace(suffix, "")
    return n.strip()

def map_nazwa(nazwa: str) -> str:
    if not isinstance(nazwa, str):
        return str(nazwa)
    if nazwa in NAZWY_MAP:
        return NAZWY_MAP[nazwa]
    def upros(s):
        return s.replace(" ", "").replace("-", "").replace("'", "").lower()
    for k, v in NAZWY_MAP.items():
        if upros(k) == upros(nazwa):
            return v
    znorm = _normalize(nazwa)
    if znorm in NAZWY_MAP:
        return NAZWY_MAP[znorm]
    return nazwa

# ─────────────────────────────────────────────────────────────────────────────
# ŁADOWANIE DANYCH
# ─────────────────────────────────────────────────────────────────────────────
def _pobierz_csv(league_code: str, sezon: str) -> pd.DataFrame:
    key = f"csv_{league_code}_{sezon}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        url = f"https://www.football-data.co.uk/mmz4281/{sezon}/{league_code}.csv"
        r = requests.get(url, timeout=15)
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
        for col in ["HST", "AST"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values("Date")
        _cache_set(key, df)
        return df
    except Exception as e:
        log.debug(f"CSV {league_code}/{sezon}: {e}")
        empty = pd.DataFrame()
        _cache_set(key, empty)
        return empty

def _waga_poprzedniego(n: int) -> float:
    return float(np.clip(0.8 - (n / 30) * 0.6, 0.2, 0.8))

def load_historical(csv_code: str) -> pd.DataFrame:
    key = f"hist_{csv_code}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    df_now  = _pobierz_csv(csv_code, BIEZACY_SEZON)
    df_prev = _pobierz_csv(csv_code, "2425")
    if df_now.empty and df_prev.empty:
        result = pd.DataFrame()
    elif df_now.empty:
        result = df_prev
    elif df_prev.empty:
        result = df_now
    else:
        n_now = len(df_now)
        wp = _waga_poprzedniego(n_now)
        n_prev = min(int(n_now * wp / (1 - wp)), len(df_prev))
        dp = df_prev.tail(n_prev).copy(); dp["_sezon"] = "poprzedni"
        dn = df_now.copy(); dn["_sezon"] = "biezacy"
        result = pd.concat([dp, dn], ignore_index=True).sort_values("Date")
    _cache_set(key, result)
    return result

def load_schedule(fd_org_id: int) -> pd.DataFrame:
    key = f"sched_{fd_org_id}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    token_fd = _FD_API_KEY or os.environ.get("FOOTBALL_DATA_API_KEY")
    if not token_fd:
        log.warning(f"load_schedule: brak FOOTBALL_DATA_API_KEY")
        _cache_set(key, pd.DataFrame())
        return pd.DataFrame()

    today = datetime.today()
    yr = today.year if today.month >= 7 else today.year - 1
    for yr_try in [yr, yr - 1]:
        try:
            url = f"https://api.football-data.org/v4/competitions/{fd_org_id}/matches?season={yr_try}"
            r = requests.get(url, headers={"X-Auth-Token": token_fd}, timeout=12)
            if r.status_code != 200:
                continue
            matches = r.json().get("matches", [])
            if not matches:
                continue
            rows = []
            for m in matches:
                ht = map_nazwa(m["homeTeam"].get("shortName") or m["homeTeam"].get("name", ""))
                at = map_nazwa(m["awayTeam"].get("shortName") or m["awayTeam"].get("name", ""))
                dt = pd.to_datetime(m["utcDate"]).tz_localize(None) \
                    if not m["utcDate"].endswith("Z") \
                    else pd.to_datetime(m["utcDate"]).tz_convert(None)
                sc = m.get("score", {}).get("fullTime", {})
                rows.append({
                    "round":     m.get("matchday", 0),
                    "date":      dt,
                    "home_team": ht,
                    "away_team": at,
                    "is_played": m.get("status") == "FINISHED",
                    "wynik_h":   sc.get("home"),
                    "wynik_a":   sc.get("away"),
                })
            df = pd.DataFrame(rows)
            df["round"] = pd.to_numeric(df["round"], errors="coerce").fillna(0).astype(int)
            result = df.sort_values("date").reset_index(drop=True)
            _cache_set(key, result)
            return result
        except Exception as e:
            log.debug(f"schedule {fd_org_id}/{yr_try}: {e}")
            continue

    _cache_set(key, pd.DataFrame())
    return pd.DataFrame()

def get_current_round(schedule: pd.DataFrame) -> int:
    if schedule.empty:
        return 0
    dzisiaj = datetime.now().date()
    sch = schedule.copy()
    sch["_d"] = sch["date"].dt.date
    for runda in sorted(sch["round"].unique()):
        mecze = sch[sch["round"] == runda]
        n_przyszle = (mecze["_d"] >= dzisiaj).sum()
        n_przeszle = len(mecze) - n_przyszle
        if n_przyszle >= n_przeszle and n_przyszle > 0:
            return int(runda)
    return int(schedule["round"].max())

# ─────────────────────────────────────────────────────────────────────────────
# MODEL STATYSTYCZNY (skopiowany z app.py, bez dekoratorów)
# ─────────────────────────────────────────────────────────────────────────────
def weighted_mean(values: pd.Series, dates: pd.Series = None, tau: float = 30.0) -> float:
    if len(values) == 0:
        return 0.0
    if dates is not None and len(dates) == len(values):
        try:
            dates_dt = pd.to_datetime(dates)
            ref = dates_dt.max()
            days_ago = (ref - dates_dt).dt.total_seconds() / 86400
            weights = np.exp(-days_ago.values / tau)
            weights = np.clip(weights, 0.01, None)
        except Exception:
            weights = np.linspace(1, 2, len(values))
    else:
        weights = np.linspace(1, 2, len(values))
    return float(np.average(values, weights=weights))

def oblicz_statystyki(df: pd.DataFrame) -> pd.DataFrame:
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
        h_sot_d = home.loc[home["HST"].notna(), "Date"] if "HST" in home.columns else None
        a_sot_d = away.loc[away["AST"].notna(), "Date"] if "AST" in away.columns else None
        dane[d] = {
            "Gole strzelone (dom)":    weighted_mean(home["FTHG"], h_dates),
            "Gole stracone (dom)":     weighted_mean(home["FTAG"], h_dates),
            "Gole strzelone (wyjazd)": weighted_mean(away["FTAG"], a_dates),
            "Gole stracone (wyjazd)":  weighted_mean(away["FTHG"], a_dates),
            "Różne (dom)":             weighted_mean(home["total_rozne"], h_dates) if "total_rozne" in home.columns else 0,
            "Różne (wyjazd)":          weighted_mean(away["total_rozne"], a_dates) if "total_rozne" in away.columns else 0,
            "Kartki (dom)":            weighted_mean(home["total_kartki"], h_dates) if "total_kartki" in home.columns else 0,
            "Kartki (wyjazd)":         weighted_mean(away["total_kartki"], a_dates) if "total_kartki" in away.columns else 0,
            "SOT (dom)":    (weighted_mean(home.loc[home["HST"].notna(), "HST"], h_sot_d) if len(home_sot) >= 2 else None),
            "SOT (wyjazd)": (weighted_mean(away.loc[away["AST"].notna(), "AST"], a_sot_d) if len(away_sot) >= 2 else None),
            "Konwersja (dom)":    (float(home["FTHG"].sum() / home_sot.sum()) if home_sot.sum() > 0 else None),
            "Konwersja (wyjazd)": (float(away["FTAG"].sum() / away_sot.sum()) if away_sot.sum() > 0 else None),
        }
    return pd.DataFrame(dane).T.round(2)

def oblicz_srednie(df: pd.DataFrame) -> dict:
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
    avg_sot_h = float(df["HST"].dropna().mean()) if "HST" in df.columns and df["HST"].notna().sum() > 10 else None
    avg_sot_a = float(df["AST"].dropna().mean()) if "AST" in df.columns and df["AST"].notna().sum() > 10 else None
    return {"avg_home": avg_h, "avg_away": avg_a, "rho": rho,
            "n_biezacy": n_biezacy, "avg_sot_home": avg_sot_h, "avg_sot_away": avg_sot_a}

def oblicz_lambdy(h: str, a: str, srl: pd.DataFrame, avg: dict) -> tuple:
    """Zwraca (lam_h, lam_a)."""
    ALPHA_OFF, ALPHA_DEF = 0.10, 0.20
    avg_h = avg.get("avg_home", 1.5)
    avg_a = avg.get("avg_away", 1.2)
    avg_sot_h = avg.get("avg_sot_home")
    avg_sot_a = avg.get("avg_sot_away")

    gs_h = float(srl.loc[h, "Gole strzelone (dom)"])   if h in srl.index else avg_h
    gc_h = float(srl.loc[h, "Gole stracone (dom)"])    if h in srl.index else avg_a
    gs_a = float(srl.loc[a, "Gole strzelone (wyjazd)"]) if a in srl.index else avg_a
    gc_a = float(srl.loc[a, "Gole stracone (wyjazd)"])  if a in srl.index else avg_h

    gs_h = gs_h * (1-ALPHA_OFF) + avg_h * ALPHA_OFF
    gc_h = gc_h * (1-ALPHA_DEF) + avg_a * ALPHA_DEF
    gs_a = gs_a * (1-ALPHA_OFF) + avg_a * ALPHA_OFF
    gc_a = gc_a * (1-ALPHA_DEF) + avg_h * ALPHA_DEF

    lam_h = max(0.3, (gs_h + gc_a) / 2)
    lam_a = max(0.3, (gs_a + gc_h) / 2)

    # SOT blend
    if SOT_BLEND_W > 0 and avg_sot_h and avg_sot_a:
        try:
            sot_h = float(srl.loc[h, "SOT (dom)"])    if h in srl.index and pd.notna(srl.loc[h, "SOT (dom)"])    else avg_sot_h
            sot_a = float(srl.loc[a, "SOT (wyjazd)"]) if a in srl.index and pd.notna(srl.loc[a, "SOT (wyjazd)"]) else avg_sot_a
            conv_h = float(srl.loc[h, "Konwersja (dom)"])    if h in srl.index and pd.notna(srl.loc[h, "Konwersja (dom)"])    else (avg_h / avg_sot_h if avg_sot_h else 0.3)
            conv_a = float(srl.loc[a, "Konwersja (wyjazd)"]) if a in srl.index and pd.notna(srl.loc[a, "Konwersja (wyjazd)"]) else (avg_a / avg_sot_a if avg_sot_a else 0.3)
            lam_sot_h = sot_h * conv_h
            lam_sot_a = sot_a * conv_a
            lam_h = lam_h * (1 - SOT_BLEND_W) + lam_sot_h * SOT_BLEND_W
            lam_a = lam_a * (1 - SOT_BLEND_W) + lam_sot_a * SOT_BLEND_W
            lam_h = max(0.3, lam_h)
            lam_a = max(0.3, lam_a)
        except Exception:
            pass
    return lam_h, lam_a

def dixon_coles_adj(M: np.ndarray, lam_h: float, lam_a: float, rho: float = -0.13) -> np.ndarray:
    M = M.copy()
    t = max(1e-6, lam_h * lam_a)
    corr = {(0,0): 1 - lam_h*lam_a*rho, (0,1): 1 + lam_h*rho,
            (1,0): 1 + lam_a*rho,        (1,1): 1 - rho}
    for (i,j), v in corr.items():
        if i < M.shape[0] and j < M.shape[1]:
            M[i,j] = max(0, M[i,j] * v)
    s = M.sum()
    return M / s if s > 0 else M

def _get_shrink(csv_code: str, n_train: int = 200) -> float:
    N_FULL  = 150
    BONUS   = 0.20
    base    = KALIBRACJA_PER_LIGA.get(csv_code, SHRINK_ALPHA)
    dynamic = BONUS * max(0.0, 1.0 - min(n_train, N_FULL) / N_FULL)
    return float(np.clip(base + dynamic, 0.0, 0.85))

def kalibruj(p_home: float, p_draw: float, p_away: float,
             csv_code: str = "E0", n_train: int = 200) -> tuple:
    a = _get_shrink(csv_code, n_train)
    ph = (1-a)*p_home + a/3
    pd = (1-a)*p_draw + a/3
    pa = (1-a)*p_away + a/3
    s  = ph + pd + pa
    return ph/s, pd/s, pa/s

def wybierz_typ(p_home: float, p_draw: float, p_away: float,
                csv_code: str = "E0", n_train: int = 200) -> tuple:
    ph, pd, pa = kalibruj(p_home, p_draw, p_away, csv_code, n_train)
    p_1x = ph + pd
    p_x2 = pa + pd
    is_bl = (csv_code == "D1")
    if not is_bl:
        if ph >= PROG_PEWNY: return "1",  ph
        if pa >= PROG_PEWNY: return "2",  pa
    if p_1x >= PROG_PODWOJNA or p_x2 >= PROG_PODWOJNA:
        return ("1X", p_1x) if p_1x >= p_x2 else ("X2", p_x2)
    probs = {"1": ph, "X": pd, "2": pa}
    t = max(probs, key=probs.get)
    return t, probs[t]

def predykcja(lam_h: float, lam_a: float, rho: float = -0.13,
              csv_code: str = "E0", n_train: int = 200) -> dict:
    max_g = int(np.clip(np.ceil(max(lam_h, lam_a) + 5), 8, 12))
    M = dixon_coles_adj(
        np.outer(poisson.pmf(range(max_g), lam_h),
                 poisson.pmf(range(max_g), lam_a)),
        lam_h, lam_a, rho=rho)
    p_h = float(np.tril(M, -1).sum())
    p_d = float(np.trace(M))
    p_a = float(np.triu(M,  1).sum())
    typ, p_typ = wybierz_typ(p_h, p_d, p_a, csv_code, n_train)
    fo_typ = round(1 / p_typ, 2) if p_typ > 0 else 999
    # Entropy macierzy — miara chaosu (niższa = bardziej zdecydowany model)
    M_flat = M.flatten()
    M_flat = M_flat[M_flat > 0]
    ent = float(-np.sum(M_flat * np.log(M_flat)))
    return {"typ": typ, "p_typ": p_typ, "fo_typ": fo_typ, "entropy": ent}


def compute_stability() -> list:
    """
    Oblicza Stability Score dla każdej ligi w bieżącej kolejce.
    Zwraca listę dictów posortowaną po stability malejąco.
    """
    results = []
    for liga_name, cfg in LIGI.items():
        try:
            hl = load_historical(cfg["csv_code"])
            sl = load_schedule(cfg["fd_org_id"])
            if hl.empty or sl.empty:
                continue
            srl = oblicz_statystyki(hl)
            avg = oblicz_srednie(hl)
            rho = avg.get("rho", -0.13)
            kol = get_current_round(sl)
            mecze = sl[sl["round"] == kol]
            idx = list(srl.index)

            ents = []
            for _, m in mecze.iterrows():
                h = map_nazwa(m.get("home_team", ""))
                a = map_nazwa(m.get("away_team", ""))
                if h not in idx or a not in idx:
                    continue
                try:
                    lam_h, lam_a = oblicz_lambdy(h, a, srl, avg, {})[0:2]
                    pr = predykcja(lam_h, lam_a, rho=rho, csv_code=cfg["csv_code"])
                    ents.append(pr["entropy"])
                except Exception:
                    continue

            if not ents:
                continue

            avg_ent = float(np.mean(ents))
            stab = max(0.0, min(1.0, 1.0 - (avg_ent - 0.5) / 1.0))
            stab_pct = int(stab * 100)

            if   stab_pct >= 65: rec = "✅ Graj";      stars = "●●●"
            elif stab_pct >= 45: rec = "⚠️ Ostrożnie"; stars = "●●○"
            else:                rec = "🚫 Odpuść";    stars = "●○○"

            results.append({
                "liga":  liga_name,
                "flag":  LIGA_EMOJI.get(liga_name, "⚽"),
                "stab":  stab_pct,
                "rec":   rec,
                "stars": stars,
                "kol":   int(kol),
                "n":     len(ents),
            })
        except Exception as e:
            log.debug(f"stability {liga_name}: {e}")
            continue

    results.sort(key=lambda x: -x["stab"])
    return results


def kelly_stake(p: float, kurs: float, bankroll: float = KELLY_BANKROLL) -> float:
    """Zwraca stawkę Kelly w PLN."""
    p_adj = 0.5 + max(0, p - 0.5) * KELLY_PROB_SCALE
    b = kurs - 1
    if b <= 0 or p_adj <= 0:
        return 0.0
    f = (p_adj * b - (1 - p_adj)) / b
    f = max(0, f) * KELLY_FRAC_SCALE
    stake = bankroll * f
    return round(min(stake, bankroll * 0.05), 1)  # max 5% bankrollu

# ─────────────────────────────────────────────────────────────────────────────
# OBLICZANIE VALUE BETÓW
# ─────────────────────────────────────────────────────────────────────────────
def compute_value_bets() -> list:
    """
    Oblicza value bety ze wszystkich 5 lig.
    Zwraca listę dictów posortowaną po p_model malejąco.
    Działa w pełni niezależnie od Streamlit.
    """
    all_bets = []
    debug_lines = []

    for liga_name, cfg in LIGI.items():
        try:
            hl = load_historical(cfg["csv_code"])
            sl = load_schedule(cfg["fd_org_id"])
            if hl.empty:
                debug_lines.append(f"{liga_name}: ❌ brak CSV")
                continue
            if sl.empty:
                debug_lines.append(f"{liga_name}: ❌ brak terminarza (sprawdź FOOTBALL_DATA_API_KEY)")
                continue

            srl = oblicz_statystyki(hl)
            avg = oblicz_srednie(hl)
            if srl.empty:
                debug_lines.append(f"{liga_name}: ❌ brak statystyk")
                continue

            rho   = avg.get("rho", -0.13)
            kol   = get_current_round(sl)
            mecze = sl[sl["round"] == kol]   # tylko bieżąca kolejka
            idx   = list(srl.index)

            # Kursy z cache SQLite (pobrane przez aplikację) — zero nowych requestów
            oa_cached = {}
            if _OA_OK:
                try:
                    oa_cached = _oa.get_cached_odds(cfg["csv_code"], OA_DB)
                except Exception:
                    pass
            n_added = 0; n_skip_name = 0; n_skip_prog = 0

            for _, m in mecze.iterrows():
                try:
                    h_raw = map_nazwa(m["home_team"])
                    a_raw = map_nazwa(m["away_team"])
                    # Fuzzy match
                    h = h_raw if h_raw in idx else next(
                        (x for x in idx if x.lower() in h_raw.lower() or h_raw.lower() in x.lower()), None)
                    a = a_raw if a_raw in idx else next(
                        (x for x in idx if x.lower() in a_raw.lower() or a_raw.lower() in x.lower()), None)
                    if not h or not a:
                        n_skip_name += 1
                        continue
                    lam_h, lam_a = oblicz_lambdy(h, a, srl, avg)
                    n_train = avg.get("n_biezacy", 200)
                    pr = predykcja(lam_h, lam_a, rho=rho,
                                   csv_code=cfg["csv_code"], n_train=n_train)
                    p  = pr["p_typ"]
                    fo = pr["fo_typ"]
                    if p < PROG_PEWNY or fo < 1.01:
                        n_skip_prog += 1
                        continue
                    ev    = round(p * fo - 1.0, 3)

                    # Live kurs bukmachera
                    kurs_buk = None
                    ev_buk   = None
                    if oa_cached:
                        try:
                            o = _oa.znajdz_kursy(h, a, oa_cached)
                            if o:
                                kmap = {"1": o["odds_h"], "X": o["odds_d"], "2": o["odds_a"],
                                        "1X": max(o["odds_h"], o["odds_d"]),
                                        "X2": max(o["odds_d"], o["odds_a"])}
                                kb = kmap.get(pr["typ"])
                                if kb and kb > 1.01:
                                    kurs_buk = round(kb, 2)
                                    ev_buk   = round(p * kb - 1.0, 3)
                        except Exception:
                            pass
                    # Kelly od kursu buka jeśli dostępny
                    kurs_kelly = kurs_buk if kurs_buk else fo
                    stake = kelly_stake(p, kurs_kelly)

                    data_str = str(m.get("date", ""))[:10]
                    all_bets.append({
                        "liga":     liga_name,
                        "home":     str(m.get("home_team", h_raw)),
                        "away":     str(m.get("away_team", a_raw)),
                        "typ":      pr["typ"],
                        "p_model":  p,
                        "kurs":     fo,
                        "kurs_buk": kurs_buk,
                        "ev":       ev_buk if ev_buk is not None else ev,
                        "ev_fair":  ev,
                        "live":     kurs_buk is not None,
                        "stake":    stake,
                        "kolejka":  int(kol),
                        "data":     data_str,
                    })
                    n_added += 1
                except Exception:
                    continue

            debug_lines.append(
                f"{liga_name}: ✅ kol#{kol} · {len(mecze)}m · +{n_added}typ "
                f"(skip_nazwy={n_skip_name} skip_prog={n_skip_prog})")
        except Exception as e:
            debug_lines.append(f"{liga_name}: 💥 {type(e).__name__}: {str(e)[:80]}")
            continue

    log.info("compute_value_bets: " + " | ".join(debug_lines))
    return sorted(all_bets, key=lambda x: -x["p_model"])

# ─────────────────────────────────────────────────────────────────────────────
# FORMATOWANIE WIADOMOŚCI
# ─────────────────────────────────────────────────────────────────────────────
def _format_value_message(bets: list, title: str = "🔔 VALUE BETS") -> str:
    if not bets:
        return "🔍 <b>Brak value betów</b> w bieżącej kolejce (p≥55%).\nSprawdź ponownie po zaplanowaniu meczy."

    by_liga = defaultdict(list)
    for b in bets:
        by_liga[b["liga"]].append(b)

    now = datetime.now().strftime("%d.%m.%Y  %H:%M")
    has_live = any(b.get("live") for b in bets)

    def p_icon(p: float) -> str:
        return "🟢" if p >= 0.65 else ("🟡" if p >= 0.58 else "🔵")

    lines = [f"<b>{title}</b>  ·  {now}", ""]

    for liga, lb in by_liga.items():
        flag = LIGA_EMOJI.get(liga, "⚽")
        kol  = lb[0]["kolejka"]
        for b in sorted(lb, key=lambda x: -x.get("ev", 0))[:5]:
            pi       = p_icon(b["p_model"])
            ev_val   = b.get("ev", 0)
            data_fmt = b["data"][5:].replace("-", ".") if b.get("data") else ""
            date_str = f"  📅 {data_fmt}" if data_fmt else ""
            kelly_str = f"\n   💰 Stawka Kelly: <b>{b['stake']:.0f} zł</b>" if b.get("stake", 0) > 5 else ""

            # Kurs: pokaż buka jeśli dostępny, inaczej fair
            if b.get("kurs_buk"):
                kurs_line = (f"Kurs fair: {b['kurs']:.2f}  →  "
                             f"Buka: <b>{b['kurs_buk']:.2f}</b> 🔴live")
            else:
                kurs_line = f"Kurs fair: <b>{b['kurs']:.2f}</b> <i>(sprawdź u buka)</i>"

            lines.append(
                f"{pi} <b>{b['home']} vs {b['away']}</b>{date_str}"
                f"\n   {flag} {liga}  ·  kolejka {kol}"
                f"\n   Typ: <b>{b['typ']}</b>  ·  {kurs_line}"
                f"\n   P modelu: <b>{b['p_model']:.1%}</b>  ·  EV: <b>{ev_val:+.1%}</b>"
                f"{kelly_str}"
            )
            lines.append("")

    note = "<i>EV liczone od kursu buka.</i>" if has_live else "<i>EV od fair odds — sprawdź kurs u buka przed postawieniem.</i>"
    lines.append(note)
    return "\n".join(lines)


def _send_long(token: str, chat_id: str, text: str):
    if len(text) <= 4000:
        send_message(token, chat_id, text)
        return
    chunks, current = [], []
    for line in text.split("\n"):
        current.append(line)
        if len("\n".join(current)) > 3600 and not line.strip():
            chunks.append("\n".join(current))
            current = []
    if current:
        chunks.append("\n".join(current))
    for i, chunk in enumerate(chunks):
        prefix = f"<i>({i+1}/{len(chunks)})</i>\n" if len(chunks) > 1 else ""
        send_message(token, chat_id, prefix + chunk)


# ─────────────────────────────────────────────────────────────────────────────
# OBSŁUGA KOMEND TELEGRAM
# ─────────────────────────────────────────────────────────────────────────────
def handle_value(token: str, chat_id: str):
    bets = compute_value_bets()
    _send_long(token, chat_id, _format_value_message(bets, title="💰 Value Bets — na żądanie"))


def handle_status(token: str, chat_id: str):
    try:
        con = sqlite3.connect(DB_FILE)
        lines = [f"📊 <b>Status modelu</b> · {datetime.now().strftime('%d.%m %H:%M')}\n"]
        for liga in LIGI:
            row = con.execute(
                "SELECT COUNT(*), SUM(trafione), AVG(fair_odds) FROM zdarzenia "
                "WHERE liga=? AND sezon=? AND trafione IS NOT NULL AND rynek='1X2'",
                (liga, BIEZACY_SEZON)).fetchone()
            if row and row[0] and row[0] > 0:
                n, t, afo = row
                hr  = t / n
                roi = ((t * (afo - 1)) - (n - t)) / n * 100 if n else 0
                hr_c  = "🟢" if hr >= 0.60 else ("🟡" if hr >= 0.50 else "🔴")
                roi_c = "🟢" if roi >= 0 else "🔴"
                flag  = LIGA_EMOJI.get(liga, "⚽")
                lines.append(f"{flag} <b>{liga}</b>\n"
                              f"   {hr_c} Hit: {hr:.1%} · {roi_c} ROI: {roi:+.1f}% · {n} typów")
        con.close()
        send_message(token, chat_id, "\n".join(lines) if len(lines) > 1
                     else "📭 Brak danych w bazie.")
    except Exception as e:
        send_message(token, chat_id, f"❌ Błąd bazy: {e}")


def handle_debug(token: str, chat_id: str):
    lines = [f"🔧 <b>Debug</b> · {datetime.now().strftime('%d.%m %H:%M:%S')}\n"]
    lines.append(f"TG: {'✅' if _TG_TOKEN else '❌'}  FD: {'✅' if _FD_API_KEY else '❌'}  Sezon: {BIEZACY_SEZON}\n")
    for liga_name, cfg in LIGI.items():
        flag = LIGA_EMOJI.get(liga_name, "⚽")
        try:
            hl = load_historical(cfg["csv_code"])
            sl = load_schedule(cfg["fd_org_id"])
            if hl.empty or sl.empty:
                lines.append(f"{flag} {liga_name}: ❌ hl={'ok' if not hl.empty else 'brak'} sl={'ok' if not sl.empty else 'brak'}")
                continue
            srl = oblicz_statystyki(hl)
            avg = oblicz_srednie(hl)
            kol = get_current_round(sl)
            mecze = sl[sl["round"] == kol]
            idx = list(srl.index)
            n_ok = sum(1 for _, m in mecze.iterrows()
                       if (map_nazwa(m["home_team"]) in idx or
                           any(x.lower() in map_nazwa(m["home_team"]).lower() for x in idx)))
            lines.append(
                f"{flag} {liga_name}: ✅ kol#{kol} · {len(mecze)}m · "
                f"{n_ok} dopas. · {len(srl)} drużyn")
        except Exception as e:
            lines.append(f"{flag} {liga_name}: 💥 {type(e).__name__}: {str(e)[:50]}")
    try:
        pr = predykcja(1.8, 1.0, rho=-0.13, csv_code="E0", n_train=150)
        lines.append(f"\nTest predykcji: typ={pr['typ']} p={pr['p_typ']:.3f} ≥{PROG_PEWNY}? {pr['p_typ']>=PROG_PEWNY}")
    except Exception as e:
        lines.append(f"\nTest predykcji: 💥 {e}")
    send_message(token, chat_id, "\n".join(lines))


def handle_help(token: str, chat_id: str):
    send_message(token, chat_id,
        "🤖 <b>Komendy bota</b>\n\n"
        "/value  – value bety z bieżącej kolejki\n"
        "/status – hit rate i ROI per liga\n"
        "/debug  – diagnostyka danych i kluczy API\n"
        "/help   – ta wiadomość\n\n"
        "<i>Poranny digest wysyłany automatycznie o 9:00 w dzień meczu.</i>")


# ─────────────────────────────────────────────────────────────────────────────
# DIGEST HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _digest_sent_today() -> bool:
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        con = sqlite3.connect(TG_DB_FILE)
        con.execute(f"CREATE TABLE IF NOT EXISTS {DIGEST_DB_TABLE} (date TEXT PRIMARY KEY)")
        row = con.execute(f"SELECT date FROM {DIGEST_DB_TABLE} WHERE date=?", (today,)).fetchone()
        con.close()
        return row is not None
    except Exception:
        return False


def _mark_digest_sent():
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        con = sqlite3.connect(TG_DB_FILE)
        con.execute(f"CREATE TABLE IF NOT EXISTS {DIGEST_DB_TABLE} (date TEXT PRIMARY KEY)")
        con.execute(f"INSERT OR REPLACE INTO {DIGEST_DB_TABLE} (date) VALUES (?)", (today,))
        con.commit()
        con.close()
    except Exception:
        pass


def _is_matchday() -> bool:
    today = datetime.now().date()
    for cfg in LIGI.values():
        try:
            sl = load_schedule(cfg["fd_org_id"])
            if sl.empty:
                continue
            kol   = get_current_round(sl)
            mecze = sl[sl["round"] == kol]
            if mecze["date"].dt.date.eq(today).any():
                return True
        except Exception:
            continue
    return False


# ─────────────────────────────────────────────────────────────────────────────
# GŁÓWNA PĘTLA I START
# ─────────────────────────────────────────────────────────────────────────────
_digest_sent_in_memory: set = set()
_processed_uids: set = set()


def _bot_loop():
    log.info("Bot runner started — polling every %ds", POLL_INTERVAL)

    while True:
        try:
            token, chat_id = _get_credentials()
            if not token or not chat_id:
                time.sleep(60)
                continue

            offset  = _get_offset()
            updates = get_updates(token, offset)

            for upd in updates:
                uid = upd["update_id"]
                _set_offset(uid + 1)          # przesuń PRZED przetworzeniem

                if uid in _processed_uids:    # dedup w RAM
                    continue
                _processed_uids.add(uid)
                if len(_processed_uids) > 500:
                    _processed_uids.clear()

                msg     = upd.get("message", {})
                text    = msg.get("text", "").strip()
                from_id = str(msg.get("chat", {}).get("id", ""))

                if from_id != chat_id:
                    continue

                cmd = text.lower().split()[0] if text else ""
                log.info(f"Command: {cmd!r} (uid={uid})")

                if cmd == "/value":
                    handle_value(token, chat_id)
                elif cmd == "/debug":
                    handle_debug(token, chat_id)
                elif cmd == "/status":
                    handle_status(token, chat_id)
                elif cmd == "/help":
                    handle_help(token, chat_id)

            # Poranny digest — podwójny guard RAM + SQLite
            now = datetime.now()
            today_key = now.strftime("%Y-%m-%d")
            if (now.hour == DIGEST_HOUR
                    and today_key not in _digest_sent_in_memory
                    and not _digest_sent_today()):
                _digest_sent_in_memory.add(today_key)
                _mark_digest_sent()
                if _is_matchday():
                    bets = compute_value_bets()
                    _send_long(token, chat_id,
                               _format_value_message(bets, title="☀️ Poranny Digest"))
                    log.info("Morning digest sent")

        except Exception as e:
            log.error(f"Bot loop error: {e}")

        time.sleep(POLL_INTERVAL)


def start(tg_token: str = None, tg_chat: str = None, fd_api_key: str = None):
    global _BOT_STARTED, _TG_TOKEN, _TG_CHAT, _FD_API_KEY
    with _BOT_LOCK:
        if tg_token:   _TG_TOKEN   = tg_token
        if tg_chat:    _TG_CHAT    = tg_chat
        if fd_api_key: _FD_API_KEY = fd_api_key

        if _BOT_STARTED:
            return

        try:
            import fcntl
            _lf = open(_BOT_LOCKFILE, "w")
            fcntl.flock(_lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (IOError, OSError):
            log.info("Bot already running in another process — skipping")
            _BOT_STARTED = True
            return
        except ImportError:
            pass  # Windows — bez file lock

        _BOT_STARTED = True
        t = threading.Thread(target=_bot_loop, name="telegram-bot", daemon=True)
        t.start()
        log.info(f"Bot started — TG={'ok' if _TG_TOKEN else 'MISSING'} FD={'ok' if _FD_API_KEY else 'MISSING'}")


if __name__ == "__main__":
    log.info("Running bot_runner standalone")
    _bot_loop()
