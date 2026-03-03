# =============================================================================
# odds_api.py  -  The Odds API v4 / free plan (500 req/mies.)
# Strategia: cache w predykcje.db, max 3x tygodniowo per liga
# Koszt: 5 lig x 3x/tydz x 4 tygodnie = ~60 req/mies. (zostaje 440 zapasu)
# =============================================================================

import sqlite3
import requests
from datetime import datetime, timezone

API_BASE          = "https://api.the-odds-api.com/v4"
REGION            = "eu"       # europejscy bukmacherzy (Pinnacle, Bet365, itp.)
MARKET            = "h2h"      # 1X2
MIN_REFRESH_HOURS = 56         # 56h = ~3x tygodniowo
EV_PROG_VALUE     = 0.04       # 4% EV = value bet

PREFERRED_BOOKS = [
    "pinnacle", "bet365", "betfair_ex_eu",
    "williamhill", "bwin", "unibet",
]

SPORT_KEYS = {
    "E0":  "soccer_epl",
    "SP1": "soccer_spain_la_liga",
    "D1":  "soccer_germany_bundesliga",
    "I1":  "soccer_italy_serie_a",
    "F1":  "soccer_france_ligue_one",
}

# ── Baza ──────────────────────────────────────────────────────────────────────
_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS live_odds (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        liga TEXT NOT NULL, event_id TEXT NOT NULL,
        commence_ts TEXT, home_team TEXT, away_team TEXT,
        bookmaker TEXT, odds_h REAL, odds_d REAL, odds_a REAL,
        fetched_at TEXT NOT NULL,
        UNIQUE(liga, event_id, bookmaker)
    )""",
    """CREATE TABLE IF NOT EXISTS odds_fetch_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        liga TEXT NOT NULL, fetched_at TEXT NOT NULL,
        requests_used INTEGER, requests_remaining INTEGER, n_events INTEGER
    )""",
]

def _init_db(db_file):
    con = sqlite3.connect(db_file)
    for s in _SCHEMA:
        con.execute(s)
    con.commit()
    con.close()

# ── EV ────────────────────────────────────────────────────────────────────────
def _kurs_dc(typ, oh, od, oa):
    try:
        oh, od, oa = float(oh), float(od), float(oa)
        if oh <= 1 or od <= 1 or oa <= 1:
            return None, None
        s = 1/oh + 1/od + 1/oa
        ih = (1/oh)/s; id_ = (1/od)/s; ia = (1/oa)/s
        if typ == "1":  return round(oh, 3),       round(ih, 4)
        if typ == "X":  return round(od, 3),       round(id_, 4)
        if typ == "2":  return round(oa, 3),       round(ia, 4)
        if typ == "1X": idc = ih+id_; return round(1/idc, 3), round(idc, 4)
        if typ == "X2": idc = id_+ia; return round(1/idc, 3), round(idc, 4)
    except Exception:
        pass
    return None, None

def oblicz_ev(p_model, typ, oh, od, oa):
    kdc, idc = _kurs_dc(typ, oh, od, oa)
    if kdc is None:
        return {"kurs_dc": None, "impl_dc": None, "ev": None,
                "ev_pct": "N/A", "edge": None, "is_value": False}
    ev = round(p_model * kdc - 1, 4)
    return {"kurs_dc": kdc, "impl_dc": idc, "ev": ev,
            "ev_pct": f"{ev:+.1%}", "edge": round(p_model - idc, 4),
            "is_value": ev >= EV_PROG_VALUE}

# ── Pobieranie ─────────────────────────────────────────────────────────────────
def _wybierz_bukmachera(bookmakers, home_team=""):
    """
    Wybiera bukmachera i przypisuje kursy do home/away na podstawie nazwy
    druzyny domowej z events API (event["home_team"]).
    nd[0]/nd[1] nie sa posortowane - trzeba dopasowac po nazwie.
    """
    book_map = {b["key"]: b for b in bookmakers}
    order    = PREFERRED_BOOKS + [k for k in book_map if k not in PREFERRED_BOOKS]
    for key in order:
        if key not in book_map:
            continue
        for mkt in book_map[key].get("markets", []):
            if mkt["key"] != "h2h":
                continue
            outcomes = {o["name"]: o["price"] for o in mkt.get("outcomes", [])}
            draw     = outcomes.get("Draw")
            nd       = [k for k in outcomes if k != "Draw"]
            if len(nd) == 2 and draw:
                # Dopasuj home po nazwie (case-insensitive, partial match)
                ht = home_team.lower().strip()
                n0 = nd[0].lower().strip()
                n1 = nd[1].lower().strip()
                # home_team z events API powinien byc jednym z nd[]
                if ht and (ht == n0 or ht in n0 or n0 in ht):
                    oh, oa = outcomes[nd[0]], outcomes[nd[1]]
                elif ht and (ht == n1 or ht in n1 or n1 in ht):
                    oh, oa = outcomes[nd[1]], outcomes[nd[0]]
                else:
                    # fallback: kolejnosc z API (zwykle home jest pierwszy)
                    oh, oa = outcomes[nd[0]], outcomes[nd[1]]
                return {"bookmaker": key, "odds_h": oh, "odds_d": draw, "odds_a": oa}
    return None

def fetch_odds(liga, db_file, api_key, force=False):
    if not api_key:
        return {"ok": False, "error": "Brak klucza API."}
    sport_key = SPORT_KEYS.get(liga)
    if not sport_key:
        return {"ok": False, "error": f"Liga '{liga}' nie jest obslugiwana."}
    _init_db(db_file)

    if not force:
        con = sqlite3.connect(db_file)
        row = con.execute(
            "SELECT fetched_at FROM odds_fetch_log WHERE liga=? ORDER BY fetched_at DESC LIMIT 1",
            (liga,)).fetchone()
        con.close()
        if row:
            try:
                last = datetime.fromisoformat(row[0].replace("Z", "+00:00"))
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                age_h = (datetime.now(timezone.utc) - last).total_seconds() / 3600
                if age_h < MIN_REFRESH_HOURS:
                    return {"ok": True, "from_cache": True, "cached_at": row[0],
                            "age_h": round(age_h, 1),
                            "next_refresh_h": round(MIN_REFRESH_HOURS - age_h, 1)}
            except Exception:
                pass

    url = (f"{API_BASE}/sports/{sport_key}/odds"
           f"?apiKey={api_key}&regions={REGION}&markets={MARKET}&oddsFormat=decimal")
    try:
        resp = requests.get(url, timeout=15)
    except Exception as e:
        return {"ok": False, "error": f"Blad polaczenia: {e}"}

    if resp.status_code == 401:
        return {"ok": False, "error": "Nieprawidlowy klucz API (401)."}
    if resp.status_code == 422:
        return {"ok": False, "error": "Sport/region niedostepny w tym planie (422)."}
    if resp.status_code == 429:
        return {"ok": False, "error": "Limit wyczerpany (429). Sprobuj jutro."}
    if resp.status_code != 200:
        return {"ok": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

    data          = resp.json()
    req_used      = int(resp.headers.get("x-requests-used", 0))
    req_remaining = int(resp.headers.get("x-requests-remaining", 0))
    now_str       = datetime.now(timezone.utc).isoformat()

    con = sqlite3.connect(db_file)
    con.execute("DELETE FROM live_odds WHERE liga=?", (liga,))
    n_events = 0
    for event in data:
        bk = _wybierz_bukmachera(event.get("bookmakers", []), event.get("home_team", ""))
        if not bk:
            continue
        con.execute(
            """INSERT OR REPLACE INTO live_odds
               (liga, event_id, commence_ts, home_team, away_team,
                bookmaker, odds_h, odds_d, odds_a, fetched_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (liga, event["id"], event.get("commence_time"),
             event.get("home_team"), event.get("away_team"),
             bk["bookmaker"], bk["odds_h"], bk["odds_d"], bk["odds_a"], now_str))
        n_events += 1
    con.execute(
        "INSERT INTO odds_fetch_log (liga,fetched_at,requests_used,requests_remaining,n_events) VALUES (?,?,?,?,?)",
        (liga, now_str, req_used, req_remaining, n_events))
    con.commit()
    con.close()
    return {"ok": True, "from_cache": False, "n_events": n_events,
            "requests_used": req_used, "requests_remaining": req_remaining,
            "fetched_at": now_str}

def get_cached_odds(liga, db_file):
    try:
        _init_db(db_file)
        con  = sqlite3.connect(db_file)
        rows = con.execute(
            "SELECT home_team,away_team,odds_h,odds_d,odds_a,bookmaker,commence_ts FROM live_odds WHERE liga=?",
            (liga,)).fetchall()
        con.close()
        return {(h, a): {"odds_h": oh, "odds_d": od, "odds_a": oa,
                         "bookmaker": bk, "commence_ts": ts}
                for h, a, oh, od, oa, bk, ts in rows}
    except Exception:
        return {}

def get_usage_stats(db_file):
    try:
        _init_db(db_file)
        con = sqlite3.connect(db_file)
        ms  = datetime.now(timezone.utc).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
        rows = con.execute(
            "SELECT liga,fetched_at,requests_remaining FROM odds_fetch_log WHERE fetched_at>=? ORDER BY fetched_at DESC",
            (ms,)).fetchall()
        con.close()
        last_per_liga = {}
        for liga, fat, _ in rows:
            if liga not in last_per_liga:
                last_per_liga[liga] = fat
        return {"fetches_this_month": len(rows),
                "requests_remaining": rows[0][2] if rows else None,
                "last_per_liga": last_per_liga}
    except Exception:
        return {}

# ── Mapowanie nazw ─────────────────────────────────────────────────────────────
TEAM_NAME_MAP = {
    # Premier League
    "Manchester City": "Man City", "Manchester United": "Man United",
    "Tottenham Hotspur": "Tottenham", "Nottingham Forest": "Nott'm Forest",
    "Newcastle United": "Newcastle", "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves", "Brighton and Hove Albion": "Brighton",
    "AFC Bournemouth": "Bournemouth", "Ipswich Town": "Ipswich", "Leicester City": "Leicester",
    # La Liga
    "Athletic Club": "Ath Bilbao", "Atletico Madrid": "Ath Madrid",
    "Real Betis": "Betis", "Celta Vigo": "Celta", "Deportivo Alaves": "Alaves",
    "Rayo Vallecano": "Vallecano", "Real Sociedad": "Sociedad", "UD Las Palmas": "Las Palmas",
    "RCD Mallorca": "Mallorca", "RCD Espanyol": "Espanyol",
    "Girona FC": "Girona", "Villarreal CF": "Villarreal", "CD Leganes": "Leganes",
    # Bundesliga
    "Bayer Leverkusen": "Leverkusen", "Borussia Dortmund": "Dortmund",
    "Borussia Monchengladbach": "Gladbach", "Eintracht Frankfurt": "Ein Frankfurt",
    "Werder Bremen": "Werder", "SC Freiburg": "Freiburg", "FC Augsburg": "Augsburg",
    "FC Heidenheim 1846": "Heidenheim", "FC St. Pauli": "St Pauli",
    "Holstein Kiel": "Kiel", "VfL Bochum": "Bochum", "VfL Wolfsburg": "Wolfsburg",
    "VfB Stuttgart": "Stuttgart", "TSG Hoffenheim": "Hoffenheim", "RB Leipzig": "RB Leipzig",
    "1. FC Union Berlin": "Union Berlin", "1. FC Koln": "Koln",
    # Serie A
    "Internazionale": "Inter", "AC Milan": "Milan", "AS Roma": "Roma",
    "SS Lazio": "Lazio", "SSC Napoli": "Napoli", "Hellas Verona": "Verona",
    "Udinese Calcio": "Udinese", "US Lecce": "Lecce", "Genoa CFC": "Genoa",
    "Cagliari Calcio": "Cagliari", "Parma Calcio 1913": "Parma", "Venezia FC": "Venezia",
    "Como 1907": "Como", "AC Monza": "Monza", "Empoli FC": "Empoli",
    "Bologna FC 1909": "Bologna", "ACF Fiorentina": "Fiorentina",
    "Torino FC": "Torino", "Atalanta BC": "Atalanta",
    # Ligue 1
    "Paris Saint-Germain": "Paris SG", "Olympique de Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon", "AS Monaco": "Monaco", "OGC Nice": "Nice",
    "RC Lens": "Lens", "Stade Rennais FC": "Rennes", "Stade Brestois 29": "Brest",
    "Le Havre AC": "Le Havre", "Montpellier HSC": "Montpellier", "FC Nantes": "Nantes",
    "Toulouse FC": "Toulouse", "RC Strasbourg Alsace": "Strasbourg", "Angers SCO": "Angers",
    "AS Saint-Etienne": "St Etienne", "AJ Auxerre": "Auxerre",
    "Stade de Reims": "Reims", "Lille OSC": "Lille",
}
_REV_MAP = {v: k for k, v in TEAM_NAME_MAP.items()}

def map_api_to_model(api_name):
    return TEAM_NAME_MAP.get(api_name, api_name)

def znajdz_kursy(home_model, away_model, cached):
    home_api = _REV_MAP.get(home_model, home_model)
    away_api = _REV_MAP.get(away_model, away_model)
    if (home_api, away_api) in cached:
        return cached[(home_api, away_api)]
    if (home_model, away_model) in cached:
        return cached[(home_model, away_model)]
    cl = {(h.lower(), a.lower()): v for (h, a), v in cached.items()}
    for hk, ak in [(home_api.lower(), away_api.lower()),
                   (home_model.lower(), away_model.lower())]:
        if (hk, ak) in cl:
            return cl[(hk, ak)]
    for (h, a), v in cached.items():
        if map_api_to_model(h) == home_model and map_api_to_model(a) == away_model:
            return v
    return None
