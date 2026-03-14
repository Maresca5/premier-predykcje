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
                ht = home_team.lower().strip()
                n0 = nd[0].lower().strip()
                n1 = nd[1].lower().strip()
                if ht and (ht == n0 or ht in n0 or n0 in ht):
                    oh, oa = outcomes[nd[0]], outcomes[nd[1]]
                elif ht and (ht == n1 or ht in n1 or n1 in ht):
                    oh, oa = outcomes[nd[1]], outcomes[nd[0]]
                else:
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
    # ── PREMIER LEAGUE ───────────────────────────────────────────────────────
    "Manchester City":              "Man City",
    "Manchester United":            "Man United",
    "Tottenham Hotspur":            "Tottenham",
    "Nottingham Forest":            "Nott'm Forest",
    "Newcastle United":             "Newcastle",
    "West Ham United":              "West Ham",
    "Wolverhampton Wanderers":      "Wolves",
    "Brighton and Hove Albion":     "Brighton",
    "Brighton & Hove Albion":       "Brighton",
    "AFC Bournemouth":              "Bournemouth",
    "Bournemouth":                  "Bournemouth",
    "Ipswich Town":                 "Ipswich",
    "Leicester City":               "Leicester",
    "Arsenal":                      "Arsenal",
    "Chelsea":                      "Chelsea",
    "Liverpool":                    "Liverpool",
    "Everton":                      "Everton",
    "Aston Villa":                  "Aston Villa",
    "Crystal Palace":               "Crystal Palace",
    "Fulham":                       "Fulham",
    "Brentford":                    "Brentford",
    "Southampton":                  "Southampton",
    "Sunderland":                   "Sunderland",
    "Leeds United":                 "Leeds",
    "Burnley":                      "Burnley",
    "Sheffield United":             "Sheffield United",
    "Luton Town":                   "Luton",
    # ── LA LIGA ──────────────────────────────────────────────────────────────
    "Athletic Club":                "Ath Bilbao",
    "Athletic Bilbao":              "Ath Bilbao",
    "Atletico Madrid":              "Ath Madrid",
    "Atletico de Madrid":           "Ath Madrid",
    "Real Betis":                   "Betis",
    "Celta Vigo":                   "Celta",
    "Deportivo Alaves":             "Alaves",
    "Alaves":                       "Alaves",
    "Rayo Vallecano":               "Vallecano",
    "Real Sociedad":                "Sociedad",
    "UD Las Palmas":                "Las Palmas",
    "Las Palmas":                   "Las Palmas",
    "RCD Mallorca":                 "Mallorca",
    "Mallorca":                     "Mallorca",
    "RCD Espanyol":                 "Espanol",
    "Espanyol":                     "Espanol",
    "Girona FC":                    "Girona",
    "Girona":                       "Girona",
    "Villarreal CF":                "Villarreal",
    "Villarreal":                   "Villarreal",
    "CD Leganes":                   "Leganes",
    "Leganes":                      "Leganes",
    "FC Barcelona":                 "Barcelona",
    "Barcelona":                    "Barcelona",
    "Real Madrid":                  "Real Madrid",
    "Sevilla FC":                   "Sevilla",
    "Sevilla":                      "Sevilla",
    "Getafe CF":                    "Getafe",
    "Getafe":                       "Getafe",
    "CA Osasuna":                   "Osasuna",
    "Osasuna":                      "Osasuna",
    "Valencia CF":                  "Valencia",
    "Valencia":                     "Valencia",
    "Real Valladolid":              "Valladolid",
    # ── BUNDESLIGA ───────────────────────────────────────────────────────────
    "Bayern Munich":                "Bayern Munich",
    "FC Bayern Munich":             "Bayern Munich",
    "FC Bayern Munchen":            "Bayern Munich",
    "Bayer Leverkusen":             "Leverkusen",
    "Bayer 04 Leverkusen":          "Leverkusen",
    "Borussia Dortmund":            "Dortmund",
    "Borussia Monchengladbach":     "M'gladbach",
    "Borussia M'gladbach":          "M'gladbach",
    "Eintracht Frankfurt":          "Ein Frankfurt",
    "Werder Bremen":                "Werder Bremen",
    "SV Werder Bremen":             "Werder Bremen",
    "SC Freiburg":                  "Freiburg",
    "Freiburg":                     "Freiburg",
    "FC Augsburg":                  "Augsburg",
    "Augsburg":                     "Augsburg",
    "FC Heidenheim 1846":           "Heidenheim",
    "Heidenheim":                   "Heidenheim",
    "FC St. Pauli":                 "St Pauli",
    "St. Pauli":                    "St Pauli",
    "Holstein Kiel":                "Kiel",
    "VfL Bochum":                   "Bochum",
    "Bochum":                       "Bochum",
    "VfL Wolfsburg":                "Wolfsburg",
    "Wolfsburg":                    "Wolfsburg",
    "VfB Stuttgart":                "Stuttgart",
    "Stuttgart":                    "Stuttgart",
    "TSG Hoffenheim":               "Hoffenheim",
    "Hoffenheim":                   "Hoffenheim",
    "RB Leipzig":                   "RB Leipzig",
    "1. FC Union Berlin":           "Union Berlin",
    "Union Berlin":                 "Union Berlin",
    "1. FC Koln":                   "FC Koln",
    "FC Koln":                      "FC Koln",
    "1. FSV Mainz 05":              "Mainz",
    "FSV Mainz 05":                 "Mainz",
    "Mainz":                        "Mainz",
    "Hamburger SV":                 "Hamburg",
    # ── SERIE A ──────────────────────────────────────────────────────────────
    "Internazionale":               "Inter",
    "Inter Milan":                  "Inter",
    "AC Milan":                     "Milan",
    "AS Roma":                      "Roma",
    "SS Lazio":                     "Lazio",
    "Lazio":                        "Lazio",
    "SSC Napoli":                   "Napoli",
    "Napoli":                       "Napoli",
    "Hellas Verona":                "Verona",
    "Verona":                       "Verona",
    "Udinese Calcio":               "Udinese",
    "Udinese":                      "Udinese",
    "US Lecce":                     "Lecce",
    "Lecce":                        "Lecce",
    "Genoa CFC":                    "Genoa",
    "Genoa":                        "Genoa",
    "Cagliari Calcio":              "Cagliari",
    "Cagliari":                     "Cagliari",
    "Parma Calcio 1913":            "Parma",
    "Parma":                        "Parma",
    "Venezia FC":                   "Venezia",
    "Venezia":                      "Venezia",
    "Como 1907":                    "Como",
    "Como":                         "Como",
    "AC Monza":                     "Monza",
    "Monza":                        "Monza",
    "Empoli FC":                    "Empoli",
    "Empoli":                       "Empoli",
    "Bologna FC 1909":              "Bologna",
    "Bologna":                      "Bologna",
    "ACF Fiorentina":               "Fiorentina",
    "Fiorentina":                   "Fiorentina",
    "Torino FC":                    "Torino",
    "Torino":                       "Torino",
    "Atalanta BC":                  "Atalanta",
    "Atalanta":                     "Atalanta",
    "Juventus":                     "Juventus",
    # ── LIGUE 1 ──────────────────────────────────────────────────────────────
    "Paris Saint Germain":          "Paris SG",
    "Paris Saint-Germain":          "Paris SG",
    "Paris SG":                     "Paris SG",
    "PSG":                          "Paris SG",
    "Paris SG FC":                  "Paris SG",
    "Paris Saint-Germain FC":       "Paris SG",
    "Olympique de Marseille":       "Marseille",
    "Marseille":                    "Marseille",
    "Olympique Lyonnais":           "Lyon",
    "Lyon":                         "Lyon",
    "AS Monaco":                    "Monaco",
    "Monaco":                       "Monaco",
    "OGC Nice":                     "Nice",
    "Nice":                         "Nice",
    "RC Lens":                      "Lens",
    "Lens":                         "Lens",
    "Stade Rennais FC":             "Rennes",
    "Stade Rennais":                "Rennes",
    "Rennes":                       "Rennes",
    "Stade Brestois 29":            "Brest",
    "Stade Brestois":               "Brest",
    "Brest":                        "Brest",
    "Le Havre AC":                  "Le Havre",
    "Le Havre":                     "Le Havre",
    "Montpellier HSC":              "Montpellier",
    "Montpellier":                  "Montpellier",
    "FC Nantes":                    "Nantes",
    "Nantes":                       "Nantes",
    "Toulouse FC":                  "Toulouse",
    "Toulouse":                     "Toulouse",
    "RC Strasbourg Alsace":         "Strasbourg",
    "RC Strasbourg":                "Strasbourg",
    "Strasbourg":                   "Strasbourg",
    "Angers SCO":                   "Angers",
    "Angers":                       "Angers",
    "AS Saint-Etienne":             "St Etienne",
    "Saint-Etienne":                "St Etienne",
    "AJ Auxerre":                   "Auxerre",
    "Auxerre":                      "Auxerre",
    "Stade de Reims":               "Reims",
    "Reims":                        "Reims",
    "Lille OSC":                    "Lille",
    "Lille":                        "Lille",
}

# ── Reverse map: model_name -> [lista wszystkich możliwych nazw API] ──────────
# POPRAWKA: zamiast {v: k} (który traci duplikaty) budujemy {v: [k1, k2, ...]}
_REV_MAP_ALL: dict = {}
for _api_name, _model_name in TEAM_NAME_MAP.items():
    _REV_MAP_ALL.setdefault(_model_name, []).append(_api_name)

# Płaski dict dla kompatybilności wstecznej (pierwszy klucz)
_REV_MAP = {v: keys[0] for v, keys in _REV_MAP_ALL.items()}


def map_api_to_model(api_name: str) -> str:
    return TEAM_NAME_MAP.get(api_name, api_name)


def _normalize_fuzzy(s: str) -> str:
    """Normalizacja do fuzzy match: usuwa akcenty, sufiksy klubowe, znaki specjalne."""
    import unicodedata, re
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(
        r"\b(FC|SC|CF|AC|AS|RC|CD|RCD|SL|SS|US|SK|FK|BK|VfB|VfL|SV|FSV|TSG|BSC|"
        r"RB|KV|KRC|AFC|RFC|GFC|OGC|SM|UD|SD|CP|CA|CR|CE|1\.|04|Stade|Calcio|"
        r"Hotspur|United|City|Athletic|Albion|Wanderers|County)\b",
        "", s, flags=re.IGNORECASE)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


# Cache fuzzy żeby nie liczyć wielokrotnie
_fuzzy_cache: dict = {}


def _fuzzy_team_match(name: str, candidates: list, cutoff: float = 0.72) -> str | None:
    """Fuzzy matching nazwy drużyny. Zwraca najlepsze trafienie lub None."""
    import difflib
    if name in _fuzzy_cache:
        return _fuzzy_cache[name]

    name_l = name.lower().strip()
    cands_l = [c.lower().strip() for c in candidates]

    # Exact normalized
    for i, cl in enumerate(cands_l):
        if cl == name_l:
            _fuzzy_cache[name] = candidates[i]
            return candidates[i]

    # difflib bezpośrednio
    m = difflib.get_close_matches(name_l, cands_l, n=1, cutoff=cutoff)
    if m:
        idx = cands_l.index(m[0])
        _fuzzy_cache[name] = candidates[idx]
        return candidates[idx]

    # difflib po normalizacji (usuwa FC, akcenty itp.)
    fn   = _normalize_fuzzy(name)
    fmap = {_normalize_fuzzy(c): c for c in candidates}
    m2   = difflib.get_close_matches(fn, list(fmap.keys()), n=1, cutoff=0.55)
    if m2:
        result = fmap[m2[0]]
        _fuzzy_cache[name] = result
        return result

    # Token overlap: wszystkie słowa krótkiej nazwy zawierają się w długiej
    words_n = set(name_l.split())
    for i, cl in enumerate(cands_l):
        words_c = set(cl.split())
        if words_n and words_n.issubset(words_c):
            _fuzzy_cache[name] = candidates[i]
            return candidates[i]
        if words_c and words_c.issubset(words_n):
            _fuzzy_cache[name] = candidates[i]
            return candidates[i]

    _fuzzy_cache[name] = None
    return None


def znajdz_kursy(home_model: str, away_model: str, cached: dict):
    """
    Szuka kursów dla meczu home_model vs away_model w cache z Odds API.
    
    Pipeline (od najszybszego do najwolniejszego):
    1. Bezpośrednie trafienie (model name lub api name)
    2. Case-insensitive
    3. Wszystkie możliwe api_names z _REV_MAP_ALL (naprawia bug z PSG)
    4. map_api_to_model() dla każdego klucza w cached
    5. Fuzzy matching jako ostateczny fallback
    """
    if not cached:
        return None

    # ── Krok 1: Bezpośrednie trafienie ────────────────────────────────────
    if (home_model, away_model) in cached:
        return cached[(home_model, away_model)]

    # ── Krok 2: Wszystkie api_names z _REV_MAP_ALL ─────────────────────────
    # KLUCZOWA POPRAWKA: sprawdzamy WSZYSTKIE możliwe nazwy API (nie tylko ostatnią)
    home_api_names = _REV_MAP_ALL.get(home_model, [home_model])
    away_api_names = _REV_MAP_ALL.get(away_model, [away_model])
    for hk in home_api_names:
        for ak in away_api_names:
            if (hk, ak) in cached:
                return cached[(hk, ak)]

    # ── Krok 3: Case-insensitive ───────────────────────────────────────────
    cl_lower = {(h.lower(), a.lower()): v for (h, a), v in cached.items()}
    for hk in home_api_names + [home_model]:
        for ak in away_api_names + [away_model]:
            if (hk.lower(), ak.lower()) in cl_lower:
                return cl_lower[(hk.lower(), ak.lower())]

    # ── Krok 4: map_api_to_model() dla każdego klucza w cached ────────────
    for (h, a), v in cached.items():
        if map_api_to_model(h) == home_model and map_api_to_model(a) == away_model:
            return v

    # ── Krok 5: Fuzzy matching jako ostatni ratunek ─────────────────────────
    all_homes = list({h for (h, _) in cached})
    all_aways = list({a for (_, a) in cached})

    for search_h in [home_model] + home_api_names:
        fh = _fuzzy_team_match(search_h, all_homes)
        if fh:
            for search_a in [away_model] + away_api_names:
                fa = _fuzzy_team_match(search_a, all_aways)
                if fa and (fh, fa) in cached:
                    return cached[(fh, fa)]

    return None
