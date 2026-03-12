"""
understat_fetcher.py
====================
Pobiera dane xG/npxG/PPDA/deep z understat.com i zapisuje do SQLite.
Używany przez bot_runner.py (raz dziennie) i bezpośrednio z CLI.

Tabela: xg_cache
  team        TEXT  – nazwa drużyny (understat)
  team_mapped TEXT  – nazwa drużyny (football-data / app)
  liga        TEXT  – csv_code: E0/SP1/D1/I1/F1
  sezon       TEXT  – np. "2025" (rok startowy sezonu)
  mecz_data   TEXT  – YYYY-MM-DD
  home_team   TEXT
  away_team   TEXT
  xg_h        REAL  – xG gospodarza
  xg_a        REAL  – xG gościa
  npxg_h      REAL  – npxG (bez karnych) gospodarza
  npxg_a      REAL  – npxG gościa
  ppda_h      REAL  – PPDA gospodarza (niższy = większy pressing)
  ppda_a      REAL  – PPDA gościa
  deep_h      REAL  – podania w pole karne gospodarza
  deep_a      REAL  – podania w pole karne gościa
  wynik       TEXT  – "H"/"D"/"A"
  fetch_ts    TEXT  – timestamp pobrania

Tabela: xg_fetch_log
  liga        TEXT
  sezon       TEXT
  fetch_ts    TEXT
  n_mecze     INTEGER
  status      TEXT
"""

import sqlite3
import time
import json
import re
import logging
from datetime import datetime, timedelta
from typing import Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logger = logging.getLogger("understat_fetcher")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [XG] %(levelname)s %(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

# ── Mapowania liga: csv_code → understat name ──────────────────────────────
LIGA_UNDERSTAT = {
    "E0":  "EPL",
    "SP1": "La_Liga",
    "D1":  "Bundesliga",
    "I1":  "Serie_A",
    "F1":  "Ligue_1",
}

# ── Mapowania nazw drużyn understat → football-data ───────────────────────
# Format: "understat_name": "football-data_name"
# Niekompletne — aplikacja dopasowuje też fuzzy matchem
TEAM_NAME_MAP = {
    # EPL
    "Manchester United":    "Man United",
    "Manchester City":      "Man City",
    "Tottenham":            "Tottenham",
    "Newcastle United":     "Newcastle",
    "Wolverhampton Wanderers": "Wolves",
    "Brighton":             "Brighton",
    "West Ham":             "West Ham",
    "Aston Villa":          "Aston Villa",
    "Nottingham Forest":    "Nott'm Forest",
    "Leicester":            "Leicester",
    "Crystal Palace":       "Crystal Palace",
    "Brentford":            "Brentford",
    "Fulham":               "Fulham",
    "Everton":              "Everton",
    "Bournemouth":          "Bournemouth",
    "Southampton":          "Southampton",
    "Arsenal":              "Arsenal",
    "Chelsea":              "Chelsea",
    "Liverpool":            "Liverpool",
    "Ipswich":              "Ipswich",
    # La Liga
    "Real Madrid":          "Real Madrid",
    "Barcelona":            "Barcelona",
    "Atletico Madrid":      "Atletico Madrid",
    "Athletic Club":        "Athletic Club",
    "Real Sociedad":        "Real Sociedad",
    "Sevilla":              "Sevilla",
    "Villarreal":           "Villarreal",
    "Valencia":             "Valencia",
    "Osasuna":              "Osasuna",
    "Celta Vigo":           "Celta",
    "Getafe":               "Getafe",
    "Alaves":               "Alaves",
    "Las Palmas":           "Las Palmas",
    "Girona":               "Girona",
    "Rayo Vallecano":       "Rayo Vallecano",
    "Betis":                "Betis",
    "Mallorca":             "Mallorca",
    "Leganes":              "Leganes",
    "Espanyol":             "Espanyol",
    "Valladolid":           "Valladolid",
    # Bundesliga
    "Bayern Munich":        "Bayern Munich",
    "Borussia Dortmund":    "Dortmund",
    "RB Leipzig":           "RB Leipzig",
    "Bayer Leverkusen":     "Leverkusen",
    "Eintracht Frankfurt":  "Ein Frankfurt",
    "Wolfsburg":            "Wolfsburg",
    "Freiburg":             "Freiburg",
    "Union Berlin":         "Union Berlin",
    "Borussia Moenchengladbach": "M'gladbach",
    "Mainz":                "Mainz",
    "Hoffenheim":           "Hoffenheim",
    "Werder Bremen":        "Werder Bremen",
    "Stuttgart":            "Stuttgart",
    "Augsburg":             "Augsburg",
    "Bochum":               "Bochum",
    "Heidenheim":           "Heidenheim",
    "Holstein Kiel":        "Holstein Kiel",
    "St. Pauli":            "St. Pauli",
    # Serie A
    "AC Milan":             "Milan",
    "Inter":                "Inter",
    "Juventus":             "Juventus",
    "Napoli":               "Napoli",
    "AS Roma":              "Roma",
    "Atalanta":             "Atalanta",
    "Lazio":                "Lazio",
    "Fiorentina":           "Fiorentina",
    "Bologna":              "Bologna",
    "Torino":               "Torino",
    "Udinese":              "Udinese",
    "Genoa":                "Genoa",
    "Cagliari":             "Cagliari",
    "Hellas Verona":        "Verona",
    "Lecce":                "Lecce",
    "Parma":                "Parma",
    "Empoli":               "Empoli",
    "Monza":                "Monza",
    "Venezia":              "Venezia",
    "Como":                 "Como",
    # Ligue 1
    "Paris Saint-Germain":  "Paris SG",
    "Marseille":            "Marseille",
    "Monaco":               "Monaco",
    "Lens":                 "Lens",
    "Nice":                 "Nice",
    "Lyon":                 "Lyon",
    "Lille":                "Lille",
    "Rennes":               "Rennes",
    "Toulouse":             "Toulouse",
    "Strasbourg":           "Strasbourg",
    "Brest":                "Brest",
    "Nantes":               "Nantes",
    "Reims":                "Reims",
    "Montpellier":          "Montpellier",
    "Le Havre":             "Le Havre",
    "Auxerre":              "Auxerre",
    "Angers":               "Angers",
    "Saint-Etienne":        "St Etienne",
}

XG_DB_FILE = "xg_cache.db"
COOLDOWN_HOURS = 20       # nie pobieraj częściej niż co 20h per liga/sezon
REQUEST_DELAY  = 2.5      # sekundy między requestami (grzeczne scrapowanie)
MAX_RETRIES    = 3


def _ensure_db(db_file: str = XG_DB_FILE):
    con = sqlite3.connect(db_file)
    con.execute("""
        CREATE TABLE IF NOT EXISTS xg_cache (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            liga        TEXT NOT NULL,
            sezon       TEXT NOT NULL,
            mecz_data   TEXT,
            home_team   TEXT,
            away_team   TEXT,
            home_mapped TEXT,
            away_mapped TEXT,
            xg_h        REAL,
            xg_a        REAL,
            npxg_h      REAL,
            npxg_a      REAL,
            ppda_h      REAL,
            ppda_a      REAL,
            deep_h      REAL,
            deep_a      REAL,
            wynik       TEXT,
            fetch_ts    TEXT,
            UNIQUE(liga, sezon, mecz_data, home_team, away_team)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS xg_fetch_log (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            liga      TEXT,
            sezon     TEXT,
            fetch_ts  TEXT,
            n_mecze   INTEGER,
            status    TEXT
        )
    """)
    con.commit()
    con.close()


def _map_team(name: str) -> str:
    """Mapuje nazwę understat → football-data. Fallback: oryginalna nazwa."""
    return TEAM_NAME_MAP.get(name, name)


def _biezacy_sezon_understat() -> str:
    """Zwraca rok startowy sezonu jako string, np. '2025' dla 2025/26."""
    y = datetime.now().year if datetime.now().month >= 7 else datetime.now().year - 1
    return str(y)


def _should_fetch(liga: str, sezon: str, db_file: str = XG_DB_FILE) -> bool:
    """Sprawdza czy można pobierać (cooldown per liga/sezon)."""
    try:
        con = sqlite3.connect(db_file)
        row = con.execute(
            "SELECT fetch_ts FROM xg_fetch_log WHERE liga=? AND sezon=? "
            "AND status='ok' ORDER BY id DESC LIMIT 1",
            (liga, sezon)).fetchone()
        con.close()
        if not row:
            return True
        last = datetime.fromisoformat(row[0])
        return (datetime.now() - last).total_seconds() > COOLDOWN_HOURS * 3600
    except Exception:
        return True


def _fetch_understat_page(url: str, retries: int = MAX_RETRIES) -> Optional[str]:
    """Pobiera HTML strony understat z retry."""
    if not HAS_REQUESTS:
        logger.error("Brak biblioteki 'requests'. Zainstaluj: pip install requests")
        return None
    headers = {
        "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36"),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://understat.com/",
    }
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            if resp.status_code == 200:
                return resp.text
            elif resp.status_code == 429:
                wait = 30 * (attempt + 1)
                logger.warning(f"Rate limit (429) — czekam {wait}s")
                time.sleep(wait)
            else:
                logger.warning(f"HTTP {resp.status_code} dla {url}")
                time.sleep(5)
        except Exception as e:
            logger.warning(f"Request error (próba {attempt+1}): {e}")
            time.sleep(5 * (attempt + 1))
    return None


def _extract_matches_from_html(html: str) -> Optional[list]:
    """
    Wyciąga dane meczowe z HTML understat.com.
    Understat osadza dane jako JSON.parse('...') w tagach <script>.
    Używamy BeautifulSoup aby znaleźć właściwy skrypt (zawiera 'datesData'),
    a następnie regex + unicode_escape do dekodowania.
    Fallback: szukamy w każdym skrypcie dopóki nie znajdziemy datesData.
    """
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        scripts = soup.find_all("script")
    except ImportError:
        # Fallback bez BS4 — szukamy po regex w surowym HTML
        scripts = None

    # Wzorzec pasujący do: var datesData  = JSON.parse('...')
    pattern = re.compile(r"var\s+datesData\s*=\s*JSON\.parse\('(.+?)'\)", re.DOTALL)

    candidates = []
    if scripts:
        # Sprawdź każdy skrypt, ze szczególnym uwzględnieniem index 2
        for i, s in enumerate(scripts):
            txt = s.string or ""
            candidates.append((i, txt))
    else:
        # Bez BS4 — skanuj surowy HTML
        candidates = [(0, html)]

    for idx, txt in candidates:
        m = pattern.search(txt)
        if not m:
            continue
        raw = m.group(1)
        # Understat enkoduje dane przez JSON.stringify + encodeURIComponent
        # Wynik: unicode escapes (\uXXXX) i escape apostrofów (\')
        try:
            # Metoda 1: bezpośrednie json.loads po cleanup
            cleaned = raw.replace("\\'", "'")
            return json.loads(cleaned)
        except Exception:
            pass
        try:
            # Metoda 2: unicode_escape decode
            decoded = raw.encode("utf-8").decode("unicode_escape")
            decoded = decoded.replace("\\'", "'")
            return json.loads(decoded)
        except Exception:
            pass
        try:
            # Metoda 3: codecs
            import codecs
            decoded2 = codecs.decode(raw, "unicode_escape")
            return json.loads(decoded2)
        except Exception:
            pass

    return None


def fetch_liga_xg(liga_csv: str, sezon: Optional[str] = None,
                  db_file: str = XG_DB_FILE, force: bool = False) -> dict:
    """
    Pobiera dane xG dla całej ligi za dany sezon.
    Zwraca {"status": "ok"/"skipped"/"error", "n_mecze": N, "liga": ..., "sezon": ...}
    """
    if sezon is None:
        sezon = _biezacy_sezon_understat()

    understat_liga = LIGA_UNDERSTAT.get(liga_csv)
    if not understat_liga:
        return {"status": "error", "msg": f"Nieznana liga: {liga_csv}"}

    if not force and not _should_fetch(liga_csv, sezon, db_file):
        logger.info(f"Pomijam {liga_csv}/{sezon} — cooldown aktywny")
        return {"status": "skipped", "liga": liga_csv, "sezon": sezon}

    _ensure_db(db_file)
    url = f"https://understat.com/league/{understat_liga}/{sezon}"
    logger.info(f"Pobieranie {liga_csv}/{sezon}: {url}")

    html = _fetch_understat_page(url)
    if not html:
        return {"status": "error", "msg": "Brak odpowiedzi z understat.com", "liga": liga_csv}

    # Understat osadza datesData z wynikami meczów per kolejka
    dates_data = _extract_matches_from_html(html)
    if not dates_data:
        # Zapisz fragment HTML do logu żeby móc debugować
        logger.error(f"Nie znaleziono datesData dla {liga_csv}/{sezon}. "
                     f"Rozmiar HTML: {len(html)} znaków. "
                     f"Fragment: {html[1000:1200] if len(html)>1200 else html[:200]}")
        return {"status": "error", "msg": "Brak datesData w HTML", "liga": liga_csv}

    con = sqlite3.connect(db_file)
    n_inserted = 0
    n_updated  = 0
    ts = datetime.now().isoformat()

    for mecz in dates_data:
        try:
            # Tylko rozegrane mecze (isResult=True)
            if not mecz.get("isResult"):
                continue

            date_str  = mecz.get("datetime", "")[:10]
            home_name = mecz.get("h", {}).get("title", "")
            away_name = mecz.get("a", {}).get("title", "")
            if not home_name or not away_name:
                continue

            h_goals = int(mecz.get("goals", {}).get("h", 0) or 0)
            a_goals = int(mecz.get("goals", {}).get("a", 0) or 0)
            wynik   = "H" if h_goals > a_goals else ("A" if a_goals > h_goals else "D")

            xg_h    = float(mecz.get("xG",    {}).get("h", 0) or 0)
            xg_a    = float(mecz.get("xG",    {}).get("a", 0) or 0)
            npxg_h  = float(mecz.get("npxG",  {}).get("h", 0) or 0)
            npxg_a  = float(mecz.get("npxG",  {}).get("a", 0) or 0)
            ppda_h  = float(mecz.get("ppda",  {}).get("h", 0) or 0)
            ppda_a  = float(mecz.get("ppda",  {}).get("a", 0) or 0)
            deep_h  = float(mecz.get("deep",  {}).get("h", 0) or 0)
            deep_a  = float(mecz.get("deep",  {}).get("a", 0) or 0)

            # PPDA: understat podaje {att, def} — liczymy ratio
            ppda_raw_h = mecz.get("ppda", {}).get("h", {})
            ppda_raw_a = mecz.get("ppda", {}).get("a", {})
            if isinstance(ppda_raw_h, dict):
                att_h = float(ppda_raw_h.get("att", 0) or 0)
                def_h = float(ppda_raw_h.get("def", 1) or 1)
                ppda_h = round(att_h / def_h, 3) if def_h > 0 else 0
            if isinstance(ppda_raw_a, dict):
                att_a = float(ppda_raw_a.get("att", 0) or 0)
                def_a = float(ppda_raw_a.get("def", 1) or 1)
                ppda_a = round(att_a / def_a, 3) if def_a > 0 else 0

            home_mapped = _map_team(home_name)
            away_mapped = _map_team(away_name)

            con.execute("""
                INSERT INTO xg_cache
                  (liga, sezon, mecz_data, home_team, away_team,
                   home_mapped, away_mapped,
                   xg_h, xg_a, npxg_h, npxg_a,
                   ppda_h, ppda_a, deep_h, deep_a,
                   wynik, fetch_ts)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(liga, sezon, mecz_data, home_team, away_team)
                DO UPDATE SET
                  xg_h=excluded.xg_h, xg_a=excluded.xg_a,
                  npxg_h=excluded.npxg_h, npxg_a=excluded.npxg_a,
                  ppda_h=excluded.ppda_h, ppda_a=excluded.ppda_a,
                  deep_h=excluded.deep_h, deep_a=excluded.deep_a,
                  wynik=excluded.wynik, fetch_ts=excluded.fetch_ts,
                  home_mapped=excluded.home_mapped, away_mapped=excluded.away_mapped
            """, (liga_csv, sezon, date_str, home_name, away_name,
                  home_mapped, away_mapped,
                  xg_h, xg_a, npxg_h, npxg_a,
                  ppda_h, ppda_a, deep_h, deep_a,
                  wynik, ts))
            n_inserted += 1

        except Exception as e:
            logger.warning(f"Błąd parsowania meczu: {e} — {mecz}")
            continue

    con.execute(
        "INSERT INTO xg_fetch_log (liga, sezon, fetch_ts, n_mecze, status) VALUES (?,?,?,?,?)",
        (liga_csv, sezon, ts, n_inserted, "ok"))
    con.commit()
    con.close()

    logger.info(f"✅ {liga_csv}/{sezon}: zapisano {n_inserted} meczów xG")
    return {"status": "ok", "liga": liga_csv, "sezon": sezon, "n_mecze": n_inserted}


def fetch_all_ligas(db_file: str = XG_DB_FILE, force: bool = False,
                    include_prev: bool = True) -> list:
    """
    Pobiera xG dla wszystkich 5 lig (bieżący sezon + opcjonalnie poprzedni).
    Throttling: REQUEST_DELAY sekund między ligami.
    """
    results = []
    sezon_curr = _biezacy_sezon_understat()
    sezon_prev = str(int(sezon_curr) - 1)
    sezony = [sezon_curr]
    if include_prev:
        sezony.append(sezon_prev)

    for liga in list(LIGA_UNDERSTAT.keys()):
        for sezon in sezony:
            r = fetch_liga_xg(liga, sezon, db_file=db_file, force=force)
            results.append(r)
            if r["status"] == "ok":
                time.sleep(REQUEST_DELAY)
    return results


# ── API dla app.py i bot_runner.py ─────────────────────────────────────────

def get_team_xg_stats(team_mapped: str, liga: str,
                      db_file: str = XG_DB_FILE,
                      n_mecze: int = 10) -> Optional[dict]:
    """
    Zwraca agregowane statystyki xG dla drużyny (ostatnie n_mecze).
    team_mapped: nazwa z football-data (np. "Man City")
    Zwraca dict z kluczami:
      xg_scored_avg, xg_conceded_avg, npxg_scored_avg, npxg_conceded_avg,
      ppda_avg, deep_avg, n (liczba meczów), xg_diff (xg_scored - xg_conceded)
    """
    try:
        sezon = _biezacy_sezon_understat()
        con   = sqlite3.connect(db_file)
        _ensure_db(db_file)

        # Mecze jako gospodarz
        home = con.execute("""
            SELECT xg_h, xg_a, npxg_h, npxg_a, ppda_h, deep_h
            FROM xg_cache
            WHERE liga=? AND sezon=? AND home_mapped=?
            ORDER BY mecz_data DESC LIMIT ?
        """, (liga, sezon, team_mapped, n_mecze)).fetchall()

        # Mecze jako gość
        away = con.execute("""
            SELECT xg_a, xg_h, npxg_a, npxg_h, ppda_a, deep_a
            FROM xg_cache
            WHERE liga=? AND sezon=? AND away_mapped=?
            ORDER BY mecz_data DESC LIMIT ?
        """, (liga, sezon, team_mapped, n_mecze)).fetchall()

        con.close()

        all_rows = list(home) + list(away)
        if len(all_rows) < 3:
            return None  # za mało danych

        n = len(all_rows)
        xg_s   = sum(r[0] for r in all_rows) / n
        xg_c   = sum(r[1] for r in all_rows) / n
        npxg_s = sum(r[2] for r in all_rows) / n
        npxg_c = sum(r[3] for r in all_rows) / n
        ppda   = [r[4] for r in all_rows if r[4] and r[4] > 0]
        deep   = [r[5] for r in all_rows if r[5] is not None]

        return {
            "n":                  n,
            "xg_scored_avg":      round(xg_s, 3),
            "xg_conceded_avg":    round(xg_c, 3),
            "npxg_scored_avg":    round(npxg_s, 3),
            "npxg_conceded_avg":  round(npxg_c, 3),
            "ppda_avg":           round(sum(ppda)/len(ppda), 2) if ppda else None,
            "deep_avg":           round(sum(deep)/len(deep), 1) if deep else None,
            "xg_diff":            round(xg_s - xg_c, 3),
            "npxg_diff":          round(npxg_s - npxg_c, 3),
        }
    except Exception as e:
        logger.warning(f"get_team_xg_stats error ({team_mapped}): {e}")
        return None


def get_league_xg_table(liga: str, db_file: str = XG_DB_FILE) -> list:
    """
    Zwraca tabelę xG per drużyna dla całej ligi (bieżący sezon).
    Lista dict posortowana po xg_diff desc.
    """
    try:
        sezon = _biezacy_sezon_understat()
        con   = sqlite3.connect(db_file)
        _ensure_db(db_file)

        rows = con.execute("""
            SELECT home_mapped, away_mapped,
                   xg_h, xg_a, npxg_h, npxg_a,
                   ppda_h, ppda_a, deep_h, deep_a
            FROM xg_cache
            WHERE liga=? AND sezon=?
        """, (liga, sezon)).fetchall()
        con.close()

        if not rows:
            return []

        teams = {}
        for r in rows:
            h_m, a_m = r[0], r[1]
            for team, xg_s, xg_c, npxg_s, npxg_c, ppda, deep in [
                (h_m, r[2], r[3], r[4], r[5], r[6], r[8]),
                (a_m, r[3], r[2], r[5], r[4], r[7], r[9]),
            ]:
                if not team: continue
                if team not in teams:
                    teams[team] = {"team": team, "n": 0,
                                   "xg_s": 0, "xg_c": 0,
                                   "npxg_s": 0, "npxg_c": 0,
                                   "ppda": [], "deep": []}
                t = teams[team]
                t["n"]      += 1
                t["xg_s"]   += xg_s or 0
                t["xg_c"]   += xg_c or 0
                t["npxg_s"] += npxg_s or 0
                t["npxg_c"] += npxg_c or 0
                if ppda and ppda > 0: t["ppda"].append(ppda)
                if deep is not None:  t["deep"].append(deep)

        result = []
        for t in teams.values():
            n = max(t["n"], 1)
            ppda_list = t["ppda"]
            deep_list = t["deep"]
            result.append({
                "team":        t["team"],
                "n":           t["n"],
                "xg_scored":   round(t["xg_s"]   / n, 2),
                "xg_conceded": round(t["xg_c"]   / n, 2),
                "npxg_scored": round(t["npxg_s"] / n, 2),
                "npxg_diff":   round((t["npxg_s"] - t["npxg_c"]) / n, 2),
                "xg_diff":     round((t["xg_s"]   - t["xg_c"])   / n, 2),
                "ppda":        round(sum(ppda_list)/len(ppda_list), 1) if ppda_list else None,
                "deep":        round(sum(deep_list)/len(deep_list), 1) if deep_list else None,
            })

        return sorted(result, key=lambda x: x["xg_diff"], reverse=True)
    except Exception as e:
        logger.warning(f"get_league_xg_table error ({liga}): {e}")
        return []


def get_last_fetch_info(db_file: str = XG_DB_FILE) -> dict:
    """Zwraca info o ostatnim pobraniu per liga."""
    try:
        con = sqlite3.connect(db_file)
        _ensure_db(db_file)
        rows = con.execute("""
            SELECT liga, sezon, fetch_ts, n_mecze FROM xg_fetch_log
            WHERE status='ok'
            ORDER BY id DESC
        """).fetchall()
        con.close()
        result = {}
        for liga, sezon, ts, n in rows:
            if liga not in result:
                result[liga] = {"sezon": sezon, "ts": ts, "n_mecze": n}
        return result
    except Exception:
        return {}


# ── CLI entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Understat xG fetcher")
    parser.add_argument("--liga",    default=None, help="E0/SP1/D1/I1/F1 lub 'all'")
    parser.add_argument("--sezon",   default=None, help="rok startowy, np. 2025")
    parser.add_argument("--force",   action="store_true", help="Ignoruj cooldown")
    parser.add_argument("--db",      default=XG_DB_FILE, help="Ścieżka do SQLite")
    parser.add_argument("--no-prev", action="store_true", help="Nie pobieraj poprzedniego sezonu")
    args = parser.parse_args()

    if args.liga == "all" or args.liga is None:
        res = fetch_all_ligas(db_file=args.db, force=args.force,
                              include_prev=not args.no_prev)
        for r in res:
            print(r)
    else:
        r = fetch_liga_xg(args.liga, args.sezon, db_file=args.db, force=args.force)
        print(r)
