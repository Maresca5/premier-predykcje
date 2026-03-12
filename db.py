"""
db.py – baza SQLite: inicjalizacja, zapis/odczyt predykcji, bankroll, paper trades.
"""
import sqlite3
import pandas as pd
import numpy as np
import requests
from datetime import datetime

from config import DB_FILE, LIGI, BIEZACY_SEZON

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
            sezon       TEXT,
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
            UNIQUE(liga, sezon, kolejnosc, mecz, rynek, typ, linia)
        )
    """)
    # Migracja: dodaj kolumnę sezon do istniejących baz (bez niej ALTER TABLE nie przejdzie)
    try:
        con.execute("ALTER TABLE zdarzenia ADD COLUMN sezon TEXT")
        # Wypełnij stare rekordy (bez sezonu) wartością poprzedniego sezonu
        _prev = f"{str(int(BIEZACY_SEZON[:2])-1):02d}{BIEZACY_SEZON[:2]}"
        con.execute("UPDATE zdarzenia SET sezon=? WHERE sezon IS NULL", (_prev,))
        con.commit()
    except Exception:
        pass  # kolumna już istnieje
    
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

    # Paper trading – zakłady sugerowane przez Kelly
    con.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            liga        TEXT NOT NULL,
            kolejnosc   INTEGER NOT NULL,
            mecz        TEXT NOT NULL,
            home        TEXT NOT NULL,
            away        TEXT NOT NULL,
            rynek       TEXT NOT NULL,
            typ         TEXT NOT NULL,
            p_model     REAL,
            fair_odds   REAL,
            kelly_frac  REAL,
            stawka      REAL NOT NULL,
            bankroll_przed REAL,
            status      TEXT DEFAULT 'oczekuje',
            trafiony    INTEGER,
            wynik_meczu TEXT,
            pnl         REAL,
            bankroll_po REAL,
            data_zapisu TEXT,
            data_wyniku TEXT,
            UNIQUE(liga, kolejnosc, mecz, rynek, typ)
        )
    """)
    con.commit()
    con.close()

# ===========================================================================
# ŁADOWANIE DANYCH – BLEND SEZONÓW
# ===========================================================================

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

def pobierz_aktualny_bankroll(liga: str, start: float = 1000.0) -> float:
    """Zwraca aktualny stan bankrollu (po paper trades i korektach)."""
    con = sqlite3.connect(DB_FILE)
    _ensure_paper_trades_table(con)
    # Sprawdź paper trades które mają wynik
    row = con.execute(
        "SELECT bankroll_po FROM paper_trades WHERE liga=? AND bankroll_po IS NOT NULL "
        "ORDER BY id DESC LIMIT 1", (liga,)).fetchone()
    if row:
        cap = float(row[0])
    else:
        # Fallback: bankroll z tabeli bankroll
        row2 = con.execute(
            "SELECT kapital_po FROM bankroll WHERE liga=? ORDER BY id DESC LIMIT 1",
            (liga,)).fetchone()
        cap = float(row2[0]) if row2 else start
    con.close()
    return cap

def zapisz_paper_trades(liga: str, kolejnosc: int, trades: list, bankroll_przed: float) -> int:
    """
    Zapisuje listę zakładów Kelly do paper_trades.
    trades = [{"mecz", "home", "away", "rynek", "typ", "p_model",
               "fair_odds", "kelly_frac", "stawka"}, ...]
    Zwraca liczbę zapisanych rekordów.
    """
    con = sqlite3.connect(DB_FILE)
    _ensure_paper_trades_table(con)
    saved = 0
    for t in trades:
        try:
            con.execute(
                """INSERT OR IGNORE INTO paper_trades
                   (liga, kolejnosc, mecz, home, away, rynek, typ,
                    p_model, fair_odds, kelly_frac, stawka, bankroll_przed,
                    status, data_zapisu)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'oczekuje',?)""",
                (liga, kolejnosc, t["mecz"], t["home"], t["away"],
                 t["rynek"], t["typ"], t["p_model"], t["fair_odds"],
                 t["kelly_frac"], t["stawka"], bankroll_przed,
                 datetime.now().strftime("%Y-%m-%d %H:%M"))
            )
            saved += con.execute("SELECT changes()").fetchone()[0]
        except Exception:
            pass
    con.commit()
    con.close()
    return saved

def zapisz_real_odds(trade_id: int, real_odds: float) -> bool:
    """Zapisuje rzeczywisty kurs osiągnięty u bukmachera dla paper trade."""
    con = sqlite3.connect(DB_FILE)
    _ensure_paper_trades_table(con)
    try:
        # Pobierz dane do przeliczenia pnl_real
        row = con.execute(
            "SELECT stawka, trafiony, pnl, fair_odds FROM paper_trades WHERE id=?",
            (trade_id,)).fetchone()
        if row:
            stawka, trafiony, pnl_fair, fair_o = row
            if trafiony is not None:  # już rozliczone – przelicz pnl_real
                pnl_real = round(stawka * (real_odds - 1), 2) if trafiony else round(-stawka, 2)
            else:
                pnl_real = None
            con.execute(
                "UPDATE paper_trades SET real_odds=?, pnl_real=? WHERE id=?",
                (real_odds, pnl_real, trade_id))
            con.commit()
            return True
        return False
    except Exception:
        return False
    finally:
        con.close()


def usun_paper_trade(trade_id: int) -> bool:
    """Usuwa pojedynczy paper trade (tylko ze statusem oczekuje)."""
    con = sqlite3.connect(DB_FILE)
    _ensure_paper_trades_table(con)
    try:
        con.execute(
            "DELETE FROM paper_trades WHERE id=? AND status='oczekuje'", (trade_id,))
        con.commit()
        deleted = con.execute("SELECT changes()").fetchone()[0]
        return deleted > 0
    except Exception:
        return False
    finally:
        con.close()

def pobierz_paper_trades(liga: str, kolejnosc: int = None,
                          status: str = None) -> pd.DataFrame:
    """Pobiera paper trades z DB."""
    con = sqlite3.connect(DB_FILE)
    _ensure_paper_trades_table(con)
    q = "SELECT * FROM paper_trades WHERE liga=?"
    params = [liga]
    if kolejnosc:
        q += " AND kolejnosc=?"; params.append(kolejnosc)
    if status:
        q += " AND status=?"; params.append(status)
    q += " ORDER BY id DESC"
    df = pd.read_sql_query(q, con, params=params)
    con.close()
    return df

def _ensure_paper_trades_table(con) -> None:
    """
    Tworzy tabelę paper_trades jeśli nie istnieje.
    Jeśli istnieje – dodaje brakujące kolumny (migracja starych baz).
    """
    con.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            liga        TEXT NOT NULL,
            kolejnosc   INTEGER NOT NULL,
            mecz        TEXT NOT NULL,
            home        TEXT NOT NULL,
            away        TEXT NOT NULL,
            rynek       TEXT NOT NULL,
            typ         TEXT NOT NULL,
            p_model     REAL,
            fair_odds   REAL,
            kelly_frac  REAL,
            stawka      REAL NOT NULL,
            bankroll_przed REAL,
            status      TEXT DEFAULT 'oczekuje',
            trafiony    INTEGER,
            wynik_meczu TEXT,
            pnl         REAL,
            bankroll_po REAL,
            data_zapisu TEXT,
            data_wyniku TEXT,
            linia       REAL
        )
    """)
    # Migracja: dodaj kolumny których może brakować w starych bazach
    existing = {row[1] for row in con.execute("PRAGMA table_info(paper_trades)").fetchall()}
    for col, definition in [
        ("linia",          "REAL"),
        ("bankroll_przed", "REAL"),
        ("bankroll_po",    "REAL"),
        ("data_zapisu",    "TEXT"),
        ("data_wyniku",    "TEXT"),
        ("wynik_meczu",    "TEXT"),
        ("pnl",            "REAL"),
        ("trafiony",       "INTEGER"),
        ("kelly_frac",     "REAL"),
        ("p_model",        "REAL"),
        ("fair_odds",      "REAL"),
        ("real_odds",      "REAL"),   # kurs rzeczywisty u bukmachera (slippage)
        ("pnl_real",       "REAL"),   # PnL liczony z real_odds
    ]:
        if col not in existing:
            try:
                con.execute(f"ALTER TABLE paper_trades ADD COLUMN {col} {definition}")
            except Exception:
                pass
    con.commit()


def rozlicz_paper_trades(liga: str, hist: pd.DataFrame) -> dict:
    """
    Po aktualizacji wyników sprawdza oczekujące paper trades i rozlicza je.
    Aktualizuje pnl, bankroll_po, status.
    Zwraca {"rozliczone": N, "trafione": M, "pnl_total": X, "bankroll_po": Y}
    """
    if hist.empty:
        return {"rozliczone": 0, "trafione": 0, "pnl_total": 0.0, "bankroll_po": None}
    hist_biezacy = hist[hist["_sezon"] == "biezacy"] if "_sezon" in hist.columns else hist

    con = sqlite3.connect(DB_FILE)
    _ensure_paper_trades_table(con)
    oczekujace = con.execute(
        "SELECT id, home, away, rynek, typ, linia, stawka, fair_odds, bankroll_przed "
        "FROM paper_trades WHERE liga=? AND status='oczekuje'",
        (liga,)).fetchall()

    if not oczekujace:
        con.close()
        return {"rozliczone": 0, "trafione": 0, "pnl_total": 0.0, "bankroll_po": None}

    rozliczone = 0; trafione_cnt = 0; pnl_total = 0.0
    bk_po = None

    for row in oczekujace:
        tid, home, away, rynek, typ, linia, stawka, fair_odds, bk_przed = row
        # Znajdź wynik meczu
        match = hist_biezacy[
            (hist_biezacy["HomeTeam"] == home) &
            (hist_biezacy["AwayTeam"] == away)]
        if match.empty:
            continue

        m = match.iloc[-1]
        hg = int(m["FTHG"]); ag = int(m["FTAG"])
        wynik_str = f"{hg}:{ag}"
        wynik_1x2 = "1" if hg > ag else ("2" if ag > hg else "X")
        rzuty = int(m.get("HC", 0) or 0) + int(m.get("AC", 0) or 0)
        kartki = (int(m.get("HY", 0) or 0) + int(m.get("AY", 0) or 0) +
                  (int(m.get("HR", 0) or 0) + int(m.get("AR", 0) or 0)) * 2)

        traf = False
        if rynek == "1X2":
            traf = typ in (wynik_1x2,) or (typ == "1X" and wynik_1x2 in ("1","X")) or (typ == "X2" and wynik_1x2 in ("X","2"))
        elif rynek == "Gole":
            lin = float(linia) if linia else 2.5
            traf = (hg + ag) > lin if "Over" in typ else (hg + ag) < lin
        elif rynek == "BTTS":
            traf = (hg > 0 and ag > 0) if "Tak" in typ else (hg == 0 or ag == 0)
        elif rynek == "Rożne":
            traf = rzuty > float(linia or 8.5)
        elif rynek == "Kartki":
            traf = kartki > float(linia or 3.5)

        pnl = round(stawka * (fair_odds - 1), 2) if traf else round(-stawka, 2)
        # Pobierz bieżący bankroll przed tym trade
        last_bk = con.execute(
            "SELECT bankroll_po FROM paper_trades WHERE liga=? AND bankroll_po IS NOT NULL "
            "AND id < ? ORDER BY id DESC LIMIT 1", (liga, tid)).fetchone()
        bk_aktualny = float(last_bk[0]) if last_bk else float(bk_przed or 1000.0)
        bk_nowy = round(bk_aktualny + pnl, 2)

        # Sprawdź real_odds – jeśli podane, przelicz pnl_real
        real_o = con.execute("SELECT real_odds FROM paper_trades WHERE id=?", (tid,)).fetchone()
        real_o = float(real_o[0]) if real_o and real_o[0] else None
        pnl_real = round(stawka * (real_o - 1), 2) if (traf and real_o) else (round(-stawka, 2) if real_o else None)
        con.execute(
            """UPDATE paper_trades SET status='rozliczony', trafiony=?, wynik_meczu=?,
               pnl=?, bankroll_po=?, data_wyniku=?, pnl_real=? WHERE id=?""",
            (int(traf), wynik_str, pnl, bk_nowy,
             datetime.now().strftime("%Y-%m-%d %H:%M"), pnl_real, tid))

        rozliczone += 1
        if traf: trafione_cnt += 1
        pnl_total += pnl
        bk_po = bk_nowy

    con.commit()
    con.close()
    return {"rozliczone": rozliczone, "trafione": trafione_cnt,
            "pnl_total": round(pnl_total, 2), "bankroll_po": bk_po}


def zapisz_zdarzenia(liga: str, kolejnosc: int, mecz: str, home: str, away: str,
                     rynek: str, typ: str, linia: float, p_model: float, fair_odds: float,
                     sezon: str = None):
    """Zapisuje pojedyncze zdarzenie do bazy (do późniejszej weryfikacji)"""
    if sezon is None:
        sezon = BIEZACY_SEZON
    init_db()
    con = sqlite3.connect(DB_FILE)
    try:
        con.execute(
            """INSERT OR IGNORE INTO zdarzenia 
               (liga, sezon, kolejnosc, mecz, home, away, rynek, linia, typ, p_model, fair_odds, data)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (liga, sezon, kolejnosc, mecz, home, away, rynek, linia, typ, p_model, fair_odds,
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


def aktualizuj_wyniki_z_api(fd_org_id: int, api_key: str) -> int:
    """Pobiera wyniki zakończonych meczów z fd.org API i aktualizuje bazę.
    Szybsze niż CSV – dane dostępne ~15min po meczu.
    Zwraca liczbę zaktualizowanych meczów."""
    if not api_key:
        return 0
    try:
        import requests as _req
        from datetime import date as _date, timedelta as _td
        headers = {"X-Auth-Token": api_key}
        # Pobierz mecze z ostatnich 7 dni które mają status FINISHED
        _date_from = (_date.today() - _td(days=7)).isoformat()
        _date_to   = _date.today().isoformat()
        url = (f"https://api.football-data.org/v4/competitions/{fd_org_id}/matches"
               f"?status=FINISHED&dateFrom={_date_from}&dateTo={_date_to}")
        resp = _req.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        matches = resp.json().get("matches", [])

        init_db()
        con = sqlite3.connect(DB_FILE)
        n_updated = 0

        for m in matches:
            score = m.get("score", {}).get("fullTime", {})
            hg = score.get("home")
            ag = score.get("away")
            if hg is None or ag is None:
                continue
            hg, ag = int(hg), int(ag)
            ht_name = m.get("homeTeam", {}).get("shortName") or m.get("homeTeam", {}).get("name", "")
            at_name = m.get("awayTeam", {}).get("shortName") or m.get("awayTeam", {}).get("name", "")
            # Spróbuj znaleźć w bazie przez map_nazwa
            ht_mapped = map_nazwa(ht_name)
            at_mapped = map_nazwa(at_name)
            wynik_1x2 = "1" if hg > ag else ("2" if ag > hg else "X")

            zdarzenia = con.execute(
                "SELECT id, rynek, typ, linia FROM zdarzenia WHERE home=? AND away=? AND trafione IS NULL",
                (ht_mapped, at_mapped)
            ).fetchall()
            if not zdarzenia:
                # Spróbuj też oryginalne nazwy
                zdarzenia = con.execute(
                    "SELECT id, rynek, typ, linia FROM zdarzenia WHERE home=? AND away=? AND trafione IS NULL",
                    (ht_name, at_name)
                ).fetchall()

            for zid, rynek, typ, linia in zdarzenia:
                trafione = False
                if rynek == "1X2":
                    if typ == "1":  trafione = (wynik_1x2 == "1")
                    elif typ == "X": trafione = (wynik_1x2 == "X")
                    elif typ == "2": trafione = (wynik_1x2 == "2")
                    elif typ == "1X": trafione = (wynik_1x2 in ("1","X"))
                    elif typ == "X2": trafione = (wynik_1x2 in ("X","2"))
                elif rynek == "Gole":
                    trafione = ((hg+ag) > linia) if "Over" in typ else ((hg+ag) < linia)
                elif rynek == "BTTS":
                    trafione = (hg>0 and ag>0) if "Tak" in typ else (hg==0 or ag==0)
                con.execute(
                    "UPDATE zdarzenia SET wynik=?, trafione=? WHERE id=?",
                    (f"{hg}:{ag}", trafione, zid)
                )
                n_updated += 1

        con.commit()
        con.close()
        return n_updated // max(1, len(matches) or 1)  # approx mecze
    except Exception:
        return 0


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
        # Buduj czytelną nazwę: "Gole Over 2.5", "Gole Under 2.5", "BTTS Tak" etc.
        # typ może być: "Over", "Under", "Tak", "Nie", "Over 2.5 goli", "Under 1.5 goli"
        # Normalizuj: wyciągnij Over/Under z długiej nazwy
        typ_norm = typ or ""
        if "Over" in typ_norm and typ_norm != "Over":
            typ_norm = "Over"
        elif "Under" in typ_norm and typ_norm != "Under":
            typ_norm = "Under"

        nazwa = f"{rynek}"
        if typ_norm in ("Tak", "Nie"):
            nazwa += f" {typ_norm}"
        elif typ_norm in ("Over", "Under") and linia and linia > 0:
            nazwa += f" {typ_norm} {float(linia):.1f}"
        elif linia and linia > 0:
            nazwa += f" {float(linia):.1f}"
        elif typ_norm:
            nazwa += f" {typ_norm}"
        
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
            lh, la, lr, lk, _, lsot = oblicz_lambdy(h, a, srednie_df, srednie_lig, forma_dict,
                               csv_code=LIGI[wybrana_liga]["csv_code"])
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
# FORMA + KONTEKST (zastępuje AI komentarz)
# ===========================================================================