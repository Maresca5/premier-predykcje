"""
backtest.py – Dixon-Coles walk-forward backtest
Pobiera dane z football-data.co.uk (jeden plik CSV per sezon).
Nie wymaga pliku terminarza. Nie importuje nic z app.py.
"""

import sqlite3
import numpy as np
import pandas as pd
import requests
from io import StringIO
from scipy.stats import poisson
from typing import Optional, Callable

# ── Parametry identyczne z app.py ─────────────────────────────────────────
SOT_BLEND_W   = 0.30
PROG_PEWNY    = 0.55   # zaktualizowane po backteście (było 0.42)
PROG_PODWOJNA = 0.55   # zaktualizowane po backteście (było 0.62)
SHRINK_ALPHA  = 0.20   # shrinkage kalibracyjny – identyczny z app.py
TAU_DAYS      = 30.0

# ── Nazwa tabeli w SQLite ──────────────────────────────────────────────────
_TABLE = "backtest"


# =============================================================================
# POBIERANIE DANYCH
# =============================================================================

def _pobierz(kod: str, sezon: str) -> pd.DataFrame:
    """
    Pobiera CSV z football-data.co.uk i zwraca znormalizowany DataFrame.
    Kolumna daty zawsze nazywa się 'Date' (duże D).
    Przy błędzie zwraca pusty DataFrame – nigdy nie rzuca wyjątku.
    """
    url = f"https://www.football-data.co.uk/mmz4281/{sezon}/{kod}.csv"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))

        # Normalizuj nazwy kolumn – usuń białe znaki, ujednolicaj Date
        df.columns = df.columns.str.strip()

        # Upewnij się że kolumna daty istnieje i nazywa się "Date"
        if "Date" not in df.columns:
            for alt in ["date", "DATE", "Datetime"]:
                if alt in df.columns:
                    df = df.rename(columns={alt: "Date"})
                    break

        if "Date" not in df.columns:
            return pd.DataFrame()

        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

        # Wymagane kolumny
        required = ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]
        for col in required:
            if col not in df.columns:
                return pd.DataFrame()

        df = df.dropna(subset=["Date"] + required)
        df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
        df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
        df = df.dropna(subset=["FTHG", "FTAG"])
        df["FTHG"] = df["FTHG"].astype(int)
        df["FTAG"] = df["FTAG"].astype(int)

        # Opcjonalne kolumny – uzupełnij NaN jeśli brakuje
        for col in ["HC", "AC", "HY", "AY", "HR", "AR", "HST", "AST"]:
            if col not in df.columns:
                df[col] = np.nan
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["total_gole"]   = df["FTHG"] + df["FTAG"]
        df["total_kartki"] = df["HY"].fillna(0) + df["AY"].fillna(0) \
                           + (df["HR"].fillna(0) + df["AR"].fillna(0)) * 2
        df["total_rozne"]  = df["HC"].fillna(0) + df["AC"].fillna(0)

        return df.sort_values("Date").reset_index(drop=True)

    except Exception:
        return pd.DataFrame()


def _kolejki(df: pd.DataFrame) -> list:
    """
    Dzieli DataFrame sezonu na kolejki.
    Mecze w oknie ≤4 dni → ta sama kolejka (jeden weekend grania).
    Zwraca listę DataFrames. Zawsze kolumna daty = 'Date'.
    """
    if df.empty:
        return []
    df = df.sort_values("Date").reset_index(drop=True)
    dates = df["Date"].dt.date.values
    grp = np.zeros(len(df), dtype=int)
    cur = 0
    prev = None
    for i, d in enumerate(dates):
        if prev is not None and (d - prev).days > 4:
            cur += 1
        grp[i] = cur
        prev = d
    return [df[grp == g].copy() for g in range(cur + 1)]


def _waga_prev(n: int) -> float:
    return float(np.clip(0.8 - (n / 30) * 0.6, 0.2, 0.8))


def _polacz(df_prev: pd.DataFrame, df_biezacy: pd.DataFrame) -> pd.DataFrame:
    """
    Łączy dane poprzedniego i bieżącego sezonu.
    Gwarantuje że wynik ma kolumnę 'Date' i jest posortowany.
    """
    frames = [f for f in [df_prev, df_biezacy] if not f.empty]
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    # Upewnij się że kolumna daty istnieje
    if "Date" not in result.columns:
        for alt in ["date", "DATE"]:
            if alt in result.columns:
                result = result.rename(columns={alt: "Date"})
                break
    if "Date" in result.columns:
        result = result.sort_values("Date").reset_index(drop=True)
    return result


# =============================================================================
# MODEL STATYSTYCZNY – identyczny z app.py
# =============================================================================

def _weighted_mean(values: pd.Series, dates: pd.Series = None) -> float:
    if len(values) == 0:
        return 0.0
    try:
        if dates is not None and len(dates) == len(values):
            dates_dt = pd.to_datetime(dates)
            ref = dates_dt.max()
            days_ago = (ref - dates_dt).dt.total_seconds() / 86400
            weights = np.clip(np.exp(-days_ago.values / TAU_DAYS), 0.01, None)
        else:
            weights = np.linspace(1, 2, len(values))
        return float(np.average(values, weights=weights))
    except Exception:
        return float(values.mean()) if len(values) > 0 else 0.0


def _statystyki(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "HomeTeam" not in df.columns:
        return pd.DataFrame()
    druzyny = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
    dane = {}
    for d in druzyny:
        home = df[df["HomeTeam"] == d].tail(10)
        away = df[df["AwayTeam"] == d].tail(10)
        if len(home) < 2 or len(away) < 2:
            continue
        h_d = home["Date"] if "Date" in home.columns else None
        a_d = away["Date"] if "Date" in away.columns else None
        h_sot = home["HST"].dropna() if "HST" in home.columns else pd.Series(dtype=float)
        a_sot = away["AST"].dropna() if "AST" in away.columns else pd.Series(dtype=float)
        h_sd = home.loc[home["HST"].notna(), "Date"] if "HST" in home.columns and h_d is not None else None
        a_sd = away.loc[away["AST"].notna(), "Date"] if "AST" in away.columns and a_d is not None else None
        dane[d] = {
            "Gole strzelone (dom)":    _weighted_mean(home["FTHG"], h_d),
            "Gole stracone (dom)":     _weighted_mean(home["FTAG"], h_d),
            "Gole strzelone (wyjazd)": _weighted_mean(away["FTAG"], a_d),
            "Gole stracone (wyjazd)":  _weighted_mean(away["FTHG"], a_d),
            "Rozne (dom)":   _weighted_mean(home["total_rozne"], h_d) if "total_rozne" in home.columns else 0.0,
            "Rozne (wyjazd)":_weighted_mean(away["total_rozne"], a_d) if "total_rozne" in away.columns else 0.0,
            "Kartki (dom)":  _weighted_mean(home["total_kartki"], h_d) if "total_kartki" in home.columns else 0.0,
            "Kartki (wyjazd)":_weighted_mean(away["total_kartki"], a_d) if "total_kartki" in away.columns else 0.0,
            "SOT (dom)":    (_weighted_mean(home.loc[home["HST"].notna(), "HST"], h_sd)
                             if len(h_sot) >= 2 else None),
            "SOT (wyjazd)": (_weighted_mean(away.loc[away["AST"].notna(), "AST"], a_sd)
                             if len(a_sot) >= 2 else None),
        }
    return pd.DataFrame(dane).T if dane else pd.DataFrame()


def _srednie(df: pd.DataFrame) -> dict:
    default = {"avg_home": 1.5, "avg_away": 1.2, "rho": -0.13,
               "avg_sot_home": None, "avg_sot_away": None}
    if df.empty or "FTHG" not in df.columns:
        return default
    try:
        avg_h = float(df["FTHG"].mean())
        avg_a = float(df["FTAG"].mean())
        n = len(df)
        obs_00 = len(df[(df["FTHG"] == 0) & (df["FTAG"] == 0)]) / n
        obs_11 = len(df[(df["FTHG"] == 1) & (df["FTAG"] == 1)]) / n
        exp_00 = poisson.pmf(0, avg_h) * poisson.pmf(0, avg_a)
        exp_11 = poisson.pmf(1, avg_h) * poisson.pmf(1, avg_a)
        rho_00 = (obs_00 / exp_00 - 1) / (avg_h * avg_a) if exp_00 > 0 else -0.13
        rho_11 = -(obs_11 / exp_11 - 1) if exp_11 > 0 else -0.13
        rho = float(np.clip(np.mean([rho_00, rho_11]), -0.25, 0.0))
        avg_sot_h = float(df["HST"].dropna().mean()) if "HST" in df.columns and df["HST"].notna().sum() > 10 else None
        avg_sot_a = float(df["AST"].dropna().mean()) if "AST" in df.columns and df["AST"].notna().sum() > 10 else None
        return {"avg_home": avg_h, "avg_away": avg_a, "rho": rho,
                "avg_sot_home": avg_sot_h, "avg_sot_away": avg_sot_a}
    except Exception:
        return default


def _forma(df: pd.DataFrame, team: str) -> str:
    if df.empty or "HomeTeam" not in df.columns:
        return ""
    mecze = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].tail(5)
    wyniki = []
    for _, m in mecze.iterrows():
        try:
            hg, ag = int(m["FTHG"]), int(m["FTAG"])
            if m["HomeTeam"] == team:
                wyniki.append("W" if hg > ag else ("L" if hg < ag else "D"))
            else:
                wyniki.append("W" if ag > hg else ("L" if ag < hg else "D"))
        except Exception:
            pass
    return "".join(wyniki)


def _lambdy(h: str, a: str, st: pd.DataFrame, sl: dict, fh: str, fa: str):
    """Zwraca (lam_h, lam_a, rho) lub (None, None, rho) gdy brak danych."""
    if st.empty or h not in st.index or a not in st.index:
        return None, None, sl["rho"]
    try:
        avg_h = max(sl["avg_home"], 0.5)
        avg_a = max(sl["avg_away"], 0.5)

        def fw(f): return float(np.clip(1.0 + (f.count("W") - f.count("L")) * 0.03, 0.85, 1.15))

        atak_h   = st.loc[h, "Gole strzelone (dom)"]    / avg_h
        obrona_a = st.loc[a, "Gole stracone (wyjazd)"]  / avg_a
        atak_a   = st.loc[a, "Gole strzelone (wyjazd)"] / avg_a
        obrona_h = st.loc[h, "Gole stracone (dom)"]     / avg_h

        lh = avg_h * atak_h * obrona_a * fw(fh)
        la = avg_a * atak_a * obrona_h * fw(fa)

        # SOT blend
        if SOT_BLEND_W > 0 and "SOT (dom)" in st.columns and "SOT (wyjazd)" in st.columns:
            sot_h = st.loc[h, "SOT (dom)"]
            sot_a = st.loc[a, "SOT (wyjazd)"]
            ash   = sl.get("avg_sot_home")
            asa   = sl.get("avg_sot_away")
            if (sot_h is not None and sot_a is not None and ash and asa):
                sh, sa = float(sot_h), float(sot_a)
                if not (np.isnan(sh) or np.isnan(sa)):
                    lsot_h = sh * (avg_h / ash) * obrona_a * fw(fh)
                    lsot_a = sa * (avg_a / asa) * obrona_h * fw(fa)
                    lh = (1 - SOT_BLEND_W) * lh + SOT_BLEND_W * lsot_h
                    la = (1 - SOT_BLEND_W) * la + SOT_BLEND_W * lsot_a

        return float(np.clip(lh, 0.3, 4.5)), float(np.clip(la, 0.3, 4.5)), sl["rho"]
    except Exception:
        return None, None, sl["rho"]


# =============================================================================
# DIXON-COLES – identyczny z app.py
# =============================================================================

def _dc(lh: float, la: float, rho: float) -> np.ndarray:
    mg = int(np.clip(np.ceil(max(lh, la) + 4), 6, 10))
    M = np.outer(poisson.pmf(range(mg), lh), poisson.pmf(range(mg), la))
    tau = {(0,0): 1 - lh*la*rho, (1,0): 1 + la*rho,
           (0,1): 1 + lh*rho,    (1,1): 1 - rho}
    for (i, j), t in tau.items():
        if i < mg and j < mg:
            M[i, j] *= max(t, 0.001)
    return M / M.sum()


def _kalibruj(ph: float, pd: float, pa: float) -> tuple:
    """Shrinkage identyczny z kalibruj_prawdopodobienstwa() w app.py."""
    a = SHRINK_ALPHA
    ph2 = (1-a)*ph + a/3
    pd2 = (1-a)*pd + a/3
    pa2 = (1-a)*pa + a/3
    s = ph2 + pd2 + pa2
    return ph2/s, pd2/s, pa2/s

def _pred(lh: float, la: float, rho: float) -> dict:
    M  = _dc(lh, la, rho)
    ph = float(np.tril(M, -1).sum())
    pd_= float(np.trace(M))
    pa = float(np.triu(M, 1).sum())
    # Shrinkage – identyczny z app.py (kalibruj_prawdopodobienstwa)
    ph, pd_, pa = _kalibruj(ph, pd_, pa)
    p1x = ph + pd_; px2 = pa + pd_
    if   ph >= PROG_PEWNY:  typ, pt = "1",  ph
    elif pa >= PROG_PEWNY:  typ, pt = "2",  pa
    elif p1x >= PROG_PODWOJNA or px2 >= PROG_PODWOJNA:
        typ, pt = ("1X", p1x) if p1x >= px2 else ("X2", px2)
    else:
        d = {"1": ph, "X": pd_, "2": pa}
        typ = max(d, key=d.get); pt = d[typ]
    return {"ph": ph, "pd": pd_, "pa": pa, "typ": typ, "pt": float(pt), "lh": lh, "la": la}


def _wynik(fthg: int, ftag: int) -> str:
    return "1" if fthg > ftag else ("2" if fthg < ftag else "X")


def _traf(typ: str, wynik: str) -> int:
    if typ in ("1", "X", "2"):  return 1 if typ == wynik else 0
    if typ == "1X":             return 1 if wynik in ("1", "X") else 0
    if typ == "X2":             return 1 if wynik in ("X", "2") else 0
    return 0


# =============================================================================
# BAZA DANYCH
# =============================================================================

def _init_db(db: str):
    con = sqlite3.connect(db)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {_TABLE} (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            liga       TEXT NOT NULL,
            sezon      TEXT NOT NULL,
            kolejka    INTEGER,
            data       TEXT,
            home       TEXT,
            away       TEXT,
            fthg       INTEGER,
            ftag       INTEGER,
            wynik      TEXT,
            typ        TEXT,
            p_typ      REAL,
            p_home     REAL,
            p_draw     REAL,
            p_away     REAL,
            lam_h      REAL,
            lam_a      REAL,
            trafiony   INTEGER,
            brier      REAL,
            n_train    INTEGER,
            UNIQUE(liga, sezon, kolejka, home, away)
        )
    """)
    con.commit()
    con.close()


def _clear(liga: str, sezon: str, db: str):
    con = sqlite3.connect(db)
    con.execute(f"DELETE FROM {_TABLE} WHERE liga=? AND sezon=?", (liga, sezon))
    con.commit()
    con.close()


def _insert_bulk(rows: list, db: str):
    if not rows:
        return
    con = sqlite3.connect(db)
    con.executemany(f"""
        INSERT OR REPLACE INTO {_TABLE}
        (liga,sezon,kolejka,data,home,away,fthg,ftag,wynik,
         typ,p_typ,p_home,p_draw,p_away,lam_h,lam_a,trafiony,brier,n_train)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    con.commit()
    con.close()


# =============================================================================
# PUBLICZNE API
# =============================================================================

def run_backtest(
    liga_code: str,
    sezon_test: str,
    sezon_prev: str,
    db_path: str,
    progress_cb: Optional[Callable] = None,
) -> dict:
    """
    Uruchamia walk-forward backtest dla jednego sezonu.

    Parametry:
        liga_code   – kod ligi z football-data.co.uk (np. "E0", "SP1", "D1")
        sezon_test  – sezon do testowania, format XXYY (np. "2324" = 2023/24)
        sezon_prev  – poprzedni sezon jako dane startowe (np. "2223")
        db_path     – ścieżka do pliku SQLite
        progress_cb – callback(fraction: float, message: str) dla Streamlit

    Zwraca:
        dict z polami: ok, n, trafione, hit_rate, brier, bss, roi_pct, n_kolejek, df
        lub: {"error": "..."}
    """
    def cb(f, m):
        if progress_cb:
            try:
                progress_cb(min(float(f), 1.0), str(m))
            except Exception:
                pass

    cb(0.0, "Inicjalizacja bazy danych...")
    _init_db(db_path)
    _clear(liga_code, sezon_test, db_path)

    cb(0.05, f"Pobieram sezon testowy {sezon_test} ({liga_code})...")
    df_test = _pobierz(liga_code, sezon_test)
    if df_test.empty:
        return {"error": (
            f"Nie udało się pobrać danych dla {liga_code}/{sezon_test}.\n"
            "Sprawdź: kod ligi (E0=PL, SP1=LaLiga, D1=Bundesliga, I1=SerieA, F1=Ligue1) "
            "oraz format sezonu (np. 2324 = 2023/24)."
        )}

    cb(0.12, f"Pobieram sezon bazowy {sezon_prev}...")
    df_prev = _pobierz(liga_code, sezon_prev)
    if df_prev.empty:
        cb(0.12, f"Sezon bazowy {sezon_prev} niedostępny – model startuje bez poprzedniego sezonu")

    cb(0.18, "Dzielę sezon na kolejki...")
    kolejki = _kolejki(df_test)
    n_k = len(kolejki)
    if n_k == 0:
        return {"error": "Nie udało się podzielić sezonu na kolejki."}

    n_mecze = sum(len(k) for k in kolejki)
    cb(0.20, f"Pobrano {n_mecze} meczów · {n_k} kolejek · start walk-forward")

    rows_db  = []
    rows_csv = []
    skipped  = 0

    for k_idx, k_df in enumerate(kolejki):
        k_nr = k_idx + 1
        frac = 0.20 + 0.74 * (k_idx / n_k)

        # Dane treningowe = kolejki 1..K-1
        if k_idx > 0:
            mecze_przed = pd.concat(kolejki[:k_idx], ignore_index=True)
        else:
            mecze_przed = pd.DataFrame()
        n_biezacy = len(mecze_przed)

        cb(frac, f"Kolejka {k_nr}/{n_k}  ·  trening: {n_biezacy} bież. mecz.")

        # Blending z poprzednim sezonem
        if not df_prev.empty and n_biezacy > 0:
            w = _waga_prev(n_biezacy)
            n_p = min(int(n_biezacy * w / max(1 - w, 1e-9)), len(df_prev))
            df_p = df_prev.tail(n_p).copy() if n_p > 0 else pd.DataFrame()
        elif not df_prev.empty:
            # Pierwsza kolejka – użyj całego poprzedniego sezonu jako start
            df_p = df_prev.copy()
        else:
            df_p = pd.DataFrame()

        df_train = _polacz(df_p, mecze_przed)

        if len(df_train) < 10:
            skipped += len(k_df)
            continue

        # Buduj model
        st = _statystyki(df_train)
        sl = _srednie(df_train)

        if st.empty:
            skipped += len(k_df)
            continue

        # Predykcje dla każdego meczu kolejki K
        for _, mecz in k_df.iterrows():
            try:
                h    = str(mecz["HomeTeam"])
                a    = str(mecz["AwayTeam"])
                fthg = int(mecz["FTHG"])
                ftag = int(mecz["FTAG"])
                data = str(mecz["Date"].date()) if pd.notna(mecz.get("Date")) else ""

                fh = _forma(df_train, h)
                fa = _forma(df_train, a)
                lh, la, rho = _lambdy(h, a, st, sl, fh, fa)

                if lh is None:
                    skipped += 1
                    continue

                p    = _pred(lh, la, rho)
                wn   = _wynik(fthg, ftag)
                tr   = _traf(p["typ"], wn)
                fo   = round(1 / p["pt"], 2) if p["pt"] > 0 else 999.0

                # Brier Score (trój-klasowy 1X2)
                y1 = 1 if wn == "1" else 0
                yx = 1 if wn == "X" else 0
                y2 = 1 if wn == "2" else 0
                brier = ((p["ph"] - y1)**2 + (p["pd"] - yx)**2 + (p["pa"] - y2)**2) / 3

                rows_db.append((
                    liga_code, sezon_test, k_nr, data, h, a, fthg, ftag, wn,
                    p["typ"], p["pt"], p["ph"], p["pd"], p["pa"],
                    p["lh"], p["la"], tr, round(brier, 6), len(df_train)
                ))
                rows_csv.append({
                    "liga": liga_code, "sezon": sezon_test,
                    "kolejka": k_nr, "data": data,
                    "dom": h, "gosc": a,
                    "wynik": f"{fthg}:{ftag} ({wn})",
                    "typ": p["typ"],
                    "p_typ": round(p["pt"], 4),
                    "fair": fo,
                    "p_home": round(p["ph"], 4),
                    "p_draw": round(p["pd"], 4),
                    "p_away": round(p["pa"], 4),
                    "trafiony": "TAK" if tr else "NIE",
                    "brier": round(brier, 6),
                })
            except Exception:
                skipped += 1
                continue

    cb(0.95, f"Zapisuję {len(rows_db)} rekordów do bazy...")
    _insert_bulk(rows_db, db_path)

    if not rows_csv:
        return {"error": "Brak wyników – za mało danych treningowych lub błędy pobierania."}

    # Metryki globalne
    n     = len(rows_csv)
    traf  = sum(1 for r in rows_csv if r["trafiony"] == "TAK")
    brier = float(np.mean([r["brier"] for r in rows_csv]))
    bn    = float(((1/3 - 1)**2 + (1/3)**2 + (1/3)**2) / 3)
    bss   = float(1 - brier / bn) if bn > 0 else 0.0
    roi   = sum((r["fair"] - 1) if r["trafiony"] == "TAK" else -1 for r in rows_csv)
    roi_pct = roi / n * 100

    cb(1.0, f"Gotowe! {n} meczów · hit {traf/n:.1%} · Brier {brier:.4f} · ROI {roi_pct:+.1f}%")

    return {
        "ok": True,
        "liga": liga_code, "sezon": sezon_test,
        "n": n, "trafione": traf, "skipped": skipped,
        "hit_rate": traf / n,
        "brier": brier, "bss": bss,
        "roi_pct": roi_pct, "n_kolejek": n_k,
        "df": pd.DataFrame(rows_csv),
    }


def load_results(liga: str, sezon: str, db: str) -> pd.DataFrame:
    """Wczytuje zapisane predykcje backtestowe z SQLite."""
    try:
        con = sqlite3.connect(db)
        df = pd.read_sql(
            f"SELECT * FROM {_TABLE} WHERE liga=? AND sezon=? ORDER BY kolejka, home",
            con, params=(liga, sezon)
        )
        con.close()
        return df
    except Exception:
        return pd.DataFrame()


def has_results(liga: str, sezon: str, db: str) -> bool:
    return not load_results(liga, sezon, db).empty


def summary(liga: str, sezon: str, db: str) -> dict:
    """Pełne metryki z zapisanych wyników: globalne + per kolejka + kalibracja + equity."""
    df = load_results(liga, sezon, db)
    if df.empty:
        return {}

    n    = len(df)
    traf = int(df["trafiony"].sum())
    brier = float(df["brier"].mean())
    bn   = float(((1/3 - 1)**2 + (1/3)**2 + (1/3)**2) / 3)
    bss  = float(1 - brier / bn) if bn > 0 else 0.0

    roi_s = 0.0
    for _, r in df.iterrows():
        fo = round(1 / r["p_typ"], 2) if 0 < r["p_typ"] <= 1 else 999.0
        roi_s += (fo - 1) if r["trafiony"] == 1 else -1
    roi_pct = roi_s / n * 100

    # Per kolejka
    per_k = (df.groupby("kolejka")
               .agg(mecze=("home", "count"),
                    trafione=("trafiony", "sum"),
                    brier=("brier", "mean"))
               .reset_index())
    per_k["hit_rate"] = per_k["trafione"] / per_k["mecze"]

    # Kalibracja
    bins = [0.40, 0.48, 0.54, 0.60, 0.65, 0.70, 0.75, 0.80, 1.01]
    kal = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        g = df[(df["p_typ"] >= lo) & (df["p_typ"] < hi)]
        if len(g) >= 3:
            kal.append({
                "przedzial": f"{lo:.0%}–{hi:.0%}",
                "n":      len(g),
                "p_mean": float(g["p_typ"].mean()),
                "hit":    float(g["trafiony"].mean()),
                "rozb":   float(g["trafiony"].mean()) - float(g["p_typ"].mean()),
            })

    # Per typ
    per_typ = (df.groupby("typ")
                 .agg(n=("trafiony", "count"),
                      traf=("trafiony", "sum"),
                      brier=("brier", "mean"))
                 .reset_index())
    per_typ["hit"] = per_typ["traf"] / per_typ["n"]

    # Equity curve
    df_s = df.sort_values(["kolejka", "home"]).reset_index(drop=True)
    kap = 0.0
    eq  = []
    for _, r in df_s.iterrows():
        fo = round(1 / r["p_typ"], 2) if 0 < r["p_typ"] <= 1 else 999.0
        kap += (fo - 1) if r["trafiony"] == 1 else -1
        eq.append(kap)
    df_s["equity"] = eq

    return {
        "n": n, "trafione": traf, "hit_rate": traf / n,
        "brier": brier, "bss": bss, "roi_pct": roi_pct,
        "per_kolejka": per_k,
        "kalibracja": pd.DataFrame(kal),
        "per_typ": per_typ,
        "equity_df": df_s[["kolejka", "equity"]],
    }
