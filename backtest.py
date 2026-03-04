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
PROG_PEWNY    = 0.42
PROG_PODWOJNA = 0.62
TAU_DAYS      = 30.0
# Kelly fractions po analizie backtestów – obniżone o 50%
KELLY_FRACTIONS_BT = {
    "Gole": 0.075, "BTTS": 0.075,
    "Rożne": 0.050, "Kartki": 0.050, "SOT": 0.050,
}
# Kalibracja liniowa p_model przed Kelly (wynik analizy E0 2425)
CALIB_SLOPE_BT     = 0.88
CALIB_INTERCEPT_BT = 0.06
MAX_EXPOSURE_BT = 0.05

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
        return None, None, sl["rho"], 0.0, 0.0, None
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

        lh = float(np.clip(lh, 0.3, 4.5))
        la = float(np.clip(la, 0.3, 4.5))

        # Alt market lambdy
        lam_r = 0.0; lam_k = 0.0; lam_sot_ret = None
        try:
            if "Rozne (dom)" in st.columns:
                lam_r = (float(st.loc[h, "Rozne (dom)"]) + float(st.loc[a, "Rozne (wyjazd)"])) / 2
            if "Kartki (dom)" in st.columns:
                lam_k = (float(st.loc[h, "Kartki (dom)"]) + float(st.loc[a, "Kartki (wyjazd)"])) / 2
            sot_h2 = st.loc[h, "SOT (dom)"]   if "SOT (dom)"    in st.columns else None
            sot_a2 = st.loc[a, "SOT (wyjazd)"] if "SOT (wyjazd)" in st.columns else None
            ash2   = sl.get("avg_sot_home"); asa2 = sl.get("avg_sot_away")
            if sot_h2 and sot_a2 and ash2 and asa2:
                _sh2, _sa2 = float(sot_h2), float(sot_a2)
                if not (np.isnan(_sh2) or np.isnan(_sa2)):
                    lam_sot_ret = float(np.clip(
                        (_sh2*(avg_h/ash2)*obrona_a + _sa2*(avg_a/asa2)*obrona_h) / 2,
                        0.5, 12.0))
        except Exception:
            pass

        return lh, la, sl["rho"], lam_r, lam_k, lam_sot_ret
    except Exception:
        return None, None, sl["rho"], 0.0, 0.0, None


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


def _pred(lh: float, la: float, rho: float) -> dict:
    M  = _dc(lh, la, rho)
    ph = float(np.tril(M, -1).sum())
    pd_= float(np.trace(M))
    pa = float(np.triu(M, 1).sum())
    if   ph >= PROG_PEWNY:   typ, pt = "1",  ph
    elif pa >= PROG_PEWNY:   typ, pt = "2",  pa
    elif pd_ >= PROG_PEWNY:  typ, pt = "X",  pd_
    else:
        p1x = ph + pd_; px2 = pa + pd_
        if p1x >= PROG_PODWOJNA or px2 >= PROG_PODWOJNA:
            typ, pt = ("1X", p1x) if p1x >= px2 else ("X2", px2)
        else:
            d = {"1": ph, "X": pd_, "2": pa}
            typ = max(d, key=d.get); pt = d[typ]
    return {"ph": ph, "pd": pd_, "pa": pa, "typ": typ, "pt": float(pt), "lh": lh, "la": la}


def _alt_zdarzenia(lh: float, la: float, lam_r: float, lam_k: float,
                    lam_sot, rho: float, fthg: int, ftag: int,
                    hc: int, ac: int, hy: int, ay: int, hr: int, ar: int,
                    hst, ast_val, kelly_fraction: float = 0.25) -> list:
    """
    Generuje alt market predictions z wynikami i Kelly PnL.
    Zwraca listę dict per zdarzenie.
    """
    from scipy.stats import poisson as _poi
    rows = []
    mg = int(min(max(int(lh + la) + 5, 8), 12))

    def _fair(p):
        return round(1 / p, 3) if p > 0.01 else 999.0

    _rynek_fracs = {"Gole":0.075,"BTTS":0.075,"Rożne":0.050,"Kartki":0.050,"SOT":0.050}

    def _kelly_pnl(p, fo, traf, rynek="Gole"):
        b = fo - 1.0
        if b <= 0 or p <= 0 or p >= 1: return 0.0, 0.0, 0.0
        # Kalibracja liniowa (analiza E0 2425: model zawyża p przy faworytach)
        p_adj = max(0.01, min(0.99, p * CALIB_SLOPE_BT + CALIB_INTERCEPT_BT))
        ev = p_adj * b - (1 - p_adj)
        if ev < 0.05: return 0.0, 0.0, 0.0   # EV < 5% filter
        frac = _rynek_fracs.get(rynek, kelly_fraction)
        f_full = max(0.0, (p_adj * b - (1 - p_adj)) / b)
        f_frac = f_full * frac
        pnl = (fo - 1) * f_frac if traf else -f_frac
        return round(f_full, 4), round(f_frac, 4), round(pnl, 4)

    def _brier(p, traf):
        return round((p - float(traf)) ** 2, 6)

    try:
        M = None
        from scipy.stats import poisson as _poi2
        import numpy as np
        M_raw = np.outer(_poi2.pmf(range(mg), lh), _poi2.pmf(range(mg), la))
        # Simple DC adjustment (inline)
        tau = lambda x, y, l, m, r: (
            1 - l*m*r if x==0 and y==0 else
            1 + l*r    if x==0 and y==1 else
            1 + m*r    if x==1 and y==0 else
            1 - r      if x==1 and y==1 else 1.0)
        for i in range(min(2, mg)):
            for j in range(min(2, mg)):
                M_raw[i,j] *= tau(i, j, lh, la, rho)
        M = M_raw / M_raw.sum()
    except Exception:
        M = None

    total_gole = fthg + ftag
    total_rozne = hc + ac
    total_kartki = hy + ay + (hr + ar) * 2

    # Gole Over/Under
    if M is not None:
        for linia in [1.5, 2.5, 3.5]:
            p_over = float(sum(M[i,j] for i in range(mg) for j in range(mg) if i+j > linia))
            p_under = 1 - p_over
            for p, typ, traf in [
                (p_over,  f"Over {linia} goli",  total_gole > linia),
                (p_under, f"Under {linia} goli", total_gole < linia),
            ]:
                fo = _fair(p)
                kf, kfrac, kpnl = _kelly_pnl(p, fo, traf)
                rows.append({"rynek":"Gole","linia":linia,"typ":typ,
                             "p_model":round(p,4),"fair_odds":fo,"trafiony":int(traf),
                             "brier_bin":_brier(p,traf),"kelly_full":kf,
                             "kelly_frac":kfrac,"kelly_pnl":kpnl})

    # BTTS
    if M is not None:
        p_btts = float(1 - M[0,:].sum() - M[:,0].sum() + M[0,0])
        p_nobtts = 1 - p_btts
        for p, typ, traf in [
            (p_btts,   "BTTS – Tak", fthg > 0 and ftag > 0),
            (p_nobtts, "BTTS – Nie", fthg == 0 or ftag == 0),
        ]:
            fo = _fair(p)
            kf, kfrac, kpnl = _kelly_pnl(p, fo, traf)
            rows.append({"rynek":"BTTS","linia":0,"typ":typ,
                         "p_model":round(p,4),"fair_odds":fo,"trafiony":int(traf),
                         "brier_bin":_brier(p,traf),"kelly_full":kf,
                         "kelly_frac":kfrac,"kelly_pnl":kpnl})

    # Rożne
    if lam_r and lam_r > 0:
        for linia in [7.5, 8.5, 9.5, 10.5]:
            try:
                p_over = float(1 - _poi.cdf(int(linia), lam_r))
                traf = total_rozne > linia
                fo = _fair(p_over)
                kf, kfrac, kpnl = _kelly_pnl(p_over, fo, traf)
                rows.append({"rynek":"Rożne","linia":linia,
                             "typ":f"Over {linia} rożnych",
                             "p_model":round(p_over,4),"fair_odds":fo,"trafiony":int(traf),
                             "brier_bin":_brier(p_over,traf),"kelly_full":kf,
                             "kelly_frac":kfrac,"kelly_pnl":kpnl})
            except Exception: pass

    # Kartki
    if lam_k and lam_k > 0:
        for linia in [2.5, 3.5, 4.5]:
            try:
                p_over = float(1 - _poi.cdf(int(linia), lam_k))
                traf = total_kartki > linia
                fo = _fair(p_over)
                kf, kfrac, kpnl = _kelly_pnl(p_over, fo, traf)
                rows.append({"rynek":"Kartki","linia":linia,
                             "typ":f"Over {linia} kartek",
                             "p_model":round(p_over,4),"fair_odds":fo,"trafiony":int(traf),
                             "brier_bin":_brier(p_over,traf),"kelly_full":kf,
                             "kelly_frac":kfrac,"kelly_pnl":kpnl})
            except Exception: pass

    # SOT
    if lam_sot and lam_sot > 0:
        hst_v = int(hst) if hst is not None and str(hst) not in ('','nan','None') else None
        ast_v = int(ast_val) if ast_val is not None and str(ast_val) not in ('','nan','None') else None
        if hst_v is not None and ast_v is not None:
            total_sot = hst_v + ast_v
            for linia in [3.5, 4.5, 5.5]:
                try:
                    p_over = float(1 - _poi.cdf(int(linia), lam_sot))
                    traf = total_sot > linia
                    fo = _fair(p_over)
                    kf, kfrac, kpnl = _kelly_pnl(p_over, fo, traf)
                    rows.append({"rynek":"SOT","linia":linia,
                                 "typ":f"Over {linia} celnych",
                                 "p_model":round(p_over,4),"fair_odds":fo,"trafiony":int(traf),
                                 "brier_bin":_brier(p_over,traf),"kelly_full":kf,
                                 "kelly_frac":kfrac,"kelly_pnl":kpnl})
                except Exception: pass

    return rows


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
            b365h      REAL, b365d REAL, b365a REAL,
            psh        REAL, psd  REAL, psa  REAL,
            UNIQUE(liga, sezon, kolejka, home, away)
        )
    """)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {_TABLE}_alt (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            liga        TEXT NOT NULL,
            sezon       TEXT NOT NULL,
            kolejka     INTEGER,
            data        TEXT,
            home        TEXT,
            away        TEXT,
            rynek       TEXT,
            linia       REAL,
            typ         TEXT,
            p_model     REAL,
            fair_odds   REAL,
            trafiony    INTEGER,
            brier_bin   REAL,
            kelly_full  REAL,
            kelly_frac  REAL,
            kelly_pnl   REAL,
            UNIQUE(liga, sezon, kolejka, home, away, rynek, linia, typ)
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
         typ,p_typ,p_home,p_draw,p_away,lam_h,lam_a,trafiony,brier,n_train,
         b365h,b365d,b365a,psh,psd,psa)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
    rows_alt = []
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
                _lret = _lambdy(h, a, st, sl, fh, fa)
                lh, la, rho = _lret[0], _lret[1], _lret[2]
                lam_r = _lret[3] if len(_lret) > 3 else 0.0
                lam_k = _lret[4] if len(_lret) > 4 else 0.0
                lam_sot_bt = _lret[5] if len(_lret) > 5 else None

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

                def _g(col):
                    v = mecz.get(col)
                    try: return float(v) if v is not None and str(v).strip() not in ('','nan') else None
                    except: return None
                _b3h=_g("B365H"); _b3d=_g("B365D"); _b3a=_g("B365A")
                _psh=_g("PSH");   _psd=_g("PSD");   _psa=_g("PSA")
                rows_db.append((
                    liga_code, sezon_test, k_nr, data, h, a, fthg, ftag, wn,
                    p["typ"], p["pt"], p["ph"], p["pd"], p["pa"],
                    p["lh"], p["la"], tr, round(brier, 6), len(df_train),
                    _b3h, _b3d, _b3a, _psh, _psd, _psa
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
                # Dopisz real odds do ostatniego rekordu
                def _kdc(typ, oh, od, oa):
                    try:
                        if None in (oh,od,oa) or min(oh,od,oa)<=1: return None
                        s=1/oh+1/od+1/oa; ih=(1/oh)/s; id_=(1/od)/s; ia=(1/oa)/s
                        if typ=="1": return oh
                        if typ=="X": return od
                        if typ=="2": return oa
                        if typ=="1X": return round(1/(ih+id_),3)
                        if typ=="X2": return round(1/(id_+ia),3)
                    except: return None
                rows_csv[-1]["kurs_ps"]   = _kdc(p["typ"], _psh, _psd, _psa)
                rows_csv[-1]["kurs_b365"] = _kdc(p["typ"], _b3h, _b3d, _b3a)

                # Alt markets – gole, BTTS, rożne, kartki, SOT
                try:
                    _hc  = int(mecz.get("HC",  0) or 0)
                    _ac  = int(mecz.get("AC",  0) or 0)
                    _hy  = int(mecz.get("HY",  0) or 0)
                    _ay  = int(mecz.get("AY",  0) or 0)
                    _hr  = int(mecz.get("HR",  0) or 0)
                    _ar  = int(mecz.get("AR",  0) or 0)
                    _hst = mecz.get("HST"); _ast = mecz.get("AST")
                    _alt_rows = _alt_zdarzenia(
                        lh, la, lam_r, lam_k, lam_sot_bt, rho,
                        fthg, ftag, _hc, _ac, _hy, _ay, _hr, _ar, _hst, _ast)
                    for _ar_row in _alt_rows:
                        rows_alt.append({
                            "liga": liga_code, "sezon": sezon_test,
                            "kolejka": k_nr, "data": data,
                            "home": h, "away": a,
                            **_ar_row,
                        })
                except Exception:
                    pass

            except Exception:
                skipped += 1
                continue

    cb(0.95, f"Zapisuję {len(rows_db)} + {len(rows_alt)} alt rekordów do bazy...")
    _insert_bulk(rows_db, db_path)
    # Save alt market rows
    if rows_alt:
        _alt_tbl = f"{_TABLE}_alt"
        _con_alt = sqlite3.connect(db_path)
        _con_alt.execute(f"DELETE FROM {_alt_tbl} WHERE liga=? AND sezon=?",
                         (liga_code, sezon_test))
        _con_alt.executemany(
            f"INSERT OR REPLACE INTO {_alt_tbl} "
            f"(liga,sezon,kolejka,data,home,away,rynek,linia,typ,"
            f"p_model,fair_odds,trafiony,brier_bin,kelly_full,kelly_frac,kelly_pnl) "
            f"VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(r["liga"],r["sezon"],r["kolejka"],r["data"],r["home"],r["away"],
              r["rynek"],r["linia"],r["typ"],r["p_model"],r["fair_odds"],
              r["trafiony"],r["brier_bin"],r["kelly_full"],r["kelly_frac"],r["kelly_pnl"])
             for r in rows_alt]
        )
        _con_alt.commit()
        _con_alt.close()

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

    def _real_roi(field):
        vals = [r for r in rows_csv if r.get(field) is not None]
        if not vals: return None
        s = sum((r[field]-1) if r["trafiony"]=="TAK" else -1 for r in vals)
        return round(s / len(vals) * 100, 2)
    roi_ps_pct   = _real_roi("kurs_ps")
    roi_b365_pct = _real_roi("kurs_b365")
    ps_info = f" · PS {roi_ps_pct:+.1f}%" if roi_ps_pct is not None else ""
    cb(1.0, f"Gotowe! {n} meczów · hit {traf/n:.1%} · Brier {brier:.4f} · ROI fair {roi_pct:+.1f}%{ps_info}")
    # Alt markets summary
    _alt_df = pd.DataFrame(rows_alt) if rows_alt else pd.DataFrame()
    _alt_summary = {}
    if not _alt_df.empty:
        for _rynek in _alt_df["rynek"].unique():
            _rdf = _alt_df[_alt_df["rynek"] == _rynek]
            _rn = len(_rdf)
            _rhit = int(_rdf["trafiony"].sum())
            _rroi_flat = float(sum(
                (r["fair_odds"]-1) if r["trafiony"] else -1
                for _, r in _rdf.iterrows())) / _rn * 100
            _rroi_kelly = float(_rdf["kelly_pnl"].sum())
            _alt_summary[_rynek] = {
                "n": _rn, "hit": _rhit,
                "hit_rate": _rhit/_rn,
                "roi_flat": round(_rroi_flat, 2),
                "roi_kelly": round(_rroi_kelly, 4),
            }

    return {
        "ok": True,
        "liga": liga_code, "sezon": sezon_test,
        "n": n, "trafione": traf, "skipped": skipped,
        "hit_rate": traf / n,
        "brier": brier, "bss": bss,
        "roi_pct": roi_pct,
        "roi_ps_pct": roi_ps_pct,
        "roi_b365_pct": roi_b365_pct,
        "n_kolejek": n_k,
        "df": pd.DataFrame(rows_csv),
        "alt_df": _alt_df,
        "alt_summary": _alt_summary,
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


def load_alt_results(liga: str, sezon: str, db: str) -> pd.DataFrame:
    """Ładuje wyniki alt markets z DB."""
    try:
        con = sqlite3.connect(db)
        df = pd.read_sql_query(
            f"SELECT * FROM {_TABLE}_alt WHERE liga=? AND sezon=? ORDER BY kolejka,id",
            con, params=(liga, sezon))
        con.close()
        return df
    except Exception:
        return pd.DataFrame()


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
