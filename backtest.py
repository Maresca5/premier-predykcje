# ===========================================================================
# backtest.py  –  Walk-forward backtest dla Football Analytics System
# Wersja zsynchronizowana z app.py (asymetryczny shrink lambdy, per-liga
# kalibracja, SOT blend, Dixon-Coles, waga poprzedniego sezonu).
#
# Interfejs wymagany przez app.py:
#   run_backtest(liga, sezon_test, sezon_prev, db_file, progress_cb) -> int
#   load_results(liga, sezon_test, db_file) -> pd.DataFrame
#   summary(liga, sezon_test, db_file) -> dict
# ===========================================================================

import sqlite3
import requests
import numpy as np
import pandas as pd
from io import StringIO
from scipy.stats import poisson

# ---------------------------------------------------------------------------
# PARAMETRY MODELU  –  identyczne z app.py
# ---------------------------------------------------------------------------

# Per-liga shrinkage prawdopodobieństwa (kalibracja)
KALIBRACJA_PER_LIGA = {
    "E0":  {"shrink": 0.25, "prog_pewny": 0.55, "prog_podwojna": 0.55},
    "SP1": {"shrink": 0.25, "prog_pewny": 0.55, "prog_podwojna": 0.55},
    "D1":  {"shrink": 0.25, "prog_pewny": 0.55, "prog_podwojna": 0.55},
    "I1":  {"shrink": 0.28, "prog_pewny": 0.55, "prog_podwojna": 0.55},
    "F1":  {"shrink": 0.28, "prog_pewny": 0.55, "prog_podwojna": 0.55},
}

# Asymetryczny shrink lambdy (potwierdzony backtestem na 3 datasetach)
# atak jest zmienny  → mocniejszy shrink (0.10)
# obrona jest stabilna → słabszy shrink (0.20)
ALPHA_OFF = 0.10
ALPHA_DEF = 0.20

SOT_BLEND_W     = 0.30
MIN_MECZE_DOM   = 2
MIN_MECZE_WYJAZD = 2


def waga_poprzedniego(n_biezacy: int) -> float:
    return float(np.clip(0.8 - (n_biezacy / 30) * 0.6, 0.2, 0.8))


# ---------------------------------------------------------------------------
# SCHEMAT BAZY
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS backtest_results (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    liga       TEXT,
    sezon      TEXT,
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
    UNIQUE(liga, sezon, home, away, data)
);
"""

# ---------------------------------------------------------------------------
# POBIERANIE CSV
# ---------------------------------------------------------------------------
def _pobierz_csv(liga: str, sezon: str) -> pd.DataFrame:
    url = f"https://www.football-data.co.uk/mmz4281/{sezon}/{liga}.csv"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        df["FTHG"] = df["FTHG"].astype(int)
        df["FTAG"] = df["FTAG"].astype(int)
        for col in ["HC", "AC", "HY", "AY", "HR", "AR", "HST", "AST"]:
            if col not in df.columns:
                df[col] = np.nan
        df["HST"] = pd.to_numeric(df["HST"], errors="coerce")
        df["AST"] = pd.to_numeric(df["AST"], errors="coerce")
        return df.sort_values("Date").reset_index(drop=True)
    except Exception as e:
        print(f"  [WARN] Nie mozna pobrac {liga}/{sezon}: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# MODEL  –  identyczny z app.py
# ---------------------------------------------------------------------------
def _dc_adj(M: np.ndarray, lh: float, la: float, rho: float = -0.13) -> np.ndarray:
    tau = {(0,0): 1-lh*la*rho, (1,0): 1+la*rho, (0,1): 1+lh*rho, (1,1): 1-rho}
    M = M.copy()
    for (i, j), t in tau.items():
        if i < M.shape[0] and j < M.shape[1]:
            M[i, j] *= max(t, 0.001)
    return M / M.sum()


def _predykcja(lh: float, la: float, rho: float = -0.13) -> tuple:
    mg = int(np.clip(np.ceil(max(lh, la) + 4), 6, 10))
    M  = _dc_adj(np.outer(poisson.pmf(range(mg), lh),
                           poisson.pmf(range(mg), la)), lh, la, rho)
    return (float(np.tril(M, -1).sum()),
            float(np.trace(M)),
            float(np.triu(M, 1).sum()))


def _kalibruj(ph: float, pd_: float, pa: float, shrink: float) -> tuple:
    a = shrink
    ph2 = (1-a)*ph + a/3; pd2 = (1-a)*pd_ + a/3; pa2 = (1-a)*pa + a/3
    s = ph2 + pd2 + pa2
    return ph2/s, pd2/s, pa2/s


def _wybierz_typ(ph, pd_, pa, prog_pewny, prog_podw):
    p1x = ph + pd_; px2 = pa + pd_
    if ph >= prog_pewny:  return "1",  ph
    if pa >= prog_pewny:  return "2",  pa
    if p1x >= prog_podw or px2 >= prog_podw:
        return ("1X", p1x) if p1x >= px2 else ("X2", px2)
    d = {"1": ph, "X": pd_, "2": pa}
    t = max(d, key=d.get)
    return t, d[t]


def _brier(ph, pd_, pa, wynik):
    y1 = 1 if wynik=="1" else 0
    yx = 1 if wynik=="X" else 0
    y2 = 1 if wynik=="2" else 0
    return ((ph-y1)**2 + (pd_-yx)**2 + (pa-y2)**2) / 3


def _trafiony(typ, wynik):
    if typ in ("1","X","2"):  return 1 if typ==wynik else 0
    if typ == "1X": return 1 if wynik in ("1","X") else 0
    if typ == "X2": return 1 if wynik in ("X","2") else 0
    return 0


# ---------------------------------------------------------------------------
# STATYSTYKI
# ---------------------------------------------------------------------------
def _wm(values: pd.Series, dates: pd.Series, tau: float = 30.0) -> float:
    if values.empty:
        return 0.0
    if dates is None or dates.empty:
        return float(values.mean())
    now = dates.max()
    d   = (now - dates).dt.total_seconds() / 86400
    w   = np.exp(-d / tau)
    w  /= w.sum()
    return float((values * w).sum())


def _stat(df_train: pd.DataFrame) -> pd.DataFrame:
    druzyny = pd.unique(df_train[["HomeTeam", "AwayTeam"]].values.ravel())
    dane = {}
    for d in druzyny:
        home = df_train[df_train["HomeTeam"] == d]
        away = df_train[df_train["AwayTeam"] == d]
        if len(home) < MIN_MECZE_DOM or len(away) < MIN_MECZE_WYJAZD:
            continue
        hd = home["Date"]; ad = away["Date"]
        hs = home["HST"].dropna(); as_ = away["AST"].dropna()
        hsd = home.loc[home["HST"].notna(), "Date"]
        asd = away.loc[away["AST"].notna(), "Date"]
        dane[d] = {
            "Gole strzelone (dom)":    _wm(home["FTHG"], hd),
            "Gole stracone (dom)":     _wm(home["FTAG"], hd),
            "Gole strzelone (wyjazd)": _wm(away["FTAG"], ad),
            "Gole stracone (wyjazd)":  _wm(away["FTHG"], ad),
            "SOT (dom)":    (_wm(home.loc[home["HST"].notna(),"HST"], hsd)
                             if len(hs) >= 2 else None),
            "SOT (wyjazd)": (_wm(away.loc[away["AST"].notna(),"AST"], asd)
                             if len(as_) >= 2 else None),
        }
    return pd.DataFrame(dane).T


def _srednie_lig(df: pd.DataFrame) -> dict:
    avg_h = float(df["FTHG"].mean()); avg_a = float(df["FTAG"].mean())
    n = len(df)
    e00 = poisson.pmf(0,avg_h)*poisson.pmf(0,avg_a)
    e11 = poisson.pmf(1,avg_h)*poisson.pmf(1,avg_a)
    o00 = len(df[(df["FTHG"]==0)&(df["FTAG"]==0)])/n
    o11 = len(df[(df["FTHG"]==1)&(df["FTAG"]==1)])/n
    r00 = (o00/e00-1)/(avg_h*avg_a) if e00 > 0 else -0.13
    r11 = -(o11/e11-1) if e11 > 0 else -0.13
    rho = float(np.clip(np.mean([r00,r11]), -0.25, 0.0))
    sh = float(df["HST"].dropna().mean()) if df["HST"].notna().sum()>10 else None
    sa = float(df["AST"].dropna().mean()) if df["AST"].notna().sum()>10 else None
    return {"avg_home":avg_h, "avg_away":avg_a, "rho":rho,
            "avg_sot_home":sh, "avg_sot_away":sa}


def _forma(df: pd.DataFrame) -> dict:
    forma = {}
    for d in pd.unique(df[["HomeTeam","AwayTeam"]].values.ravel()):
        mecze = df[(df["HomeTeam"]==d)|(df["AwayTeam"]==d)].tail(5)
        wyniki = []
        for _, m in mecze.iterrows():
            if m["HomeTeam"]==d:
                wyniki.append("W" if m["FTHG"]>m["FTAG"] else
                              "D" if m["FTHG"]==m["FTAG"] else "L")
            else:
                wyniki.append("W" if m["FTAG"]>m["FTHG"] else
                              "D" if m["FTAG"]==m["FTHG"] else "L")
        forma[d] = "".join(wyniki)
    return forma


def _lambdy(h, a, sdf, slig, fdict, sot_w=SOT_BLEND_W):
    """Asymetryczny shrink – identyczny z app.py."""
    avg_h = max(slig["avg_home"], 0.5)
    avg_a = max(slig["avg_away"], 0.5)

    atak_h   = sdf.loc[h,"Gole strzelone (dom)"]    / avg_h
    obrona_a = sdf.loc[a,"Gole stracone (wyjazd)"]  / avg_a
    atak_a   = sdf.loc[a,"Gole strzelone (wyjazd)"] / avg_a
    obrona_h = sdf.loc[h,"Gole stracone (dom)"]     / avg_h

    def fw(team):
        f = fdict.get(team,"")
        return float(np.clip(1.0+(f.count("W")-f.count("L"))*0.03, 0.85, 1.15))

    lh_g = avg_h * atak_h * obrona_a * fw(h)
    la_g = avg_a * atak_a * obrona_h * fw(a)

    # Asymetryczny shrink lambdy
    lh_g = (1-ALPHA_OFF)*lh_g + ALPHA_OFF*avg_h
    la_g = (1-ALPHA_DEF)*la_g + ALPHA_DEF*avg_a

    lh, la = lh_g, la_g

    # SOT blend
    if sot_w > 0:
        sh = sdf.loc[h,"SOT (dom)"]    if "SOT (dom)"    in sdf.columns else None
        sa = sdf.loc[a,"SOT (wyjazd)"] if "SOT (wyjazd)" in sdf.columns else None
        ash = slig.get("avg_sot_home"); asa = slig.get("avg_sot_away")
        if (sh is not None and sa is not None and ash and asa
                and not np.isnan(float(sh)) and not np.isnan(float(sa))):
            ch = avg_h/ash; ca = avg_a/asa
            lsh = float(sh)*ch*obrona_a*fw(h)
            lsa = float(sa)*ca*obrona_h*fw(a)
            lsh = (1-ALPHA_OFF)*lsh + ALPHA_OFF*avg_h
            lsa = (1-ALPHA_DEF)*lsa + ALPHA_DEF*avg_a
            lh  = (1-sot_w)*lh_g + sot_w*lsh
            la  = (1-sot_w)*la_g + sot_w*lsa

    return float(np.clip(lh,0.3,4.5)), float(np.clip(la,0.3,4.5))


# ---------------------------------------------------------------------------
# BAZA
# ---------------------------------------------------------------------------
def _init_db(db_file):
    con = sqlite3.connect(db_file)
    con.execute(SCHEMA); con.commit(); con.close()


def _zapisz(db_file, rows):
    con = sqlite3.connect(db_file)
    con.executemany(
        """INSERT OR IGNORE INTO backtest_results
           (liga,sezon,kolejka,data,home,away,fthg,ftag,wynik,
            typ,p_typ,p_home,p_draw,p_away,lam_h,lam_a,trafiony,brier,n_train)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
    con.commit(); con.close()


# ---------------------------------------------------------------------------
# PUBLICZNY INTERFEJS
# ---------------------------------------------------------------------------
def run_backtest(liga: str, sezon_test: str, sezon_prev: str,
                 db_file: str, progress_cb=None) -> int:
    """
    Walk-forward backtest bez look-ahead bias.
    Dla każdej kolejki K: trenuje na danych do K-1, przewiduje K.
    Zwraca liczbę zapisanych meczów.
    """
    _init_db(db_file)
    kal = KALIBRACJA_PER_LIGA.get(liga, KALIBRACJA_PER_LIGA["E0"])
    shrink     = kal["shrink"]
    prog_pewny = kal["prog_pewny"]
    prog_podw  = kal["prog_podwojna"]

    if progress_cb: progress_cb(0.05, "Pobieranie danych...")
    df_test = _pobierz_csv(liga, sezon_test)
    df_prev = _pobierz_csv(liga, sezon_prev)

    if df_test.empty:
        print(f"  [ERR] Brak danych dla {liga}/{sezon_test}"); return 0

    df_test = df_test.sort_values("Date").reset_index(drop=True)
    unique_d = sorted(df_test["Date"].dt.date.unique())
    df_test["_k"] = df_test["Date"].dt.date.map(
        {d: i+1 for i, d in enumerate(unique_d)}
    )
    kolejki = sorted(df_test["_k"].unique())
    rows = []

    for ki, k in enumerate(kolejki):
        if progress_cb:
            progress_cb(0.05 + 0.90*ki/len(kolejki), f"Kolejka {k}/{kolejki[-1]}...")

        df_known = df_test[df_test["_k"] < k]
        n_biezacy = len(df_known)

        if not df_prev.empty:
            w_p = waga_poprzedniego(n_biezacy)
            n_p = min(int(n_biezacy * w_p / max(1-w_p, 0.01)), len(df_prev))
            df_train = pd.concat([df_prev.tail(n_p), df_known], ignore_index=True)
        else:
            df_train = df_known.copy()

        if len(df_train) < 10: continue

        sdf  = _stat(df_train)
        slig = _srednie_lig(df_train)
        fdict = _forma(df_train)
        rho  = slig["rho"]

        for _, row in df_test[df_test["_k"]==k].iterrows():
            h = row["HomeTeam"]; a = row["AwayTeam"]
            if h not in sdf.index or a not in sdf.index: continue
            try:
                lh, la = _lambdy(h, a, sdf, slig, fdict)
                ph, pd_, pa = _predykcja(lh, la, rho)
                phk, pdk, pak = _kalibruj(ph, pd_, pa, shrink)
                typ, p_typ = _wybierz_typ(phk, pdk, pak, prog_pewny, prog_podw)
                fthg=int(row["FTHG"]); ftag=int(row["FTAG"])
                wynik = "1" if fthg>ftag else ("2" if ftag>fthg else "X")
                rows.append((
                    liga, sezon_test, int(k),
                    row["Date"].strftime("%Y-%m-%d"),
                    h, a, fthg, ftag, wynik,
                    typ, round(p_typ,4),
                    round(phk,4), round(pdk,4), round(pak,4),
                    round(lh,3), round(la,3),
                    _trafiony(typ,wynik),
                    round(_brier(phk,pdk,pak,wynik),5),
                    n_biezacy
                ))
            except Exception as e:
                print(f"  [WARN] {h} vs {a}: {e}"); continue

    if not rows:
        return {"error": "Brak danych – sprawdz kody ligi i sezonu."}

    _zapisz(db_file, rows)
    if progress_cb: progress_cb(1.0, f"Gotowe – {len(rows)} meczow")

    cols = ["liga","sezon","kolejka","data","home","away","fthg","ftag","wynik",
            "typ","p_typ","p_home","p_draw","p_away","lam_h","lam_a",
            "trafiony","brier","n_train"]
    df_res = pd.DataFrame(rows, columns=cols)
    n      = len(df_res)
    hit    = float(df_res["trafiony"].mean())
    brier  = float(df_res["brier"].mean())
    bn     = ((1/3-1)**2 + (1/3)**2 + (1/3)**2) / 3
    roi_s  = sum(
        (round(1/r["p_typ"], 2) - 1 if r["trafiony"] == 1 else -1)
        for _, r in df_res.iterrows()
    )
    return {
        "n":        n,
        "hit_rate": hit,
        "brier":    brier,
        "bss":      1 - brier / bn,
        "roi_pct":  roi_s / n * 100,
        "df":       df_res,
    }


def load_results(liga: str, sezon: str, db_file: str) -> pd.DataFrame:
    try:
        _init_db(db_file)
        con = sqlite3.connect(db_file)
        df  = pd.read_sql(
            "SELECT * FROM backtest_results WHERE liga=? AND sezon=? ORDER BY kolejka,data",
            con, params=(liga, sezon))
        con.close()
        return df
    except Exception:
        return pd.DataFrame()


def summary(liga: str, sezon: str, db_file: str) -> dict:
    df = load_results(liga, sezon, db_file)
    if df.empty: return {}

    n     = len(df)
    hit   = float(df["trafiony"].mean())
    brier = float(df["brier"].mean())
    bn    = ((1/3-1)**2 + (1/3)**2 + (1/3)**2) / 3
    bss   = 1 - brier / bn

    # ROI per mecz
    roi_vals = []
    for _, r in df.iterrows():
        fo = round(1 / r["p_typ"], 2) if r["p_typ"] > 0 else 999
        roi_vals.append((fo - 1) if r["trafiony"] == 1 else -1)
    roi_pct = sum(roi_vals) / n * 100

    # Per typ – DataFrame z kolumnami: typ, n, hit, brier
    per_typ_rows = []
    for typ, g in df.groupby("typ"):
        nt = len(g); ht = float(g["trafiony"].mean())
        rt = sum((round(1/r["p_typ"],2)-1 if r["trafiony"]==1 else -1)
                 for _, r in g.iterrows())
        per_typ_rows.append({
            "typ":   typ,
            "n":     nt,
            "hit":   round(ht, 4),
            "brier": round(float(g["brier"].mean()), 5),
            "roi":   round(rt/nt*100, 2),
        })
    per_typ = pd.DataFrame(per_typ_rows)

    # Per kolejka
    per_k_rows = []
    for k, g in df.groupby("kolejka"):
        nt = len(g); ht = float(g["trafiony"].mean())
        rt = sum((round(1/r["p_typ"],2)-1 if r["trafiony"]==1 else -1)
                 for _, r in g.iterrows())
        per_k_rows.append({"kolejka": int(k), "n": nt,
                           "hit_rate": round(ht, 4), "roi": round(rt/nt*100, 2),
                           "brier": round(float(g["brier"].mean()), 5)})
    per_kolejka = pd.DataFrame(per_k_rows).sort_values("kolejka").reset_index(drop=True)

    # Equity curve
    equity = 0.0
    eq_rows = []
    for k, rv in zip(df["kolejka"], roi_vals):
        equity += rv
        eq_rows.append({"kolejka": int(k), "equity": round(equity, 3)})
    equity_df = pd.DataFrame(eq_rows)

    # Kalibracja – kolumny: przedzial, n, p_mean, hit, rozb
    bins = [0.40, 0.50, 0.57, 0.63, 0.68, 0.73, 0.80, 1.01]
    kal_rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        g = df[(df["p_typ"] >= lo) & (df["p_typ"] < hi)]
        if len(g) >= 3:
            p_mean = float(g["p_typ"].mean())
            hit_k  = float(g["trafiony"].mean())
            kal_rows.append({
                "przedzial": f"{lo:.0%}–{hi:.0%}",
                "n":         len(g),
                "p_mean":    round(p_mean, 4),
                "hit":       round(hit_k, 4),
                "rozb":      round(hit_k - p_mean, 4),
            })
    kalibracja = pd.DataFrame(kal_rows)

    return {
        "liga": liga, "sezon": sezon, "n": n,
        "hit_rate":    round(hit, 4),
        "brier":       round(brier, 4),
        "bss":         round(bss, 4),
        "roi_pct":     round(roi_pct, 2),
        "per_typ":     per_typ,
        "per_kolejka": per_kolejka,
        "equity_df":   equity_df,
        "kalibracja":  kalibracja,
    }


# ---------------------------------------------------------------------------
# STANDALONE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Football Analytics – Backtest")
    parser.add_argument("liga",       help="E0 / SP1 / D1 / I1 / F1")
    parser.add_argument("sezon_test", help="np. 2425")
    parser.add_argument("sezon_prev", help="np. 2324")
    parser.add_argument("--db",  default="backtest_results.db")
    parser.add_argument("--csv", default="")
    args = parser.parse_args()

    def cb(pct, msg):
        bar = "█"*int(pct*30) + "░"*(30-int(pct*30))
        print(f"\r  [{bar}] {pct:.0%}  {msg}", end="", flush=True)

    print(f"\n=== BACKTEST: {args.liga} {args.sezon_test} ===")
    kal = KALIBRACJA_PER_LIGA.get(args.liga, KALIBRACJA_PER_LIGA["E0"])
    print(f"  ALPHA_OFF={ALPHA_OFF}  ALPHA_DEF={ALPHA_DEF}  SOT_W={SOT_BLEND_W}")
    print(f"  shrink={kal['shrink']}  prog_pewny={kal['prog_pewny']}\n")

    n = run_backtest(args.liga, args.sezon_test, args.sezon_prev, args.db, cb)
    print()
    s = summary(args.liga, args.sezon_test, args.db)
    if s:
        print(f"\n  Meczow:    {s['n']}")
        print(f"  Hit Rate:  {s['hit']:.1%}")
        print(f"  Brier:     {s['brier']:.4f}  (BSS {s['bss']:+.4f})")
        print(f"  ROI(fair): {s['roi']:+.1f}%")
        print("\n  Per typ:")
        for typ, t in sorted(s["per_typ"].items()):
            print(f"    {typ:3s}: {t['n']:3d}x  hit {t['hit']:.1%}  ROI {t['roi']:+.1f}%")
    if args.csv:
        load_results(args.liga, args.sezon_test, args.db).to_csv(
            args.csv, index=False, decimal=",")
        print(f"\n  CSV zapisany: {args.csv}")
