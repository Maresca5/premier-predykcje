"""
backtest.py – silnik backtestowy dla modelu Dixon-Coles + SOT
=============================================================
Źródło danych: JEDEN plik CSV z football-data.co.uk (np. 2324/E0.csv).
Zawiera kompletny sezon – daty, drużyny i wyniki wszystkich meczów.
Zero scrappingu terminarza. Zero zależności od app.py.

Użycie z Streamlit:
    import backtest as bt
    result = bt.run_backtest("E0", "2324", "2223", "backtest.db", cb)
    df     = bt.load_results("E0", "2324", "backtest.db")
    summ   = bt.summary("E0", "2324", "backtest.db")
"""

import sqlite3
import numpy as np
import pandas as pd
import requests
from io import StringIO
from scipy.stats import poisson
from typing import Callable, Optional

# ═══════════════════════════════════════════════════════════════════════════
# PARAMETRY MODELU – identyczne z app.py
# ═══════════════════════════════════════════════════════════════════════════
SOT_BLEND_W    = 0.30
PROG_PEWNY     = 0.42
PROG_PODWOJNA  = 0.62
TAU_DAYS       = 30.0
MIN_MECZE_STAT = 2

# ═══════════════════════════════════════════════════════════════════════════
# POBIERANIE DANYCH
# ═══════════════════════════════════════════════════════════════════════════
def _pobierz_csv(league_code: str, sezon: str) -> pd.DataFrame:
    url = f"https://www.football-data.co.uk/mmz4281/{sezon}/{league_code}.csv"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        for col in ["HC", "AC", "HY", "AY", "HR", "AR", "HST", "AST"]:
            if col not in df.columns:
                df[col] = np.nan
        df["HST"] = pd.to_numeric(df["HST"], errors="coerce")
        df["AST"] = pd.to_numeric(df["AST"], errors="coerce")
        df["total_gole"]   = df["FTHG"] + df["FTAG"]
        df["total_kartki"] = df["HY"] + df["AY"] + (df["HR"] + df["AR"]) * 2
        df["total_rozne"]  = df["HC"] + df["AC"]
        return df.sort_values("Date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _podziel_na_kolejki(df: pd.DataFrame) -> list:
    """
    Kluczowa funkcja: grupuje mecze z jednego pliku CSV w kolejki.
    Mecze dzielące <= 4 dni -> jeden weekend grania = jedna kolejka.
    Zero potrzeby osobnego pliku terminarza.
    """
    if df.empty:
        return []
    df = df.sort_values("Date").reset_index(drop=True)
    dates = df["Date"].dt.date.values
    grup = np.zeros(len(df), dtype=int)
    current = 0
    prev = None
    for i, d in enumerate(dates):
        if prev is not None and (d - prev).days > 4:
            current += 1
        grup[i] = current
        prev = d
    return [df[grup == g].copy() for g in range(current + 1)]


def _waga_poprzedniego(n_biezacy: int) -> float:
    return float(np.clip(0.8 - (n_biezacy / 30) * 0.6, 0.2, 0.8))


# ═══════════════════════════════════════════════════════════════════════════
# STATYSTYKI – skopiowane 1:1 z app.py
# ═══════════════════════════════════════════════════════════════════════════
def _weighted_mean(values: pd.Series, dates: pd.Series = None) -> float:
    if len(values) == 0:
        return 0.0
    if dates is not None and len(dates) == len(values):
        try:
            dates_dt = pd.to_datetime(dates)
            ref = dates_dt.max()
            days_ago = (ref - dates_dt).dt.total_seconds() / 86400
            weights = np.clip(np.exp(-days_ago.values / TAU_DAYS), 0.01, None)
        except Exception:
            weights = np.linspace(1, 2, len(values))
    else:
        weights = np.linspace(1, 2, len(values))
    return float(np.average(values, weights=weights))


def _statystyki(df: pd.DataFrame) -> pd.DataFrame:
    druzyny = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
    dane = {}
    for d in druzyny:
        home = df[df["HomeTeam"] == d].tail(10)
        away = df[df["AwayTeam"] == d].tail(10)
        if len(home) < MIN_MECZE_STAT or len(away) < MIN_MECZE_STAT:
            continue
        h_d = home["Date"] if "Date" in home.columns else None
        a_d = away["Date"] if "Date" in away.columns else None
        h_sot = home["HST"].dropna()
        a_sot = away["AST"].dropna()
        h_sd = home.loc[home["HST"].notna(), "Date"] if "Date" in home.columns else None
        a_sd = away.loc[away["AST"].notna(), "Date"] if "Date" in away.columns else None
        dane[d] = {
            "Gole strzelone (dom)":    _weighted_mean(home["FTHG"], h_d),
            "Gole stracone (dom)":     _weighted_mean(home["FTAG"], h_d),
            "Gole strzelone (wyjazd)": _weighted_mean(away["FTAG"], a_d),
            "Gole stracone (wyjazd)":  _weighted_mean(away["FTHG"], a_d),
            "Rozne (dom)":             _weighted_mean(home["total_rozne"], h_d),
            "Rozne (wyjazd)":          _weighted_mean(away["total_rozne"], a_d),
            "Kartki (dom)":            _weighted_mean(home["total_kartki"], h_d),
            "Kartki (wyjazd)":         _weighted_mean(away["total_kartki"], a_d),
            "SOT (dom)":    (_weighted_mean(home.loc[home["HST"].notna(),"HST"], h_sd)
                             if len(h_sot) >= MIN_MECZE_STAT else None),
            "SOT (wyjazd)": (_weighted_mean(away.loc[away["AST"].notna(),"AST"], a_sd)
                             if len(a_sot) >= MIN_MECZE_STAT else None),
        }
    return pd.DataFrame(dane).T if dane else pd.DataFrame()


def _srednie_lig(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"avg_home": 1.5, "avg_away": 1.2, "rho": -0.13,
                "avg_sot_home": None, "avg_sot_away": None}
    avg_h = float(df["FTHG"].mean())
    avg_a = float(df["FTAG"].mean())
    n = len(df)
    obs_00 = len(df[(df["FTHG"]==0)&(df["FTAG"]==0)]) / n
    obs_11 = len(df[(df["FTHG"]==1)&(df["FTAG"]==1)]) / n
    exp_00 = poisson.pmf(0, avg_h) * poisson.pmf(0, avg_a)
    exp_11 = poisson.pmf(1, avg_h) * poisson.pmf(1, avg_a)
    rho_00 = (obs_00/exp_00 - 1) / (avg_h * avg_a) if exp_00 > 0 else -0.13
    rho_11 = -(obs_11/exp_11 - 1) if exp_11 > 0 else -0.13
    rho = float(np.clip(np.mean([rho_00, rho_11]), -0.25, 0.0))
    avg_sot_h = float(df["HST"].dropna().mean()) if df["HST"].notna().sum() > 10 else None
    avg_sot_a = float(df["AST"].dropna().mean()) if df["AST"].notna().sum() > 10 else None
    return {"avg_home": avg_h, "avg_away": avg_a, "rho": rho,
            "avg_sot_home": avg_sot_h, "avg_sot_away": avg_sot_a}


def _forma(df: pd.DataFrame, team: str) -> str:
    mecze = df[(df["HomeTeam"]==team)|(df["AwayTeam"]==team)].tail(5)
    wyniki = []
    for _, m in mecze.iterrows():
        if m["HomeTeam"] == team:
            wyniki.append("W" if m["FTHG"]>m["FTAG"] else ("L" if m["FTHG"]<m["FTAG"] else "D"))
        else:
            wyniki.append("W" if m["FTAG"]>m["FTHG"] else ("L" if m["FTAG"]<m["FTHG"] else "D"))
    return "".join(wyniki)


def _oblicz_lambdy(h, a, st_df, sl, fh, fa):
    avg_h = max(sl["avg_home"], 0.5)
    avg_a = max(sl["avg_away"], 0.5)
    if h not in st_df.index or a not in st_df.index:
        return None, None, sl["rho"]
    def fw(f): return float(np.clip(1.0 + (f.count("W")-f.count("L"))*0.03, 0.85, 1.15))
    atak_h   = st_df.loc[h, "Gole strzelone (dom)"]    / avg_h
    obrona_a = st_df.loc[a, "Gole stracone (wyjazd)"]  / avg_a
    atak_a   = st_df.loc[a, "Gole strzelone (wyjazd)"] / avg_a
    obrona_h = st_df.loc[h, "Gole stracone (dom)"]     / avg_h
    lam_h = avg_h * atak_h * obrona_a * fw(fh)
    lam_a = avg_a * atak_a * obrona_h * fw(fa)
    if SOT_BLEND_W > 0:
        sot_h = st_df.loc[h, "SOT (dom)"]    if "SOT (dom)"    in st_df.columns else None
        sot_a = st_df.loc[a, "SOT (wyjazd)"] if "SOT (wyjazd)" in st_df.columns else None
        ash = sl.get("avg_sot_home"); asa = sl.get("avg_sot_away")
        if sot_h is not None and sot_a is not None and ash and asa:
            try:
                if not (np.isnan(float(sot_h)) or np.isnan(float(sot_a))):
                    lsot_h = float(sot_h)*(avg_h/ash)*obrona_a*fw(fh)
                    lsot_a = float(sot_a)*(avg_a/asa)*obrona_h*fw(fa)
                    lam_h = (1-SOT_BLEND_W)*lam_h + SOT_BLEND_W*lsot_h
                    lam_a = (1-SOT_BLEND_W)*lam_a + SOT_BLEND_W*lsot_a
            except (TypeError, ValueError):
                pass
    return float(np.clip(lam_h,0.3,4.5)), float(np.clip(lam_a,0.3,4.5)), sl["rho"]


# ═══════════════════════════════════════════════════════════════════════════
# DIXON-COLES – identyczny z app.py
# ═══════════════════════════════════════════════════════════════════════════
def _dc(lam_h, lam_a, rho):
    mg = int(np.clip(np.ceil(max(lam_h,lam_a)+4), 6, 10))
    M = np.outer(poisson.pmf(range(mg),lam_h), poisson.pmf(range(mg),lam_a))
    tau = {(0,0):1-lam_h*lam_a*rho,(1,0):1+lam_a*rho,(0,1):1+lam_h*rho,(1,1):1-rho}
    for (i,j),t in tau.items():
        if i<mg and j<mg: M[i,j] *= max(t,0.001)
    return M/M.sum()


def _pred(lam_h, lam_a, rho):
    M = _dc(lam_h, lam_a, rho)
    ph=float(np.tril(M,-1).sum()); pd_=float(np.trace(M)); pa=float(np.triu(M,1).sum())
    if ph>=PROG_PEWNY:     typ,pt="1",ph
    elif pa>=PROG_PEWNY:   typ,pt="2",pa
    elif pd_>=PROG_PEWNY:  typ,pt="X",pd_
    else:
        p1x=ph+pd_; px2=pa+pd_
        if p1x>=PROG_PODWOJNA or px2>=PROG_PODWOJNA:
            typ,pt=("1X",p1x) if p1x>=px2 else ("X2",px2)
        else:
            d={"1":ph,"X":pd_,"2":pa}; typ=max(d,key=d.get); pt=d[typ]
    return {"ph":ph,"pd":pd_,"pa":pa,"typ":typ,"pt":float(pt),"lh":lam_h,"la":lam_a}


def _wynik(fthg,ftag): return "1" if fthg>ftag else ("2" if fthg<ftag else "X")
def _traf(typ,wynik):
    if typ in("1","X","2"): return 1 if typ==wynik else 0
    if typ=="1X": return 1 if wynik in("1","X") else 0
    if typ=="X2": return 1 if wynik in("X","2") else 0
    return 0


# ═══════════════════════════════════════════════════════════════════════════
# BAZA DANYCH
# ═══════════════════════════════════════════════════════════════════════════
_T = "backtest"

def _init_db(db):
    con=sqlite3.connect(db)
    con.execute(f"""CREATE TABLE IF NOT EXISTS {_T}(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        liga TEXT, sezon TEXT, kolejka INTEGER, data TEXT,
        home TEXT, away TEXT, fthg INTEGER, ftag INTEGER, wynik TEXT,
        typ TEXT, p_typ REAL, p_home REAL, p_draw REAL, p_away REAL,
        lam_h REAL, lam_a REAL, trafiony INTEGER, brier REAL, n_train INTEGER,
        UNIQUE(liga,sezon,kolejka,home,away))""")
    con.commit(); con.close()

def _clear(liga,sezon,db):
    con=sqlite3.connect(db)
    con.execute(f"DELETE FROM {_T} WHERE liga=? AND sezon=?",(liga,sezon))
    con.commit(); con.close()

def _insert(rows,db):
    if not rows: return
    con=sqlite3.connect(db)
    con.executemany(f"""INSERT OR REPLACE INTO {_T}
        (liga,sezon,kolejka,data,home,away,fthg,ftag,wynik,
         typ,p_typ,p_home,p_draw,p_away,lam_h,lam_a,trafiony,brier,n_train)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
    con.commit(); con.close()


# ═══════════════════════════════════════════════════════════════════════════
# GŁÓWNA FUNKCJA
# ═══════════════════════════════════════════════════════════════════════════
def run_backtest(liga_code, sezon_test, sezon_prev, db_path, progress_cb=None):
    """
    Walk-forward backtest z jednego pliku CSV.
    Zwraca dict z metrykami + df (do generowania CSV) lub {"error": "..."}.
    """
    def cb(f,m):
        if progress_cb: progress_cb(min(float(f),1.0), m)

    cb(0.0, "Inicjalizacja...")
    _init_db(db_path)
    _clear(liga_code, sezon_test, db_path)

    cb(0.04, f"Pobieram sezon testowy {sezon_test}...")
    df_test = _pobierz_csv(liga_code, sezon_test)
    if df_test.empty:
        return {"error": f"Brak danych dla {liga_code}/{sezon_test}. "
                         "Sprawdź kod ligi (np. E0, SP1, D1) i format sezonu (np. 2324)."}

    cb(0.10, f"Pobieram sezon bazowy {sezon_prev}...")
    df_prev = _pobierz_csv(liga_code, sezon_prev)

    cb(0.15, "Dzielę sezon na kolejki po datach...")
    kolejki = _podziel_na_kolejki(df_test)
    n_k = len(kolejki)
    if n_k == 0:
        return {"error": "Nie udało się podzielić sezonu na kolejki."}

    n_mecze = sum(len(k) for k in kolejki)
    cb(0.18, f"{n_mecze} meczów · {n_k} kolejek · start symulacji")

    rows_db = []; rows_csv = []; skipped = 0

    for k_idx, k_df in enumerate(kolejki):
        k_nr = k_idx + 1
        mecze_przed = pd.concat(kolejki[:k_idx], ignore_index=True) if k_idx>0 else pd.DataFrame()
        n_biezacy = len(mecze_przed)

        cb(0.18 + 0.75*(k_idx/n_k),
           f"Kolejka {k_nr}/{n_k}  ·  trening: {n_biezacy} mecz. bież. + blend z {sezon_prev}")

        # Blending z poprzednim sezonem
        if not df_prev.empty:
            w = _waga_poprzedniego(n_biezacy)
            n_p = min(int(n_biezacy * w / max(1-w,1e-9)), len(df_prev))
            df_p = df_prev.tail(n_p).copy() if n_p>0 else pd.DataFrame()
        else:
            df_p = pd.DataFrame()

        # Ujednolicenie nazwy kolumny daty przed concat (Date vs date)
        def _norm_date_col(d):
            if d.empty:
                return d
            if "date" in d.columns and "Date" not in d.columns:
                d = d.rename(columns={"date": "Date"})
            return d
        df_p        = _norm_date_col(df_p)
        mecze_przed = _norm_date_col(mecze_przed)
        df_train = pd.concat([df_p, mecze_przed], ignore_index=True)
        if "Date" in df_train.columns:
            df_train = df_train.sort_values("Date")
        elif "date" in df_train.columns:
            df_train = df_train.sort_values("date").rename(columns={"date": "Date"})

        if len(df_train) < 10:
            skipped += len(k_df); continue

        st_df = _statystyki(df_train)
        sl    = _srednie_lig(df_train)
        if st_df.empty:
            skipped += len(k_df); continue

        for _, mecz in k_df.iterrows():
            h=str(mecz["HomeTeam"]); a=str(mecz["AwayTeam"])
            fthg=int(mecz["FTHG"]); ftag=int(mecz["FTAG"])
            data=str(mecz["Date"].date()) if pd.notna(mecz["Date"]) else ""
            fh=_forma(df_train,h); fa=_forma(df_train,a)
            lh,la,rho = _oblicz_lambdy(h,a,st_df,sl,fh,fa)
            if lh is None: skipped+=1; continue
            p = _pred(lh,la,rho)
            wn = _wynik(fthg,ftag)
            tr = _traf(p["typ"],wn)
            y1=1 if wn=="1" else 0; yx=1 if wn=="X" else 0; y2=1 if wn=="2" else 0
            brier = ((p["ph"]-y1)**2+(p["pd"]-yx)**2+(p["pa"]-y2)**2)/3
            fo = round(1/p["pt"],2) if p["pt"]>0 else 999.0

            rows_db.append((liga_code,sezon_test,k_nr,data,h,a,fthg,ftag,wn,
                            p["typ"],p["pt"],p["ph"],p["pd"],p["pa"],
                            p["lh"],p["la"],tr,round(brier,6),len(df_train)))
            rows_csv.append({
                "liga":liga_code,"sezon":sezon_test,"kolejka":k_nr,"data":data,
                "dom":h,"gosc":a,"wynik_rzecz":f"{fthg}:{ftag} ({wn})",
                "typ_modelu":p["typ"],"p_typ":round(p["pt"],4),"fair_odds":fo,
                "p_home":round(p["ph"],4),"p_draw":round(p["pd"],4),"p_away":round(p["pa"],4),
                "lam_h":round(p["lh"],3),"lam_a":round(p["la"],3),
                "trafiony":"TAK" if tr else "NIE","brier":round(brier,6),
                "n_trening":len(df_train),
            })

    cb(0.94, f"Zapisuję {len(rows_db)} rekordów do SQLite...")
    _insert(rows_db, db_path)

    if not rows_csv:
        return {"error": "Brak wyników – za mało danych treningowych na wszystkich kolejkach."}

    n=len(rows_csv); traf=sum(1 for r in rows_csv if r["trafiony"]=="TAK")
    brier_avg=float(np.mean([r["brier"] for r in rows_csv]))
    bn=float(((1/3-1)**2+(1/3)**2+(1/3)**2)/3)
    bss=float(1-brier_avg/bn) if bn>0 else 0.0
    roi_s=sum((r["fair_odds"]-1) if r["trafiony"]=="TAK" else -1 for r in rows_csv)
    roi_pct=roi_s/n*100

    cb(1.0, f"Gotowe!  {n} meczów  ·  hit {traf/n:.1%}  ·  Brier {brier_avg:.4f}  ·  ROI {roi_pct:+.1f}%")
    return {"ok":True,"liga":liga_code,"sezon":sezon_test,"n":n,"trafione":traf,
            "skipped":skipped,"hit_rate":traf/n,"brier":brier_avg,"bss":bss,
            "roi_pct":roi_pct,"n_kolejek":n_k,"df":pd.DataFrame(rows_csv)}


# ═══════════════════════════════════════════════════════════════════════════
# ODCZYT I METRYKI
# ═══════════════════════════════════════════════════════════════════════════
def load_results(liga, sezon, db):
    try:
        con=sqlite3.connect(db)
        df=pd.read_sql(f"SELECT * FROM {_T} WHERE liga=? AND sezon=? ORDER BY kolejka,home",
                       con, params=(liga,sezon))
        con.close(); return df
    except Exception: return pd.DataFrame()

def has_results(liga, sezon, db):
    return not load_results(liga, sezon, db).empty

def summary(liga, sezon, db):
    df=load_results(liga,sezon,db)
    if df.empty: return {}
    n=len(df); traf=int(df["trafiony"].sum())
    brier=float(df["brier"].mean())
    bn=float(((1/3-1)**2+(1/3)**2+(1/3)**2)/3)
    bss=float(1-brier/bn) if bn>0 else 0.0
    roi_s=sum((round(1/r["p_typ"],2)-1) if r["trafiony"]==1 else -1
              for _,r in df.iterrows())
    roi_pct=roi_s/n*100

    per_k=(df.groupby("kolejka").agg(mecze=("home","count"),trafione=("trafiony","sum"),
                                      brier=("brier","mean")).reset_index())
    per_k["hit_rate"]=per_k["trafione"]/per_k["mecze"]

    bins=[0.40,0.48,0.54,0.60,0.65,0.70,0.75,0.80,1.01]
    kal=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        g=df[(df["p_typ"]>=lo)&(df["p_typ"]<hi)]
        if len(g)>=3:
            kal.append({"przedzial":f"{lo:.0%}–{hi:.0%}","n":len(g),
                        "p_mean":float(g["p_typ"].mean()),"hit":float(g["trafiony"].mean()),
                        "rozb":float(g["trafiony"].mean())-float(g["p_typ"].mean())})

    per_typ=(df.groupby("typ").agg(n=("trafiony","count"),traf=("trafiony","sum"),
                                    brier=("brier","mean")).reset_index())
    per_typ["hit"]=per_typ["traf"]/per_typ["n"]

    df_s=df.sort_values(["kolejka","home"]).reset_index(drop=True)
    kap=0.0; eq=[]
    for _,r in df_s.iterrows():
        fo=round(1/r["p_typ"],2) if 0<r["p_typ"]<=1 else 999.0
        kap+=(fo-1) if r["trafiony"]==1 else -1
        eq.append(kap)
    df_s["equity"]=eq

    return {"n":n,"trafione":traf,"hit_rate":traf/n,"brier":brier,"bss":bss,"roi_pct":roi_pct,
            "per_kolejka":per_k,"kalibracja":pd.DataFrame(kal),"per_typ":per_typ,
            "equity_df":df_s[["kolejka","equity"]]}
