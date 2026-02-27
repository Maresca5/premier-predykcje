import sqlite3
import pandas as pd
import numpy as np

_T = "backtest"


# =========================
# DATABASE
# =========================

def _init_db(db):
    con = sqlite3.connect(db)

    # Zawsze świeża tabela (eliminuje problemy strukturalne)
    con.execute(f"DROP TABLE IF EXISTS {_T}")

    con.execute(f"""
        CREATE TABLE {_T}(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            liga TEXT,
            sezon TEXT,
            kolejka INTEGER,
            data TEXT,
            home TEXT,
            away TEXT,
            p_home REAL,
            p_draw REAL,
            p_away REAL,
            pred TEXT,
            wynik TEXT,
            correct INTEGER
        )
    """)
    con.commit()
    con.close()


def _insert(db, row):
    con = sqlite3.connect(db)
    con.execute(f"""
        INSERT INTO {_T}
        (liga,sezon,kolejka,data,home,away,
         p_home,p_draw,p_away,pred,wynik,correct)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, row)
    con.commit()
    con.close()


def load_results(liga, sezon, db):
    con = sqlite3.connect(db)

    df = pd.read_sql_query(
        f"""
        SELECT * FROM {_T}
        WHERE liga = ? AND sezon = ?
        """,
        con,
        params=(liga, sezon)
    )

    con.close()
    return df


# =========================
# BACKTEST CORE
# =========================

def run_backtest(liga, sezon_test, sezon_prev, db, callback=None):

    _init_db(db)

    # ======= Wczytanie danych =======

    df = pd.read_csv(f"{sezon_test}.csv")
    df.columns = df.columns.str.strip()

    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Brak kolumny {col} w CSV")

    df = df.sort_values("Date").reset_index(drop=True)

    # Tworzymy numer kolejki na podstawie kolejności dat
    df["kolejka"] = (
        df["Date"]
        .rank(method="dense")
        .astype(int)
    )

    # ======= PROSTY MODEL TESTOWY =======
    # (możesz tu później podpiąć swój realny model)

    for idx, row in df.iterrows():

        home = row["HomeTeam"]
        away = row["AwayTeam"]

        hg = row["FTHG"]
        ag = row["FTAG"]

        # Prosty baseline (placeholder)
        p_home = 0.45
        p_draw = 0.25
        p_away = 0.30

        if p_home > p_away and p_home > p_draw:
            pred = "H"
        elif p_away > p_home and p_away > p_draw:
            pred = "A"
        else:
            pred = "D"

        if hg > ag:
            wynik = "H"
        elif ag > hg:
            wynik = "A"
        else:
            wynik = "D"

        correct = 1 if pred == wynik else 0

        _insert(db, (
            liga,
            sezon_test,
            int(row["kolejka"]),
            row["Date"],
            home,
            away,
            p_home,
            p_draw,
            p_away,
            pred,
            wynik,
            correct
        ))

        if callback:
            callback(idx + 1, len(df))

    return summary(liga, sezon_test, db)


# =========================
# SUMMARY
# =========================

def summary(liga, sezon, db):

    df = load_results(liga, sezon, db)

    if df.empty:
        return None

    # Bezpieczne sortowanie
    sort_cols = []
    if "kolejka" in df.columns:
        sort_cols.append("kolejka")
    if "home" in df.columns:
        sort_cols.append("home")

    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    accuracy = df["correct"].mean()

    # Brier score (dla p_home jako przykład)
    brier = np.mean(
        (df["correct"] - df["p_home"]) ** 2
    )

    equity_curve = df["correct"].cumsum()

    return {
        "accuracy": round(float(accuracy), 4),
        "brier": round(float(brier), 4),
        "total_predictions": int(len(df)),
        "equity_curve": equity_curve.tolist()
    }