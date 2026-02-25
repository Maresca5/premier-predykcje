import streamlit as st
import pandas as pd
from scipy.stats import poisson
from datetime import datetime
import requests
from io import StringIO
import numpy as np
import unicodedata
import sqlite3
import os

# ===========================================================================
# KONFIGURACJA
# ===========================================================================
LIGI = {
    "Premier League": {"csv_code": "E0",  "file": "terminarz_premier_2025.csv"},
    "La Liga":        {"csv_code": "SP1", "file": "terminarz_la_liga_2025.csv"},
    "Bundesliga":     {"csv_code": "D1",  "file": "terminarz_bundesliga_2025.csv"},
    "Serie A":        {"csv_code": "I1",  "file": "terminarz_serie_a_2025.csv"},
    "Ligue 1":        {"csv_code": "F1",  "file": "terminarz_ligue_1_2025.csv"},
}

DB_FILE = "predykcje.db"

# Waga poprzedniego sezonu – maleje w miarę gromadzenia danych bieżącego
# Przy <5 meczach bieżącego sezonu: 80% poprzedni, przy >30: 20%
def waga_poprzedniego(n_biezacy: int) -> float:
    return float(np.clip(0.8 - (n_biezacy / 30) * 0.6, 0.2, 0.8))

st.set_page_config(page_title="Predykcje Piłkarskie Top 5", layout="wide")

# ===========================================================================
# MAPOWANIA NAZW
# ===========================================================================

def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return str(name)
    name = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode("ASCII")
    for suffix in [" FC", " CF", " UD", " CD", " RCD", " AS", " Stade", " 1."]:
        name = name.replace(suffix, "")
    return name.strip()

NAZWY_MAP = {
    # PREMIER LEAGUE
    "Brighton & Hove Albion":  "Brighton",
    "West Ham United":         "West Ham",
    "Newcastle United":        "Newcastle",
    "Tottenham Hotspur":       "Tottenham",
    "Leeds United":            "Leeds",
    "Manchester United":       "Man United",
    "Manchester City":         "Man City",
    "Nottingham Forest":       "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "Wolverhampton":           "Wolves",
    "Leicester City":          "Leicester",
    "Sheffield United":        "Sheffield Utd",
    # LA LIGA
    "Girona FC":               "Girona",
    "Rayo Vallecano":          "Vallecano",
    "RCD Mallorca":            "Mallorca",
    "FC Barcelona":            "Barcelona",
    "Deportivo Alavés":        "Alaves",
    "Deportivo Alaves":        "Alaves",
    "Levante UD":              "Levante",
    "Valencia CF":             "Valencia",
    "Real Sociedad":           "Sociedad",
    "Celta Vigo":              "Celta",
    "Getafe CF":               "Getafe",
    "Athletic Club":           "Ath Bilbao",
    "Athletic Bilbao":         "Ath Bilbao",
    "Sevilla FC":              "Sevilla",
    "RCD Espanyol":            "Espanol",
    "Espanyol":                "Espanol",
    "Atlético Madrid":         "Ath Madrid",
    "Atletico Madrid":         "Ath Madrid",
    "Elche CF":                "Elche",
    "Real Betis":              "Betis",
    "CA Osasuna":              "Osasuna",
    "Real Valladolid":         "Valladolid",
    "UD Las Palmas":           "Las Palmas",
    "Cadiz CF":                "Cadiz",
    "Granada CF":              "Granada",
    "UD Almeria":              "Almeria",
    # BUNDESLIGA
    "FC Bayern München":              "Bayern Munich",
    "Bayern":                         "Bayern Munich",
    "Borussia Dortmund":              "Dortmund",
    "Leipzig":                        "RB Leipzig",
    "Bayer 04 Leverkusen":            "Leverkusen",
    "Eintracht Frankfurt":            "Ein Frankfurt",
    "VfB Stuttgart":                  "Stuttgart",
    "VfL Wolfsburg":                  "Wolfsburg",
    "Borussia Mönchengladbach":       "M'gladbach",
    "Borussia Monchengladbach":       "M'gladbach",
    "1. FC Union Berlin":             "Union Berlin",
    "SC Freiburg":                    "Freiburg",
    "1. FC Köln":                     "FC Koln",
    "FC Köln":                        "FC Koln",
    "1. FSV Mainz 05":                "Mainz",
    "FSV Mainz 05":                   "Mainz",
    "TSG Hoffenheim":                 "Hoffenheim",
    "SV Werder Bremen":               "Werder Bremen",
    "VfL Bochum":                     "Bochum",
    "FC Augsburg":                    "Augsburg",
    "FC St. Pauli":                   "St Pauli",
    "St. Pauli":                      "St Pauli",
    "1. FC Heidenheim":               "Heidenheim",
    "Hamburger SV":                   "Hamburg",
    # SERIE A
    "AC Milan":               "Milan",
    "Internazionale":         "Inter",
    "AS Roma":                "Roma",
    "Hellas Verona":          "Verona",
    # LIGUE 1
    "Paris Saint-Germain":    "Paris SG",
    "PSG":                    "Paris SG",
    "Olympique de Marseille": "Marseille",
    "AS Monaco":              "Monaco",
    "Olympique Lyonnais":     "Lyon",
    "LOSC Lille":             "Lille",
    "Stade Rennais":          "Rennes",
    "OGC Nice":               "Nice",
    "RC Lens":                "Lens",
    "RC Strasbourg":          "Strasbourg",
    "FC Nantes":              "Nantes",
    "Montpellier HSC":        "Montpellier",
    "FC Toulouse":            "Toulouse",
    "Stade de Reims":         "Reims",
    "Stade Brestois 29":      "Brest",
    "Stade Brestois":         "Brest",
    "Clermont Foot":          "Clermont",
    "AJ Auxerre":             "Auxerre",
    "Le Havre AC":            "Le Havre",
    "FC Metz":                "Metz",
}

_niezmapowane: set = set()

def map_nazwa(nazwa: str) -> str:
    if not isinstance(nazwa, str):
        return str(nazwa)
    if nazwa in NAZWY_MAP:
        return NAZWY_MAP[nazwa]
    def uproszcz(s):
        return s.replace(" ", "").replace("-", "").replace("'", "").lower()
    for key, val in NAZWY_MAP.items():
        if uproszcz(key) == uproszcz(nazwa):
            return val
    znorm = normalize_name(nazwa)
    if znorm in NAZWY_MAP:
        return NAZWY_MAP[znorm]
    _niezmapowane.add(nazwa)
    return nazwa

# ===========================================================================
# ŁADOWANIE DANYCH – BIEŻĄCY + POPRZEDNI SEZON (BLEND)
# ===========================================================================

def _pobierz_csv(league_code: str, sezon: str) -> pd.DataFrame:
    """Pobiera i parsuje CSV z football-data.co.uk dla danego sezonu (np. '2526')."""
    try:
        url = f"https://www.football-data.co.uk/mmz4281/{sezon}/{league_code}.csv"
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        # Rożne i kartki – kolumny mogą być nieobecne w starszych sezonach
        for col in ["HC", "AC", "HY", "AY", "HR", "AR"]:
            if col not in df.columns:
                df[col] = 0
        df["total_gole"]   = df["FTHG"] + df["FTAG"]
        df["total_kartki"] = df["HY"] + df["AY"] + (df["HR"] + df["AR"]) * 2
        df["total_rozne"]  = df["HC"] + df["AC"]
        return df.sort_values("Date")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900)
def load_historical(league_code: str) -> pd.DataFrame:
    """
    Łączy dane bieżącego sezonu (2025/26) z poprzednim (2024/25).
    Poprzedni sezon jest ważony malejąco wraz z przyrostem bieżących danych,
    co stabilizuje model na początku sezonu kiedy mamy <10 meczów.
    """
    df_now  = _pobierz_csv(league_code, "2526")
    df_prev = _pobierz_csv(league_code, "2425")

    if df_now.empty and df_prev.empty:
        return pd.DataFrame()
    if df_now.empty:
        return df_prev
    if df_prev.empty:
        return df_now

    n_now = len(df_now)
    w_prev = waga_poprzedniego(n_now)

    # Próbkujemy wiersze z poprzedniego sezonu proporcjonalnie do wagi
    n_prev_target = int(n_now * w_prev / (1 - w_prev))
    n_prev_target = min(n_prev_target, len(df_prev))

    # Bierzemy ostatnie n_prev_target meczów poprzedniego sezonu (najnowsze = najbardziej trafne)
    df_prev_sample = df_prev.tail(n_prev_target).copy()
    df_prev_sample["_sezon"] = "poprzedni"
    df_now = df_now.copy()
    df_now["_sezon"] = "biezacy"

    df = pd.concat([df_prev_sample, df_now], ignore_index=True).sort_values("Date")
    return df


@st.cache_data(ttl=86400)
def load_schedule(filename: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filename)
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
        if "round" not in df.columns:
            df = df.sort_values("date")
            unique_dates = df["date"].dt.date.unique()
            df["round"] = df["date"].dt.date.map(
                {d: i + 1 for i, d in enumerate(unique_dates)}
            )
        return df.dropna(subset=["date"]).sort_values("date")
    except Exception as e:
        st.error(f"Problem z plikiem terminarza {filename}: {e}")
        return pd.DataFrame()

# ===========================================================================
# STATYSTYKI
# ===========================================================================

def weighted_mean(values: pd.Series) -> float:
    if len(values) == 0:
        return 0.0
    weights = np.linspace(1, 2, len(values))
    return float(np.average(values, weights=weights))


@st.cache_data
def oblicz_wszystkie_statystyki(df_json: str) -> pd.DataFrame:
    df = pd.read_json(df_json)
    if df.empty:
        return pd.DataFrame()
    druzyny = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
    dane = {}
    for d in druzyny:
        home = df[df["HomeTeam"] == d].tail(10)
        away = df[df["AwayTeam"] == d].tail(10)
        if len(home) < 2 or len(away) < 2:
            continue
        dane[d] = {
            "Gole strzelone (dom)":    weighted_mean(home["FTHG"]),
            "Gole stracone (dom)":     weighted_mean(home["FTAG"]),
            "Gole strzelone (wyjazd)": weighted_mean(away["FTAG"]),
            "Gole stracone (wyjazd)":  weighted_mean(away["FTHG"]),
            "Różne (dom)":             weighted_mean(home["total_rozne"]),
            "Różne (wyjazd)":          weighted_mean(away["total_rozne"]),
            "Kartki (dom)":            weighted_mean(home["total_kartki"]),
            "Kartki (wyjazd)":         weighted_mean(away["total_kartki"]),
        }
    return pd.DataFrame(dane).T.round(2)


@st.cache_data
def oblicz_srednie_ligowe(df_json: str) -> dict:
    """
    Średnie ligowe + kalibrowany rho Dixon-Coles per liga.
    rho obliczamy empirycznie porównując obserwowany % wyników 0:0 i 1:1
    z oczekiwaniem czystego Poissona.
    """
    df = pd.read_json(df_json)
    if df.empty:
        return {"avg_home": 1.5, "avg_away": 1.2, "rho": -0.13, "n_biezacy": 0}

    # Statystyki tylko z bieżącego sezonu do wyświetlenia n_biezacy
    n_biezacy = int((df.get("_sezon", pd.Series()) == "biezacy").sum()) if "_sezon" in df.columns else len(df)

    avg_h = float(df["FTHG"].mean())
    avg_a = float(df["FTAG"].mean())
    n     = len(df)

    obs_00 = len(df[(df["FTHG"] == 0) & (df["FTAG"] == 0)]) / n
    obs_11 = len(df[(df["FTHG"] == 1) & (df["FTAG"] == 1)]) / n

    from scipy.stats import poisson as _p
    exp_00 = _p.pmf(0, avg_h) * _p.pmf(0, avg_a)
    exp_11 = _p.pmf(1, avg_h) * _p.pmf(1, avg_a)

    rho_00 = (obs_00 / exp_00 - 1) / (avg_h * avg_a) if exp_00 > 0 else -0.13
    rho_11 = -(obs_11 / exp_11 - 1) if exp_11 > 0 else -0.13
    rho    = float(np.clip(np.mean([rho_00, rho_11]), -0.25, 0.0))

    return {"avg_home": avg_h, "avg_away": avg_a, "rho": rho, "n_biezacy": n_biezacy}


def oblicz_lambdy(h: str, a: str, srednie_df: pd.DataFrame,
                  srednie_lig: dict, forma_dict: dict) -> tuple:
    avg_h = max(srednie_lig["avg_home"], 0.5)
    avg_a = max(srednie_lig["avg_away"], 0.5)

    atak_h   = srednie_df.loc[h, "Gole strzelone (dom)"]    / avg_h
    obrona_a = srednie_df.loc[a, "Gole stracone (wyjazd)"]  / avg_a
    atak_a   = srednie_df.loc[a, "Gole strzelone (wyjazd)"] / avg_a
    obrona_h = srednie_df.loc[h, "Gole stracone (dom)"]     / avg_h

    def form_weight(team: str) -> float:
        f = forma_dict.get(team, "")
        w = f.count("W"); l = f.count("L")
        return float(np.clip(1.0 + (w - l) * 0.03, 0.85, 1.15))

    lam_h = avg_h * atak_h * obrona_a * form_weight(h)
    lam_a = avg_a * atak_a * obrona_h * form_weight(a)
    return float(np.clip(lam_h, 0.3, 4.5)), float(np.clip(lam_a, 0.3, 4.5))


def oblicz_forme(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    forma = {}
    for d in pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel()):
        mecze = df[(df["HomeTeam"] == d) | (df["AwayTeam"] == d)].tail(5)
        wyniki = []
        for _, m in mecze.iterrows():
            if m["HomeTeam"] == d:
                wyniki.append("W" if m["FTHG"] > m["FTAG"] else ("L" if m["FTHG"] < m["FTAG"] else "D"))
            else:
                wyniki.append("W" if m["FTAG"] > m["FTHG"] else ("L" if m["FTAG"] < m["FTHG"] else "D"))
        forma[d] = "".join(wyniki)
    return forma


def tabela_ligowa(df: pd.DataFrame) -> pd.DataFrame:
    # Tylko bieżący sezon do tabeli
    if "_sezon" in df.columns:
        df = df[df["_sezon"] == "biezacy"]
    if df.empty:
        return pd.DataFrame()
    table: dict = {}
    for _, m in df.iterrows():
        home, away = m["HomeTeam"], m["AwayTeam"]
        hg, ag = m["FTHG"], m["FTAG"]
        for t in [home, away]:
            if t not in table:
                table[t] = {"Pkt": 0, "GZ": 0, "GS": 0, "M": 0}
        table[home]["GZ"] += hg; table[home]["GS"] += ag; table[home]["M"] += 1
        table[away]["GZ"] += ag; table[away]["GS"] += hg; table[away]["M"] += 1
        if   hg > ag: table[home]["Pkt"] += 3
        elif hg < ag: table[away]["Pkt"] += 3
        else:         table[home]["Pkt"] += 1; table[away]["Pkt"] += 1
    res = pd.DataFrame(table).T
    res["RG"] = res["GZ"] - res["GS"]
    return res.sort_values(["Pkt", "RG", "GZ"], ascending=False)

# ===========================================================================
# MODEL POISSONA
# ===========================================================================

def dixon_coles_adj(M: np.ndarray, lam_h: float, lam_a: float,
                    rho: float = -0.13) -> np.ndarray:
    M = M.copy()
    tau = {
        (0, 0): 1 - lam_h * lam_a * rho,
        (1, 0): 1 + lam_a * rho,
        (0, 1): 1 + lam_h * rho,
        (1, 1): 1 - rho,
    }
    for (i, j), t in tau.items():
        if i < M.shape[0] and j < M.shape[1]:
            M[i, j] *= max(t, 0.001)
    M /= M.sum()
    return M


def wybierz_typ(p_home: float, p_draw: float, p_away: float) -> tuple:
    PROG_PEWNY    = 0.42
    PROG_PODWOJNA = 0.62
    if p_home >= PROG_PEWNY: return "1",  p_home
    if p_away >= PROG_PEWNY: return "2",  p_away
    if p_draw >= PROG_PEWNY: return "X",  p_draw
    p_1x = p_home + p_draw
    p_x2 = p_away + p_draw
    if p_1x >= PROG_PODWOJNA or p_x2 >= PROG_PODWOJNA:
        return ("1X", p_1x) if p_1x >= p_x2 else ("X2", p_x2)
    probs = {"1": p_home, "X": p_draw, "2": p_away}
    t = max(probs, key=probs.get)
    return t, probs[t]


def wybierz_wynik(M: np.ndarray, lam_h: float, lam_a: float) -> tuple:
    idx_max = np.unravel_index(M.argmax(), M.shape)
    p_max   = float(M[idx_max])
    flat    = M.flatten()
    top3    = np.sort(flat)[::-1][:3]
    spread  = float(top3[0] - top3[2])
    if spread < 0.04:
        g_h = min(int(round(lam_h)), M.shape[0] - 1)
        g_a = min(int(round(lam_a)), M.shape[1] - 1)
        return g_h, g_a, float(M[g_h, g_a])
    return int(idx_max[0]), int(idx_max[1]), p_max


def confidence_score(p_home: float, p_draw: float, p_away: float) -> tuple:
    vals  = sorted([p_home, p_draw, p_away], reverse=True)
    spread = vals[0] - vals[2]
    edge   = vals[0] - vals[1]
    if edge > 0.18:
        return "High",     "🟢", f"Wyraźny faworyt (+{edge:.0%} nad 2. opcją)"
    elif edge > 0.08:
        return "Medium",   "🟡", f"Umiarkowana przewaga (+{edge:.0%})"
    else:
        return "Coinflip", "🔴", f"Mecz bardzo wyrównany (spread {spread:.0%})"


def predykcja_meczu(lam_h: float, lam_a: float, rho: float = -0.13) -> dict:
    max_gole = int(np.clip(np.ceil(max(lam_h, lam_a) + 4), 6, 10))
    M_raw = np.outer(
        poisson.pmf(range(max_gole), lam_h),
        poisson.pmf(range(max_gole), lam_a),
    )
    M = dixon_coles_adj(M_raw, lam_h, lam_a, rho=rho)

    p_home = float(np.tril(M, -1).sum())
    p_draw = float(np.trace(M))
    p_away = float(np.triu(M, 1).sum())

    wynik_h, wynik_a, p_exact = wybierz_wynik(M, lam_h, lam_a)

    def fo(p): return round(1 / p, 2) if p > 0.001 else 999.0

    typ, p_typ = wybierz_typ(p_home, p_draw, p_away)
    conf_level, conf_emoji, conf_opis = confidence_score(p_home, p_draw, p_away)

    return {
        "lam_h": lam_h, "lam_a": lam_a,
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "wynik_h": wynik_h, "wynik_a": wynik_a, "p_exact": p_exact,
        "fo_home": fo(p_home), "fo_draw": fo(p_draw), "fo_away": fo(p_away),
        "typ": typ, "p_typ": p_typ, "fo_typ": fo(p_typ),
        "conf_level": conf_level, "conf_emoji": conf_emoji, "conf_opis": conf_opis,
        "macierz": M,
    }

# ===========================================================================
# KOMENTARZ AI / STATYSTYCZNY
# ===========================================================================

def _ocen_forme(f: str) -> str:
    if not f or f == "?": return "forma nieznana"
    w = f.count("W"); l = f.count("L")
    if w >= 4: return "doskonała forma"
    if w >= 3: return "dobra forma"
    if l >= 3: return "słaba forma"
    return "nieregularna forma"

def generuj_komentarz(home: str, away: str, pred: dict, forma_dict: dict) -> str:
    fh = forma_dict.get(home, "?")
    fa = forma_dict.get(away, "?")
    try:
        import anthropic
        key = st.secrets.get("ANTHROPIC_API_KEY", None)
        if key:
            client = anthropic.Anthropic(api_key=key)
            roznica_sil  = abs(pred["lam_h"] - pred["lam_a"])
            forma_vs_mod = (
                "forma kłóci się z modelem"
                if (pred["p_home"] > 0.5 and fh.count("L") >= 2)
                   or (pred["p_away"] > 0.5 and fa.count("L") >= 2)
                else "forma spójna z modelem"
            )
            upset_risk = pred["p_draw"] > 0.28 and roznica_sil > 0.4
            trap_game  = pred["conf_level"] == "High" and (fh.count("W") <= 1 or fa.count("W") >= 3)
            prompt = (
                f"Jesteś analitykiem piłkarskim piszącym w stylu 'Narrative Mode'.\n"
                f"Mecz: {home} vs {away}\n\n"
                f"Dane modelu (Dixon-Coles + blend sezonów + korekta formy):\n"
                f"- λ gosp: {pred['lam_h']:.2f} | λ gości: {pred['lam_a']:.2f}\n"
                f"- Szanse 1X2: {pred['p_home']:.1%} / {pred['p_draw']:.1%} / {pred['p_away']:.1%}\n"
                f"- Typ: {pred['typ']} @ {pred['fo_typ']:.2f} | Pewność: {pred['conf_level']}\n"
                f"- Wynik modelu: {pred['wynik_h']}:{pred['wynik_a']}\n"
                f"- Forma {home}: {fh} | {away}: {fa}\n"
                f"- Sygnały: {forma_vs_mod}"
                f"{', ⚠️ ryzyko niespodzianki' if upset_risk else ''}"
                f"{', 🪤 trap game?' if trap_game else ''}\n\n"
                f"Napisz 2-3 zdania po polsku. Narracyjny styl, konkretny i analityczny."
            )
            msg = client.messages.create(
                model="claude-opus-4-6", max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
    except Exception:
        pass

    roznica = pred["p_home"] - pred["p_away"]
    if   roznica >  0.20: faw = f"{home} jest wyraźnym faworytem ({pred['p_home']:.0%})."
    elif roznica >  0.08: faw = f"{home} jest lekkim faworytem ({pred['p_home']:.0%} vs {pred['p_away']:.0%})."
    elif roznica < -0.20: faw = f"{away} jest wyraźnym faworytem ({pred['p_away']:.0%})."
    elif roznica < -0.08: faw = f"{away} jest lekkim faworytem ({pred['p_away']:.0%} vs {pred['p_home']:.0%})."
    else:                 faw = f"Mecz bardzo wyrównany – remis ma {pred['p_draw']:.0%} szans."
    gole = pred["lam_h"] + pred["lam_a"]
    gole_opis = (
        f"Model spodziewa się bramkostrzelnego meczu (śr. {gole:.1f} goli)."
        if gole >= 2.8 else
        f"Model przewiduje defensywny mecz (śr. {gole:.1f} goli)."
    )
    return f"{faw} Forma: {home} [{_ocen_forme(fh)}: {fh}], {away} [{_ocen_forme(fa)}: {fa}]. {gole_opis}"

# ===========================================================================
# WERYFIKACJA – SQLITE (trwałe, przeżywa restart serwera)
# ===========================================================================

def init_db():
    con = sqlite3.connect(DB_FILE)
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
    con.commit()
    con.close()

def wczytaj_predykcje() -> list:
    init_db()
    con = sqlite3.connect(DB_FILE)
    rows = con.execute(
        "SELECT liga,home,away,round,typ,fo_typ,p_home,p_draw,p_away,data FROM predykcje ORDER BY data,round"
    ).fetchall()
    con.close()
    keys = ["liga","home","away","round","typ","fo_typ","p_home","p_draw","p_away","data"]
    return [dict(zip(keys, r)) for r in rows]

def zapisz_predykcje_db(nowe: list):
    init_db()
    con = sqlite3.connect(DB_FILE)
    dodane = 0
    for p in nowe:
        try:
            con.execute(
                "INSERT OR IGNORE INTO predykcje (liga,home,away,round,typ,fo_typ,p_home,p_draw,p_away,data) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (p["liga"], p["home"], p["away"], p["round"],
                 p["typ"], p["fo_typ"], p["p_home"], p["p_draw"], p["p_away"], p["data"])
            )
            dodane += con.execute("SELECT changes()").fetchone()[0]
        except Exception:
            pass
    con.commit()
    con.close()
    return dodane

def usun_wszystkie_predykcje():
    init_db()
    con = sqlite3.connect(DB_FILE)
    con.execute("DELETE FROM predykcje")
    con.commit()
    con.close()

def weryfikuj_predykcje(predykcje: list, hist: pd.DataFrame) -> pd.DataFrame:
    # Tylko bieżący sezon do matchowania
    if "_sezon" in hist.columns:
        hist = hist[hist["_sezon"] == "biezacy"]
    wyniki = []
    for p in predykcje:
        h, a = p["home"], p["away"]
        match = hist[(hist["HomeTeam"] == h) & (hist["AwayTeam"] == a)]
        brier = None
        if match.empty:
            status = "⏳ oczekuje"; wynik_r = "–"; trafiony = None
        else:
            row = match.iloc[-1]
            hg, ag   = int(row["FTHG"]), int(row["FTAG"])
            wynik_r  = f"{hg}:{ag}"
            rzecz    = "1" if hg > ag else ("2" if hg < ag else "X")
            typ_pred = p["typ"]
            if   typ_pred == "1X": trafiony = rzecz in ("1", "X")
            elif typ_pred == "X2": trafiony = rzecz in ("X", "2")
            else:                  trafiony = (rzecz == typ_pred)
            status = "✅ trafiony" if trafiony else "❌ chybiony"
            if all(p.get(k) is not None for k in ("p_home", "p_draw", "p_away")):
                r1 = 1.0 if rzecz == "1" else 0.0
                rx = 1.0 if rzecz == "X" else 0.0
                r2 = 1.0 if rzecz == "2" else 0.0
                brier = round((p["p_home"]-r1)**2 + (p["p_draw"]-rx)**2 + (p["p_away"]-r2)**2, 3)
        wyniki.append({
            "Liga": p.get("liga","–"), "Mecz": f"{h} vs {a}",
            "Kolejka": p.get("round","–"), "Typ": p["typ"],
            "Fair Odds": p["fo_typ"], "Wynik": wynik_r,
            "Status": status, "Trafiony": trafiony,
            "Brier": brier, "Data": p.get("data","–"),
        })
    return pd.DataFrame(wyniki) if wyniki else pd.DataFrame()

# ===========================================================================
# HELPERS UI
# ===========================================================================

def koloruj(p: float) -> str:
    return "🟢" if p > 0.65 else ("🟡" if p > 0.50 else "🔴")

def oblicz_p(typ: str, linia: float, lam: float) -> float:
    return 1 - poisson.cdf(linia, lam) if typ == "Over" else poisson.cdf(linia, lam)

def badge_typ(typ: str) -> str:
    if   typ == "1X": styl = "background:linear-gradient(90deg,#2196F3,#FF9800)"
    elif typ == "X2": styl = "background:linear-gradient(90deg,#FF9800,#E91E63)"
    else:
        kolory = {"1": "#2196F3", "X": "#FF9800", "2": "#E91E63"}
        styl   = f"background:{kolory.get(typ, '#888')}"
    return (f"<span style='{styl};color:white;padding:2px 14px;"
            f"border-radius:12px;font-weight:bold;font-size:0.95em'>{typ}</span>")

def render_macierz_html(M: np.ndarray, home: str, away: str) -> str:
    max_val = M.max() if M.max() > 0 else 1
    best    = np.unravel_index(M.argmax(), M.shape)
    n       = M.shape[0]
    rows    = [
        "<table style='border-collapse:collapse;font-size:0.70em;width:100%;margin-top:6px'>",
        "<thead><tr>",
        f"<th style='padding:2px 5px;color:#999;font-size:0.85em'>{home[:7]}↓ {away[:7]}→</th>",
    ]
    for j in range(n):
        rows.append(f"<th style='padding:2px 5px;text-align:center;color:#aaa'>{j}</th>")
    rows.append("</tr></thead><tbody>")
    for i in range(n):
        rows.append("<tr>")
        rows.append(f"<td style='padding:2px 5px;font-weight:bold;color:#aaa;text-align:center'>{i}</td>")
        for j in range(n):
            v     = M[i, j]
            ratio = v / max_val
            g     = int(255 * (1 - ratio * 0.85))
            bold  = "font-weight:bold;" if (i, j) == best else ""
            bord  = "border:2px solid #E91E63;" if (i, j) == best else ""
            rows.append(
                f"<td style='padding:2px 5px;text-align:center;"
                f"background:rgb(255,{g},{g});{bold}{bord}'>{v:.1%}</td>"
            )
        rows.append("</tr>")
    rows.append("</tbody></table>")
    return "".join(rows)

def macierz_goli(lam_h, lam_a, rho, linia_int, typ_gole):
    """Pomocnicza – zwraca macierz D-C i p_over dla combo buildera."""
    mg  = int(np.clip(np.ceil(max(lam_h, lam_a) + 4), 6, 10))
    M   = dixon_coles_adj(
        np.outer(poisson.pmf(range(mg), lam_h), poisson.pmf(range(mg), lam_a)),
        lam_h, lam_a, rho=rho
    )
    p_over = float(sum(M[i, j] for i in range(mg) for j in range(mg) if i + j > linia_int))
    p_g    = p_over if typ_gole == "Over" else 1 - p_over
    return M, p_g

# ===========================================================================
# ŁADOWANIE DANYCH
# ===========================================================================

st.sidebar.header("🌍 Wybór Rozgrywek")
wybrana_liga = st.sidebar.selectbox("Wybierz ligę", list(LIGI.keys()))
debug_mode   = st.sidebar.checkbox("🔧 Debug – niezmapowane nazwy", value=False)

historical = load_historical(LIGI[wybrana_liga]["csv_code"])
schedule   = load_schedule(LIGI[wybrana_liga]["file"])

st.title(f"Predykcje {wybrana_liga} 2025/26")
st.markdown("Dixon-Coles • Blend sezonów • Forma • Fair Odds • SQLite weryfikacja")

if not historical.empty:
    srednie_df  = oblicz_wszystkie_statystyki(historical.to_json())
    srednie_lig = oblicz_srednie_ligowe(historical.to_json())
    forma_dict  = oblicz_forme(historical)
    tabela      = tabela_ligowa(historical)
    rho         = srednie_lig["rho"]
    n_biezacy   = srednie_lig["n_biezacy"]
    w_prev      = waga_poprzedniego(n_biezacy)

    # ── SIDEBAR: szybki podgląd kolejki ──────────────────────────────────────
    if not schedule.empty and not srednie_df.empty:
        dzisiaj  = datetime.now().date()
        przyszle = schedule[schedule["date"].dt.date >= dzisiaj]
        if not przyszle.empty:
            nb_side = przyszle["round"].min()
            mecze_side = schedule[schedule["round"] == nb_side]
            st.sidebar.divider()
            st.sidebar.markdown(f"**⚽ Kolejka {int(nb_side)} – typy modelu**")
            for _, m in mecze_side.iterrows():
                h_s = map_nazwa(m["home_team"])
                a_s = map_nazwa(m["away_team"])
                if h_s not in srednie_df.index or a_s not in srednie_df.index:
                    continue
                lh_s, la_s = oblicz_lambdy(h_s, a_s, srednie_df, srednie_lig, forma_dict)
                pr_s = predykcja_meczu(lh_s, la_s, rho=rho)
                kolory_t = {"1": "🔵", "X": "🟠", "2": "🔴", "1X": "🟣", "X2": "🟣"}
                ikona    = kolory_t.get(pr_s["typ"], "⚪")
                st.sidebar.markdown(
                    f"{ikona} **{h_s[:10]}** vs **{a_s[:10]}**  \n"
                    f"&nbsp;&nbsp;&nbsp;`{pr_s['typ']}` @ {pr_s['fo_typ']:.2f} "
                    f"· {pr_s['conf_emoji']} {pr_s['conf_level']}",
                    unsafe_allow_html=True,
                )

    # ── TABS ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Bet Builder",
        "⚽ Przewidywane Wyniki",
        "✅ Weryfikacja",
        "📊 Tabela i Forma",
        "📈 Model & Dane",
    ])

    # =========================================================================
    # TAB 1 – BET BUILDER
    # =========================================================================
    with tab1:
        st.subheader("🎛️ Combo Builder")
        c1, c2, c3 = st.columns(3)
        with c1:
            linia_gole   = st.selectbox("Linia goli",    [1.5, 2.5, 3.5], index=1)
            typ_gole     = st.selectbox("Typ goli",      ["Over", "Under"])
        with c2:
            linia_rogi   = st.selectbox("Linia rożnych", [7.5, 8.5, 9.5, 10.5], index=1)
            typ_rogi     = st.selectbox("Typ rożnych",   ["Over", "Under"])
        with c3:
            linia_kartki = st.selectbox("Linia kartek",  [2.5, 3.5, 4.5, 5.5], index=1)
            typ_kartki   = st.selectbox("Typ kartek",    ["Over", "Under"])
        min_prob = st.slider("Min. prawdopodobieństwo combo", 0.0, 1.0, 0.40, 0.05)

        if not schedule.empty and not srednie_df.empty:
            dzisiaj  = datetime.now().date()
            przyszle = schedule[schedule["date"].dt.date >= dzisiaj]
            if not przyszle.empty:
                nb    = przyszle["round"].min()
                mecze = schedule[schedule["round"] == nb]
                st.caption(f"Kolejka {int(nb)} – {len(mecze)} meczów")
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Combo Builder** *(Over/Under przez macierz D-C)*")
                    combo_count = 0
                    for _, mecz in mecze.iterrows():
                        h = map_nazwa(mecz["home_team"])
                        a = map_nazwa(mecz["away_team"])
                        if h not in srednie_df.index or a not in srednie_df.index:
                            continue
                        lam_h, lam_a = oblicz_lambdy(h, a, srednie_df, srednie_lig, forma_dict)
                        lam_r = (srednie_df.loc[h,"Różne (dom)"] + srednie_df.loc[a,"Różne (wyjazd)"]) / 2
                        lam_k = (srednie_df.loc[h,"Kartki (dom)"] + srednie_df.loc[a,"Kartki (wyjazd)"]) / 2
                        _, p_g = macierz_goli(lam_h, lam_a, rho, int(linia_gole), typ_gole)
                        p_r   = oblicz_p(typ_rogi,   linia_rogi,   lam_r)
                        p_k   = oblicz_p(typ_kartki, linia_kartki, lam_k)
                        p_combo = p_g * p_r * p_k
                        if p_combo >= min_prob:
                            combo_count += 1
                            with st.expander(f"{h} vs {a} ({p_combo:.1%})"):
                                st.write(f"{koloruj(p_g)} Gole {typ_gole} {linia_gole}: {p_g:.1%}")
                                st.write(f"{koloruj(p_r)} Rożne {typ_rogi} {linia_rogi}: {p_r:.1%}")
                                st.write(f"{koloruj(p_k)} Kartki {typ_kartki} {linia_kartki}: {p_k:.1%}")
                    if combo_count == 0:
                        st.info("Brak meczów spełniających kryteria. Zmniejsz próg.")

                with col2:
                    st.write("**BTTS Ranking** *(z macierzy D-C)*")
                    btts_data = []
                    for _, mecz in mecze.iterrows():
                        h = map_nazwa(mecz["home_team"])
                        a = map_nazwa(mecz["away_team"])
                        if h not in srednie_df.index or a not in srednie_df.index:
                            continue
                        lam_h, lam_a = oblicz_lambdy(h, a, srednie_df, srednie_lig, forma_dict)
                        mg  = int(np.clip(np.ceil(max(lam_h, lam_a) + 4), 6, 10))
                        M_b = dixon_coles_adj(
                            np.outer(poisson.pmf(range(mg), lam_h), poisson.pmf(range(mg), lam_a)),
                            lam_h, lam_a, rho=rho
                        )
                        p_btts = float(1 - M_b[0,:].sum() - M_b[:,0].sum() + M_b[0,0])
                        btts_data.append((f"{h} - {a}", p_btts))
                    for ms, p in sorted(btts_data, key=lambda x: x[1], reverse=True):
                        st.write(f"{koloruj(p)} **{ms}**: {p:.1%}")

                st.divider()
                vc1, vc2 = st.columns(2)
                with vc1:
                    st.write("**🔥 Power Index – największa różnica sił**")
                    power_data = []
                    for _, mecz in mecze.iterrows():
                        h2 = map_nazwa(mecz["home_team"]); a2 = map_nazwa(mecz["away_team"])
                        if h2 not in srednie_df.index or a2 not in srednie_df.index: continue
                        lh2, la2 = oblicz_lambdy(h2, a2, srednie_df, srednie_lig, forma_dict)
                        power_data.append((f"{h2} vs {a2}", abs(lh2-la2), lh2, la2))
                    for label, diff, lh2, la2 in sorted(power_data, key=lambda x: x[1], reverse=True)[:5]:
                        st.write(f"⚡ **{label}** – gap {diff:.2f} (λ {lh2:.2f} vs {la2:.2f})")
                with vc2:
                    st.write("**⚽ Najbardziej bramkowy mecz**")
                    gole_data = []
                    for _, mecz in mecze.iterrows():
                        h2 = map_nazwa(mecz["home_team"]); a2 = map_nazwa(mecz["away_team"])
                        if h2 not in srednie_df.index or a2 not in srednie_df.index: continue
                        lh2, la2 = oblicz_lambdy(h2, a2, srednie_df, srednie_lig, forma_dict)
                        gole_data.append((f"{h2} vs {a2}", lh2 + la2))
                    for label, total in sorted(gole_data, key=lambda x: x[1], reverse=True)[:5]:
                        st.write(f"🎯 **{label}** – śr. {total:.2f} goli")
            else:
                st.info("Brak nadchodzących meczów w terminarzu.")

    # =========================================================================
    # TAB 2 – PRZEWIDYWANE WYNIKI (dwie kolumny, expander)
    # =========================================================================
    with tab2:
        st.subheader("⚽ Przewidywane wyniki – najbliższa kolejka")
        st.caption(
            "**Fair Odds** = kurs bez marży bukmachera. "
            "**Typ modelu** = optymalny typ wg modelu (1/X/2/1X/X2)."
        )

        tgl1, tgl2, tgl3 = st.columns(3)
        with tgl1: pokaz_komentarz = st.toggle("💬 Komentarz analityczny", value=True)
        with tgl2: pokaz_macierz  = st.toggle("🔢 Macierz wyników",        value=False)
        with tgl3:
            zapisuj_flag = st.toggle(
                "💾 Zapisz do weryfikacji", value=False,
                help="Zapisuje predykcje do SQLite – trwałe między sesjami."
            )

        if not schedule.empty and not srednie_df.empty:
            dzisiaj  = datetime.now().date()
            przyszle = schedule[schedule["date"].dt.date >= dzisiaj]
            if not przyszle.empty:
                nb    = przyszle["round"].min()
                mecze = schedule[schedule["round"] == nb]
                st.caption(f"Kolejka {int(nb)} – {len(mecze)} meczów")

                nowe_predykcje = []
                dopasowane     = 0

                # Dwie kolumny kart
                kol_a, kol_b = st.columns(2)
                mecze_list   = list(mecze.iterrows())

                for idx, (_, mecz) in enumerate(mecze_list):
                    h = map_nazwa(mecz["home_team"])
                    a = map_nazwa(mecz["away_team"])
                    if h not in srednie_df.index or a not in srednie_df.index:
                        continue

                    lam_h, lam_a = oblicz_lambdy(h, a, srednie_df, srednie_lig, forma_dict)
                    pred         = predykcja_meczu(lam_h, lam_a, rho=rho)
                    dopasowane  += 1
                    data_meczu   = mecz["date"].strftime("%d.%m %H:%M") if pd.notna(mecz["date"]) else ""

                    nowe_predykcje.append({
                        "liga": wybrana_liga, "home": h, "away": a,
                        "round": int(nb), "typ": pred["typ"], "fo_typ": pred["fo_typ"],
                        "p_home": round(pred["p_home"], 4),
                        "p_draw": round(pred["p_draw"], 4),
                        "p_away": round(pred["p_away"], 4),
                        "data": datetime.now().strftime("%Y-%m-%d"),
                    })

                    # Karta w odpowiedniej kolumnie
                    kolumna = kol_a if idx % 2 == 0 else kol_b

                    with kolumna:
                        with st.expander(
                            f"{'🟢' if pred['conf_level']=='High' else '🟡' if pred['conf_level']=='Medium' else '🔴'} "
                            f"**{h}** vs **{a}**  ·  {badge_typ(pred['typ'])} @ {pred['fo_typ']:.2f}  ·  {data_meczu}",
                            expanded=True
                        ):
                            # Nagłówek drużyny
                            ch, cmid, ca = st.columns([5, 2, 5])
                            with ch:
                                st.markdown(f"<div style='font-size:1.05em;font-weight:bold'>{h}</div>",
                                            unsafe_allow_html=True)
                            with cmid:
                                st.markdown(f"<div style='text-align:center;color:#888;font-size:0.78em'>{data_meczu}</div>",
                                            unsafe_allow_html=True)
                            with ca:
                                st.markdown(f"<div style='font-size:1.05em;font-weight:bold;text-align:right'>{a}</div>",
                                            unsafe_allow_html=True)

                            # Przewidywany wynik
                            st.markdown(
                                f"<div style='text-align:center;font-size:1.7em;font-weight:bold;margin:4px 0'>"
                                f"⚽ {pred['wynik_h']}:{pred['wynik_a']}"
                                f"<span style='font-size:0.5em;color:#888;font-weight:normal;margin-left:8px'>"
                                f"({pred['p_exact']:.1%})</span></div>",
                                unsafe_allow_html=True,
                            )

                            # Typ + confidence
                            conf_colors = {"High": "#4CAF50", "Medium": "#FF9800", "Coinflip": "#F44336"}
                            conf_c = conf_colors.get(pred["conf_level"], "#888")
                            st.markdown(
                                f"<div style='text-align:center;margin-bottom:4px'>"
                                f"Typ: {badge_typ(pred['typ'])}&nbsp;&nbsp;"
                                f"<span style='font-size:0.88em;color:#888'>"
                                f"Fair Odds: <b>{pred['fo_typ']:.2f}</b> ({pred['p_typ']:.1%})"
                                f"</span></div>"
                                f"<div style='text-align:center;font-size:0.80em;color:{conf_c};margin-bottom:8px'>"
                                f"{pred['conf_emoji']} <b>{pred['conf_level']}</b> · {pred['conf_opis']}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                            # Metryki 1X2
                            mc1, mc2, mc3 = st.columns(3)
                            with mc1:
                                st.metric(f"1 – {h[:12]}", f"{pred['p_home']:.1%}", f"@ {pred['fo_home']:.2f}",
                                          delta_color="normal" if pred["typ"] in ("1","1X") else "off")
                            with mc2:
                                st.metric("X – Remis", f"{pred['p_draw']:.1%}", f"@ {pred['fo_draw']:.2f}",
                                          delta_color="normal" if pred["typ"] in ("X","1X","X2") else "off")
                            with mc3:
                                st.metric(f"2 – {a[:12]}", f"{pred['p_away']:.1%}", f"@ {pred['fo_away']:.2f}",
                                          delta_color="normal" if pred["typ"] in ("2","X2") else "off")

                            # Oczekiwane gole (wyeksponowane)
                            st.markdown(
                                f"<div style='text-align:center;font-size:0.82em;color:#888;margin-top:2px'>"
                                f"λ {h[:8]}: <b>{pred['lam_h']:.2f}</b> &nbsp;|&nbsp; "
                                f"λ {a[:8]}: <b>{pred['lam_a']:.2f}</b> &nbsp;|&nbsp; "
                                f"Łącznie: <b>{pred['lam_h']+pred['lam_a']:.2f}</b></div>",
                                unsafe_allow_html=True,
                            )

                            if pokaz_komentarz:
                                st.info(f"💬 {generuj_komentarz(h, a, pred, forma_dict)}")

                            if pokaz_macierz:
                                st.markdown("**Macierz dokładnych wyników**")
                                st.markdown(render_macierz_html(pred["macierz"], h, a),
                                            unsafe_allow_html=True)

                # Zapis + Export
                if zapisuj_flag and nowe_predykcje:
                    dodane = zapisz_predykcje_db(nowe_predykcje)
                    st.success(f"✅ Zapisano {dodane} nowych predykcji do bazy." if dodane
                               else "Predykcje tej kolejki już są w bazie.")

                # Export CSV
                if dopasowane > 0:
                    df_export = pd.DataFrame(nowe_predykcje)[
                        ["liga","home","away","round","typ","fo_typ","p_home","p_draw","p_away","data"]
                    ]
                    st.download_button(
                        label="⬇️ Pobierz predykcje kolejki (CSV)",
                        data=df_export.to_csv(index=False, decimal=","),
                        file_name=f"predykcje_{wybrana_liga.replace(' ','_')}_kolejka{int(nb)}.csv",
                        mime="text/csv",
                    )

                if dopasowane == 0:
                    st.warning("Żaden mecz nie mógł być dopasowany – sprawdź Debug w sidebarze.")
            else:
                st.info("Brak nadchodzących meczów w terminarzu.")

    # =========================================================================
    # TAB 3 – WERYFIKACJA (SQLite, Brier, wykres BS w czasie)
    # =========================================================================
    with tab3:
        st.subheader("✅ Weryfikacja skuteczności modelu")
        st.caption(
            "Predykcje zapisujesz w zakładce **⚽ Przewidywane Wyniki** (toggle 💾). "
            "Dane trzymane są w SQLite – przeżywają restart serwera."
        )

        predykcje = wczytaj_predykcje()

        if not predykcje:
            st.info("Brak zapisanych predykcji. Zapisz je przed kolejką, wróć tu po meczach.")
        else:
            df_wer = weryfikuj_predykcje(predykcje, historical)

            if not df_wer.empty:
                zakonczone = df_wer[df_wer["Trafiony"].notna()]

                if not zakonczone.empty:
                    trafione    = int(zakonczone["Trafiony"].sum())
                    wszystkie   = len(zakonczone)
                    skutecznosc = trafione / wszystkie
                    sr_odds_tr  = zakonczone[zakonczone["Trafiony"] == True]["Fair Odds"].mean()
                    brier_vals  = zakonczone["Brier"].dropna()
                    sr_brier    = brier_vals.mean() if len(brier_vals) > 0 else float("nan")
                    brier_delta = f"{0.667 - sr_brier:+.3f} vs random" if not np.isnan(sr_brier) else None

                    st.markdown("### 📊 Podsumowanie")
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Rozegrane",        wszystkie)
                    m2.metric("Trafione",          trafione)
                    m3.metric("Skuteczność",       f"{skutecznosc:.1%}")
                    m4.metric("Śr. odds (hit)",   f"{sr_odds_tr:.2f}" if not np.isnan(sr_odds_tr) else "–")
                    m5.metric("Brier Score ↓",    f"{sr_brier:.3f}" if not np.isnan(sr_brier) else "–",
                              delta=brier_delta, delta_color="normal",
                              help="Niżej = lepiej. Losowy model ~0.667, dobry model <0.50")

                    # Wykres Brier Score w czasie (kalibracja w czasie)
                    if len(brier_vals) >= 3:
                        st.markdown("**📈 Brier Score kolejka po kolejce** *(niżej = model lepiej skalibrowany)*")
                        bs_time = (
                            zakonczone[zakonczone["Brier"].notna()]
                            .groupby("Kolejka")["Brier"]
                            .mean()
                            .reset_index()
                            .rename(columns={"Brier": "Śr. Brier Score"})
                        )
                        bs_time["Kolejka"] = bs_time["Kolejka"].astype(str)
                        st.line_chart(bs_time.set_index("Kolejka"))

                    # Skuteczność per liga
                    per_liga = (
                        zakonczone.groupby("Liga")["Trafiony"]
                        .agg(["sum","count"])
                        .rename(columns={"sum":"Trafione","count":"Mecze"})
                    )
                    per_liga["Skuteczność %"] = (per_liga["Trafione"]/per_liga["Mecze"]*100).round(1)
                    if len(per_liga) > 1:
                        st.bar_chart(per_liga["Skuteczność %"])

                    st.divider()

                # Tabela szczegółowa
                st.markdown("### 📋 Szczegółowe predykcje")
                filtr = st.selectbox("Filtruj", ["Wszystkie","✅ Trafione","❌ Chybione","⏳ Oczekujące"])
                df_show = df_wer.copy()
                if filtr == "✅ Trafione":   df_show = df_show[df_show["Status"] == "✅ trafiony"]
                elif filtr == "❌ Chybione": df_show = df_show[df_show["Status"] == "❌ chybiony"]
                elif filtr == "⏳ Oczekujące": df_show = df_show[df_show["Status"] == "⏳ oczekuje"]

                st.dataframe(df_show.drop(columns=["Trafiony"]), use_container_width=True, hide_index=True)

                # Export historii
                st.download_button(
                    "⬇️ Pobierz historię predykcji (CSV)",
                    data=df_wer.drop(columns=["Trafiony"]).to_csv(index=False, decimal=","),
                    file_name="historia_predykcji.csv",
                    mime="text/csv",
                )

                st.divider()
                if st.button("🗑️ Wyczyść wszystkie predykcje", type="secondary"):
                    usun_wszystkie_predykcje()
                    st.success("Baza wyczyszczona.")
                    st.rerun()

    # =========================================================================
    # TAB 4 – TABELA I FORMA
    # =========================================================================
    with tab4:
        st.subheader("📊 Aktualna Sytuacja")
        c_l, c_f = st.columns([2, 1])
        with c_l:
            st.write("**Tabela Ligowa** *(tylko bieżący sezon)*")
            st.dataframe(tabela, use_container_width=True)
        with c_f:
            st.write("**Forma (5 ostatnich)**")
            st.dataframe(
                pd.DataFrame.from_dict(forma_dict, orient="index", columns=["Forma"]),
                use_container_width=True,
            )

    # =========================================================================
    # TAB 5 – MODEL & DANE
    # =========================================================================
    with tab5:
        st.subheader("📈 Parametry modelu i dane")

        # Info o blend sezonów
        with st.container(border=True):
            st.markdown("**🔀 Blend sezonów**")
            bc1, bc2, bc3 = st.columns(3)
            bc1.metric("Mecze bieżącego sezonu", n_biezacy)
            bc2.metric("Waga poprzedniego sezonu", f"{w_prev:.0%}",
                       help="Maleje automatycznie w miarę gromadzenia danych bieżącego sezonu")
            bc3.metric("Łącznie rekordów w modelu", len(historical))

        # Parametry D-C
        with st.container(border=True):
            st.markdown("**⚙️ Parametry Dixon-Coles**")
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("ρ (rho) kalibrowane",    f"{rho:.4f}",
                       help="Kalibrowane empirycznie z historii tej ligi. Bliżej 0 = Poisson dokładniejszy.")
            dc2.metric("Śr. gole (dom)",          f"{srednie_lig['avg_home']:.2f}")
            dc3.metric("Śr. gole (wyjazd)",       f"{srednie_lig['avg_away']:.2f}")

        st.divider()
        st.write("**Średnie ważone drużyn** *(ostatnie 10 meczów, wagi rosnące)*")
        st.dataframe(srednie_df.sort_index(), use_container_width=True)

        st.caption(f"📅 Ostatnia aktualizacja danych: {historical['Date'].max().strftime('%d.%m.%Y')}")
        if st.button("🔄 Odśwież dane"):
            st.cache_data.clear()
            st.rerun()

    # Debug
    if debug_mode:
        st.sidebar.divider()
        st.sidebar.write("**🔧 Niezmapowane nazwy**")
        if _niezmapowane:
            for n in sorted(_niezmapowane): st.sidebar.warning(f"`{n}`")
        else:
            st.sidebar.success("Wszystkie zmapowane ✅")

else:
    st.error("Nie udało się pobrać danych. Sprawdź połączenie z internetem.")
