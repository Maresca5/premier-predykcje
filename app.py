import streamlit as st
import pandas as pd
from scipy.stats import poisson
from datetime import datetime
import requests
from io import StringIO
import numpy as np
import unicodedata
import json
import os

# ===========================================================================
# KONFIGURACJA LIG
# ===========================================================================
LIGI = {
    "Premier League": {"csv_code": "E0",  "file": "terminarz_premier_2025.csv"},
    "La Liga":        {"csv_code": "SP1", "file": "terminarz_la_liga_2025.csv"},
    "Bundesliga":     {"csv_code": "D1",  "file": "terminarz_bundesliga_2025.csv"},
    "Serie A":        {"csv_code": "I1",  "file": "terminarz_serie_a_2025.csv"},
    "Ligue 1":        {"csv_code": "F1",  "file": "terminarz_ligue_1_2025.csv"},
}

# Plik do przechowywania predykcji (na potrzeby weryfikacji skuteczności)
PREDYKCJE_FILE = "predykcje_historia.json"

st.set_page_config(page_title="Predykcje Piłkarskie Top 5", layout="wide")

# ===========================================================================
# SIDEBAR
# ===========================================================================
st.sidebar.header("🌍 Wybór Rozgrywek")
wybrana_liga = st.sidebar.selectbox("Wybierz ligę", list(LIGI.keys()))
debug_mode   = st.sidebar.checkbox("🔧 Debug – niezmapowane nazwy", value=False)

st.title(f"Predykcje {wybrana_liga} 2025/26")
st.markdown("Model Poissona + wagi formy • Fair Odds • Weryfikacja skuteczności")

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

# Klucz → wartość = ZAWSZE nazwa z football-data.co.uk
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
# ŁADOWANIE DANYCH
# ===========================================================================

@st.cache_data(ttl=900)
def load_historical(league_code: str) -> pd.DataFrame:
    try:
        url = f"https://www.football-data.co.uk/mmz4281/2526/{league_code}.csv"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        df = df.sort_values("Date")
        df["total_gole"]   = df["FTHG"] + df["FTAG"]
        df["total_kartki"] = df["HY"] + df["AY"] + (df["HR"] + df["AR"]) * 2
        df["total_rozne"]  = df["HC"] + df["AC"]
        return df
    except Exception as e:
        st.error(f"Błąd pobierania danych historycznych: {e}")
        return pd.DataFrame()


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
# OBLICZENIA STATYSTYK
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
    """Srednie ligowe potrzebne do normalizacji lambd."""
    df = pd.read_json(df_json)
    if df.empty:
        return {"avg_home": 1.5, "avg_away": 1.2}
    return {
        "avg_home": float(df["FTHG"].mean()),
        "avg_away": float(df["FTAG"].mean()),
    }


def oblicz_lambdy(h: str, a: str, srednie_df: pd.DataFrame, srednie_lig: dict) -> tuple:
    """
    Dixon-Coles style: lam = avg_liga * sila_ataku * slabosc_obrony.
    Normalizacja do sredniej ligowej sprawia ze lambdy sa zroznicowane
    i nie spieszczaja sie do ~1 przy slabszych parach druzyn.
    """
    avg_h = max(srednie_lig["avg_home"], 0.5)
    avg_a = max(srednie_lig["avg_away"], 0.5)

    atak_h   = srednie_df.loc[h, "Gole strzelone (dom)"]    / avg_h
    obrona_a = srednie_df.loc[a, "Gole stracone (wyjazd)"]  / avg_a
    atak_a   = srednie_df.loc[a, "Gole strzelone (wyjazd)"] / avg_a
    obrona_h = srednie_df.loc[h, "Gole stracone (dom)"]     / avg_h

    lam_h = avg_h * atak_h   * obrona_a
    lam_a = avg_a * atak_a   * obrona_h

    # Clamp – zabezpieczenie na start sezonu gdy malo danych
    lam_h = float(np.clip(lam_h, 0.3, 4.5))
    lam_a = float(np.clip(lam_a, 0.3, 4.5))
    return lam_h, lam_a



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
# MODEL POISSONA – PREDYKCJE
# ===========================================================================

MAX_GOLE = 7  # 0..6

def wybierz_typ(p_home: float, p_draw: float, p_away: float) -> tuple[str, float]:
    """
    Logika wyboru typu:
    - Jeśli jeden wynik ma p >= PROG_PEWNY  → typ czysty: 1 / X / 2
    - Jeśli żaden nie jest wyraźny, ale    → podwójna szansa:
        p_home + p_draw >= PROG_PODWOJNA   → 1X
        p_away + p_draw >= PROG_PODWOJNA   → X2
        (w razie remisu między 1X i X2 → wybieramy wyższy)
    - Ostateczny fallback: czysty typ z najwyższym p
    """
    PROG_PEWNY     = 0.42   # próg dla czystego typu (1/X/2)
    PROG_PODWOJNA  = 0.62   # próg dla podwójnej szansy (1X/X2)

    # Sprawdzamy czyste typy
    if p_home >= PROG_PEWNY:
        return "1",  p_home
    if p_away >= PROG_PEWNY:
        return "2",  p_away
    if p_draw >= PROG_PEWNY:
        return "X",  p_draw

    # Podwójne szanse
    p_1x = p_home + p_draw
    p_x2 = p_away + p_draw
    if p_1x >= PROG_PODWOJNA or p_x2 >= PROG_PODWOJNA:
        if p_1x >= p_x2:
            return "1X", p_1x
        else:
            return "X2", p_x2

    # Fallback: najwyższe p
    probs = {"1": p_home, "X": p_draw, "2": p_away}
    t = max(probs, key=probs.get)
    return t, probs[t]


def wybierz_wynik(M: np.ndarray, lam_h: float, lam_a: float) -> tuple:
    """
    Wybiera "przewidywany wynik" jako:
    - Wartosc oczekiwana lambdy zaokraglona (np. lam=1.8 -> 2 gole)
      JESLI roznica pomiedzy top-3 wynikami jest mala (maks spread < 4pp)
      co wskazuje ze argmax jest przypadkowy
    - W przeciwnym razie: argmax macierzy (jest wyrazny lider)
    Pokazuje tez osobno p_exact dla wybranego wyniku.
    """
    idx_max = np.unravel_index(M.argmax(), M.shape)
    p_max   = float(M[idx_max])

    # Top-3 prawdopodobienstw
    flat  = M.flatten()
    top3  = np.sort(flat)[::-1][:3]
    spread = float(top3[0] - top3[2])  # roznica miedzy 1. a 3.

    # Jesli spread maly -> uzywamy zaokraglonej wartosci oczekiwanej
    if spread < 0.04:
        g_h = int(round(lam_h))
        g_a = int(round(lam_a))
        # Clamp do zakresu macierzy
        g_h = min(g_h, M.shape[0] - 1)
        g_a = min(g_a, M.shape[1] - 1)
        return g_h, g_a, float(M[g_h, g_a])
    else:
        return int(idx_max[0]), int(idx_max[1]), p_max


def predykcja_meczu(lam_h: float, lam_a: float) -> dict:
    M = np.outer(
        poisson.pmf(range(MAX_GOLE), lam_h),
        poisson.pmf(range(MAX_GOLE), lam_a),
    )
    p_home = float(np.tril(M, -1).sum())
    p_draw = float(np.trace(M))
    p_away = float(np.triu(M, 1).sum())

    wynik_h, wynik_a, p_exact = wybierz_wynik(M, lam_h, lam_a)

    def fo(p: float) -> float:
        return round(1 / p, 2) if p > 0.001 else 999.0

    typ, p_typ = wybierz_typ(p_home, p_draw, p_away)

    return {
        "lam_h":   lam_h,
        "lam_a":   lam_a,
        "p_home":  p_home,
        "p_draw":  p_draw,
        "p_away":  p_away,
        "wynik_h": wynik_h,
        "wynik_a": wynik_a,
        "p_exact": p_exact,
        "fo_home": fo(p_home),
        "fo_draw": fo(p_draw),
        "fo_away": fo(p_away),
        "typ":     typ,
        "p_typ":   p_typ,
        "fo_typ":  fo(p_typ),
        "macierz": M,
    }

# ===========================================================================
# KOMENTARZ STATYSTYCZNY / AI
# ===========================================================================

def _ocen_forme(f: str) -> str:
    if not f or f == "?":
        return "forma nieznana"
    w = f.count("W"); l = f.count("L")
    if w >= 4: return "doskonała forma"
    if w >= 3: return "dobra forma"
    if l >= 3: return "słaba forma"
    return "nieregularna forma"

def generuj_komentarz(home: str, away: str, pred: dict, forma_dict: dict) -> str:
    fh = forma_dict.get(home, "?")
    fa = forma_dict.get(away, "?")

    # Próba Claude API (wymaga ANTHROPIC_API_KEY w st.secrets)
    try:
        import anthropic
        key = st.secrets.get("ANTHROPIC_API_KEY", None)
        if key:
            client = anthropic.Anthropic(api_key=key)
            prompt = (
                f"Jesteś zwięzłym analitykiem piłkarskim. Analizujesz mecz {home} vs {away}.\n\n"
                f"Dane modelu:\n"
                f"- Oczekiwane gole: {home} {pred['lam_h']:.2f} | {away} {pred['lam_a']:.2f}\n"
                f"- Szanse 1X2: {pred['p_home']:.1%} / {pred['p_draw']:.1%} / {pred['p_away']:.1%}\n"
                f"- Typ modelu: {pred['typ']} @ {pred['fo_typ']:.2f}\n"
                f"- Przewidywany wynik: {pred['wynik_h']}:{pred['wynik_a']}\n"
                f"- Forma {home}: {fh} | Forma {away}: {fa}\n\n"
                f"Napisz 2-3 zdania po polsku. Wspomnij kto jest faworytem, "
                f"co mówi forma i czy spodziewasz się niespodzianki. "
                f"Bądź konkretny, bez ogólników."
            )
            msg = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
    except Exception:
        pass

    # Fallback – czysto statystyczny
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
    forma_opis = (
        f"Forma: {home} [{_ocen_forme(fh)}: {fh}], "
        f"{away} [{_ocen_forme(fa)}: {fa}]."
    )
    return f"{faw} {forma_opis} {gole_opis}"

# ===========================================================================
# WERYFIKACJA PREDYKCJI
# ===========================================================================

def wczytaj_predykcje() -> list:
    if os.path.exists(PREDYKCJE_FILE):
        try:
            with open(PREDYKCJE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def zapisz_predykcje(lista: list) -> None:
    with open(PREDYKCJE_FILE, "w", encoding="utf-8") as f:
        json.dump(lista, f, ensure_ascii=False, indent=2)

def weryfikuj_predykcje(predykcje: list, hist: pd.DataFrame) -> pd.DataFrame:
    wyniki = []
    for p in predykcje:
        h, a = p["home"], p["away"]
        match = hist[(hist["HomeTeam"] == h) & (hist["AwayTeam"] == a)]
        if match.empty:
            status   = "⏳ oczekuje"
            wynik_r  = "–"
            trafiony = None
        else:
            row = match.iloc[-1]
            hg, ag = int(row["FTHG"]), int(row["FTAG"])
            wynik_r  = f"{hg}:{ag}"
            rzecz    = "1" if hg > ag else ("2" if hg < ag else "X")
            # Obsługa podwójnych szans
            typ_pred = p["typ"]
            if   typ_pred == "1X": trafiony = rzecz in ("1", "X")
            elif typ_pred == "X2": trafiony = rzecz in ("X", "2")
            else:                  trafiony = (rzecz == typ_pred)
            status   = "✅ trafiony" if trafiony else "❌ chybiony"
        wyniki.append({
            "Liga":           p.get("liga", "–"),
            "Mecz":           f"{h} vs {a}",
            "Kolejka":        p.get("round", "–"),
            "Typ modelu":     p["typ"],
            "Fair Odds":      p["fo_typ"],
            "Wynik":          wynik_r,
            "Status":         status,
            "Trafiony":       trafiony,
            "Data predykcji": p.get("data", "–"),
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
    if typ == "1X":
        styl = "background:linear-gradient(90deg,#2196F3,#FF9800)"
    elif typ == "X2":
        styl = "background:linear-gradient(90deg,#FF9800,#E91E63)"
    else:
        kolory = {"1": "#2196F3", "X": "#FF9800", "2": "#E91E63"}
        styl   = f"background:{kolory.get(typ, '#888')}"
    return (
        f"<span style='{styl};color:white;padding:2px 14px;"
        f"border-radius:12px;font-weight:bold;font-size:0.95em'>{typ}</span>"
    )

def render_macierz_html(M: np.ndarray, home: str, away: str) -> str:
    """Renderuje macierz wyników jako HTML table – unika problemów z st.dataframe + .style"""
    max_val = M.max() if M.max() > 0 else 1
    n = M.shape[0]
    rows = [
        "<table style='border-collapse:collapse;font-size:0.70em;width:100%;margin-top:6px'>",
        "<thead><tr>",
        f"<th style='padding:2px 5px;color:#999;font-size:0.85em'>{home[:7]}↓ {away[:7]}→</th>",
    ]
    for j in range(n):
        rows.append(f"<th style='padding:2px 5px;text-align:center;color:#aaa'>{j}</th>")
    rows.append("</tr></thead><tbody>")

    for i in range(n):
        rows.append("<tr>")
        rows.append(
            f"<td style='padding:2px 5px;font-weight:bold;color:#aaa;text-align:center'>{i}</td>"
        )
        for j in range(n):
            v = M[i, j]
            ratio = v / max_val
            # Odcień od białego do czerwonego
            r = 255
            g = int(255 * (1 - ratio * 0.85))
            b = int(255 * (1 - ratio * 0.85))
            bold  = "font-weight:bold;" if (i, j) == np.unravel_index(M.argmax(), M.shape) else ""
            border = "border:2px solid #E91E63;" if (i, j) == np.unravel_index(M.argmax(), M.shape) else ""
            rows.append(
                f"<td style='padding:2px 5px;text-align:center;"
                f"background:rgb({r},{g},{b});{bold}{border}'>"
                f"{v:.1%}</td>"
            )
        rows.append("</tr>")

    rows.append("</tbody></table>")
    return "".join(rows)

# ===========================================================================
# GŁÓWNA LOGIKA APLIKACJI
# ===========================================================================

historical = load_historical(LIGI[wybrana_liga]["csv_code"])
schedule   = load_schedule(LIGI[wybrana_liga]["file"])

if not historical.empty:
    srednie_df  = oblicz_wszystkie_statystyki(historical.to_json())
    srednie_lig = oblicz_srednie_ligowe(historical.to_json())
    forma_dict  = oblicz_forme(historical)
    tabela      = tabela_ligowa(historical)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Bet Builder",
        "⚽ Przewidywane Wyniki",
        "✅ Weryfikacja Skuteczności",
        "📊 Tabela i Forma",
        "📈 Statystyki Modelu",
    ])

    # =========================================================================
    # TAB 1 – BET BUILDER
    # =========================================================================
    with tab1:
        st.subheader("🎛️ Zbuduj własne combo")
        c1, c2, c3 = st.columns(3)
        with c1:
            linia_gole = st.selectbox("Linia goli",   [1.5, 2.5, 3.5], index=1)
            typ_gole   = st.selectbox("Typ goli",     ["Over", "Under"])
        with c2:
            linia_rogi = st.selectbox("Linia rożnych", [7.5, 8.5, 9.5, 10.5], index=1)
            typ_rogi   = st.selectbox("Typ rożnych",  ["Over", "Under"])
        with c3:
            linia_kartki = st.selectbox("Linia kartek", [2.5, 3.5, 4.5, 5.5], index=1)
            typ_kartki   = st.selectbox("Typ kartek",  ["Over", "Under"])

        min_prob = st.slider("Minimalne prawdopodobieństwo combo", 0.0, 1.0, 0.40, 0.05)
        st.subheader("📅 Predykcje – najbliższa kolejka")

        if not schedule.empty and not srednie_df.empty:
            dzisiaj  = datetime.now().date()
            przyszle = schedule[schedule["date"].dt.date >= dzisiaj]
            if not przyszle.empty:
                nb    = przyszle["round"].min()
                mecze = schedule[schedule["round"] == nb]
                st.caption(f"Kolejka {int(nb)} – {len(mecze)} meczów")

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Combo Builder**")
                    combo_count = 0
                    for _, mecz in mecze.iterrows():
                        h = map_nazwa(mecz["home_team"])
                        a = map_nazwa(mecz["away_team"])
                        if h not in srednie_df.index or a not in srednie_df.index:
                            continue
                        lam_h, lam_a = oblicz_lambdy(h, a, srednie_df, srednie_lig)
                        lam_r = (srednie_df.loc[h, "Różne (dom)"]             + srednie_df.loc[a, "Różne (wyjazd)"]) / 2
                        lam_k = (srednie_df.loc[h, "Kartki (dom)"]            + srednie_df.loc[a, "Kartki (wyjazd)"]) / 2
                        p_g = oblicz_p(typ_gole,   linia_gole,   lam_h + lam_a)
                        p_r = oblicz_p(typ_rogi,   linia_rogi,   lam_r)
                        p_k = oblicz_p(typ_kartki, linia_kartki, lam_k)
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
                    st.write("**BTTS Ranking**")
                    btts_data = []
                    for _, mecz in mecze.iterrows():
                        h = map_nazwa(mecz["home_team"])
                        a = map_nazwa(mecz["away_team"])
                        if h not in srednie_df.index or a not in srednie_df.index:
                            continue
                        lam_h, lam_a = oblicz_lambdy(h, a, srednie_df, srednie_lig)
                        p_btts = (1 - poisson.pmf(0, lam_h)) * (1 - poisson.pmf(0, lam_a))
                        btts_data.append((f"{h} - {a}", p_btts))
                    for ms, p in sorted(btts_data, key=lambda x: x[1], reverse=True):
                        st.write(f"{koloruj(p)} **{ms}**: {p:.1%}")
            else:
                st.info("Brak nadchodzących meczów w terminarzu.")

    # =========================================================================
    # TAB 2 – PRZEWIDYWANE WYNIKI
    # =========================================================================
    with tab2:
        st.subheader("⚽ Przewidywane wyniki – najbliższa kolejka")
        st.caption(
            "**Fair Odds** = kurs bez marży bukmachera wyliczony z modelu. "
            "**Typ modelu** = wynik z najwyższym p (1 / X / 2)."
        )

        ct1, ct2, ct3 = st.columns(3)
        with ct1:
            pokaz_komentarz = st.toggle("💬 Komentarz analityczny", value=True)
        with ct2:
            pokaz_macierz = st.toggle("🔢 Macierz wyników", value=False)
        with ct3:
            zapisuj_flag = st.toggle(
                "💾 Zapisz do weryfikacji", value=False,
                help="Zapisz predykcje tej kolejki – po meczach sprawdzisz skuteczność.",
            )

        if not schedule.empty and not srednie_df.empty:
            dzisiaj  = datetime.now().date()
            przyszle = schedule[schedule["date"].dt.date >= dzisiaj]

            if not przyszle.empty:
                nb    = przyszle["round"].min()
                mecze = schedule[schedule["round"] == nb]
                st.caption(f"Kolejka {int(nb)} – {len(mecze)} meczów")

                nowe_predykcje = []
                dopasowane = 0

                for _, mecz in mecze.iterrows():
                    h = map_nazwa(mecz["home_team"])
                    a = map_nazwa(mecz["away_team"])
                    if h not in srednie_df.index or a not in srednie_df.index:
                        continue

                    lam_h, lam_a = oblicz_lambdy(h, a, srednie_df, srednie_lig)
                    pred  = predykcja_meczu(lam_h, lam_a)
                    dopasowane += 1
                    data_meczu = mecz["date"].strftime("%d.%m %H:%M") if pd.notna(mecz["date"]) else ""

                    nowe_predykcje.append({
                        "liga":   wybrana_liga,
                        "home":   h,
                        "away":   a,
                        "round":  int(nb),
                        "typ":    pred["typ"],
                        "fo_typ": pred["fo_typ"],
                        "data":   datetime.now().strftime("%Y-%m-%d"),
                    })

                    # --- KARTA MECZU ---
                    with st.container(border=True):
                        # Wiersz: drużyna domowa | data | drużyna gości
                        col_h, col_mid, col_a = st.columns([5, 2, 5])
                        with col_h:
                            st.markdown(
                                f"<div style='font-size:1.05em;font-weight:bold;padding:2px 0'>{h}</div>",
                                unsafe_allow_html=True,
                            )
                        with col_mid:
                            st.markdown(
                                f"<div style='text-align:center;color:#888;font-size:0.78em;"
                                f"padding-top:5px'>{data_meczu}</div>",
                                unsafe_allow_html=True,
                            )
                        with col_a:
                            st.markdown(
                                f"<div style='font-size:1.05em;font-weight:bold;"
                                f"text-align:right;padding:2px 0'>{a}</div>",
                                unsafe_allow_html=True,
                            )

                        # Przewidywany wynik
                        st.markdown(
                            f"<div style='text-align:center;font-size:1.7em;"
                            f"font-weight:bold;margin:6px 0'>"
                            f"⚽ {pred['wynik_h']}:{pred['wynik_a']}"
                            f"<span style='font-size:0.5em;color:#888;"
                            f"font-weight:normal;margin-left:8px'>"
                            f"({pred['p_exact']:.1%})</span></div>",
                            unsafe_allow_html=True,
                        )

                        # Typ + fair odds
                        st.markdown(
                            f"<div style='text-align:center;margin-bottom:10px'>"
                            f"Typ modelu:&nbsp;{badge_typ(pred['typ'])}"
                            f"&nbsp;&nbsp;"
                            f"<span style='font-size:0.88em;color:#888'>"
                            f"Fair Odds:&nbsp;<b>{pred['fo_typ']:.2f}</b>"
                            f"&nbsp;({pred['p_typ']:.1%})"
                            f"</span></div>",
                            unsafe_allow_html=True,
                        )

                        # Metryki 1X2
                        mc1, mc2, mc3 = st.columns(3)
                        with mc1:
                            aktywny = pred["typ"] in ("1", "1X")
                            st.metric(
                                f"1 – {h[:13]}",
                                f"{pred['p_home']:.1%}",
                                f"@ {pred['fo_home']:.2f}",
                                delta_color="normal" if aktywny else "off",
                            )
                        with mc2:
                            aktywny = pred["typ"] in ("X", "1X", "X2")
                            st.metric(
                                "X – Remis",
                                f"{pred['p_draw']:.1%}",
                                f"@ {pred['fo_draw']:.2f}",
                                delta_color="normal" if aktywny else "off",
                            )
                        with mc3:
                            aktywny = pred["typ"] in ("2", "X2")
                            st.metric(
                                f"2 – {a[:13]}",
                                f"{pred['p_away']:.1%}",
                                f"@ {pred['fo_away']:.2f}",
                                delta_color="normal" if aktywny else "off",
                            )

                        # Komentarz
                        if pokaz_komentarz:
                            kom = generuj_komentarz(h, a, pred, forma_dict)
                            st.info(f"💬 {kom}")

                        # Macierz (HTML, nie st.dataframe)
                        if pokaz_macierz:
                            st.markdown("**Macierz dokładnych wyników** (wiersze = gole {h[:6]}, kolumny = gole {a[:6]})")
                            st.markdown(
                                render_macierz_html(pred["macierz"], h, a),
                                unsafe_allow_html=True,
                            )

                # Zapis predykcji
                if zapisuj_flag and nowe_predykcje:
                    istniejace = wczytaj_predykcje()
                    klucze     = {(p["liga"], p["home"], p["away"], p["round"]) for p in istniejace}
                    dodane     = 0
                    for p in nowe_predykcje:
                        k = (p["liga"], p["home"], p["away"], p["round"])
                        if k not in klucze:
                            istniejace.append(p)
                            dodane += 1
                    zapisz_predykcje(istniejace)
                    st.success(f"✅ Zapisano {dodane} nowych predykcji." if dodane else "Predykcje tej kolejki już zapisano.")

                if dopasowane == 0:
                    st.warning("Żaden mecz nie mógł być dopasowany – sprawdź Debug w sidebarze.")
            else:
                st.info("Brak nadchodzących meczów w terminarzu.")

    # =========================================================================
    # TAB 3 – WERYFIKACJA SKUTECZNOŚCI
    # =========================================================================
    with tab3:
        st.subheader("✅ Weryfikacja skuteczności modelu")
        st.caption(
            "Predykcje zapisujesz w zakładce **⚽ Przewidywane Wyniki** przed kolejką. "
            "Po rozegraniu meczów dane z football-data.co.uk są porównywane automatycznie."
        )

        predykcje = wczytaj_predykcje()

        if not predykcje:
            st.info(
                "Brak zapisanych predykcji. Przejdź do zakładki **⚽ Przewidywane Wyniki**, "
                "włącz przełącznik 💾 i wróć tu po kolejce."
            )
        else:
            df_wer = weryfikuj_predykcje(predykcje, historical)

            if not df_wer.empty:
                zakonczone = df_wer[df_wer["Trafiony"].notna()]

                if not zakonczone.empty:
                    trafione    = int(zakonczone["Trafiony"].sum())
                    wszystkie   = len(zakonczone)
                    skutecznosc = trafione / wszystkie
                    sr_odds_tr  = zakonczone[zakonczone["Trafiony"] == True]["Fair Odds"].mean()

                    st.markdown("### 📊 Podsumowanie")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Rozegrane mecze",      wszystkie)
                    m2.metric("Trafione typy",         trafione)
                    m3.metric("Skuteczność",           f"{skutecznosc:.1%}")
                    m4.metric("Śr. odds trafionych",  f"{sr_odds_tr:.2f}" if not np.isnan(sr_odds_tr) else "–")

                    # Wykres per liga (jeśli więcej niż jedna)
                    per_liga = (
                        zakonczone.groupby("Liga")["Trafiony"]
                        .agg(["sum", "count"])
                        .rename(columns={"sum": "Trafione", "count": "Mecze"})
                    )
                    per_liga["Skuteczność %"] = (per_liga["Trafione"] / per_liga["Mecze"] * 100).round(1)
                    if len(per_liga) > 1:
                        st.bar_chart(per_liga["Skuteczność %"])

                    st.divider()

                # Tabela szczegółowa
                st.markdown("### 📋 Szczegółowe wyniki")
                filtr = st.selectbox(
                    "Filtruj wyniki",
                    ["Wszystkie", "✅ Trafione", "❌ Chybione", "⏳ Oczekujące"],
                )
                df_show = df_wer.copy()
                if filtr == "✅ Trafione":
                    df_show = df_show[df_show["Status"] == "✅ trafiony"]
                elif filtr == "❌ Chybione":
                    df_show = df_show[df_show["Status"] == "❌ chybiony"]
                elif filtr == "⏳ Oczekujące":
                    df_show = df_show[df_show["Status"] == "⏳ oczekuje"]

                st.dataframe(
                    df_show.drop(columns=["Trafiony"]),
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()
                if st.button("🗑️ Wyczyść wszystkie predykcje", type="secondary"):
                    zapisz_predykcje([])
                    st.success("Predykcje wyczyszczone.")
                    st.rerun()

    # =========================================================================
    # TAB 4 – TABELA I FORMA
    # =========================================================================
    with tab4:
        st.subheader("📊 Aktualna Sytuacja")
        c_l, c_f = st.columns([2, 1])
        with c_l:
            st.write("**Tabela Ligowa**")
            st.dataframe(tabela, use_container_width=True)
        with c_f:
            st.write("**Forma (5 ostatnich)**")
            st.dataframe(
                pd.DataFrame.from_dict(forma_dict, orient="index", columns=["Forma"]),
                use_container_width=True,
            )

    # =========================================================================
    # TAB 5 – STATYSTYKI MODELU
    # =========================================================================
    with tab5:
        st.subheader("📊 Średnie ważone drużyn")
        st.write("Dane uwzględniają atut własnego boiska + wagę ostatnich meczów (10 ostatnich).")
        st.dataframe(srednie_df.sort_index(), use_container_width=True)
        st.divider()
        st.caption(f"📊 Liczba meczów w bazie: {len(historical)}")
        st.caption(f"📅 Ostatnia aktualizacja: {historical['Date'].max().strftime('%d.%m.%Y')}")
        if st.button("🔄 Odśwież dane"):
            st.cache_data.clear()
            st.rerun()

    # Debug mode
    if debug_mode:
        st.sidebar.divider()
        st.sidebar.write("**🔧 Niezmapowane nazwy drużyn**")
        if _niezmapowane:
            for n in sorted(_niezmapowane):
                st.sidebar.warning(f"`{n}`")
        else:
            st.sidebar.success("Wszystkie nazwy zmapowane ✅")

else:
    st.error("Nie udało się pobrać danych historycznych. Sprawdź połączenie z internetem.")
