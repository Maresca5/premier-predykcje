import streamlit as st
import pandas as pd
from scipy.stats import poisson
from datetime import datetime, timedelta
import requests
from io import StringIO
import numpy as np
import unicodedata

# --- KONFIGURACJA LIG ---
LIGI = {
    "Premier League": {"csv_code": "E0", "file": "terminarz_premier_2025.csv"},
    "La Liga":        {"csv_code": "SP1", "file": "terminarz_la_liga_2025.csv"},
    "Bundesliga":     {"csv_code": "D1",  "file": "terminarz_bundesliga_2025.csv"},
    "Serie A":        {"csv_code": "I1",  "file": "terminarz_serie_a_2025.csv"},
    "Ligue 1":        {"csv_code": "F1",  "file": "terminarz_ligue_1_2025.csv"},
}

st.set_page_config(page_title="Predykcje Piłkarskie Top 5", layout="wide")

# --- SIDEBAR ---
st.sidebar.header("🌍 Wybór Rozgrywek")
wybrana_liga = st.sidebar.selectbox("Wybierz ligę", list(LIGI.keys()))
debug_mode = st.sidebar.checkbox("🔧 Debug – niezmapowane nazwy", value=False)

st.title(f"Predykcje {wybrana_liga} 2025/26")
st.markdown("Model Poissona + wagi formy • Fair Odds • Komentarze AI")

# ===========================================================================
# MAPOWANIA
# ===========================================================================

def normalize_name(name):
    if not isinstance(name, str):
        return str(name)
    name = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode("ASCII")
    for suffix in [" FC", " CF", " UD", " CD", " RCD", " AS", " Stade", " 1."]:
        name = name.replace(suffix, "")
    return name.strip()

# Klucz -> wartość jest ZAWSZE nazwą używaną w football-data.co.uk
# Zasada: żaden klucz nie może być równy wartości innego klucza (zapobiega pętlom)
NAZWY_MAP = {
    # ===== PREMIER LEAGUE =====
    "Brighton & Hove Albion": "Brighton",
    "West Ham United":        "West Ham",
    "Newcastle United":       "Newcastle",
    "Tottenham Hotspur":      "Tottenham",
    "Leeds United":           "Leeds",
    "Manchester United":      "Man United",
    "Manchester City":        "Man City",
    "Nottingham Forest":      "Nott'm Forest",
    "Wolverhampton Wanderers":"Wolves",
    "Wolverhampton":          "Wolves",
    "Leicester City":         "Leicester",
    "Sheffield United":       "Sheffield Utd",

    # ===== LA LIGA =====
    "Girona FC":              "Girona",
    "Rayo Vallecano":         "Vallecano",
    "RCD Mallorca":           "Mallorca",
    "FC Barcelona":           "Barcelona",
    "Deportivo Alavés":       "Alaves",
    "Deportivo Alaves":       "Alaves",
    "Levante UD":             "Levante",
    "Valencia CF":            "Valencia",
    "Real Sociedad":          "Sociedad",
    "Celta Vigo":             "Celta",
    "Getafe CF":              "Getafe",
    "Athletic Club":          "Ath Bilbao",
    "Athletic Bilbao":        "Ath Bilbao",
    "Sevilla FC":             "Sevilla",
    "RCD Espanyol":           "Espanol",
    "Espanyol":               "Espanol",
    "Atlético Madrid":        "Ath Madrid",
    "Atletico Madrid":        "Ath Madrid",
    "Elche CF":               "Elche",
    "Real Betis":             "Betis",
    "CA Osasuna":             "Osasuna",
    "Real Valladolid":        "Valladolid",
    "UD Las Palmas":          "Las Palmas",
    "Cadiz CF":               "Cadiz",
    "Granada CF":             "Granada",
    "UD Almeria":             "Almeria",

    # ===== BUNDESLIGA =====
    "FC Bayern München":              "Bayern Munich",
    "Bayern":                         "Bayern Munich",
    "Borussia Dortmund":              "Dortmund",
    "RB Leipzig":                     "RB Leipzig",
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

    # ===== SERIE A =====
    "AC Milan":               "Milan",
    "Internazionale":         "Inter",
    "AS Roma":                "Roma",
    "Hellas Verona":          "Verona",

    # ===== LIGUE 1 =====
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

_niezmapowane: set = set()  # do debug mode

def map_nazwa(nazwa_z_csv: str) -> str:
    if not isinstance(nazwa_z_csv, str):
        return str(nazwa_z_csv)

    # 1. Bezpośrednie
    if nazwa_z_csv in NAZWY_MAP:
        return NAZWY_MAP[nazwa_z_csv]

    # 2. Uproszczone (bez spacji/myślników/apostrofów)
    def uproszcz(s):
        return s.replace(" ", "").replace("-", "").replace("'", "").lower()

    for key, val in NAZWY_MAP.items():
        if uproszcz(key) == uproszcz(nazwa_z_csv):
            return val

    # 3. Znormalizowana (ASCII)
    znorm = normalize_name(nazwa_z_csv)
    if znorm in NAZWY_MAP:
        return NAZWY_MAP[znorm]

    # 4. Brak dopasowania – rejestruj i zwróć oryginał
    _niezmapowane.add(nazwa_z_csv)
    return nazwa_z_csv


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
            date_to_round = {d: i + 1 for i, d in enumerate(unique_dates)}
            df["round"] = df["date"].dt.date.map(date_to_round)
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


# UWAGA: cache na hash stringa (nazwy ligi) żeby uniknąć problemów z hash DataFrame
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


def oblicz_forme(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    druzyny = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
    forma = {}
    for d in druzyny:
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
        for team in [home, away]:
            if team not in table:
                table[team] = {"Pkt": 0, "GZ": 0, "GS": 0, "M": 0}
        table[home]["GZ"] += hg
        table[home]["GS"] += ag
        table[home]["M"]  += 1
        table[away]["GZ"] += ag
        table[away]["GS"] += hg
        table[away]["M"]  += 1
        if hg > ag:
            table[home]["Pkt"] += 3
        elif hg < ag:
            table[away]["Pkt"] += 3
        else:
            table[home]["Pkt"] += 1
            table[away]["Pkt"] += 1
    res = pd.DataFrame(table).T
    res["RG"] = res["GZ"] - res["GS"]
    return res.sort_values(["Pkt", "RG", "GZ"], ascending=False)


# ===========================================================================
# MODEL POISSONA – PREDYKCJE WYNIKÓW
# ===========================================================================

MAX_GOLE = 7  # 0..6 dla każdej drużyny

def oblicz_macierz(lam_h: float, lam_a: float) -> np.ndarray:
    """Macierz prawdopodobieństw wyników [gole_h x gole_a]"""
    return np.outer(
        poisson.pmf(range(MAX_GOLE), lam_h),
        poisson.pmf(range(MAX_GOLE), lam_a),
    )


def predykcja_meczu(lam_h: float, lam_a: float) -> dict:
    M = oblicz_macierz(lam_h, lam_a)
    p_home = float(np.tril(M, -1).sum())
    p_draw = float(np.trace(M))
    p_away = float(np.triu(M, 1).sum())

    # Najbardziej prawdopodobny dokładny wynik
    idx = np.unravel_index(M.argmax(), M.shape)
    p_exact = float(M[idx])

    # Fair odds (bez marży)
    def fair_odds(p: float) -> str:
        return f"{1/p:.2f}" if p > 0 else "∞"

    return {
        "lam_h": lam_h,
        "lam_a": lam_a,
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "wynik_h": int(idx[0]),
        "wynik_a": int(idx[1]),
        "p_exact": p_exact,
        "fo_home": fair_odds(p_home),
        "fo_draw": fair_odds(p_draw),
        "fo_away": fair_odds(p_away),
        "macierz": M,
    }


# ===========================================================================
# KOMENTARZ AI (Claude API przez Anthropic w artifacts)
# Jeśli klucz API nie jest dostępny, wyświetla statystyczny fallback
# ===========================================================================

def generuj_komentarz(home: str, away: str, pred: dict, forma_dict: dict) -> str:
    """Generuje komentarz analityczny. Używa Claude API jeśli dostępny, 
    w przeciwnym razie generuje komentarz na podstawie samych danych."""

    forma_h = forma_dict.get(home, "brak danych")
    forma_a = forma_dict.get(away, "brak danych")

    # Próba użycia Claude API
    try:
        client = _get_anthropic_client()
        if client:
            prompt = f"""Jesteś zwięzłym analitykiem piłkarskim. Analizujesz mecz {home} vs {away}.

Dane modelu Poissona:
- Oczekiwane gole: {home} {pred['lam_h']:.2f} | {away} {pred['lam_a']:.2f}
- Szanse: Wygrana {home} {pred['p_home']:.1%} | Remis {pred['p_draw']:.1%} | Wygrana {away} {pred['p_away']:.1%}
- Najbardziej prawdopodobny wynik: {pred['wynik_h']}:{pred['wynik_a']} ({pred['p_exact']:.1%})
- Forma {home} (ostatnie 5): {forma_h}
- Forma {away} (ostatnie 5): {forma_a}

Napisz komentarz analityczny 2-3 zdania po polsku. Uwzględnij:
- Kto jest faworytem i z jakiego powodu
- Co mówi forma o aktualnej dyspozycji
- Ewentualne niespodzianki lub wyrównany mecz
Bądź konkretny, bez ogólników. Nie używaj słowa 'jednoznacznie'."""

            import anthropic
            message = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text.strip()
    except Exception:
        pass

    # Fallback – komentarz statystyczny bez AI
    return _komentarz_statystyczny(home, away, pred, forma_h, forma_a)


def _get_anthropic_client():
    """Zwraca klienta Anthropic jeśli klucz jest w st.secrets, inaczej None."""
    try:
        import anthropic
        key = st.secrets.get("ANTHROPIC_API_KEY", None)
        if key:
            return anthropic.Anthropic(api_key=key)
    except Exception:
        pass
    return None


def _komentarz_statystyczny(home: str, away: str, pred: dict, forma_h: str, forma_a: str) -> str:
    """Komentarz generowany czysto na podstawie danych – bez zewnętrznego API."""
    roznica = pred["p_home"] - pred["p_away"]

    if roznica > 0.20:
        faworyt = f"{home} jest wyraźnym faworytem ({pred['p_home']:.0%} szans na wygraną)."
    elif roznica > 0.08:
        faworyt = f"{home} jest nieznacznym faworytem ({pred['p_home']:.0%} vs {pred['p_away']:.0%})."
    elif roznica < -0.20:
        faworyt = f"{away} jest wyraźnym faworytem ({pred['p_away']:.0%} szans na wygraną)."
    elif roznica < -0.08:
        faworyt = f"{away} jest nieznacznym faworytem ({pred['p_away']:.0%} vs {pred['p_home']:.0%})."
    else:
        faworyt = f"Mecz bardzo wyrównany – remis ma {pred['p_draw']:.0%} szans."

    # Forma
    def ocen_forme(f: str) -> str:
        if not f or f == "brak danych":
            return "forma nieznana"
        wins = f.count("W")
        losses = f.count("L")
        if wins >= 4:
            return "doskonała forma"
        elif wins >= 3:
            return "dobra forma"
        elif losses >= 3:
            return "słaba forma"
        else:
            return "nieregularna forma"

    forma_opis = f"{home} ({ocen_forme(forma_h)}: {forma_h}), {away} ({ocen_forme(forma_a)}: {forma_a})."

    gole_oczekiwane = pred["lam_h"] + pred["lam_a"]
    gole_opis = (
        f"Model spodziewa się bramkostrzelnego meczu (śr. {gole_oczekiwane:.1f} goli)."
        if gole_oczekiwane >= 2.8
        else f"Model przewiduje mecz z niewielu bramkami (śr. {gole_oczekiwane:.1f} goli)."
    )

    return f"{faworyt} Forma: {forma_opis} {gole_opis}"


# ===========================================================================
# HELPERS UI
# ===========================================================================

def koloruj(p: float) -> str:
    return "🟢" if p > 0.65 else ("🟡" if p > 0.50 else "🔴")


def oblicz_p(typ: str, linia: float, lam: float) -> float:
    return 1 - poisson.cdf(linia, lam) if typ == "Over" else poisson.cdf(linia, lam)


# ===========================================================================
# GŁÓWNA LOGIKA
# ===========================================================================

historical = load_historical(LIGI[wybrana_liga]["csv_code"])
schedule   = load_schedule(LIGI[wybrana_liga]["file"])

if not historical.empty:
    srednie_df  = oblicz_wszystkie_statystyki(historical.to_json())
    forma_dict  = oblicz_forme(historical)
    tabela      = tabela_ligowa(historical)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Bet Builder",
        "⚽ Przewidywane Wyniki",
        "📊 Tabela i Forma",
        "📈 Statystyki Modelu",
    ])

    # -----------------------------------------------------------------------
    # TAB 1 – BET BUILDER
    # -----------------------------------------------------------------------
    with tab1:
        st.subheader("🎛️ Zbuduj własne combo")
        c1, c2, c3 = st.columns(3)
        with c1:
            linia_gole = st.selectbox("Linia goli", [1.5, 2.5, 3.5], index=1)
            typ_gole   = st.selectbox("Typ goli", ["Over", "Under"], index=0)
        with c2:
            linia_rogi = st.selectbox("Linia rożnych", [7.5, 8.5, 9.5, 10.5], index=1)
            typ_rogi   = st.selectbox("Typ rożnych", ["Over", "Under"], index=0)
        with c3:
            linia_kartki = st.selectbox("Linia kartek", [2.5, 3.5, 4.5, 5.5], index=1)
            typ_kartki   = st.selectbox("Typ kartek", ["Over", "Under"], index=0)

        min_prob = st.slider("Minimalne prawdopodobieństwo combo", 0.0, 1.0, 0.40, 0.05)

        st.subheader("📅 Predykcje – najbliższa kolejka")

        if not schedule.empty and not srednie_df.empty:
            dzisiaj = datetime.now().date()
            przyszle = schedule[schedule["date"].dt.date >= dzisiaj]

            if not przyszle.empty:
                najblizsza = przyszle["round"].min()
                mecze = schedule[schedule["round"] == najblizsza]
                st.caption(f"Kolejka {int(najblizsza)} – {len(mecze)} meczów")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Combo Builder**")
                    combo_count = 0
                    for _, mecz in mecze.iterrows():
                        h = map_nazwa(mecz["home_team"])
                        a = map_nazwa(mecz["away_team"])
                        if h not in srednie_df.index or a not in srednie_df.index:
                            continue

                        lam_h = (srednie_df.loc[h, "Gole strzelone (dom)"]    + srednie_df.loc[a, "Gole stracone (wyjazd)"]) / 2
                        lam_a = (srednie_df.loc[a, "Gole strzelone (wyjazd)"] + srednie_df.loc[h, "Gole stracone (dom)"]) / 2
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
                        lam_h = (srednie_df.loc[h, "Gole strzelone (dom)"]    + srednie_df.loc[a, "Gole stracone (wyjazd)"]) / 2
                        lam_a = (srednie_df.loc[a, "Gole strzelone (wyjazd)"] + srednie_df.loc[h, "Gole stracone (dom)"]) / 2
                        p_btts = (1 - poisson.pmf(0, lam_h)) * (1 - poisson.pmf(0, lam_a))
                        btts_data.append((f"{h} - {a}", p_btts))

                    for mecz_str, p in sorted(btts_data, key=lambda x: x[1], reverse=True):
                        st.write(f"{koloruj(p)} **{mecz_str}**: {p:.1%}")
            else:
                st.info("Brak nadchodzących meczów w terminarzu.")
        else:
            if schedule.empty:
                st.warning("Brak danych terminarza.")
            if srednie_df.empty:
                st.warning("Brak statystyk drużyn (za mało danych historycznych).")

    # -----------------------------------------------------------------------
    # TAB 2 – PRZEWIDYWANE WYNIKI (NOWY)
    # -----------------------------------------------------------------------
    with tab2:
        st.subheader("⚽ Przewidywane wyniki – najbliższa kolejka")
        st.caption("Fair Odds = kurs bez marży bukmachera, wyliczony z modelu Poissona.")

        pokaz_komentarz = st.toggle("💬 Pokaż komentarz analityczny", value=True)
        pokaz_macierz   = st.toggle("🔢 Pokaż macierz wyników", value=False)

        if not schedule.empty and not srednie_df.empty:
            dzisiaj = datetime.now().date()
            przyszle = schedule[schedule["date"].dt.date >= dzisiaj]

            if not przyszle.empty:
                najblizsza = przyszle["round"].min()
                mecze = schedule[schedule["round"] == najblizsza]
                st.caption(f"Kolejka {int(najblizsza)} – {len(mecze)} meczów")

                dopasowane = 0
                for _, mecz in mecze.iterrows():
                    h_raw = mecz["home_team"]
                    a_raw = mecz["away_team"]
                    h = map_nazwa(h_raw)
                    a = map_nazwa(a_raw)

                    if h not in srednie_df.index or a not in srednie_df.index:
                        continue

                    lam_h = (srednie_df.loc[h, "Gole strzelone (dom)"]    + srednie_df.loc[a, "Gole stracone (wyjazd)"]) / 2
                    lam_a = (srednie_df.loc[a, "Gole strzelone (wyjazd)"] + srednie_df.loc[h, "Gole stracone (dom)"]) / 2
                    pred  = predykcja_meczu(lam_h, lam_a)
                    dopasowane += 1

                    # Data meczu
                    data_meczu = mecz["date"].strftime("%d.%m %H:%M") if pd.notna(mecz["date"]) else ""

                    with st.container(border=True):
                        # Nagłówek
                        col_h, col_vs, col_a = st.columns([5, 2, 5])
                        with col_h:
                            st.markdown(f"### {h}")
                        with col_vs:
                            st.markdown(f"<div style='text-align:center;padding-top:8px;color:gray'>{data_meczu}</div>", unsafe_allow_html=True)
                        with col_a:
                            st.markdown(f"<div style='text-align:right'>### {a}</div>", unsafe_allow_html=True)
                            st.markdown(f"### {a}")

                        # Przewidywany wynik
                        st.markdown(
                            f"<h2 style='text-align:center;margin:0'>⚽ {pred['wynik_h']} : {pred['wynik_a']} "
                            f"<span style='font-size:0.6em;color:gray'>({pred['p_exact']:.1%})</span></h2>",
                            unsafe_allow_html=True,
                        )

                        # Prawdopodobieństwa i fair odds
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric(f"1 – {h[:10]}", f"{pred['p_home']:.1%}", f"Fair Odds: {pred['fo_home']}")
                        with c2:
                            st.metric("X – Remis", f"{pred['p_draw']:.1%}", f"Fair Odds: {pred['fo_draw']}")
                        with c3:
                            st.metric(f"2 – {a[:10]}", f"{pred['p_away']:.1%}", f"Fair Odds: {pred['fo_away']}")

                        # Komentarz AI / statystyczny
                        if pokaz_komentarz:
                            with st.spinner("Generuję komentarz..."):
                                kom = generuj_komentarz(h, a, pred, forma_dict)
                            st.info(f"💬 {kom}")

                        # Macierz wyników
                        if pokaz_macierz:
                            st.write("**Macierz prawdopodobieństw dokładnych wyników**")
                            M = pred["macierz"]
                            df_m = pd.DataFrame(
                                M,
                                index=[f"{h[:6]}={i}" for i in range(MAX_GOLE)],
                                columns=[f"{a[:6]}={j}" for j in range(MAX_GOLE)],
                            )
                            st.dataframe(df_m.style.format("{:.1%}").background_gradient(cmap="YlOrRd"), use_container_width=True)

                if dopasowane == 0:
                    st.warning("Żaden mecz z kolejki nie mógł być zmapowany na dane historyczne. Sprawdź Debug w sidebarze.")
            else:
                st.info("Brak nadchodzących meczów w terminarzu.")

    # -----------------------------------------------------------------------
    # TAB 3 – TABELA I FORMA
    # -----------------------------------------------------------------------
    with tab3:
        st.subheader("📊 Aktualna Sytuacja")
        c_l, c_f = st.columns([2, 1])
        with c_l:
            st.write("**Tabela Ligowa**")
            st.dataframe(tabela, use_container_width=True)
        with c_f:
            st.write("**Forma (5 ostatnich)**")
            forma_df = pd.DataFrame.from_dict(forma_dict, orient="index", columns=["Forma"])
            st.dataframe(forma_df, use_container_width=True)

    # -----------------------------------------------------------------------
    # TAB 4 – STATYSTYKI MODELU
    # -----------------------------------------------------------------------
    with tab4:
        st.subheader("📊 Średnie ważone drużyn")
        st.write("Dane uwzględniają atut własnego boiska oraz wagę ostatnich meczów (10 ostatnich spotkań).")
        st.dataframe(srednie_df.sort_index(), use_container_width=True)

        st.divider()
        st.caption(f"📊 Liczba meczów w bazie: {len(historical)}")
        st.caption(f"📅 Ostatnia aktualizacja: {historical['Date'].max().strftime('%d.%m.%Y')}")

        if st.button("🔄 Odśwież dane"):
            st.cache_data.clear()
            st.rerun()

    # -----------------------------------------------------------------------
    # DEBUG MODE
    # -----------------------------------------------------------------------
    if debug_mode:
        st.sidebar.divider()
        st.sidebar.write("**🔧 Niezmapowane nazwy drużyn**")
        if _niezmapowane:
            for n in sorted(_niezmapowane):
                st.sidebar.warning(f"`{n}`")
        else:
            st.sidebar.success("Wszystkie nazwy zmapowane poprawnie ✅")

else:
    st.error("Nie udało się pobrać danych historycznych. Sprawdź połączenie z internetem lub spróbuj później.")
