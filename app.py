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
    "La Liga": {"csv_code": "SP1", "file": "terminarz_la_liga_2025.csv"},
    "Bundesliga": {"csv_code": "D1", "file": "terminarz_bundesliga_2025.csv"},
    "Serie A": {"csv_code": "I1", "file": "terminarz_serie_a_2025.csv"},
    "Ligue 1": {"csv_code": "F1", "file": "terminarz_ligue_1_2025.csv"}
}

st.set_page_config(page_title="Football Predictor Pro", layout="wide")

# --- MAPOWANIE NAZW (Zoptymalizowane pod football-data.co.uk) ---
NAZWY_MAP = {
    # La Liga
    "Atletico Madrid": "Ath Madrid", "Atlético Madrid": "Ath Madrid",
    "Athletic Club": "Ath Bilbao", "Athletic Bilbao": "Ath Bilbao",
    "Real Sociedad": "Sociedad", "Villarreal CF": "Villarreal",
    "Girona FC": "Girona", "Getafe CF": "Getafe",
    "Sevilla FC": "Sevilla", "Valencia CF": "Valencia",
    "RCD Mallorca": "Mallorca", "RCD Espanyol": "Espanyol",
    "Rayo Vallecano": "Vallecano", "CA Osasuna": "Osasuna",
    "Deportivo Alaves": "Alaves", "Deportivo Alavés": "Alaves",
    "Real Betis": "Betis", "Real Valladolid": "Valladolid",
    # Bundesliga
    "Bayern Munich": "Bayern Munich", "Bayer Leverkusen": "Leverkusen",
    "Borussia Dortmund": "Dortmund", "RB Leipzig": "Leipzig",
    "Borussia Monchengladbach": "M'gladbach", "Eintracht Frankfurt": "Eintracht Frankfurt",
    "Werder Bremen": "Werder Bremen", "VfB Stuttgart": "Stuttgart",
    "1. FC Union Berlin": "Union Berlin", "1. FSV Mainz 05": "Mainz",
    "TSG Hoffenheim": "Hoffenheim", "FC Augsburg": "Augsburg",
    "VfL Wolfsburg": "Wolfsburg", "SC Freiburg": "Freiburg",
    "VfL Bochum": "Bochum", "Heidenheim": "Heidenheim",
    "St. Pauli": "St Pauli", "Holstein Kiel": "Holstein Kiel",
    # Ligue 1
    "Paris Saint-Germain": "PSG", "Olympique Lyonnais": "Lyon",
    "Olympique de Marseille": "Marseille", "Stade Rennais": "Rennes",
    "Stade de Reims": "Reims", "Stade Brestois 29": "Brest",
    "AS Saint-Etienne": "St Etienne", "Le Havre AC": "Le Havre",
    # Serie A
    "Hellas Verona": "Verona", "Inter Milan": "Inter", "AC Milan": "Milan"
}

# --- FUNKCJE POMOCNICZE ---
def normalize_name(name):
    """Usuwa polskie/obce znaki i zbędne końcówki dla lepszego dopasowania"""
    if not isinstance(name, str): return name
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
    for suffix in [" FC", " CF", " AS", " RCD", " 1.", " 05", " UD"]:
        name = name.replace(suffix, "")
    return name.strip()

# --- POBIERANIE DANYCH ---
@st.cache_data(ttl=3600)
def load_historical(league_code):
    try:
        url = f"https://www.football-data.co.uk/mmz4281/2526/{league_code}.csv"
        r = requests.get(url, timeout=10)
        df = pd.read_csv(StringIO(r.text))
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam'])
        df['total_gole'] = df['FTHG'] + df['FTAG']
        df['total_kartki'] = df['HY'] + df['AY'] + (df['HR'] + df['AR']) * 2
        df['total_rozne'] = df['HC'] + df['AC']
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=43200)
def load_schedule(filename):
    try:
        df = pd.read_csv(filename)
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.tz_localize(None)
        return df.dropna(subset=['date'])
    except: return pd.DataFrame()

# --- LOGIKA ---
st.sidebar.header("🌍 Wybór Rozgrywek")
wybrana_liga = st.sidebar.selectbox("Wybierz ligę", list(LIGI.keys()))

historical = load_historical(LIGI[wybrana_liga]["csv_code"])
schedule = load_schedule(LIGI[wybrana_liga]["file"])

@st.cache_data
def get_stats(df):
    if df.empty: return pd.DataFrame()
    teams = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel())
    res = {}
    for t in teams:
        h = df[df['HomeTeam'] == t].tail(8)
        a = df[df['AwayTeam'] == t].tail(8)
        if len(h) < 2 or len(a) < 2: continue
        res[t] = {
            "att_h": weighted_mean(h['FTHG']), "def_h": weighted_mean(h['FTAG']),
            "att_a": weighted_mean(a['FTAG']), "def_a": weighted_mean(a['FTHG']),
            "corners": weighted_mean(pd.concat([h['total_rozne'], a['total_rozne']])),
            "cards": weighted_mean(pd.concat([h['total_kartki'], a['total_kartki']]))
        }
    return pd.DataFrame(res).T

def weighted_mean(v):
    if len(v) == 0: return 0
    return np.average(v, weights=np.linspace(1, 2, len(v)))

# --- INTERFEJS ---
if not historical.empty:
    stats_df = get_stats(historical)
    
    st.title(f"Analiza: {wybrana_liga}")
    t1, t2, t3 = st.tabs(["🎯 Bet Builder", "📊 Tabela", "📈 Statystyki"])

    with t1:
        c1, c2, c3 = st.columns(3)
        l_g = c1.selectbox("Gole", [1.5, 2.5, 3.5], index=1)
        l_r = c2.selectbox("Rożne", [8.5, 9.5, 10.5], index=1)
        l_k = c3.selectbox("Kartki", [3.5, 4.5, 5.5], index=1)
        min_p = st.slider("Min. Prawdopodobieństwo", 0.0, 1.0, 0.4)

        dzis = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        if not schedule.empty:
            # Kluczowa zmiana: bierzemy mecze z najbliższej rundy, ale nie starsze niż dzisiaj
            nadchodzace = schedule[schedule['date'] >= (dzis - timedelta(days=1))]
            if not nadchodzace.empty:
                min_round = nadchodzace['round'].min()
                mecze = nadchodzace[nadchodzace['round'] == min_round]
                
                col_a, col_b = st.columns(2)
                for i, m in mecze.iterrows():
                    # Próba dopasowania nazwy
                    h_raw, a_raw = m['home_team'], m['away_team']
                    h = NAZWY_MAP.get(h_raw, normalize_name(h_raw))
                    a = NAZWY_MAP.get(a_raw, normalize_name(a_raw))

                    # Jeśli nadal nie ma w stats_df, spróbuj znaleźć po fragmencie nazwy
                    if h not in stats_df.index:
                        found = [x for x in stats_df.index if h in x or x in h]
                        if found: h = found[0]
                    if a not in stats_df.index:
                        found = [x for x in stats_df.index if a in x or x in a]
                        if found: a = found[0]

                    if h in stats_df.index and a in stats_df.index:
                        # Model Poisson
                        lam_h = (stats_df.loc[h, "att_h"] + stats_df.loc[a, "def_a"]) / 2
                        lam_a = (stats_df.loc[a, "att_a"] + stats_df.loc[h, "def_h"]) / 2
                        p_g = 1 - poisson.cdf(l_g, lam_h + lam_a)
                        p_r = 1 - poisson.cdf(l_r, (stats_df.loc[h, "corners"] + stats_df.loc[a, "corners"])/2)
                        p_k = 1 - poisson.cdf(l_k, (stats_df.loc[h, "cards"] + stats_df.loc[a, "cards"])/2)
                        p_combo = p_g * p_r * p_k
                        
                        if p_combo >= min_p:
                            with col_a if i % 2 == 0 else col_b:
                                with st.expander(f"⚽ {h} - {a} ({p_combo:.1%})"):
                                    st.write(f"Gole > {l_g}: {p_g:.1%}")
                                    st.write(f"Rożne > {l_r}: {p_r:.1%}")
                                    st.write(f"Kartki > {l_k}: {p_k:.1%}")
                    else:
                        # Pokaż błędy tylko w konsoli/debugu
                        pass
        else:
            st.warning("Brak terminarza.")

    with t3:
        st.write("Nazwy drużyn w bazie statystyk (użyj ich w słowniku, jeśli meczu brakuje):")
        st.write(list(stats_df.index))
        st.dataframe(stats_df)
