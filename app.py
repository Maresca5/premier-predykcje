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

st.set_page_config(page_title="Predykcje Piłkarskie Top 5", layout="wide")

# --- MAPOWANIE I NORMALIZACJA (Twoje dotychczasowe funkcje) ---
def normalize_name(name):
    if not isinstance(name, str): return str(name)
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
    for suffix in [" FC", " CF", " UD", " CD", " RCD", " AS", " Stade", " 1."]:
        name = name.replace(suffix, "")
    return name.strip()

# (Słownik NAZWY_MAP skrócony dla czytelności kodu, zachowaj swój pełny w swojej wersji)
NAZWY_MAP = {
# ===== PREMIER LEAGUE =====
"Brighton & Hove Albion": "Brighton",
"Brighton": "Brighton",
"West Ham United": "West Ham",
"West Ham": "West Ham",
"Newcastle United": "Newcastle",
"Newcastle": "Newcastle",
"Tottenham Hotspur": "Tottenham",
"Tottenham": "Tottenham",
"Leeds United": "Leeds",
"Leeds": "Leeds",
"Manchester United": "Man United",
"Man United": "Man United",
"Manchester City": "Man City",
"Man City": "Man City",
"Nottingham Forest": "Nott'm Forest",
"Nott'm Forest": "Nott'm Forest",
"Wolverhampton Wanderers": "Wolves",
"Wolverhampton": "Wolves",
"Wolves": "Wolves",
"Leicester City": "Leicester",
"Leicester": "Leicester",
"Southampton": "Southampton",
"Crystal Palace": "Crystal Palace",
"Everton": "Everton",
"Aston Villa": "Aston Villa",
"Fulham": "Fulham",
"Brentford": "Brentford",
"Burnley": "Burnley",
"Bournemouth": "Bournemouth",
"Sheffield United": "Sheffield Utd",
"Sheffield Utd": "Sheffield Utd",

# ===== LA LIGA =====  
"Girona FC": "Girona",  
"Girona": "Girona",  
"Rayo Vallecano": "Vallecano",  
"Vallecano": "Rayo Vallecano",  
"Villarreal": "Villarreal",  
"Real Oviedo": "Oviedo",  
"Oviedo": "Real Oviedo",  
"Mallorca": "Mallorca",  
"RCD Mallorca": "Mallorca",  
"Barcelona": "Barcelona",  
"FC Barcelona": "Barcelona",  
"Deportivo Alavés": "Alaves",  
"Deportivo Alaves": "Alaves",  
"Alaves": "Alaves",  
"Levante UD": "Levante",  
"Levante": "Levante",  
"Valencia": "Valencia",  
"Valencia CF": "Valencia",  
"Real Sociedad": "Sociedad",  
"Sociedad": "Real Sociedad",  
"Celta Vigo": "Celta",  
"Celta": "Celta",  
"Getafe": "Getafe",  
"Getafe CF": "Getafe",  
"Athletic Club": "Ath Bilbao",  
"Athletic Bilbao": "Ath Bilbao",  
"Ath Bilbao": "Ath Bilbao",  
"Sevilla": "Sevilla",  
"Sevilla FC": "Sevilla",  
"Espanyol": "Espanol",  
"Esponol": "Espanyol",  
"RCD Espanyol": "Espanol",  
"Atlético Madrid": "Ath Madrid",  
"Ath Madrid": "Atlético Madrid",  
"Atletico Madrid": "Ath Madrid",  
"Ath Madrid": "Ath Madrid",  
"Elche": "Elche",  
"Elche CF": "Elche",  
"Real Betis": "Betis",  
"Betis": "Betis",  
"Real Madrid": "Real Madrid",  
"Osasuna": "Osasuna",  
"CA Osasuna": "Osasuna",  
"Real Valladolid": "Valladolid",  
"Valladolid": "Valladolid",  
"UD Las Palmas": "Las Palmas",  
"Las Palmas": "Las Palmas",  
"Cadiz": "Cadiz",  
"Cadiz CF": "Cadiz",  
"Granada": "Granada",  
"Granada CF": "Granada",  
"Almeria": "Almeria",  
"UD Almeria": "Almeria",  
  
# ===== BUNDESLIGA =====  
"Bayern Munich": "Bayern Munich",  
"Bayern": "Bayern Munich",  
"FC Bayern München": "Bayern Munich",  
"Borussia Dortmund": "Dortmund",  
"Dortmund": "Dortmund",  
"RB Leipzig": "RB Leipzig",  
"Leipzig": "RB Leipzig",  
"Bayer 04 Leverkusen": "Leverkusen",  
"Leverkusen": "Leverkusen",  
"Eintracht Frankfurt": "Ein Frankfurt",  
"Eintracht": "Eintracht",  
"VfB Stuttgart": "Stuttgart",  
"Stuttgart": "Stuttgart",  
"VfL Wolfsburg": "Wolfsburg",  
"Wolfsburg": "Wolfsburg",  
"Borussia Mönchengladbach": "M'gladbach",  
"Borussia Monchengladbach": "M'gladbach",  
"Borussia M'gladbach": "M'gladbach",  
"M'gladbach": "M'gladbach",  
"1. FC Union Berlin": "Union Berlin",  
"Union Berlin": "Union Berlin",  
"SC Freiburg": "Freiburg",  
"Freiburg": "Freiburg",  
"1. FC Köln": "FC Koln",  
"FC Köln": "Koln",  
"Koln": "FC Köln",  
"Köln": "Köln",  
"1. FSV Mainz 05": "Mainz",  
"Mainz": "FSV Mainz 05",  
"Mainz": "Mainz",  
"TSG Hoffenheim": "Hoffenheim",  
"Hoffenheim": "Hoffenheim",  
"Werder Bremen": "Werder Bremen",  
"SV Werder Bremen": "Werder Bremen",  
"VfL Bochum": "Bochum",  
"Bochum": "Bochum",  
"FC Augsburg": "Augsburg",  
"Augsburg": "Augsburg",  
"FC St. Pauli": "St Pauli",  
"St. Pauli": "St Pauli",  
"St Pauli": "St Pauli",  
"1. FC Heidenheim": "Heidenheim",  
"Heidenheim": "Heidenheim",  
"Hamburger SV": "Hamburg",  
"Hamburg": "Hamburg",  
  
# ===== SERIE A =====  
"AC Milan": "AC Milan",  
"Milan": "AC Milan",  
"Milan": "Milan",  
"Inter": "Inter",  
"Internazionale": "Inter",  
"Juventus": "Juventus",  
"AS Roma": "Roma",  
"Roma": "Roma",  
"Lazio": "Lazio",  
"Napoli": "Napoli",  
"Atalanta": "Atalanta",  
"Fiorentina": "Fiorentina",  
"Torino": "Torino",  
"Bologna": "Bologna",  
"Udinese": "Udinese",  
"Sassuolo": "Sassuolo",  
"Empoli": "Empoli",  
"Salernitana": "Salernitana",  
"Hellas Verona": "Verona",  
"Verona": "Verona",  
"Lecce": "Lecce",  
"Cagliari": "Cagliari",  
"Genoa": "Genoa",  
"Monza": "Monza",  
"Frosinone": "Frosinone",  
  
# ===== LIGUE 1 =====  
"Paris Saint-Germain": "Paris SG",  
"PSG": "Paris SG",  
"Paris SG": "Paris SG",  
"Olympique de Marseille": "Marseille",  
"Marseille": "Marseille",  
"AS Monaco": "Monaco",  
"Monaco": "Monaco",  
"Olympique Lyonnais": "Lyon",  
"Lyon": "Lyon",  
"LOSC Lille": "Lille",  
"Lille": "Lille",  
"Stade Rennais": "Rennes",  
"Rennes": "Rennes",  
"OGC Nice": "Nice",  
"Nice": "Nice",  
"RC Lens": "Lens",  
"Lens": "Lens",  
"RC Strasbourg": "Strasbourg",  
"Strasbourg": "Strasbourg",  
"FC Nantes": "Nantes",  
"Nantes": "Nantes",  
"Montpellier HSC": "Montpellier",  
"Montpellier": "Montpellier",  
"FC Toulouse": "Toulouse",  
"Toulouse": "Toulouse",  
"Stade de Reims": "Reims",  
"Reims": "Reims",  
"Stade Brestois 29": "Brest",  
"Stade Brestois": "Brest",  
"Brest": "Brest",  
"Clermont Foot": "Clermont",  
"Clermont": "Clermont",  
"AJ Auxerre": "Auxerre",  
"Auxerre": "Auxerre",  
"Le Havre AC": "Le Havre",  
"Le Havre": "Le Havre",  
"FC Metz": "Metz",  
"Metz": "Metz"
}

def map_nazwa(nazwa_z_csv):
    if not isinstance(nazwa_z_csv, str): return str(nazwa_z_csv)
    if nazwa_z_csv in NAZWY_MAP: return NAZWY_MAP[nazwa_z_csv]
    znormalizowana = normalize_name(nazwa_z_csv)
    if znormalizowana in NAZWY_MAP: return NAZWY_MAP[znormalizowana]
    return nazwa_z_csv

# --- NOWA FUNKCJA: FAIR ODDS I WYNIKI ---
def get_match_predictions(lam_h, lam_a):
    max_g = 6
    # Macierz prawdopodobieństw wyników
    p_matrix = np.outer(
        [poisson.pmf(i, lam_h) for i in range(max_g)],
        [poisson.pmf(j, lam_a) for j in range(max_g)]
    )
    
    # Najbardziej prawdopodobny wynik
    res_idx = np.unravel_index(p_matrix.argmax(), p_matrix.shape)
    top_score = f"{res_idx[0]}:{res_idx[1]}"
    top_score_p = p_matrix.max()
    
    # Prawdopodobieństwa 1X2
    p_home = np.sum(np.tril(p_matrix, -1))
    p_draw = np.sum(np.diag(p_matrix))
    p_away = np.sum(np.triu(p_matrix, 1))
    
    # Fair Odds (1/p) - zabezpieczenie przed dzieleniem przez 0
    fair_h = 1/p_home if p_home > 0 else 100
    fair_d = 1/p_draw if p_draw > 0 else 100
    fair_a = 1/p_away if p_away > 0 else 100
    
    return {
        "score": top_score, "score_p": top_score_p,
        "1": p_home, "X": p_draw, "2": p_away,
        "odds_1": fair_h, "odds_X": fair_d, "odds_2": fair_a
    }

# --- FUNKCJE DANYCH (Twoje dotychczasowe) ---
@st.cache_data(ttl=900)
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

@st.cache_data(ttl=86400)
def load_schedule(filename):
    try:
        df = pd.read_csv(filename)
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.tz_localize(None)
        if 'round' not in df.columns:
            df = df.sort_values('date')
            df['round'] = (np.arange(len(df)) // 10) + 1 # Uproszczenie
        return df.dropna(subset=['date'])
    except: return pd.DataFrame()

# --- OBLICZENIA STATYSTYK ---
def weighted_mean(values):
    if len(values) == 0: return 0
    weights = np.linspace(1, 2, len(values))
    return np.average(values, weights=weights)

@st.cache_data
def oblicz_wszystkie_statystyki(df):
    if df.empty: return pd.DataFrame()
    druzyny = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel())
    dane = {}
    for d in druzyny:
        home = df[df['HomeTeam'] == d].tail(10)
        away = df[df['AwayTeam'] == d].tail(10)
        if len(home) < 2 or len(away) < 2: continue
        dane[d] = {
            "Gole strzelone (dom)": weighted_mean(home['FTHG']),
            "Gole stracone (dom)": weighted_mean(home['FTAG']),
            "Gole strzelone (wyjazd)": weighted_mean(away['FTAG']),
            "Gole stracone (wyjazd)": weighted_mean(away['FTHG']),
            "Różne (dom)": weighted_mean(home['total_rozne']),
            "Różne (wyjazd)": weighted_mean(away['total_rozne']),
            "Kartki (dom)": weighted_mean(home['total_kartki']),
            "Kartki (wyjazd)": weighted_mean(away['total_kartki']),
        }
    return pd.DataFrame(dane).T.round(2)

# --- GŁÓWNA LOGIKA RENDEROWANIA ---
st.sidebar.header("🌍 Wybór Rozgrywek")
wybrana_liga = st.sidebar.selectbox("Wybierz ligę", list(LIGI.keys()))

historical = load_historical(LIGI[wybrana_liga]["csv_code"])
schedule = load_schedule(LIGI[wybrana_liga]["file"])

if not historical.empty:
    srednie_df = oblicz_wszystkie_statystyki(historical)
    
    # ZAKŁADKI
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Bet Builder", 
        "📊 Analiza Wyników & Value", 
        "📈 Tabela i Forma", 
        "⚙️ Statystyki Modelu"
    ])

    # Pobranie meczów najbliższej kolejki
    dzisiaj = datetime.now().date()
    przyszle = schedule[schedule['date'].dt.date >= dzisiaj]
    if not przyszle.empty:
        najblizsza_kolejka = przyszle['round'].min()
        mecze = schedule[schedule['round'] == najblizsza_kolejka]
    else:
        mecze = pd.DataFrame()

    with tab1:
        st.subheader("🎛️ Combo Builder")
        # (Tutaj pozostaje Twój kod z Selectboxami i pętlą dla Combo)
        st.info("Sekcja Combo Builder działa bez zmian.")

    with tab2:
        st.subheader("🔮 Przewidywane Wyniki i Fair Odds (1X2)")
        st.caption("Kursy sprawiedliwe (Fair Odds) nie zawierają marży bukmachera.")
        
        if not mecze.empty:
            for _, m in mecze.iterrows():
                h, a = map_nazwa(m['home_team']), map_nazwa(m['away_team'])
                if h in srednie_df.index and a in srednie_df.index:
                    lam_h = (srednie_df.loc[h, "Gole strzelone (dom)"] + srednie_df.loc[a, "Gole stracone (wyjazd)"]) / 2
                    lam_a = (srednie_df.loc[a, "Gole strzelone (wyjazd)"] + srednie_df.loc[h, "Gole stracone (dom)"]) / 2
                    
                    pred = get_match_predictions(lam_h, lam_a)
                    
                    with st.container():
                        col_m, col_s, col_o = st.columns([2, 1, 2])
                        with col_m:
                            st.markdown(f"**{h} vs {a}**")
                            st.caption(f"Średnie gole: {lam_h:.1f} - {lam_a:.1f}")
                        with col_s:
                            st.metric("Top Wynik", pred['score'], f"{pred['score_p']:.1%}")
                        with col_o:
                            # Wyświetlanie Fair Odds
                            st.write(f"Fair Odds: **1:** {pred['odds_1']:.2f} | **X:** {pred['odds_X']:.2f} | **2:** {pred['odds_2']:.2f}")
                        
                        # AI ANALYST PLACEHOLDER
                        with st.expander("🤖 Analiza kontekstowa AI"):
                            st.write(f"*Analiza dla meczu {h}-{a}:*")
                            st.write(f"Statystycznie najbardziej prawdopodobny wynik to {pred['score']}. "
                                     f"Szansa na zwycięstwo gospodarzy ({h}) wynosi {pred['1']:.1%}. "
                                     f"Jeśli u bukmachera kurs na {h} jest wyższy niż {pred['odds_1']:.2f}, warto rozważyć zakład.")
                            st.button("Pobierz nastroje i kontuzje (Live AI)", key=f"ai_{h}")
                        st.divider()

    with tab3:
        # (Tutaj Twój kod tabeli i formy)
        st.write("Tabela i Forma")

    with tab4:
        st.dataframe(srednie_df)
