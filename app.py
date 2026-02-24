import streamlit as st
import pandas as pd
from scipy.stats import poisson
from datetime import datetime
import requests
from io import StringIO
import numpy as np

# --- KONFIGURACJA ---
st.set_page_config(page_title="Premier League Pro Predictor", layout="wide")

NAZWY_MAP = {
    "Brighton & Hove Albion": "Brighton",
    "West Ham United": "West Ham",
    "Newcastle United": "Newcastle",
    "Tottenham Hotspur": "Tottenham",
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "Sheffield United": "Sheffield Utd",
    "Luton Town": "Luton"
}

# --- ŁADOWANIE DANYCH ---
@st.cache_data(ttl=3600)
def load_all_data():
    try:
        # Dane historyczne
        url = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df = df.sort_values('Date')
        
        # Dodatkowe kolumny statystyczne
        df['total_gole'] = df['FTHG'] + df['FTAG']
        df['total_kartki'] = df['HY'] + df['AY'] + (df['HR'] + df['AR']) * 2
        df['total_rozne'] = df['HC'] + df['AC']
        
        # Terminarz (załóżmy, że plik istnieje lokalnie)
        # Jeśli nie masz pliku, kod obsłuży brak nadchodzących meczów
        try:
            sch = pd.read_csv("terminarz_premier_2025.csv")
            sch['date'] = pd.to_datetime(sch['date'])
        except FileNotFoundError:
            sch = pd.DataFrame(columns=['date', 'home_team', 'away_team', 'round'])
            
        return df, sch
    except Exception as e:
        st.error(f"Błąd podczas pobierania danych: {e}")
        return pd.DataFrame(), pd.DataFrame()

historical, schedule = load_all_data()

# --- LOGIKA ANALITYCZNA ---
def weighted_mean(values):
    if len(values) == 0: return 0
    weights = np.linspace(1, 2, len(values))
    return np.average(values, weights=weights)

@st.cache_data
def calculate_team_stats(df):
    teams = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel())
    stats = {}

    for team in teams:
        home_games = df[df['HomeTeam'] == team].tail(10)
        away_games = df[df['AwayTeam'] == team].tail(10)
        
        if len(home_games) < 2 or len(away_games) < 2: continue

        stats[team] = {
            "Gole strzelone (dom)": weighted_mean(home_games['FTHG']),
            "Gole stracone (dom)": weighted_mean(home_games['FTAG']),
            "Gole strzelone (wyjazd)": weighted_mean(away_games['FTAG']),
            "Gole stracone (wyjazd)": weighted_mean(away_games['FTHG']),
            "Różne (dom)": weighted_mean(home_games['HC']), # tylko własne rożne
            "Różne (wyjazd)": weighted_mean(away_games['AC']),
            "Kartki (średnia)": weighted_mean(pd.concat([home_games['HY'], away_games['AY']]))
        }
    return pd.DataFrame(stats).T.round(2)

def generate_league_table(df):
    # Wykorzystanie wektoryzacji zamiast pętli iterrows
    home = df.groupby('HomeTeam').agg({'FTHG':'sum', 'FTAG':'sum', 'Date':'count'}).rename(columns={'Date':'P', 'FTHG':'GF', 'FTAG':'GA'})
    away = df.groupby('AwayTeam').agg({'FTAG':'sum', 'FTHG':'sum', 'Date':'count'}).rename(columns={'Date':'P', 'FTAG':'GF', 'FTHG':'GA'})
    
    table = home.add(away, fill_value=0)
    table['GD'] = table['GF'] - table['GA']
    
    # Punkty (uproszczone z agregacji wyników)
    df['h_pts'] = np.where(df['FTHG'] > df['FTAG'], 3, np.where(df['FTHG'] == df['FTAG'], 1, 0))
    df['a_pts'] = np.where(df['FTAG'] > df['FTHG'], 3, np.where(df['FTHG'] == df['FTAG'], 1, 0))
    
    pts_home = df.groupby('HomeTeam')['h_pts'].sum()
    pts_away = df.groupby('AwayTeam')['a_pts'].sum()
    table['PTS'] = pts_home.add(pts_away, fill_value=0).astype(int)
    
    return table.sort_values(['PTS', 'GD', 'GF'], ascending=False)

# --- INTERFEJS UŻYTKOWNIKA ---
st.title("⚽ Premier League Advanced Analytics")
if historical.empty:
    st.stop()

stats_df = calculate_team_stats(historical)
league_table = generate_league_table(historical)

# Sidebar - Filtry i ustawienia
with st.sidebar:
    st.header("Ustawienia modelu")
    min_prob = st.slider("Minimalne Prawdopodobieństwo", 0.3, 0.9, 0.5)
    if st.button("Wyczyść Cache i Odśwież"):
        st.cache_data.clear()
        st.rerun()

# --- MODUŁ PREDYKCJI ---
tab1, tab2, tab3 = st.tabs(["🎯 Predykcje Kolejki", "📊 Tabela i Forma", "🛠️ Bet Builder"])

with tab1:
    st.subheader("Analiza nadchodzących spotkań")
    future_matches = schedule[schedule['date'] > datetime.now()]
    
    if not future_matches.empty:
        curr_round = future_matches['round'].min()
        matches = future_matches[future_matches['round'] == curr_round]
        
        for _, match in matches.iterrows():
            h_name = NAZWY_MAP.get(match['home_team'], match['home_team'])
            a_name = NAZWY_MAP.get(match['away_team'], match['away_team'])
            
            if h_name in stats_df.index and a_name in stats_df.index:
                # Obliczanie parametrów Poissona
                l_h = (stats_df.loc[h_name, "Gole strzelone (dom)"] + stats_df.loc[a_name, "Gole stracone (wyjazd)"]) / 2
                l_a = (stats_df.loc[a_name, "Gole strzelone (wyjazd)"] + stats_df.loc[h_name, "Gole stracone (dom)"]) / 2
                
                # Prawdopodobieństwa
                p_ov25 = 1 - poisson.cdf(2, l_h + l_a)
                p_btts = (1 - poisson.pmf(0, l_h)) * (1 - poisson.pmf(0, l_a))
                
                with st.expander(f"🏟️ {h_name} vs {a_name}"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Over 2.5", f"{p_ov25:.1%}", f"Kurs min: {1/p_ov25:.2f}" if p_ov25 > 0 else None)
                    c2.metric("BTTS (Tak)", f"{p_btts:.1%}", f"Kurs min: {1/p_btts:.2f}" if p_btts > 0 else None)
                    c3.write(f"**Prognoza xG:** {l_h:.2f} - {l_a:.2f}")
    else:
        st.info("Brak nadchodzących meczów w pliku terminarza.")

with tab2:
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        st.subheader("Tabela Ligowa")
        st.dataframe(league_table, use_container_width=True)
    with col_t2:
        st.subheader("Statystyki Dom/Wyjazd")
        st.dataframe(stats_df[['Gole strzelone (dom)', 'Gole strzelone (wyjazd)']], use_container_width=True)

with tab3:
    st.subheader("Kalkulator Szans (Poisson)")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        team_h = st.selectbox("Gospodarz", stats_df.index)
    with col_b2:
        team_a = st.selectbox("Gość", stats_df.index)
    
    # Prosta symulacja macierzy wyników
    l_h = (stats_df.loc[team_h, "Gole strzelone (dom)"] + stats_df.loc[team_a, "Gole stracone (wyjazd)"]) / 2
    l_a = (stats_df.loc[team_a, "Gole strzelone (wyjazd)"] + stats_df.loc[team_h, "Gole stracone (dom)"]) / 2
    
    # Wyświetlenie rozkładu goli
    st.write(f"Przewidywany wynik: **{l_h:.1f} - {l_a:.1f}**")
