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

# --- SIDEBAR - WYBÓR LIGI ---
st.sidebar.header("🌍 Wybór Rozgrywek")
wybrana_liga = st.sidebar.selectbox("Wybierz ligę", list(LIGI.keys()))

st.title(f"Predykcje {wybrana_liga} 2025/26")
st.markdown("Model Poissona + wagi formy")

# --- FUNKCJA NORMALIZACJI NAZW ---
def normalize_name(name):
    if not isinstance(name, str): return str(name)
    # Usuwanie akcentów i ogonków (np. é -> e)
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
    # Usuwanie typowych skrótów dla lepszego dopasowania
    for suffix in [" FC", " CF", " UD", " CD", " RCD", " AS", " Stade", " 1."]:
        name = name.replace(suffix, "")
    return name.strip()

# Rozszerzone mapowanie nazw (Klucz: nazwa z Twojego CSV, Wartość: nazwa z football-data.co.uk)
NAZWY_MAP = {
    # Premier League
    "Brighton & Hove Albion": "Brighton",
    "West Ham United": "West Ham",
    "Newcastle United": "Newcastle",
    "Tottenham Hotspur": "Tottenham",
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "Wolverhampton": "Wolves",
    "Leicester City": "Leicester",
    # La Liga
    "Girona FC": "Girona",
    "Rayo Vallecano": "Vallecano",
    "Villarreal": "Villarreal",
    "FC Barcelona": "Barcelona",
    "Deportivo Alaves": "Alaves",
    "Deportivo Alaves": "Alaves",
    "Real Sociedad": "Sociedad",
    "Celta Vigo": "Celta",
    "Athletic Club": "Ath Bilbao",
    "Athletic Bilbao": "Ath Bilbao",
    "Sevilla FC": "Sevilla",
    "RCD Espanyol": "Espanyol",
    "Atletico Madrid": "Ath Madrid",
    "Atletico Madrid": "Ath Madrid",
    "Real Betis": "Betis",
    "CA Osasuna": "Osasuna",
    # Bundesliga
    "Bayern Munich": "Bayern Munich",
    "RB Leipzig": "Leipzig",
    "Bayer Leverkusen": "Leverkusen",
    "Eintracht Frankfurt": "Eintracht Frankfurt",
    "Borussia Monchengladbach": "M'gladbach",
    "Borussia Monchengladbach": "M'gladbach",
    "Werder Bremen": "Werder Bremen",
    "St. Pauli": "St Pauli",
    # Serie A
    "Hellas Verona": "Verona",
    "Inter Milan": "Inter",
    # Ligue 1
    "Paris Saint-Germain": "PSG",
    "Stade Rennais": "Rennes",
    "Stade Brestois 29": "Brest",
    "Stade de Reims": "Reims",
    "Le Havre AC": "Le Havre"
}

# --- FUNKCJE DANYCH ---
@st.cache_data(ttl=900)
def load_historical(league_code):
    try:
        url = f"https://www.football-data.co.uk/mmz4281/2526/{league_code}.csv"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam'])
        df = df.sort_values('Date')
        df['total_gole'] = df['FTHG'] + df['FTAG']
        df['total_kartki'] = df['HY'] + df['AY'] + (df['HR'] + df['AR']) * 2
        df['total_rozne'] = df['HC'] + df['AC']
        return df
    except Exception as e:
        st.error(f"Błąd pobierania danych historycznych: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def load_schedule(filename):
    try:
        df = pd.read_csv(filename)
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.tz_localize(None)
        return df.dropna(subset=['date']).sort_values('date')
    except Exception as e:
        st.error(f"Problem z plikiem terminarza {filename}: {e}")
        return pd.DataFrame()

historical = load_historical(LIGI[wybrana_liga]["csv_code"])
schedule = load_schedule(LIGI[wybrana_liga]["file"])

# --- OBLICZENIA ---
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

def oblicz_forme(df):
    if df.empty: return {}
    druzyny = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel())
    forma = {}
    for d in druzyny:
        mecze = df[(df['HomeTeam'] == d) | (df['AwayTeam'] == d)].tail(5)
        wyniki = "".join(["W" if ((m['HomeTeam']==d and m['FTHG']>m['FTAG']) or (m['AwayTeam']==d and m['FTAG']>m['FTHG']))
                          else "L" if ((m['HomeTeam']==d and m['FTHG']<m['FTAG']) or (m['AwayTeam']==d and m['FTAG']<m['FTHG']))
                          else "D" for _, m in mecze.iterrows()])
        forma[d] = wyniki
    return forma

def tabela_ligowa(df):
    if df.empty: return pd.DataFrame()
    table = {}
    for _, m in df.iterrows():
        for team, g, gs in [(m['HomeTeam'], m['FTHG'], m['FTAG']), (m['AwayTeam'], m['FTAG'], m['FTHG'])]:
            if team not in table: table[team] = {"pts":0,"gf":0,"ga":0,"played":0}
            table[team]["gf"] += g
            table[team]["ga"] += gs
            table[team]["played"] += 1
        if m['FTHG'] > m['FTAG']: table[m['HomeTeam']]["pts"] += 3
        elif m['FTHG'] < m['FTAG']: table[m['AwayTeam']]["pts"] += 3
        else:
            table[m['HomeTeam']]["pts"] += 1
            table[m['AwayTeam']]["pts"] += 1
    res = pd.DataFrame(table).T
    res["diff"] = res["gf"] - res["ga"]
    return res.sort_values(["pts","diff","gf"], ascending=False)

def koloruj(p):
    return "🟢" if p > 0.65 else "🟡" if p > 0.50 else "🔴"

# --- RENDEROWANIE ---
if not historical.empty:
    srednie_df = oblicz_wszystkie_statystyki(historical)
    forma_dict = oblicz_forme(historical)
    tabela = tabela_ligowa(historical)
   
    tab1, tab2, tab3 = st.tabs(["🎯 Bet Builder & Predykcje", "📊 Tabela i Forma", "📈 Statystyki Modelu"])
    
    with tab1:
        st.subheader("🎛️ Zbuduj własne combo")
        c1, c2, c3 = st.columns(3)
        linia_gole = c1.selectbox("Linia goli", [1.5, 2.5, 3.5], index=1)
        linia_rogi = c2.selectbox("Linia rożnych", [7.5, 8.5, 9.5, 10.5], index=1)
        linia_kartki = c3.selectbox("Linia kartek", [2.5, 3.5, 4.5, 5.5], index=1)
       
        min_prob = st.slider("Minimalne prawdopodobieństwo combo", 0.0, 1.0, 0.40, 0.05)
        
        st.subheader("📅 Predykcje – najbliższa kolejka")
        
        # Poprawka daty: szukamy meczów od wczoraj, aby złapać trwającą kolejkę
        wczoraj = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        
        if not schedule.empty:
            # Filtrujemy mecze od wczoraj wzwyż
            nadchodzace = schedule[schedule['date'] >= wczoraj]
            
            if not nadchodzace.empty:
                # Bierzemy najniższy dostępny numer rundy z tych, które jeszcze się nie odbyły/trwają
                min_round = nadchodzace['round'].min()
                mecze = schedule[schedule['round'] == min_round]
               
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Combo Builder (Kolejka {int(min_round)})**")
                    for _, mecz in mecze.iterrows():
                        # Pobieranie nazw z uwzględnieniem mapowania i normalizacji
                        h_raw = mecz['home_team']
                        a_raw = mecz['away_team']
                        h = NAZWY_MAP.get(h_raw, normalize_name(h_raw))
                        a = NAZWY_MAP.get(a_raw, normalize_name(a_raw))
                       
                        if h in srednie_df.index and a in srednie_df.index:
                            lam_h = (srednie_df.loc[h, "Gole strzelone (dom)"] + srednie_df.loc[a, "Gole stracone (wyjazd)"]) / 2
                            lam_a = (srednie_df.loc[a, "Gole strzelone (wyjazd)"] + srednie_df.loc[h, "Gole stracone (dom)"]) / 2
                            lam_r = (srednie_df.loc[h, "Różne (dom)"] + srednie_df.loc[a, "Różne (wyjazd)"]) / 2
                            lam_k = (srednie_df.loc[h, "Kartki (dom)"] + srednie_df.loc[a, "Kartki (wyjazd)"]) / 2

                            p_g = 1 - poisson.cdf(linia_gole, lam_h + lam_a)
                            p_r = 1 - poisson.cdf(linia_rogi, lam_r)
                            p_k = 1 - poisson.cdf(linia_kartki, lam_k)
                            p_combo = p_g * p_r * p_k

                            if p_combo >= min_prob:
                                with st.expander(f"{h} vs {a} ({p_combo:.1%})"):
                                    st.write(f"{koloruj(p_g)} Gole: {p_g:.1%}")
                                    st.write(f"{koloruj(p_r)} Rożne: {p_r:.1%}")
                                    st.write(f"{koloruj(p_k)} Kartki: {p_k:.1%}")
                
                with col2:
                    st.write("**BTTS Ranking**")
                    for _, mecz in mecze.iterrows():
                        h_raw = mecz['home_team']
                        a_raw = mecz['away_team']
                        h = NAZWY_MAP.get(h_raw, normalize_name(h_raw))
                        a = NAZWY_MAP.get(a_raw, normalize_name(a_raw))
                        
                        if h in srednie_df.index and a in srednie_df.index:
                            lam_h = (srednie_df.loc[h, "Gole strzelone (dom)"] + srednie_df.loc[a, "Gole stracone (wyjazd)"]) / 2
                            lam_a = (srednie_df.loc[a, "Gole strzelone (wyjazd)"] + srednie_df.loc[h, "Gole stracone (dom)"]) / 2
                            p_btts = (1 - poisson.pmf(0, lam_h)) * (1 - poisson.pmf(0, lam_a))
                            st.write(f"{koloruj(p_btts)} **{h} - {a}**: {p_btts:.1%}")
            else:
                st.info("Brak nadchodzących meczów w najbliższym terminie.")
        else:
            st.warning("Brak danych w pliku terminarza.")

    with tab2:
        st.subheader("📊 Aktualna Sytuacja")
        c_l, c_f = st.columns([2, 1])
        with c_l: st.dataframe(tabela, use_container_width=True)
        with c_f: st.dataframe(pd.DataFrame.from_dict(forma_dict, orient="index", columns=["Forma"]), use_container_width=True)

    with tab3:
        st.subheader("📊 Średnie ważone drużyn")
        st.dataframe(srednie_df.sort_index(), use_container_width=True)
        if st.button("🔄 Odśwież dane"):
            st.cache_data.clear()
            st.rerun()
