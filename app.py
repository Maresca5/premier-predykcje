import streamlit as st
import pandas as pd
from scipy.stats import poisson
from datetime import datetime
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

st.sidebar.header("🌍 Wybór Rozgrywek")
wybrana_liga = st.sidebar.selectbox("Wybierz ligę", list(LIGI.keys()))

st.title(f"Predykcje {wybrana_liga} 2025/26")
st.markdown("Model Poissona + wagi formy + Relative Strength")

# ---------------------------
# NORMALIZACJA NAZW
# ---------------------------
def normalize_name(name):
    if not isinstance(name, str): return str(name)
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
    for suffix in [" FC", " CF", " UD", " CD", " RCD", " AS", " Stade", " 1."]:
        name = name.replace(suffix, "")
    return name.strip()

# (Twoje NAZWY_MAP zostaje bez zmian)
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
"Metz": "Metz"}  # ← ZOSTAW SWOJE MAPOWANIE BEZ ZMIAN

def map_nazwa(nazwa_z_csv):
    if not isinstance(nazwa_z_csv, str):
        return str(nazwa_z_csv)
    if nazwa_z_csv in NAZWY_MAP:
        return NAZWY_MAP[nazwa_z_csv]
    return nazwa_z_csv

# ---------------------------
# DANE
# ---------------------------
@st.cache_data(ttl=900)
def load_historical(league_code):
    url = f"https://www.football-data.co.uk/mmz4281/2526/{league_code}.csv"
    r = requests.get(url)
    df = pd.read_csv(StringIO(r.text))
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    df['total_gole'] = df['FTHG'] + df['FTAG']
    return df

@st.cache_data(ttl=86400)
def load_schedule(filename):
    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
    if 'round' not in df.columns:
        df = df.sort_values('date')
        unique_dates = df['date'].dt.date.unique()
        date_to_round = {date: i+1 for i, date in enumerate(unique_dates)}
        df['round'] = df['date'].dt.date.map(date_to_round)
    return df

historical = load_historical(LIGI[wybrana_liga]["csv_code"])
schedule = load_schedule(LIGI[wybrana_liga]["file"])

# ---------------------------
# NOWY MODEL RELATIVE STRENGTH
# ---------------------------
def model_relative_strength(df):

    league_avg_home = df["FTHG"].mean()
    league_avg_away = df["FTAG"].mean()

    teams = pd.unique(df[['HomeTeam','AwayTeam']].values.ravel())
    data = {}

    for team in teams:
        home = df[df["HomeTeam"] == team]
        away = df[df["AwayTeam"] == team]

        if len(home) < 5 or len(away) < 5:
            continue

        home_scored = home["FTHG"].mean()
        home_conceded = home["FTAG"].mean()
        away_scored = away["FTAG"].mean()
        away_conceded = away["FTHG"].mean()

        data[team] = {
            "att_home": home_scored / league_avg_home,
            "def_home": home_conceded / league_avg_away,
            "att_away": away_scored / league_avg_away,
            "def_away": away_conceded / league_avg_home
        }

    return pd.DataFrame(data).T, league_avg_home, league_avg_away

# ---------------------------
# UI
# ---------------------------
if not historical.empty:

    strength_df, league_avg_home, league_avg_away = model_relative_strength(historical)

    tab1, tab2 = st.tabs(["🎯 Predykcje", "📊 Power Ranking"])

    with tab1:

        st.subheader("🔮 Predykcja Wyników (Model Zaawansowany)")

        dzisiaj = datetime.now().date()
        przyszle = schedule[schedule['date'].dt.date >= dzisiaj]

        if not przyszle.empty:

            kolejka = przyszle["round"].min()
            mecze = schedule[schedule["round"] == kolejka]

            for _, mecz in mecze.iterrows():

                h = map_nazwa(mecz["home_team"])
                a = map_nazwa(mecz["away_team"])

                if h in strength_df.index and a in strength_df.index:

                    lam_h = (
                        strength_df.loc[h,"att_home"] *
                        strength_df.loc[a,"def_away"] *
                        league_avg_home
                    )

                    lam_a = (
                        strength_df.loc[a,"att_away"] *
                        strength_df.loc[h,"def_home"] *
                        league_avg_away
                    )

                    # 1X2
                    p_home = 0
                    p_draw = 0
                    p_away = 0

                    for i in range(6):
                        for j in range(6):
                            p = poisson.pmf(i,lam_h)*poisson.pmf(j,lam_a)
                            if i>j: p_home+=p
                            elif i==j: p_draw+=p
                            else: p_away+=p

                    # BTTS
                    p_btts = (1-poisson.pmf(0,lam_h))*(1-poisson.pmf(0,lam_a))

                    # Fair odds
                    fair_home = 1/p_home if p_home>0 else 0
                    fair_draw = 1/p_draw if p_draw>0 else 0
                    fair_away = 1/p_away if p_away>0 else 0

                    with st.expander(f"{h} vs {a}"):

                        col1,col2 = st.columns(2)

                        with col1:
                            st.write("### 1X2")
                            st.write(f"Home: {p_home:.1%} (Fair: {fair_home:.2f})")
                            st.write(f"Draw: {p_draw:.1%} (Fair: {fair_draw:.2f})")
                            st.write(f"Away: {p_away:.1%} (Fair: {fair_away:.2f})")

                        with col2:
                            st.write("### Inne")
                            st.write(f"BTTS: {p_btts:.1%}")
                            st.write(f"Śr. gole H: {lam_h:.2f}")
                            st.write(f"Śr. gole A: {lam_a:.2f}")

                        # Top 5 wyników
                        wyniki = []
                        for i in range(5):
                            for j in range(5):
                                p = poisson.pmf(i,lam_h)*poisson.pmf(j,lam_a)
                                wyniki.append((f"{i}-{j}",p))

                        top5 = sorted(wyniki,key=lambda x:x[1],reverse=True)[:5]

                        st.write("### Top 5 wyników")
                        for w,p in top5:
                            st.write(f"{w} → {p:.1%}")

    with tab2:

        st.subheader("📊 Power Ranking (Modelowy)")

        power = strength_df.copy()
        power["Power Index"] = (
            power["att_home"] + power["att_away"]
        ) - (
            power["def_home"] + power["def_away"]
        )

        power = power.sort_values("Power Index",ascending=False)

        st.dataframe(power.round(3), use_container_width=True)

else:
    st.error("Brak danych historycznych.")