import streamlit as st
import pandas as pd
from scipy.stats import poisson
from datetime import datetime
import requests
from io import StringIO
import numpy as np

st.set_page_config(page_title="Predykcje Premier League", layout="wide")

st.title("Predykcje Premier League 2025/26")
st.markdown("Model Poissona + home/away + wagi formy")

NAZWY_MAP = {
    "Brighton & Hove Albion": "Brighton",
    "West Ham United": "West Ham",
    "Newcastle United": "Newcastle",
    "Tottenham Hotspur": "Tottenham",
    "Leeds United": "Leeds",
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton": "Wolves",
}

# ----------------------------------
# DANE
# ----------------------------------

@st.cache_data(ttl=3600)
def load_historical():
    url = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
    r = requests.get(url)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values('Date')
    df['total_gole'] = df['FTHG'] + df['FTAG']
    df['total_kartki'] = df['HY'] + df['AY'] + df['HR']*2 + df['AR']*2
    df['total_rozne'] = df['HC'] + df['AC']
    return df

@st.cache_data(ttl=86400)
def load_schedule():
    df = pd.read_csv("terminarz_premier_2025.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date')

historical = load_historical()
schedule = load_schedule()

# ----------------------------------
# ŚREDNIE Z WAGAMI + HOME/AWAY
# ----------------------------------

def weighted_mean(values):
    weights = np.linspace(1, 2, len(values))  # rosnąca waga
    return np.average(values, weights=weights)

@st.cache_data
def oblicz_srednie():
    druzyny = pd.unique(historical[['HomeTeam', 'AwayTeam']].values.ravel())
    dane = {}

    for d in druzyny:
        home = historical[historical['HomeTeam'] == d].tail(10)
        away = historical[historical['AwayTeam'] == d].tail(10)

        if len(home) < 3 or len(away) < 3:
            continue

        dane[d] = {
            "gole_home_scored": weighted_mean(home['FTHG']),
            "gole_home_conceded": weighted_mean(home['FTAG']),
            "gole_away_scored": weighted_mean(away['FTAG']),
            "gole_away_conceded": weighted_mean(away['FTHG']),
            "rozne_home": weighted_mean(home['total_rozne']),
            "rozne_away": weighted_mean(away['total_rozne']),
            "kartki_home": weighted_mean(home['total_kartki']),
            "kartki_away": weighted_mean(away['total_kartki']),
        }

    return pd.DataFrame(dane).T.round(2)

srednie_df = oblicz_srednie()

# ----------------------------------
# TABELA ŚREDNICH
# ----------------------------------

st.subheader("📊 Średnie (ostatnie 10 meczów z wagą formy)")
st.dataframe(srednie_df)

# ----------------------------------
# BET BUILDER
# ----------------------------------

st.subheader("🎛️ Zbuduj własne combo")

col1, col2, col3 = st.columns(3)

with col1:
    linia_gole = st.selectbox("Linia goli", [1.5, 2.5, 3.5, 4.5])
    typ_gole = st.selectbox("Typ goli", ["Over", "Under"])

with col2:
    linia_rogi = st.selectbox("Linia rożnych", 
                              [5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5])
    typ_rogi = st.selectbox("Typ rożnych", ["Over", "Under"])

with col3:
    linia_kartki = st.selectbox("Linia kartek", [2.5,3.5,4.5,5.5,6.5])
    typ_kartki = st.selectbox("Typ kartek", ["Over", "Under"])

# ----------------------------------
# PREDYKCJE
# ----------------------------------

st.subheader("Predykcje na najbliższą kolejkę")

dzisiaj = datetime.now()
nadchodzace = schedule[schedule['date'] > dzisiaj]

if not nadchodzace.empty:
    min_round = nadchodzace['round'].min()
    mecze = nadchodzace[nadchodzace['round'] == min_round]

    for _, mecz in mecze.iterrows():

        home = NAZWY_MAP.get(mecz['home_team'], mecz['home_team'])
        away = NAZWY_MAP.get(mecz['away_team'], mecz['away_team'])

        if home in srednie_df.index and away in srednie_df.index:

            # GOLE (home attack + away defence)
            lambda_gole = (
                srednie_df.loc[home, "gole_home_scored"] +
                srednie_df.loc[away, "gole_away_conceded"]
            ) / 2

            # ROŻNE
            lambda_rogi = (
                srednie_df.loc[home, "rozne_home"] +
                srednie_df.loc[away, "rozne_away"]
            ) / 2

            # KARTKI
            lambda_kartki = (
                srednie_df.loc[home, "kartki_home"] +
                srednie_df.loc[away, "kartki_away"]
            ) / 2

            # Liczenie prawdopodobieństw
            def licz_prob(typ, linia, lam):
                if typ == "Over":
                    return 1 - poisson.cdf(linia, lam)
                else:
                    return poisson.cdf(linia, lam)

            p_gole = licz_prob(typ_gole, linia_gole, lambda_gole)
            p_rogi = licz_prob(typ_rogi, linia_rogi, lambda_rogi)
            p_kartki = licz_prob(typ_kartki, linia_kartki, lambda_kartki)

            p_combo = p_gole * p_rogi * p_kartki

            with st.expander(f"{home} vs {away}"):

                st.write(f"⚽ {typ_gole} {linia_gole}: **{p_gole*100:.1f}%**")
                st.write(f"🚩 {typ_rogi} {linia_rogi}: **{p_rogi*100:.1f}%**")
                st.write(f"🟨 {typ_kartki} {linia_kartki}: **{p_kartki*100:.1f}%**")

                st.markdown("---")
                st.markdown(f"🎯 **Prawdopodobieństwo całego combo: {p_combo*100:.2f}%**")

else:
    st.warning("Brak nadchodzących meczów.")

if st.button("Odśwież dane"):
    st.cache_data.clear()
    st.rerun()