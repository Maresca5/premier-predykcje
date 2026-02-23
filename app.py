import streamlit as st
import pandas as pd
from scipy.stats import poisson
from datetime import datetime
import requests
from io import StringIO
import os

st.set_page_config(page_title="Predykcje Premier League", layout="wide")

st.title("Predykcje Premier League 2025/26")
st.markdown("Dane automatycznie z football-data.co.uk + terminarz lokalny")

SKUTECZNOSC_FILE = "skutecznosc_predykcji.csv"

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

@st.cache_data
def oblicz_srednie():
    druzyny = pd.unique(historical[['HomeTeam', 'AwayTeam']].values.ravel())
    srednie = {}
    for druzyna in druzyny:
        mecze = historical[
            (historical['HomeTeam'] == druzyna) |
            (historical['AwayTeam'] == druzyna)
        ].tail(10)

        if len(mecze) < 3:
            continue

        strzelone = mecze.apply(lambda r: r['FTHG'] if r['HomeTeam'] == druzyna else r['FTAG'], axis=1).mean()
        stracone = mecze.apply(lambda r: r['FTAG'] if r['HomeTeam'] == druzyna else r['FTHG'], axis=1).mean()
        kartki = mecze['total_kartki'].mean()
        rozne = mecze['total_rozne'].mean()

        srednie[druzyna] = {
            'total_gole': round(strzelone + stracone, 2),
            'kartki': round(kartki, 2),
            'rozne': round(rozne, 2),
        }

    return pd.DataFrame(srednie).T

srednie_df = oblicz_srednie()

# -------------------------------
# 🎛️ WYBÓR LINII (BET BUILDER)
# -------------------------------

st.subheader("🎛️ Zbuduj własne combo")

col1, col2, col3 = st.columns(3)

with col1:
    linia_gole = st.selectbox("Linia goli", [1.5, 2.5, 3.5, 4.5], index=1)
    typ_gole = st.selectbox("Typ goli", ["Over", "Under"])

with col2:
    linia_rogi = st.selectbox("Linia rożnych", [7.5, 8.5, 9.5, 10.5])
    typ_rogi = st.selectbox("Typ rożnych", ["Over", "Under"])

with col3:
    linia_kartki = st.selectbox("Linia kartek", [2.5, 3.5, 4.5, 5.5])
    typ_kartki = st.selectbox("Typ kartek", ["Over", "Under"])

# -------------------------------
# 📊 PREDYKCJE
# -------------------------------

st.subheader("Predykcje na najbliższą kolejkę")

dzisiaj = datetime.now()
nadchodzace = schedule[schedule['date'] > dzisiaj]

if nadchodzace.empty:
    st.warning("Brak nadchodzących meczów.")
else:
    min_round = nadchodzace['round'].min()
    mecze_kolejki = nadchodzace[nadchodzace['round'] == min_round]

    for _, mecz in mecze_kolejki.iterrows():
        home = mecz['home_team']
        away = mecz['away_team']

        home_map = NAZWY_MAP.get(home, home)
        away_map = NAZWY_MAP.get(away, away)

        if home_map in srednie_df.index and away_map in srednie_df.index:

            lambda_gole = (srednie_df.loc[home_map, 'total_gole'] +
                           srednie_df.loc[away_map, 'total_gole']) / 2

            lambda_rogi = (srednie_df.loc[home_map, 'rozne'] +
                           srednie_df.loc[away_map, 'rozne']) / 2

            lambda_kartki = (srednie_df.loc[home_map, 'kartki'] +
                             srednie_df.loc[away_map, 'kartki']) / 2

            # Gole
            if typ_gole == "Over":
                p_gole = 1 - poisson.cdf(linia_gole, lambda_gole)
            else:
                p_gole = poisson.cdf(linia_gole, lambda_gole)

            # Rożne
            if typ_rogi == "Over":
                p_rogi = 1 - poisson.cdf(linia_rogi, lambda_rogi)
            else:
                p_rogi = poisson.cdf(linia_rogi, lambda_rogi)

            # Kartki
            if typ_kartki == "Over":
                p_kartki = 1 - poisson.cdf(linia_kartki, lambda_kartki)
            else:
                p_kartki = poisson.cdf(linia_kartki, lambda_kartki)

            # Combo (zakładamy niezależność)
            p_combo = p_gole * p_rogi * p_kartki

            with st.expander(f"{home} vs {away}"):

                st.write(f"⚽ {typ_gole} {linia_gole}: **{p_gole*100:.1f}%**")
                st.write(f"🚩 {typ_rogi} {linia_rogi}: **{p_rogi*100:.1f}%**")
                st.write(f"🟨 {typ_kartki} {linia_kartki}: **{p_kartki*100:.1f}%**")

                st.markdown("---")
                st.markdown(f"🎯 **Prawdopodobieństwo całego combo: {p_combo*100:.2f}%**")

        else:
            st.write(f"{home} vs {away} — brak danych")

if st.button("Odśwież dane"):
    st.cache_data.clear()
    st.rerun()