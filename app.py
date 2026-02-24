import streamlit as st
import pandas as pd
from scipy.stats import poisson
from datetime import datetime
import requests
from io import StringIO
import numpy as np

st.set_page_config(page_title="Predykcje Premier League", layout="wide")

st.title("Predykcje Premier League 2025/26")
st.markdown("Model Poissona + home/away + wagi formy + analiza trendów")

# ---------------------------------------------------
# MAPOWANIE NAZW
# ---------------------------------------------------

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

# ---------------------------------------------------
# DANE
# ---------------------------------------------------

@st.cache_data(ttl=900)
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

# ---------------------------------------------------
# WAGA FORMY
# ---------------------------------------------------

def weighted_mean(values):
    weights = np.linspace(1, 2, len(values))
    return np.average(values, weights=weights)

# ---------------------------------------------------
# ŚREDNIE DRUŻYN
# ---------------------------------------------------

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
            "Gole strzelone (dom)": weighted_mean(home['FTHG']),
            "Gole stracone (dom)": weighted_mean(home['FTAG']),
            "Gole strzelone (wyjazd)": weighted_mean(away['FTAG']),
            "Gole stracone (wyjazd)": weighted_mean(away['FTHG']),
        }

    return pd.DataFrame(dane).T.round(2)

srednie_df = oblicz_srednie()

# ---------------------------------------------------
# FORMA OSTATNIE 5
# ---------------------------------------------------

def oblicz_forme():
    druzyny = pd.unique(historical[['HomeTeam', 'AwayTeam']].values.ravel())
    forma = {}

    for d in druzyny:
        mecze = historical[
            (historical['HomeTeam'] == d) |
            (historical['AwayTeam'] == d)
        ].tail(5)

        wyniki = []
        for _, m in mecze.iterrows():
            if m['HomeTeam'] == d:
                if m['FTHG'] > m['FTAG']:
                    wyniki.append("W")
                elif m['FTHG'] < m['FTAG']:
                    wyniki.append("L")
                else:
                    wyniki.append("D")
            else:
                if m['FTAG'] > m['FTHG']:
                    wyniki.append("W")
                elif m['FTAG'] < m['FTHG']:
                    wyniki.append("L")
                else:
                    wyniki.append("D")

        forma[d] = "".join(wyniki)

    return forma

forma_dict = oblicz_forme()

# ---------------------------------------------------
# TABELA LIGOWA
# ---------------------------------------------------

def tabela_ligowa():
    table = {}

    for _, m in historical.iterrows():
        home, away = m['HomeTeam'], m['AwayTeam']
        hg, ag = m['FTHG'], m['FTAG']

        for team in [home, away]:
            if team not in table:
                table[team] = {"pts":0,"gf":0,"ga":0,"played":0}

        table[home]["gf"] += hg
        table[home]["ga"] += ag
        table[home]["played"] += 1

        table[away]["gf"] += ag
        table[away]["ga"] += hg
        table[away]["played"] += 1

        if hg > ag:
            table[home]["pts"] += 3
        elif hg < ag:
            table[away]["pts"] += 3
        else:
            table[home]["pts"] += 1
            table[away]["pts"] += 1

    df = pd.DataFrame(table).T
    df["diff"] = df["gf"] - df["ga"]
    return df.sort_values(["pts","diff","gf"], ascending=False)

# ---------------------------------------------------
# HIGH PROBABILITY – MODEL
# ---------------------------------------------------

st.markdown("---")
st.subheader("🔥 High Probability – model Poissona")

dzisiaj = datetime.now()
nadchodzace = schedule[schedule['date'] > dzisiaj]

if not nadchodzace.empty:

    min_round = nadchodzace['round'].min()
    mecze = nadchodzace[nadchodzace['round'] == min_round]

    ranking = []

    for _, mecz in mecze.iterrows():

        home = NAZWY_MAP.get(mecz['home_team'], mecz['home_team'])
        away = NAZWY_MAP.get(mecz['away_team'], mecz['away_team'])

        if home in srednie_df.index and away in srednie_df.index:

            lambda_home = (
                srednie_df.loc[home,"Gole strzelone (dom)"] +
                srednie_df.loc[away,"Gole stracone (wyjazd)"]
            ) / 2

            lambda_away = (
                srednie_df.loc[away,"Gole strzelone (wyjazd)"] +
                srednie_df.loc[home,"Gole stracone (dom)"]
            ) / 2

            p_home_0 = poisson.pmf(0, lambda_home)
            p_away_0 = poisson.pmf(0, lambda_away)

            p_btts = 1 - p_home_0 - p_away_0 + (p_home_0 * p_away_0)

            p_over25 = 1 - poisson.cdf(2, lambda_home + lambda_away)

            ranking.append({
                "Mecz": f"{home} vs {away}",
                "BTTS %": round(p_btts*100,1),
                "Over 2.5 %": round(p_over25*100,1)
            })

    ranking_df = pd.DataFrame(ranking).sort_values(
        "BTTS %", ascending=False
    )

    st.dataframe(ranking_df, use_container_width=True)

# ---------------------------------------------------
# TABELA LIGOWA
# ---------------------------------------------------

st.markdown("---")
st.subheader("📊 Tabela ligowa")

st.dataframe(
    tabela_ligowa(),
    use_container_width=True
)

# ---------------------------------------------------
# FORMA
# ---------------------------------------------------

st.markdown("---")
st.subheader("📈 Forma – ostatnie 5 meczów")

forma_df = pd.DataFrame.from_dict(
    forma_dict, orient="index", columns=["Forma"]
)

st.dataframe(forma_df.sort_index(), use_container_width=True)

# ---------------------------------------------------
# ŚREDNIE
# ---------------------------------------------------

st.markdown("---")
st.subheader("📊 Średnie drużyn (waga formy)")

st.dataframe(
    srednie_df.sort_index(),
    use_container_width=True
)

# ---------------------------------------------------
# ODŚWIEŻANIE
# ---------------------------------------------------

if st.button("Odśwież dane"):
    st.cache_data.clear()
    st.rerun()