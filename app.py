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

# ----------------------------------
# ŚREDNIE Z WAGAMI
# ----------------------------------

def weighted_mean(values):
    weights = np.linspace(1, 2, len(values))
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

srednie_df = oblicz_srednie()

# ----------------------------------
# FUNKCJA KOLORUJĄCA
# ----------------------------------

def koloruj(p):
    if p > 0.65:
        return "🟢"
    elif p > 0.50:
        return "🟡"
    else:
        return "🔴"

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

min_prob = st.slider("Minimalne prawdopodobieństwo combo",
                     0.0, 1.0, 0.40, 0.05)

# ----------------------------------
# PREDYKCJE COMBO
# ----------------------------------

st.subheader("📅 Predykcje combo – najbliższa kolejka")

dzisiaj = datetime.now()
nadchodzace = schedule[schedule['date'] > dzisiaj]

if not nadchodzace.empty:

    min_round = nadchodzace['round'].min()
    mecze = nadchodzace[nadchodzace['round'] == min_round]

    lista_meczy = []

    for _, mecz in mecze.iterrows():

        home = NAZWY_MAP.get(mecz['home_team'], mecz['home_team'])
        away = NAZWY_MAP.get(mecz['away_team'], mecz['away_team'])

        if home in srednie_df.index and away in srednie_df.index:

            lambda_home = (
                srednie_df.loc[home, "Gole strzelone (dom)"] +
                srednie_df.loc[away, "Gole stracone (wyjazd)"]
            ) / 2

            lambda_away = (
                srednie_df.loc[away, "Gole strzelone (wyjazd)"] +
                srednie_df.loc[home, "Gole stracone (dom)"]
            ) / 2

            lambda_gole = lambda_home + lambda_away

            lambda_rogi = (
                srednie_df.loc[home, "Różne (dom)"] +
                srednie_df.loc[away, "Różne (wyjazd)"]
            ) / 2

            lambda_kartki = (
                srednie_df.loc[home, "Kartki (dom)"] +
                srednie_df.loc[away, "Kartki (wyjazd)"]
            ) / 2

            def licz_prob(typ, linia, lam):
                if typ == "Over":
                    return 1 - poisson.cdf(linia, lam)
                else:
                    return poisson.cdf(linia, lam)

            p_gole = licz_prob(typ_gole, linia_gole, lambda_gole)
            p_rogi = licz_prob(typ_rogi, linia_rogi, lambda_rogi)
            p_kartki = licz_prob(typ_kartki, linia_kartki, lambda_kartki)

            p_combo = p_gole * p_rogi * p_kartki

            lista_meczy.append({
                "home": home,
                "away": away,
                "p_gole": p_gole,
                "p_rogi": p_rogi,
                "p_kartki": p_kartki,
                "p_combo": p_combo,
                "lambda_home": lambda_home,
                "lambda_away": lambda_away
            })

    lista_meczy = sorted(lista_meczy,
                         key=lambda x: x["p_combo"],
                         reverse=True)

    lista_meczy = [m for m in lista_meczy
                   if m["p_combo"] >= min_prob]

    for mecz in lista_meczy:

        with st.expander(f"{mecz['home']} vs {mecz['away']}"):

            st.write(f"{koloruj(mecz['p_gole'])} ⚽ {typ_gole} {linia_gole}: "
                     f"**{mecz['p_gole']*100:.1f}%**")

            st.write(f"{koloruj(mecz['p_rogi'])} 🚩 {typ_rogi} {linia_rogi}: "
                     f"**{mecz['p_rogi']*100:.1f}%**")

            st.write(f"{koloruj(mecz['p_kartki'])} 🟨 {typ_kartki} {linia_kartki}: "
                     f"**{mecz['p_kartki']*100:.1f}%**")

            st.markdown("---")
            st.markdown(f"### 🎯 {koloruj(mecz['p_combo'])} "
                        f"Combo: {mecz['p_combo']*100:.2f}%")

            st.caption(
                f"Model λ → {mecz['home']}: {mecz['lambda_home']:.2f} | "
                f"{mecz['away']}: {mecz['lambda_away']:.2f}"
            )

else:
    st.warning("Brak nadchodzących meczów.")

# ----------------------------------
# SEKCJA BTTS
# ----------------------------------

st.markdown("---")
st.subheader("⚽ BTTS – najbliższa kolejka")

if not nadchodzace.empty:

    lista_btts = []

    for _, mecz in mecze.iterrows():

        home = NAZWY_MAP.get(mecz['home_team'], mecz['home_team'])
        away = NAZWY_MAP.get(mecz['away_team'], mecz['away_team'])

        if home in srednie_df.index and away in srednie_df.index:

            lambda_home = (
                srednie_df.loc[home, "Gole strzelone (dom)"] +
                srednie_df.loc[away, "Gole stracone (wyjazd)"]
            ) / 2

            lambda_away = (
                srednie_df.loc[away, "Gole strzelone (wyjazd)"] +
                srednie_df.loc[home, "Gole stracone (dom)"]
            ) / 2

            p_home_0 = poisson.pmf(0, lambda_home)
            p_away_0 = poisson.pmf(0, lambda_away)

            p_btts_yes = 1 - p_home_0 - p_away_0 + (p_home_0 * p_away_0)
            p_btts_no = 1 - p_btts_yes

            lista_btts.append({
                "home": home,
                "away": away,
                "yes": p_btts_yes,
                "no": p_btts_no
            })

    lista_btts = sorted(lista_btts,
                        key=lambda x: x["yes"],
                        reverse=True)

    for mecz in lista_btts:

        with st.expander(f"{mecz['home']} vs {mecz['away']}"):

            st.write(f"{koloruj(mecz['yes'])} ⚽ BTTS TAK: "
                     f"**{mecz['yes']*100:.1f}%**")

            st.write(f"{koloruj(mecz['no'])} ❌ BTTS NIE: "
                     f"**{mecz['no']*100:.1f}%**")

# ----------------------------------
# TABELA ŚREDNICH
# ----------------------------------

st.markdown("---")
st.subheader("📊 Średnie drużyn (ostatnie 10 meczów, waga formy)")

st.dataframe(
    srednie_df.sort_index(),
    use_container_width=True
)



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

# ----------------------------------
# ODŚWIEŻANIE
# ----------------------------------

if st.button("Odśwież dane"):
    st.cache_data.clear()
    st.rerun()