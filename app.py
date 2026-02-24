import streamlit as st
import pandas as pd
from scipy.stats import poisson
from datetime import datetime
import requests
from io import StringIO
import numpy as np

# =====================================================
# KONFIGURACJA LIGI (skalowalne)
# =====================================================

LEAGUE_CODE = "E0"       # Premier League
SEASON = "2526"          # sezon 2025/26

st.set_page_config(page_title="Model Piłkarski", layout="wide")
st.title("Model predykcyjny – Poisson + Strength Model")

# =====================================================
# DANE
# =====================================================

@st.cache_data(ttl=900)
def load_historical(league, season):
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
    r = requests.get(url)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values("Date")
    return df

@st.cache_data(ttl=86400)
def load_schedule():
    df = pd.read_csv("terminarz_premier_2025.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values("date")

historical = load_historical(LEAGUE_CODE, SEASON)
schedule = load_schedule()

# =====================================================
# FUNKCJE POMOCNICZE
# =====================================================

def weighted_mean(values):
    weights = np.linspace(1, 2, len(values))
    return np.average(values, weights=weights)

# =====================================================
# ŚREDNIE LIGOWE
# =====================================================

league_avg_home = historical["FTHG"].mean()
league_avg_away = historical["FTAG"].mean()

# =====================================================
# SIŁA DRUŻYN
# =====================================================

@st.cache_data
def compute_team_strength():
    teams = pd.unique(historical[['HomeTeam', 'AwayTeam']].values.ravel())
    data = {}

    for team in teams:
        home = historical[historical['HomeTeam'] == team].tail(10)
        away = historical[historical['AwayTeam'] == team].tail(10)

        if len(home) < 3 or len(away) < 3:
            continue

        home_scored = weighted_mean(home["FTHG"])
        home_conceded = weighted_mean(home["FTAG"])
        away_scored = weighted_mean(away["FTAG"])
        away_conceded = weighted_mean(away["FTHG"])

        data[team] = {
            "Attack_home": home_scored / league_avg_home,
            "Defence_home": home_conceded / league_avg_away,
            "Attack_away": away_scored / league_avg_away,
            "Defence_away": away_conceded / league_avg_home,
        }

    return pd.DataFrame(data).T

strength_df = compute_team_strength()

# =====================================================
# TABELA LIGOWA
# =====================================================

def league_table():
    table = {}

    for _, m in historical.iterrows():
        h, a = m['HomeTeam'], m['AwayTeam']
        hg, ag = m['FTHG'], m['FTAG']

        for t in [h, a]:
            if t not in table:
                table[t] = {"pts":0,"gf":0,"ga":0,"played":0}

        table[h]["gf"] += hg
        table[h]["ga"] += ag
        table[a]["gf"] += ag
        table[a]["ga"] += hg
        table[h]["played"] += 1
        table[a]["played"] += 1

        if hg > ag:
            table[h]["pts"] += 3
        elif hg < ag:
            table[a]["pts"] += 3
        else:
            table[h]["pts"] += 1
            table[a]["pts"] += 1

    df = pd.DataFrame(table).T
    df["diff"] = df["gf"] - df["ga"]
    return df.sort_values(["pts","diff","gf"], ascending=False)

# =====================================================
# MODEL MECZU
# =====================================================

def predict_match(home, away):

    att_home = strength_df.loc[home, "Attack_home"]
    def_away = strength_df.loc[away, "Defence_away"]

    att_away = strength_df.loc[away, "Attack_away"]
    def_home = strength_df.loc[home, "Defence_home"]

    lambda_home = att_home * def_away * league_avg_home
    lambda_away = att_away * def_home * league_avg_away

    lambda_total = lambda_home + lambda_away

    p_home_0 = poisson.pmf(0, lambda_home)
    p_away_0 = poisson.pmf(0, lambda_away)

    p_btts = 1 - p_home_0 - p_away_0 + (p_home_0 * p_away_0)
    p_over25 = 1 - poisson.cdf(2, lambda_total)

    # dokładne wyniki 0-4
    score_probs = {}
    for i in range(5):
        for j in range(5):
            score_probs[f"{i}-{j}"] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)

    top_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "lambda_total": round(lambda_total,2),
        "BTTS": round(p_btts*100,1),
        "Over25": round(p_over25*100,1),
        "TopScores": [(s, round(p*100,1)) for s,p in top_scores]
    }

# =====================================================
# NAJBLIŻSZA KOLEJKA
# =====================================================

st.markdown("---")
st.subheader("🔥 Model – najbliższa kolejka")

today = datetime.now()
upcoming = schedule[schedule["date"] > today]

if not upcoming.empty:

    next_round = upcoming["round"].min()
    matches = upcoming[upcoming["round"] == next_round]

    results = []

    for _, row in matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        if home in strength_df.index and away in strength_df.index:

            pred = predict_match(home, away)

            results.append({
                "Mecz": f"{home} vs {away}",
                "λ_total": pred["lambda_total"],
                "BTTS %": pred["BTTS"],
                "Over 2.5 %": pred["Over25"],
                "Top wynik": pred["TopScores"][0][0]
            })

            with st.expander(f"{home} vs {away} – szczegóły"):
                st.write("Top 5 wyników:")
                for score, prob in pred["TopScores"]:
                    st.write(f"{score} → {prob}%")

    df_results = pd.DataFrame(results).sort_values("BTTS %", ascending=False)
    st.dataframe(df_results, use_container_width=True)

# =====================================================
# TABELA
# =====================================================

st.markdown("---")
st.subheader("📊 Tabela ligowa")
st.dataframe(league_table(), use_container_width=True)

# =====================================================
# POWER RANKING
# =====================================================

st.markdown("---")
st.subheader("⚡ Power Ranking (modelowy)")

power = strength_df.copy()
power["Power"] = (
    power["Attack_home"] + power["Attack_away"]
    - power["Defence_home"] - power["Defence_away"]
)

st.dataframe(
    power.sort_values("Power", ascending=False)[["Power"]],
    use_container_width=True
)

# =====================================================
# ODŚWIEŻ
# =====================================================

if st.button("Odśwież dane"):
    st.cache_data.clear()
    st.rerun()