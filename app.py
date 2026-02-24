import streamlit as st
import pandas as pd
from scipy.stats import poisson
from datetime import datetime
import requests
from io import StringIO
import numpy as np

# --- KONFIGURACJA ---
st.set_page_config(page_title="PL Pro Analytics & Monte Carlo", layout="wide")

st.title("Premier League 2025/26 - Advanced Predictive Model")
st.markdown("Model: Poisson + Wagi Formy + Monte Carlo Simulation")

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

# --- DANE ---
@st.cache_data(ttl=900)
def load_historical():
    try:
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
    except:
        st.error("Błąd pobierania danych z football-data.co.uk")
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def load_schedule():
    try:
        df = pd.read_csv("terminarz_premier_2025.csv")
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date')
    except:
        return pd.DataFrame()

historical = load_historical()
schedule = load_schedule()

# --- LOGIKA OBLICZEŃ ---
def weighted_mean(values):
    if len(values) == 0: return 0
    weights = np.linspace(1, 2, len(values))
    return np.average(values, weights=weights)

@st.cache_data
def oblicz_wszystkie_statystyki(df):
    druzyny = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel())
    dane = {}
    for d in druzyny:
        home = df[df['HomeTeam'] == d].tail(10)
        away = df[df['AwayTeam'] == d].tail(10)
        if len(home) < 3 or len(away) < 3: continue
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

def symuluj_mecz_monte_carlo(l_h, l_a, n=5000):
    """Symuluje mecz n-razy używając rozkładu Poissona."""
    home_goals = np.random.poisson(l_h, n)
    away_goals = np.random.poisson(l_a, n)
    
    # Obliczanie wyników
    home_win = np.sum(home_goals > away_goals) / n
    draw = np.sum(home_goals == away_goals) / n
    away_win = np.sum(home_goals < away_goals) / n
    
    # Najczęstszy wynik (Moda)
    results = [f"{h}:{a}" for h, a in zip(home_goals, away_goals)]
    top_score = max(set(results), key=results.count)
    
    return home_win, draw, away_win, top_score

def koloruj(p):
    if p > 0.65: return "🟢"
    elif p > 0.45: return "🟡"
    else: return "🔴"

# --- GŁÓWNA LOGIKA APLIKACJI ---
if not historical.empty:
    srednie_df = oblicz_wszystkie_statystyki(historical)
    
    # TABS
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Bet Builder", 
        "🎲 Monte Carlo & Wyniki", 
        "📊 Tabela & Forma", 
        "📈 Dane Modelu"
    ])

    with tab1:
        st.subheader("🎛️ Parametry Twojego Combo")
        c1, c2, c3 = st.columns(3)
        with c1:
            linia_gole = st.selectbox("Gole", [1.5, 2.5, 3.5], index=1)
            typ_gole = st.selectbox("Typ", ["Over", "Under"], key="g")
        with c2:
            linia_rogi = st.selectbox("Rożne", [8.5, 9.5, 10.5, 11.5], index=1)
            typ_rogi = st.selectbox("Typ", ["Over", "Under"], key="r")
        with c3:
            linia_kartki = st.selectbox("Kartki", [3.5, 4.5, 5.5], index=1)
            typ_kartki = st.selectbox("Typ", ["Over", "Under"], key="k")
        
        min_p = st.slider("Minimalne prawdopodobieństwo", 0.1, 0.8, 0.3)

        # Predykcje
        dzisiaj = datetime.now()
        nadchodzace = schedule[schedule['date'] > dzisiaj] if not schedule.empty else pd.DataFrame()

        if not nadchodzace.empty:
            mecze = nadchodzace[nadchodzace['round'] == nadchodzace['round'].min()]
            for _, m in mecze.iterrows():
                h = NAZWY_MAP.get(m['home_team'], m['home_team'])
                a = NAZWY_MAP.get(m['away_team'], m['away_team'])
                
                if h in srednie_df.index and a in srednie_df.index:
                    l_h = (srednie_df.loc[h, "Gole strzelone (dom)"] + srednie_df.loc[a, "Gole stracone (wyjazd)"]) / 2
                    l_a = (srednie_df.loc[a, "Gole strzelone (wyjazd)"] + srednie_df.loc[h, "Gole stracone (dom)"]) / 2
                    l_r = (srednie_df.loc[h, "Różne (dom)"] + srednie_df.loc[a, "Różne (wyjazd)"]) / 2
                    l_k = (srednie_df.loc[h, "Kartki (dom)"] + srednie_df.loc[a, "Kartki (wyjazd)"]) / 2

                    p_g = (1 - poisson.cdf(linia_gole, l_h+l_a)) if typ_gole == "Over" else poisson.cdf(linia_gole, l_h+l_a)
                    p_r = (1 - poisson.cdf(linia_rogi, l_r)) if typ_rogi == "Over" else poisson.cdf(linia_rogi, l_r)
                    p_k = (1 - poisson.cdf(linia_kartki, l_k)) if typ_kartki == "Over" else poisson.cdf(linia_kartki, l_k)
                    p_c = p_g * p_r * p_k

                    if p_c >= min_p:
                        with st.expander(f"🔥 {h} vs {a} | Combo: {p_c:.1%}"):
                            st.write(f"Gole ({typ_gole} {linia_gole}): {p_g:.1%} | Rożne: {p_r:.1%} | Kartki: {p_k:.1%}")

    with tab2:
        st.subheader("🎲 Symulacje Monte Carlo (5000 iteracji)")
        st.info("Obliczamy najbardziej prawdopodobne wyniki oraz 'Fair Odds' (Kursy Sprawiedliwe).")
        
        if not nadchodzace.empty:
            for _, m in mecze.iterrows():
                h = NAZWY_MAP.get(m['home_team'], m['home_team'])
                a = NAZWY_MAP.get(m['away_team'], m['away_team'])
                
                if h in srednie_df.index and a in srednie_df.index:
                    l_h = (srednie_df.loc[h, "Gole strzelone (dom)"] + srednie_df.loc[a, "Gole stracone (wyjazd)"]) / 2
                    l_a = (srednie_df.loc[a, "Gole strzelone (wyjazd)"] + srednie_df.loc[h, "Gole stracone (dom)"]) / 2
                    
                    p_h, p_d, p_a, best_score = symuluj_mecz_monte_carlo(l_h, l_a)
                    
                    col_m1, col_m2, col_m3 = st.columns([2, 1, 2])
                    with col_m1:
                        st.write(f"**{h} vs {a}**")
                        st.caption(f"Przewidywany xG: {l_h:.2f} - {l_a:.2f}")
                    with col_m2:
                        st.metric("Top Wynik", best_score)
                    with col_m3:
                        st.write(f"1: {1/p_h:.2f} | X: {1/p_d:.2f} | 2: {1/p_a:.2f} (Kursy)")
                    st.divider()

    with tab3:
        # Tutaj Twoja oryginalna tabela ligowa i forma
        st.write("Sekcja tabeli i formy (zgodnie z poprzednim kodem)")
        # ... (kod tabeli)

    with tab4:
        st.subheader("📊 Średnie ważone i surowe dane")
        st.dataframe(srednie_df, use_container_width=True)

