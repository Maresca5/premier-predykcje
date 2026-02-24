import streamlit as st
import pandas as pd
from scipy.stats import poisson
from datetime import datetime, timedelta
import requests
from io import StringIO
import numpy as np

# --- KONFIGURACJA LIG ---
# Używamy ujednoliconych kluczy dla football-data (statystyki) i nazw dla terminarza
LIGI_CONFIG = {
    "Premier League": {"csv": "E0", "github_slug": "en.1"},
    "La Liga": {"csv": "SP1", "github_slug": "es.1"},
    "Bundesliga": {"csv": "D1", "github_slug": "de.1"},
    "Serie A": {"csv": "I1", "github_slug": "it.1"},
    "Ligue 1": {"csv": "F1", "github_slug": "fr.1"}
}

st.set_page_config(page_title="Football Predictor Pro 2026", layout="wide")

# --- SIDEBAR ---
st.sidebar.header("🌍 Wybór Rozgrywek")
wybrana_liga = st.sidebar.selectbox("Wybierz ligę", list(LIGI_CONFIG.keys()))
kod_csv = LIGI_CONFIG[wybrana_liga]["csv"]
slug = LIGI_CONFIG[wybrana_liga]["github_slug"]

st.title(f"Analiza: {wybrana_liga}")
st.markdown(f"Model: Poisson + GitHub Open Data (football.db) + Wagi Formy")

# --- MAPOWANIE NAZW (Kluczowe dla synchronizacji źródeł) ---
# Dopasowujemy nazwy z GitHuba/API do nazw w plikach CSV ze statystykami
NAZWY_MAP = {
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Tottenham Hotspur": "Tottenham",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "Brighton & Hove Albion": "Brighton",
    "West Ham United": "West Ham",
    "Leicester City": "Leicester",
    "Ipswich Town": "Ipswich",
    "Bayern München": "Bayern Munich",
    "Borussia Dortmund": "Dortmund",
    "Bayer Leverkusen": "Leverkusen",
    "Atlético Madrid": "Ath Madrid",
    "Real Madrid": "Real Madrid",
    "FC Barcelona": "Barcelona"
}

# --- DANE HISTORYCZNE (Football-Data.co.uk) ---
@st.cache_data(ttl=3600)
def load_historical(league_code):
    try:
        url = f"https://www.football-data.co.uk/mmz4281/2526/{league_code}.csv"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        # Obliczenia bazowe
        df['total_gole'] = df['FTHG'] + df['FTAG']
        df['total_kartki'] = df['HY'] + df['AY'] + (df['HR'] + df['AR']) * 2
        df['total_rozne'] = df['HC'] + df['AC']
        return df
    except Exception as e:
        st.error(f"Błąd danych historycznych: {e}")
        return pd.DataFrame()

# --- TERMINARZ Z GITHUBA (Open Football Data) ---
@st.cache_data(ttl=43200)
def load_schedule_github(league_slug):
    try:
        # Korzystamy z API football.db, które serwuje dane z GitHuba
        url = f"https://raw.githubusercontent.com/openfootball/world-cup/master/2026/worldcup.json"
        # DLA LIG EUROPEJSKICH: używamy endpointu repozytorium football.json
        url = f"https://raw.githubusercontent.com/openfootball/football.json/master/2025-26/{league_slug}.json"
        
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        
        fixtures = []
        for round_data in data.get('rounds', []):
            round_name = round_data.get('name', 'N/A')
            for match in round_data.get('matches', []):
                fixtures.append({
                    "Date": match.get('date'),
                    "Home": match.get('team1'),
                    "Away": match.get('team2'),
                    "Round": round_name
                })
        
        df = pd.DataFrame(fixtures)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Filtrowanie: najbliższe 10 dni
        dzis = datetime.now()
        mask = (df['Date'] >= (dzis - timedelta(days=1))) & (df['Date'] <= (dzis + timedelta(days=10)))
        return df.loc[mask].sort_values('Date')
        
    except Exception as e:
        st.warning(f"Błąd terminarza GitHub: {e}. Przełączam na tryb manualny lub brak danych.")
        return pd.DataFrame()

# --- LOGIKA OBLICZEŃ ---
def weighted_mean(values):
    if len(values) == 0: return 0
    weights = np.linspace(1, 2, len(values))
    return np.average(values, weights=weights)

@st.cache_data
def oblicz_statystyki(df):
    if df.empty: return pd.DataFrame()
    druzyny = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel())
    dane = {}
    for d in druzyny:
        h = df[df['HomeTeam'] == d].tail(10)
        a = df[df['AwayTeam'] == d].tail(10)
        if len(h) < 2 or len(a) < 2: continue
        dane[d] = {
            "G_strz_D": weighted_mean(h['FTHG']), "G_strac_D": weighted_mean(h['FTAG']),
            "G_strz_W": weighted_mean(a['FTAG']), "G_strac_W": weighted_mean(a['FTHG']),
            "R_D": weighted_mean(h['total_rozne']), "R_W": weighted_mean(a['total_rozne']),
            "K_D": weighted_mean(h['total_kartki']), "K_W": weighted_mean(a['total_kartki'])
        }
    return pd.DataFrame(dane).T.round(2)

def koloruj(p):
    return "🟢" if p > 0.65 else "🟡" if p > 0.45 else "🔴"

# --- PROCES GŁÓWNY ---
hist_df = load_historical(kod_csv)
sched_df = load_schedule_github(slug)

if not hist_df.empty:
    stats_df = oblicz_statystyki(hist_df)
    
    tab1, tab2, tab3 = st.tabs(["🎯 Bet Builder", "📊 Tabela", "📈 Statystyki"])

    with tab1:
        st.subheader("🎛️ Parametry Analizy")
        c1, c2, c3 = st.columns(3)
        l_g = c1.selectbox("Linia Goli", [1.5, 2.5, 3.5], index=1)
        l_r = c2.selectbox("Linia Rożnych", [8.5, 9.5, 10.5], index=1)
        l_k = c3.selectbox("Linia Kartek", [3.5, 4.5, 5.5], index=1)
        min_p = st.slider("Min. Prawdopodobieństwo Combo", 0.1, 0.9, 0.4)

        if not sched_df.empty:
            for _, m in sched_df.iterrows():
                h = NAZWY_MAP.get(m['Home'], m['Home'])
                a = NAZWY_MAP.get(m['Away'], m['Away'])
                
                if h in stats_df.index and a in stats_df.index:
                    # Model Poisson
                    lam_h = (stats_df.loc[h, "G_strz_D"] + stats_df.loc[a, "G_strac_W"]) / 2
                    lam_a = (stats_df.loc[a, "G_strz_W"] + stats_df.loc[h, "G_strac_D"]) / 2
                    lam_r = (stats_df.loc[h, "R_D"] + stats_df.loc[a, "R_W"]) / 2
                    lam_k = (stats_df.loc[h, "K_D"] + stats_df.loc[a, "K_W"]) / 2

                    def get_p(lin, lam): return 1 - poisson.cdf(lin, lam)

                    p_g = get_p(l_g, lam_h + lam_a)
                    p_r = get_p(l_r, lam_r)
                    p_k = get_p(l_k, lam_k)
                    p_combo = p_g * p_r * p_k

                    if p_combo >= min_p:
                        with st.expander(f"🏟️ {h} - {a} | {m['Date'].strftime('%d.%m')} | Combo: {p_combo:.1%}"):
                            st.write(f"{koloruj(p_g)} Gole Over {l_g}: {p_g:.1%}")
                            st.write(f"{koloruj(p_r)} Rożne Over {l_r}: {p_r:.1%}")
                            st.write(f"{koloruj(p_k)} Kartki Over {l_k}: {p_k:.1%}")
                            st.caption(f"xG: {lam_h:.2f} - {lam_a:.2f}")
        else:
            st.info("Brak nadchodzących meczów w bazie GitHub dla wybranego okresu.")

    with tab3:
        st.dataframe(stats_df, use_container_width=True)
else:
    st.error("Błąd ładowania danych. Spróbuj odświeżyć stronę.")
