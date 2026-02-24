import streamlit as st
import pandas as pd
from scipy.stats import poisson
from datetime import datetime
import requests
from io import StringIO
import numpy as np

# --- KONFIGURACJA ---
st.set_page_config(page_title="PL Analytics & Bet Builder", layout="wide")

# Mapowanie nazw dla spójności danych
NAZWY_MAP = {
    "Brighton & Hove Albion": "Brighton",
    "West Ham United": "West Ham",
    "Newcastle United": "Newcastle",
    "Tottenham Hotspur": "Tottenham",
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves"
}

# --- FUNKCJE POMOCNICZE ---
def weighted_mean(values):
    if len(values) == 0: return 0
    weights = np.linspace(1, 2, len(values))
    return np.average(values, weights=weights)

def koloruj_p(p):
    if p > 0.65: return "🟢"
    elif p > 0.45: return "🟡"
    else: return "🔴"

# --- ŁADOWANIE DANYCH ---
@st.cache_data(ttl=3600)
def load_data():
    try:
        url = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
        r = requests.get(url, timeout=10)
        df = pd.read_csv(StringIO(r.text))
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        
        # Agregacja statystyk
        df['total_gole'] = df['FTHG'] + df['FTAG']
        df['total_rogi'] = df['HC'] + df['AC']
        df['total_kartki'] = df['HY'] + df['AY'] + (df['HR'] + df['AR']) * 2
        
        # Próba załadowania terminarza
        try:
            sch = pd.read_csv("terminarz_premier_2025.csv")
            sch['date'] = pd.to_datetime(sch['date'])
        except:
            sch = pd.DataFrame()
            
        return df, sch
    except Exception as e:
        st.error(f"Błąd danych: {e}")
        return pd.DataFrame(), pd.DataFrame()

historical, schedule = load_data()

# --- ANALIZA DRUŻYN ---
@st.cache_data
def get_team_stats(df):
    teams = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel())
    stats = {}
    for t in teams:
        h = df[df['HomeTeam'] == t].tail(8)
        a = df[df['AwayTeam'] == t].tail(8)
        stats[t] = {
            "g_strz_d": weighted_mean(h['FTHG']), "g_strac_d": weighted_mean(h['FTAG']),
            "g_strz_w": weighted_mean(a['FTAG']), "g_strac_w": weighted_mean(a['FTHG']),
            "r_strz_d": weighted_mean(h['HC']), "r_strac_d": weighted_mean(h['AC']),
            "r_strz_w": weighted_mean(a['AC']), "r_strac_w": weighted_mean(a['HC']),
            "k_d": weighted_mean(h['HY'] + h['HR']*2),
            "k_w": weighted_mean(a['AY'] + a['AR']*2)
        }
    return pd.DataFrame(stats).T.round(2)

if not historical.empty:
    stats_df = get_team_stats(historical)

    # --- UI: SIDEBAR (BET BUILDER SETTINGS) ---
    with st.sidebar:
        st.header("🎛️ Globalny Bet Builder")
        st.info("Ustaw parametry, aby przefiltrować mecze pod kątem Twojego kuponu.")
        
        l_gole = st.selectbox("Linia goli", [1.5, 2.5, 3.5], index=1)
        t_gole = st.radio("Typ goli", ["Over", "Under"], horizontal=True)
        
        l_rogi = st.slider("Linia rzutów rożnych", 7.5, 12.5, 9.5, 0.5)
        t_rogi = st.radio("Typ rożnych", ["Over", "Under"], horizontal=True)
        
        l_kartki = st.slider("Linia kartek", 2.5, 6.5, 3.5, 0.5)
        t_kartki = st.radio("Typ kartek", ["Over", "Under"], horizontal=True)
        
        min_acc = st.slider("Min. Prawdopodobieństwo Combo", 0.1, 0.6, 0.3)
        
        if st.button("🔄 Odśwież dane"):
            st.cache_data.clear()
            st.rerun()

    # --- GŁÓWNY PANEL ---
    tab1, tab2, tab3 = st.tabs(["🔥 Typy z Buildera", "📊 Tabela i Forma", "📋 Pełne Statystyki"])

    with tab1:
        st.subheader(f"Typy na najbliższą kolejkę (Combo: {t_gole} {l_gole}, {t_rogi} {l_rogi}, {t_kartki} {l_kartki})")
        
        future = schedule[schedule['date'] > datetime.now()]
        if not future.empty:
            m_round = future['round'].min()
            matches = future[future['round'] == m_round]
            
            found_any = False
            for _, m in matches.iterrows():
                h = NAZWY_MAP.get(m['home_team'], m['home_team'])
                a = NAZWY_MAP.get(m['away_team'], m['away_team'])
                
                if h in stats_df.index and a in stats_df.index:
                    # Gole Poisson
                    lam_h = (stats_df.loc[h, "g_strz_d"] + stats_df.loc[a, "g_strac_w"]) / 2
                    lam_a = (stats_df.loc[a, "g_strz_w"] + stats_df.loc[h, "g_strac_d"]) / 2
                    p_g = 1 - poisson.cdf(l_gole, lam_h + lam_a) if t_gole == "Over" else poisson.cdf(l_gole, lam_h + lam_a)
                    
                    # Rożne Poisson
                    lam_r = (stats_df.loc[h, "r_strz_d"] + stats_df.loc[h, "r_strac_d"] + 
                             stats_df.loc[a, "r_strz_w"] + stats_df.loc[a, "r_strac_w"]) / 2
                    p_r = 1 - poisson.cdf(l_rogi, lam_r) if t_rogi == "Over" else poisson.cdf(l_rogi, lam_r)
                    
                    # Kartki Poisson
                    lam_k = stats_df.loc[h, "k_d"] + stats_df.loc[a, "k_w"]
                    p_k = 1 - poisson.cdf(l_kartki, lam_k) if t_kartki == "Over" else poisson.cdf(l_kartki, lam_k)
                    
                    p_combo = p_g * p_r * p_k
                    
                    if p_combo >= min_acc:
                        found_any = True
                        with st.expander(f"{koloruj_p(p_combo)} {h} vs {a} | Combo: {p_combo:.1%}"):
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Gole", f"{p_g:.1%}", f"{t_gole} {l_gole}")
                            col2.metric("Rożne", f"{p_r:.1%}", f"{t_rogi} {l_rogi}")
                            col3.metric("Kartki", f"{p_k:.1%}", f"{t_kartki} {l_kartki}")
                            st.progress(p_combo)
            
            if not found_any:
                st.warning("Brak meczów spełniających kryteria prawdopodobieństwa. Spróbuj obniżyć progi w Builderze.")
        else:
            st.info("Brak nadchodzących meczów w terminarzu.")

    with tab2:
        # Tabela Ligowa (Wersja zoptymalizowana)
        st.subheader("Tabela Premier League 25/26")
        h_table = historical.groupby('HomeTeam').agg({'FTHG':'sum','FTAG':'sum','Date':'count'}).rename(columns={'Date':'M','FTHG':'GS','FTAG':'GL'})
        a_table = historical.groupby('AwayTeam').agg({'FTAG':'sum','FTHG':'sum','Date':'count'}).rename(columns={'Date':'M','FTAG':'GS','FTAG':'GL'})
        full_table = h_table.add(a_table, fill_value=0)
        st.dataframe(full_table.sort_values('GS', ascending=False), use_container_width=True)

    with tab3:
        st.subheader("Surowe dane modelu (Waga formy)")
        st.dataframe(stats_df, use_container_width=True)

