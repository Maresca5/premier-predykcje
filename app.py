import streamlit as st
import pandas as pd
from scipy.stats import poisson
from datetime import datetime
import requests
from io import StringIO
import numpy as np

# --- KONFIGURACJA API ---
# Wstaw tutaj swój klucz z API-Football (RapidAPI)
API_KEY = "5be25cbf05cce4c2957d4fc89e6da4f7" 

LIGI_CONFIG = {
    "Premier League": {"id": 39, "csv": "E0"},
    "La Liga": {"id": 140, "csv": "SP1"},
    "Bundesliga": {"id": 78, "csv": "D1"},
    "Serie A": {"id": 135, "csv": "I1"},
    "Ligue 1": {"id": 61, "csv": "F1"}
}

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="Football Predictor Pro", layout="wide")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Ustawienia")
    wybrana_liga = st.selectbox("Wybierz Ligę", list(LIGI_CONFIG.keys()))
    kod_csv = LIGI_CONFIG[wybrana_liga]["csv"]
    kod_id = LIGI_CONFIG[wybrana_liga]["id"]
    
    st.divider()
    if st.button("🔄 Odśwież Dane"):
        st.cache_data.clear()
        st.rerun()

st.title(f"Predykcje: {wybrana_liga} 2025/26")
st.markdown("Model Poissona + API-Football + Wagi Formy")

# --- MAPOWANIE NAZW (ROZSZERZONE) ---
NAZWY_MAP = {
    "Brighton & Hove Albion": "Brighton",
    "West Ham United": "West Ham",
    "Newcastle United": "Newcastle",
    "Tottenham Hotspur": "Tottenham",
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "Sheffield United": "Sheffield Utd",
    "Atletico Madrid": "Ath Madrid",
    "AC Milan": "Milan",
    "Inter Milan": "Inter",
    "FC Barcelona": "Barcelona",
    "Real Madrid": "Real Madrid"
}

# --- DANE ---
@st.cache_data(ttl=900)
def load_historical(league_csv):
    try:
        url = f"https://www.football-data.co.uk/mmz4281/2526/{league_csv}.csv"
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
        st.error(f"Nie udało się pobrać danych historycznych dla {wybrana_liga}.")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_schedule_api(league_id):
    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
    # Pobieramy najbliższe 15 meczów
    headers = {
        "x-rapidapi-key": API_KEY,
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com"
    }
    params = {"league": str(league_id), "season": "2025", "next": "15"}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        data = response.json()
        fixtures = []
        for res in data.get("response", []):
            fixtures.append({
                "date": res["fixture"]["date"],
                "home_team": res["teams"]["home"]["name"],
                "away_team": res["teams"]["away"]["name"],
                "round": res["league"]["round"]
            })
        df = pd.DataFrame(fixtures)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except:
        st.warning("Nie udało się połączyć z API-Football. Upewnij się, że klucz jest poprawny.")
        return pd.DataFrame()

historical = load_historical(kod_csv)
schedule = load_schedule_api(kod_id)

# --- LOGIKA OBLICZEŃ ---
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

def oblicz_forme(df):
    if df.empty: return {}
    druzyny = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel())
    forma = {}
    for d in druzyny:
        mecze = df[(df['HomeTeam'] == d) | (df['AwayTeam'] == d)].tail(5)
        wyniki = []
        for _, m in mecze.iterrows():
            if m['HomeTeam'] == d:
                if m['FTHG'] > m['FTAG']: wyniki.append("W")
                elif m['FTHG'] < m['FTAG']: wyniki.append("L")
                else: wyniki.append("D")
            else:
                if m['FTAG'] > m['FTHG']: wyniki.append("W")
                elif m['FTAG'] < m['FTHG']: wyniki.append("L")
                else: wyniki.append("D")
        forma[d] = "".join(wyniki)
    return forma

def tabela_ligowa(df):
    if df.empty: return pd.DataFrame()
    table = {}
    for _, m in df.iterrows():
        home, away = m['HomeTeam'], m['AwayTeam']
        hg, ag = m['FTHG'], m['FTAG']
        for team in [home, away]:
            if team not in table: table[team] = {"pts":0,"gf":0,"ga":0,"played":0}
        table[home]["gf"] += hg
        table[home]["ga"] += ag
        table[home]["played"] += 1
        table[away]["gf"] += ag
        table[away]["ga"] += hg
        table[away]["played"] += 1
        if hg > ag: table[home]["pts"] += 3
        elif hg < ag: table[away]["pts"] += 3
        else:
            table[home]["pts"] += 1
            table[away]["pts"] += 1
    res = pd.DataFrame(table).T
    res["diff"] = res["gf"] - res["ga"]
    return res.sort_values(["pts","diff","gf"], ascending=False)

def koloruj(p):
    if p > 0.65: return "🟢"
    elif p > 0.50: return "🟡"
    else: return "🔴"

# Przygotowanie danych
if not historical.empty:
    srednie_df = oblicz_wszystkie_statystyki(historical)
    forma_dict = oblicz_forme(historical)
    tabela = tabela_ligowa(historical)
    
    tab1, tab2, tab3 = st.tabs(["🎯 Bet Builder & Predykcje", "📊 Tabela i Forma", "📈 Statystyki Modelu"])

    with tab1:
        st.subheader("🎛️ Zbuduj własne combo")
        c1, c2, c3 = st.columns(3)
        with c1:
            linia_gole = st.selectbox("Linia goli", [1.5, 2.5, 3.5, 4.5], index=1)
            typ_gole = st.selectbox("Typ goli", ["Over", "Under"])
        with c2:
            linia_rogi = st.selectbox("Linia rożnych", [7.5, 8.5, 9.5, 10.5, 11.5, 12.5], index=2)
            typ_rogi = st.selectbox("Typ rożnych", ["Over", "Under"])
        with c3:
            linia_kartki = st.selectbox("Linia kartek", [2.5, 3.5, 4.5, 5.5], index=1)
            typ_kartki = st.selectbox("Typ kartek", ["Over", "Under"])
        
        min_prob = st.slider("Minimalne prawdopodobieństwo combo", 0.0, 1.0, 0.40, 0.05)

        st.subheader("📅 Nadchodzące mecze (API Live)")

        if not schedule.empty:
            col_pred1, col_pred2 = st.columns(2)
            
            with col_pred1:
                st.write("**Combo Builder**")
                for _, mecz in schedule.iterrows():
                    # Próba dopasowania nazw z API do nazw z CSV
                    h_raw = mecz['home_team']
                    a_raw = mecz['away_team']
                    h = NAZWY_MAP.get(h_raw, h_raw)
                    a = NAZWY_MAP.get(a_raw, a_raw)
                    
                    if h in srednie_df.index and a in srednie_df.index:
                        lam_h = (srednie_df.loc[h, "Gole strzelone (dom)"] + srednie_df.loc[a, "Gole stracone (wyjazd)"]) / 2
                        lam_a = (srednie_df.loc[a, "Gole strzelone (wyjazd)"] + srednie_df.loc[h, "Gole stracone (dom)"]) / 2
                        lam_g = lam_h + lam_a
                        lam_r = (srednie_df.loc[h, "Różne (dom)"] + srednie_df.loc[a, "Różne (wyjazd)"]) / 2
                        lam_k = (srednie_df.loc[h, "Kartki (dom)"] + srednie_df.loc[a, "Kartki (wyjazd)"]) / 2

                        def licz_p(typ, linia, lam):
                            return 1 - poisson.cdf(linia, lam) if typ == "Over" else poisson.cdf(linia, lam)

                        p_g = licz_p(typ_gole, linia_gole, lam_g)
                        p_r = licz_p(typ_rogi, linia_rogi, lam_r)
                        p_k = licz_p(typ_kartki, linia_kartki, lam_k)
                        p_combo = p_g * p_r * p_k

                        if p_combo >= min_prob:
                            with st.expander(f"{h} vs {a} ({p_combo:.1%})"):
                                st.write(f"{koloruj(p_g)} Gole: {p_g:.1%}")
                                st.write(f"{koloruj(p_r)} Rożne: {p_r:.1%}")
                                st.write(f"{koloruj(p_k)} Kartki: {p_k:.1%}")
                                st.caption(f"Data: {mecz['date'].strftime('%d.%m %H:%M')}")
            
            with col_pred2:
                st.write("**BTTS Ranking**")
                for _, mecz in schedule.iterrows():
                    h = NAZWY_MAP.get(mecz['home_team'], mecz['home_team'])
                    a = NAZWY_MAP.get(mecz['away_team'], mecz['away_team'])
                    if h in srednie_df.index and a in srednie_df.index:
                        lam_h = (srednie_df.loc[h, "Gole strzelone (dom)"] + srednie_df.loc[a, "Gole stracone (wyjazd)"]) / 2
                        lam_a = (srednie_df.loc[a, "Gole strzelone (wyjazd)"] + srednie_df.loc[h, "Gole stracone (dom)"]) / 2
                        p_btts = (1 - poisson.pmf(0, lam_h)) * (1 - poisson.pmf(0, lam_a))
                        st.write(f"{koloruj(p_btts)} **{h} - {a}**: {p_btts:.1%}")
        else:
            st.info("Brak nadchodzących meczów do wyświetlenia.")

    with tab2:
        st.subheader(f"📊 Sytuacja w lidze: {wybrana_liga}")
        c_l, c_f = st.columns([2, 1])
        with c_l:
            st.write("Tabela Ligowa")
            st.dataframe(tabela, use_container_width=True)
        with c_f:
            st.write("Forma (5 ostatnich)")
            forma_df = pd.DataFrame.from_dict(forma_dict, orient="index", columns=["Forma"])
            st.dataframe(forma_df, use_container_width=True)

    with tab3:
        st.subheader("📊 Średnie ważone drużyn")
        st.dataframe(srednie_df.sort_index(), use_container_width=True)
