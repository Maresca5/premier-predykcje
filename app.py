import streamlit as st
import pandas as pd
from scipy.stats import poisson
from datetime import datetime, timedelta
import requests
from io import StringIO
import numpy as np

# --- KONFIGURACJA LIG ---
LIGI_CONFIG = {
    "Premier League": {"csv": "E0", "fbref_id": "9/schedule/Premier-League-Scores-and-Fixtures"},
    "La Liga": {"csv": "SP1", "fbref_id": "12/schedule/La-Liga-Scores-and-Fixtures"},
    "Bundesliga": {"csv": "D1", "fbref_id": "20/schedule/Bundesliga-Scores-and-Fixtures"},
    "Serie A": {"csv": "I1", "fbref_id": "11/schedule/Serie-A-Scores-and-Fixtures"},
    "Ligue 1": {"csv": "F1", "fbref_id": "13/schedule/Ligue-1-Scores-and-Fixtures"}
}

st.set_page_config(page_title="Football Predictor Pro 2026", layout="wide")

# --- SIDEBAR ---
st.sidebar.header("🌍 Wybór Rozgrywek")
wybrana_liga = st.sidebar.selectbox("Wybierz ligę", list(LIGI_CONFIG.keys()))
kod_csv = LIGI_CONFIG[wybrana_liga]["csv"]
fbref_path = LIGI_CONFIG[wybrana_liga]["fbref_id"]

st.title(f"Analiza Analityczna: {wybrana_liga}")
st.markdown(f"Model: Poisson + Wagi Formy + Auto-Schedule (Scraping)")

# --- MAPOWANIE NAZW (Dostosowanie źródeł) ---
NAZWY_MAP = {
    "Manchester Utd": "Man United",
    "Manchester City": "Man City",
    "Newcastle Utd": "Newcastle",
    "Nott'ham Forest": "Nott'm Forest",
    "Sheffield Utd": "Sheffield Utd",
    "Brighton & Hove Albion": "Brighton",
    "West Ham United": "West Ham",
    "Tottenham Hotspur": "Tottenham",
    "Wolverhampton Wanderers": "Wolves",
    "Leicester City": "Leicester",
    "Ipswich Town": "Ipswich"
}

# --- DANE HISTORYCZNE (CSV) ---
@st.cache_data(ttl=3600)
def load_historical(league_code):
    try:
        url = f"https://www.football-data.co.uk/mmz4281/2526/{league_code}.csv"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df = df.sort_values('Date')
        df['total_gole'] = df['FTHG'] + df['FTAG']
        df['total_kartki'] = df['HY'] + df['AY'] + (df['HR'] + df['AR']) * 2
        df['total_rozne'] = df['HC'] + df['AC']
        return df
    except Exception as e:
        st.error(f"Nie udało się pobrać danych historycznych: {e}")
        return pd.DataFrame()

# --- AUTOMATYCZNY TERMINARZ (SCRAPING) ---
@st.cache_data(ttl=43200) # Odświeżaj co 12h
def load_schedule_auto(path):
    try:
        url = f"https://fbref.com/en/comps/{path}"
        # Pobieranie tabel ze strony
        all_tables = pd.read_html(url)
        df = all_tables[0]
        
        # Filtrowanie i czyszczenie
        df = df[['Date', 'Home', 'Away', 'Wk']].dropna(subset=['Home', 'Away'])
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Pobieramy mecze od wczoraj do +7 dni
        dzis = datetime.now()
        mask = (df['Date'] >= (dzis - timedelta(days=1))) & (df['Date'] <= (dzis + timedelta(days=7)))
        return df.loc[mask].sort_values('Date')
    except Exception as e:
        st.warning(f"Błąd automatycznego terminarza: {e}. Upewnij się, że masz połączenie z internetem.")
        return pd.DataFrame()

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
        if len(home) < 2 or len(away) < 2: continue

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
        table[home]["gf"] += hg; table[home]["ga"] += ag; table[home]["played"] += 1
        table[away]["gf"] += ag; table[away]["ga"] += hg; table[away]["played"] += 1
        if hg > ag: table[home]["pts"] += 3
        elif hg < ag: table[away]["pts"] += 3
        else: table[home]["pts"] += 1; table[away]["pts"] += 1
    res = pd.DataFrame(table).T
    res["diff"] = res["gf"] - res["ga"]
    return res.sort_values(["pts","diff","gf"], ascending=False)

def koloruj(p):
    if p > 0.65: return "🟢"
    elif p > 0.50: return "🟡"
    else: return "🔴"

# --- GŁÓWNY PROCES ---
historical = load_historical(kod_csv)
schedule = load_schedule_auto(fbref_path)

if not historical.empty:
    srednie_df = oblicz_wszystkie_statystyki(historical)
    forma_dict = oblicz_forme(historical)
    tabela = tabela_ligowa(historical)
    
    tab1, tab2, tab3 = st.tabs(["🎯 Bet Builder & Predykcje", "📊 Tabela i Forma", "📈 Statystyki Modelu"])

    with tab1:
        st.subheader("🎛️ Zbuduj własne combo")
        c1, c2, c3 = st.columns(3)
        with c1:
            linia_gole = st.selectbox("Linia goli", [1.5, 2.5, 3.5], index=1)
            typ_gole = st.selectbox("Typ goli", ["Over", "Under"])
        with c2:
            linia_rogi = st.selectbox("Linia rożnych", [7.5, 8.5, 9.5, 10.5, 11.5], index=1)
            typ_rogi = st.selectbox("Typ rożnych", ["Over", "Under"])
        with c3:
            linia_kartki = st.selectbox("Linia kartek", [2.5, 3.5, 4.5, 5.5], index=1)
            typ_kartki = st.selectbox("Typ kartek", ["Over", "Under"])
        
        min_prob = st.slider("Minimalne prawdopodobieństwo combo", 0.1, 0.9, 0.40, 0.05)

        st.subheader("📅 Nadchodzące mecze (Następne 7 dni)")
        
        if not schedule.empty:
            col_pred1, col_pred2 = st.columns(2)
            
            with col_pred1:
                st.write("**Combo Builder**")
                for _, mecz in schedule.iterrows():
                    # Mapowanie nazw ze scraping-u na CSV
                    h = NAZWY_MAP.get(mecz['Home'], mecz['Home'])
                    a = NAZWY_MAP.get(mecz['Away'], mecz['Away'])
                    
                    if h in srednie_df.index and a in srednie_df.index:
                        # Obliczanie parametrów Lambda
                        l_h = (srednie_df.loc[h, "Gole strzelone (dom)"] + srednie_df.loc[a, "Gole stracone (wyjazd)"]) / 2
                        l_a = (srednie_df.loc[a, "Gole strzelone (wyjazd)"] + srednie_df.loc[h, "Gole stracone (dom)"]) / 2
                        l_r = (srednie_df.loc[h, "Różne (dom)"] + srednie_df.loc[a, "Różne (wyjazd)"]) / 2
                        l_k = (srednie_df.loc[h, "Kartki (dom)"] + srednie_df.loc[a, "Kartki (wyjazd)"]) / 2

                        def get_p(typ, linia, lam):
                            return 1 - poisson.cdf(linia, lam) if typ == "Over" else poisson.cdf(linia, lam)

                        p_g = get_p(typ_gole, linia_gole, l_h + l_a)
                        p_r = get_p(typ_rogi, linia_rogi, l_r)
                        p_k = get_p(typ_kartki, linia_kartki, l_k)
                        p_combo = p_g * p_r * p_k

                        if p_combo >= min_prob:
                            with st.expander(f"🏟️ {h} vs {a} ({p_combo:.1%}) - {mecz['Date'].strftime('%d.%m')}"):
                                st.write(f"{koloruj(p_g)} Gole {typ_gole} {linia_gole}: {p_g:.1%}")
                                st.write(f"{koloruj(p_r)} Rożne {typ_rogi} {linia_rogi}: {p_r:.1%}")
                                st.write(f"{koloruj(p_k)} Kartki {typ_kartki} {linia_kartki}: {p_k:.1%}")
                                st.caption(f"Przewidywany xG: {l_h:.2f} - {l_a:.2f}")

            with col_pred2:
                st.write("**BTTS Ranking**")
                for _, mecz in schedule.iterrows():
                    h = NAZWY_MAP.get(mecz['Home'], mecz['Home'])
                    a = NAZWY_MAP.get(mecz['Away'], mecz['Away'])
                    if h in srednie_df.index and a in srednie_df.index:
                        l_h = (srednie_df.loc[h, "Gole strzelone (dom)"] + srednie_df.loc[a, "Gole stracone (wyjazd)"]) / 2
                        l_a = (srednie_df.loc[a, "Gole strzelone (wyjazd)"] + srednie_df.loc[h, "Gole stracone (dom)"]) / 2
                        p_btts = (1 - poisson.pmf(0, l_h)) * (1 - poisson.pmf(0, l_a))
                        st.write(f"{koloruj(p_btts)} **{h} - {a}**: {p_btts:.1%}")
        else:
            st.info("Brak nadchodzących meczów w terminarzu (zakres 7 dni).")

    with tab2:
        st.subheader("📊 Aktualna Sytuacja")
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
        
        if st.button("🔄 Wymuś odświeżenie danych"):
            st.cache_data.clear()
            st.rerun()
