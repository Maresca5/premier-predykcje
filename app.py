import streamlit as st
import pandas as pd
from scipy.stats import poisson
from datetime import datetime
import requests
from io import StringIO
import numpy as np
import soccerdata as sd

# ==========================================
# KONFIGURACJA STRONY
# ==========================================
st.set_page_config(page_title="Predykcje Top 5 Lig", layout="wide")

# ==========================================
# KONFIGURACJA LIG
# ==========================================
LEAGUES = {
    "Premier League": {
        "name": "Premier League",
        "country": "Anglia",
        "icon": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
        "code": "E0",
        "league_id": "ENG-Premier League",  # Format dla Sofascore
        "season": "2025-2026",
        "data_url": "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
    },
    "La Liga": {
        "name": "La Liga",
        "country": "Hiszpania",
        "icon": "🇪🇸",
        "code": "SP1",
        "league_id": "ESP-La Liga",
        "season": "2025-2026",
        "data_url": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv"
    },
    "Bundesliga": {
        "name": "Bundesliga",
        "country": "Niemcy",
        "icon": "🇩🇪",
        "code": "D1",
        "league_id": "GER-Bundesliga",
        "season": "2025-2026",
        "data_url": "https://www.football-data.co.uk/mmz4281/2526/D1.csv"
    },
    "Serie A": {
        "name": "Serie A",
        "country": "Włochy",
        "icon": "🇮🇹",
        "code": "I1",
        "league_id": "ITA-Serie A",
        "season": "2025-2026",
        "data_url": "https://www.football-data.co.uk/mmz4281/2526/I1.csv"
    },
    "Ligue 1": {
        "name": "Ligue 1",
        "country": "Francja",
        "icon": "🇫🇷",
        "code": "F1",
        "league_id": "FRA-Ligue 1",
        "season": "2025-2026",
        "data_url": "https://www.football-data.co.uk/mmz4281/2526/F1.csv"
    }
}

# ==========================================
# MAPOWANIE NAZW DRUŻYN (rozszerzone)
# ==========================================
NAZWY_MAP = {
    # Premier League
    "Brighton & Hove Albion": "Brighton",
    "West Ham United": "West Ham",
    "Newcastle United": "Newcastle",
    "Tottenham Hotspur": "Tottenham",
    "Leeds United": "Leeds",
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "Wolverhampton": "Wolves",
    "Sheffield United": "Sheffield Utd",
    "Leicester City": "Leicester",
    "Southampton": "Southampton",
    "Crystal Palace": "Crystal Palace",
    
    # La Liga
    "Athletic Club": "Athletic Bilbao",
    "Atlético Madrid": "Atletico Madrid",
    "Barcelona": "Barcelona",
    "Real Madrid": "Real Madrid",
    "Real Sociedad": "Real Sociedad",
    "Real Betis": "Betis",
    "Villarreal": "Villarreal",
    "Valencia": "Valencia",
    "Sevilla": "Sevilla",
    "Rayo Vallecano": "Rayo Vallecano",
    "RCD Mallorca": "Mallorca",
    "RCD Espanyol": "Espanyol",
    "Getafe": "Getafe",
    "Celta Vigo": "Celta",
    "Girona": "Girona",
    "Deportivo Alavés": "Alaves",
    "UD Las Palmas": "Las Palmas",
    "CA Osasuna": "Osasuna",
    
    # Bundesliga
    "Bayern Munich": "Bayern Monachium",
    "Borussia Dortmund": "Borussia Dortmund",
    "RB Leipzig": "RB Leipzig",
    "Bayer Leverkusen": "Bayer Leverkusen",
    "Eintracht Frankfurt": "Eintracht",
    "VfB Stuttgart": "Stuttgart",
    "VfL Wolfsburg": "Wolfsburg",
    "Borussia Mönchengladbach": "Gladbach",
    "1. FC Union Berlin": "Union Berlin",
    "SC Freiburg": "Freiburg",
    "1. FC Köln": "FC Koln",
    "FSV Mainz 05": "Mainz",
    "TSG Hoffenheim": "Hoffenheim",
    "Werder Bremen": "Werder Bremen",
    "VfL Bochum": "Bochum",
    "FC Augsburg": "Augsburg",
    
    # Serie A
    "AC Milan": "AC Milan",
    "Inter": "Inter",
    "Juventus": "Juventus",
    "AS Roma": "Roma",
    "Lazio": "Lazio",
    "Napoli": "Napoli",
    "Atalanta": "Atalanta",
    "Fiorentina": "Fiorentina",
    "Torino": "Torino",
    "Bologna": "Bologna",
    "Udinese": "Udinese",
    "Sassuolo": "Sassuolo",
    "Empoli": "Empoli",
    "Salernitana": "Salernitana",
    "Lecce": "Lecce",
    
    # Ligue 1
    "Paris Saint-Germain": "PSG",
    "Marseille": "Marseille",
    "Monaco": "Monaco",
    "Lyon": "Lyon",
    "Lille": "Lille",
    "Rennes": "Rennes",
    "Nice": "Nice",
    "Lens": "Lens",
    "Strasbourg": "Strasbourg",
    "Nantes": "Nantes",
    "Montpellier": "Montpellier",
    "Toulouse": "Toulouse",
    "Reims": "Reims",
    "Brest": "Brest",
    "Clermont": "Clermont",
    "Auxerre": "Auxerre"
}

# ==========================================
# FUNKCJE POMOCNICZE
# ==========================================
def weighted_mean(values):
    """Oblicza średnią ważoną z wagami liniowo rosnącymi (1 → 2)"""
    if len(values) == 0:
        return 0
    weights = np.linspace(1, 2, len(values))
    return np.average(values, weights=weights)

def koloruj(p):
    """Zwraca emoji w zależności od prawdopodobieństwa"""
    if p > 0.65:
        return "🟢"
    elif p > 0.50:
        return "🟡"
    else:
        return "🔴"

def oblicz_prob_poisson(typ, linia, lam):
    """Oblicza prawdopodobieństwo Over/Under dla rozkładu Poissona"""
    if typ == "Over":
        return 1 - poisson.cdf(linia, lam)
    else:
        return poisson.cdf(linia, lam)

# ==========================================
# WCZYTYWANIE DANYCH HISTORYCZNYCH
# ==========================================
@st.cache_data(ttl=900)
def load_historical(league_code, data_url):
    """Pobiera dane historyczne dla wybranej ligi"""
    try:
        r = requests.get(data_url, timeout=10)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df = df.sort_values('Date')
        
        # Oblicz dodatkowe statystyki
        df['total_gole'] = df['FTHG'] + df['FTAG']
        df['total_kartki'] = df['HY'] + df['AY'] + df['HR']*2 + df['AR']*2
        df['total_rozne'] = df['HC'] + df['AC']
        df['btts'] = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int)
        df['over25'] = (df['total_gole'] > 2.5).astype(int)
        
        return df
    except Exception as e:
        st.error(f"Nie udało się pobrać danych historycznych dla {league_code}: {e}")
        return pd.DataFrame()

# ==========================================
# POBIERANIE TERMINARZA (SOFASCORE)
# ==========================================
@st.cache_data(ttl=3600)  # Cache na 1 godzinę
def load_schedule_sofascore(league_id, season):
    """Pobiera terminarz dla wybranej ligi używając soccerdata/Sofascore"""
    try:
        # Inicjalizacja scrapera Sofascore
        sofascore = sd.Sofascore(leagues=league_id, seasons=season)
        
        # Pobierz terminarz
        schedule_df = sofascore.read_schedule()
        
        if schedule_df is not None and not schedule_df.empty:
            # Reset index, żeby pozbyć się MultiIndex
            schedule_df = schedule_df.reset_index()
            
            # Przygotuj dane w formacie zgodnym z aplikacją
            schedule = pd.DataFrame({
                'date': pd.to_datetime(schedule_df['date']),
                'round': schedule_df['week'].astype(int),
                'home_team': schedule_df['home_team'],
                'away_team': schedule_df['away_team']
            })
            
            return schedule.sort_values('date')
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.warning(f"Nie udało się pobrać terminarza z Sofascore: {e}")
        return pd.DataFrame()

# ==========================================
# FUNKCJE OBLICZENIOWE
# ==========================================
@st.cache_data
def oblicz_wszystkie_statystyki(df):
    """Oblicza średnie statystyki dla każdej drużyny"""
    druzyny = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel())
    dane = {}

    for d in druzyny:
        home = df[df['HomeTeam'] == d].tail(10)
        away = df[df['AwayTeam'] == d].tail(10)

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

def oblicz_forme(df):
    """Oblicza formę drużyny (ostatnie 5 meczów)"""
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
    """Tworzy aktualną tabelę ligową"""
    table = {}
    for _, m in df.iterrows():
        home, away = m['HomeTeam'], m['AwayTeam']
        hg, ag = m['FTHG'], m['FTAG']
        for team in [home, away]:
            if team not in table: 
                table[team] = {"pts": 0, "gf": 0, "ga": 0, "played": 0}
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
    res = pd.DataFrame(table).T
    res["diff"] = res["gf"] - res["ga"]
    return res.sort_values(["pts", "diff", "gf"], ascending=False)

# ==========================================
# INTERFEJS UŻYTKOWNIKA
# ==========================================

# Wybór ligi w sidebarze
with st.sidebar:
    st.header("🏆 Wybierz ligę")
    
    selected_league = st.selectbox(
        "Liga:",
        options=list(LEAGUES.keys()),
        format_func=lambda x: f"{LEAGUES[x]['icon']} {x}"
    )
    
    league_config = LEAGUES[selected_league]
    
    st.caption(f"🇪🇺 Kraj: {league_config['country']}")
    st.caption(f"📅 Sezon: {league_config['season']}")
    
    st.divider()
    
    if st.button("🔄 Odśwież dane"):
        st.cache_data.clear()
        st.rerun()

# Główny tytuł z nazwą wybranej ligi
st.title(f"{league_config['icon']} Predykcje {selected_league} 2025/26")
st.markdown("Model Poissona + home/away + wagi formy")

# Pobierz dane dla wybranej ligi
historical = load_historical(league_config['code'], league_config['data_url'])
schedule = load_schedule_sofascore(league_config['league_id'], league_config['season'])

if historical.empty:
    st.error("Nie udało się pobrać danych historycznych. Spróbuj później.")
    st.stop()

# Oblicz statystyki
srednie_df = oblicz_wszystkie_statystyki(historical)
forma_dict = oblicz_forme(historical)
tabela = tabela_ligowa(historical)

# ==========================================
# INTERFEJS Z ZAKŁADKAMI
# ==========================================
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

    st.subheader("📅 Predykcje – najbliższa kolejka")
    dzisiaj = datetime.now()
    
    if not schedule.empty:
        # Filtruj nadchodzące mecze
        nadchodzace = schedule[schedule['date'] > dzisiaj]
        
        if not nadchodzace.empty:
            min_round = nadchodzace['round'].min()
            mecze = nadchodzace[nadchodzace['round'] == min_round]
            
            # Kontener na wyniki combo i BTTS
            col_pred1, col_pred2 = st.columns(2)
            
            with col_pred1:
                st.write("**Combo Builder**")
                combo_count = 0
                for _, mecz in mecze.iterrows():
                    h = NAZWY_MAP.get(mecz['home_team'], mecz['home_team'])
                    a = NAZWY_MAP.get(mecz['away_team'], mecz['away_team'])
                    
                    if h in srednie_df.index and a in srednie_df.index:
                        # Gole
                        lam_h = (srednie_df.loc[h, "Gole strzelone (dom)"] + srednie_df.loc[a, "Gole stracone (wyjazd)"]) / 2
                        lam_a = (srednie_df.loc[a, "Gole strzelone (wyjazd)"] + srednie_df.loc[h, "Gole stracone (dom)"]) / 2
                        lam_g = lam_h + lam_a
                        # Rożne i Kartki
                        lam_r = (srednie_df.loc[h, "Różne (dom)"] + srednie_df.loc[a, "Różne (wyjazd)"]) / 2
                        lam_k = (srednie_df.loc[h, "Kartki (dom)"] + srednie_df.loc[a, "Kartki (wyjazd)"]) / 2

                        p_g = oblicz_prob_poisson(typ_gole, linia_gole, lam_g)
                        p_r = oblicz_prob_poisson(typ_rogi, linia_rogi, lam_r)
                        p_k = oblicz_prob_poisson(typ_kartki, linia_kartki, lam_k)
                        p_combo = p_g * p_r * p_k

                        if p_combo >= min_prob:
                            combo_count += 1
                            with st.expander(f"{h} vs {a} ({p_combo:.1%})"):
                                st.write(f"{koloruj(p_g)} Gole {typ_gole} {linia_gole}: {p_g:.1%}")
                                st.write(f"{koloruj(p_r)} Rożne {typ_rogi} {linia_rogi}: {p_r:.1%}")
                                st.write(f"{koloruj(p_k)} Kartki {typ_kartki} {linia_kartki}: {p_k:.1%}")
                
                if combo_count == 0:
                    st.info("Brak meczów spełniających kryteria. Zmniejsz próg.")
            
            with col_pred2:
                st.write("**BTTS Ranking**")
                btts_data = []
                for _, mecz in mecze.iterrows():
                    h = NAZWY_MAP.get(mecz['home_team'], mecz['home_team'])
                    a = NAZWY_MAP.get(mecz['away_team'], mecz['away_team'])
                    if h in srednie_df.index and a in srednie_df.index:
                        lam_h = (srednie_df.loc[h, "Gole strzelone (dom)"] + srednie_df.loc[a, "Gole stracone (wyjazd)"]) / 2
                        lam_a = (srednie_df.loc[a, "Gole strzelone (wyjazd)"] + srednie_df.loc[h, "Gole stracone (dom)"]) / 2
                        p_btts = (1 - poisson.pmf(0, lam_h)) * (1 - poisson.pmf(0, lam_a))
                        btts_data.append({
                            "Mecz": f"{h} - {a}",
                            "BTTS": f"{p_btts:.1%}",
                            "Kolor": koloruj(p_btts)
                        })
                
                if btts_data:
                    for item in sorted(btts_data, key=lambda x: float(x["BTTS"].strip('%')), reverse=True):
                        st.write(f"{item['Kolor']} **{item['Mecz']}**: {item['BTTS']}")
        else:
            st.warning("Brak nadchodzących meczów w terminarzu.")
    else:
        st.warning("Nie udało się pobrać terminarza. Spróbuj ponownie później.")

with tab2:
    st.subheader("📊 Aktualna Sytuacja")
    c_l, c_f = st.columns([2, 1])
    with c_l:
        st.write("**Tabela Ligowa**")
        st.dataframe(tabela, use_container_width=True)
    with c_f:
        st.write("**Forma (5 ostatnich)**")
        forma_df = pd.DataFrame.from_dict(forma_dict, orient="index", columns=["Forma"])
        st.dataframe(forma_df, use_container_width=True)

with tab3:
    st.subheader("📊 Średnie ważone drużyn")
    st.write("Dane uwzględniają atut własnego boiska oraz wagę ostatnich meczów.")
    st.dataframe(srednie_df.sort_index(), use_container_width=True)
    
    # Dodaj informacje o źródle danych
    st.divider()
    st.caption(f"Źródło danych: football-data.co.uk | Sofascore via soccerdata")
    st.caption(f"Ostatnia aktualizacja danych historycznych: {historical['Date'].max().strftime('%d.%m.%Y') if not historical.empty else '—'}")