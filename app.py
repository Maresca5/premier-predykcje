import streamlit as st
import pandas as pd
from scipy.stats import poisson
from datetime import datetime
import requests
from io import StringIO
import numpy as np

# --- KONFIGURACJA LIG ---
LIGI = {
    "Premier League": {"csv_code": "E0", "file": "terminarz_premier_2025.csv"},
    "La Liga": {"csv_code": "SP1", "file": "terminarz_la_liga_2025.csv"},
    "Bundesliga": {"csv_code": "D1", "file": "terminarz_bundesliga_2025.csv"},
    "Serie A": {"csv_code": "I1", "file": "terminarz_serie_a_2025.csv"},
    "Ligue 1": {"csv_code": "F1", "file": "terminarz_ligue_1_2025.csv"}
}

st.set_page_config(page_title="Predykcje Piłkarskie Top 5", layout="wide")

# --- SIDEBAR - WYBÓR LIGI ---
st.sidebar.header("🌍 Wybór Rozgrywek")
wybrana_liga = st.sidebar.selectbox("Wybierz ligę", list(LIGI.keys()))

st.title(f"Predykcje {wybrana_liga} 2025/26")
st.markdown("Model Poissona + wagi formy")

# Rozszerzone mapowanie nazw
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
    "Everton": "Everton",
    "Aston Villa": "Aston Villa",
    "Fulham": "Fulham",
    "Brentford": "Brentford",
    "Burnley": "Burnley",
    "Luton Town": "Luton",
    "Bournemouth": "Bournemouth",
    "Wolves": "Wolves",
    
    # La 
    # --- POPRAWIONE MAPOWANIE DLA LA LIGA ---
LA_LIGA_MAP = {
    # Twoje nazwy z pliku → nazwy w football-data.co.uk
    "Rayo Vallecano": "Rayo Vallecano",  # Zostawiamy bez zmian
    "Real Sociedad": "Sociedad",  # Kluczowa zmiana!
    "Sociedad": "Sociedad",
    "Atlético Madrid": "Ath Madrid",
    "Atletico Madrid": "Ath Madrid",
    "Ath Madrid": "Ath Madrid",
    "Athletic Club": "Ath Bilbao",
    "Athletic Bilbao": "Ath Bilbao",
    "Ath Bilbao": "Ath Bilbao",
    "Barcelona": "Barcelona",
    "FC Barcelona": "Barcelona",
    "Real Madrid": "Real Madrid",
    "Real Betis": "Betis",
    "Betis": "Betis",
    "Sevilla": "Sevilla",
    "Sevilla FC": "Sevilla",
    "Valencia": "Valencia",
    "Valencia CF": "Valencia",
    "Villarreal": "Villarreal",
    "Celta Vigo": "Celta",
    "Celta": "Celta",
    "Getafe": "Getafe",
    "Getafe CF": "Getafe",
    "Osasuna": "Osasuna",
    "CA Osasuna": "Osasuna",
    "Mallorca": "Mallorca",
    "RCD Mallorca": "Mallorca",
    "Espanyol": "Espanol",  # Bez 'y'!
    "RCD Espanyol": "Espanol",
    "Espanol": "Espanol",
    "Girona FC": "Girona",
    "Girona": "Girona",
    "Alaves": "Alaves",
    "Deportivo Alavés": "Alaves",
    "Deportivo Alaves": "Alaves",
    "Levante UD": "Levante",
    "Levante": "Levante",
    "Elche": "Elche",
    "Elche CF": "Elche",
    "Real Oviedo": "Oviedo",
    "Oviedo": "Oviedo",
    "Real Valladolid": "Valladolid",
    "Valladolid": "Valladolid",
    "Las Palmas": "Las Palmas",
    "UD Las Palmas": "Las Palmas",
    "Cadiz": "Cadiz",
    "Cadiz CF": "Cadiz",
    "Granada": "Granada",
    "Granada CF": "Granada",
    "Almeria": "Almeria",
    "UD Almeria": "Almeria"
}

# --- POPRAWIONE MAPOWANIE DLA BUNDESLIGI ---
BUNDESLIGA_MAP = {
    # Twoje nazwy → nazwy w football-data.co.uk
    "Eintracht Frankfurt": "Eintracht",
    "Eintracht": "Eintracht",
    "FC Augsburg": "Augsburg",
    "Augsburg": "Augsburg",
    "VfL Wolfsburg": "Wolfsburg",
    "Wolfsburg": "Wolfsburg",
    "TSG Hoffenheim": "Hoffenheim",
    "Hoffenheim": "Hoffenheim",
    "SV Werder Bremen": "Werder Bremen",
    "Werder Bremen": "Werder Bremen",
    "Borussia Dortmund": "Dortmund",  # Ważne!
    "Dortmund": "Dortmund",
    "Borussia M'gladbach": "M'gladbach",
    "Borussia Mönchengladbach": "M'gladbach",
    "M'gladbach": "M'gladbach",
    "VfB Stuttgart": "Stuttgart",
    "Stuttgart": "Stuttgart",
    "FC St. Pauli": "St Pauli",
    "St Pauli": "St Pauli",
    "1. FC Heidenheim": "Heidenheim",
    "Heidenheim": "Heidenheim",
    "Hamburger SV": "Hamburg",  # Ważne!
    "Hamburg": "Hamburg",
    "Bayern Munich": "Bayern",
    "Bayern München": "Bayern",
    "Bayern": "Bayern",
    "RB Leipzig": "RB Leipzig",
    "Leipzig": "RB Leipzig",
    "Bayer Leverkusen": "Leverkusen",
    "Leverkusen": "Leverkusen",
    "SC Freiburg": "Freiburg",
    "Freiburg": "Freiburg",
    "1. FC Union Berlin": "Union Berlin",
    "Union Berlin": "Union Berlin",
    "1. FC Köln": "Köln",
    "FC Köln": "Köln",
    "Köln": "Köln",
    "FSV Mainz 05": "Mainz",
    "Mainz": "Mainz",
    "VfL Bochum": "Bochum",
    "Bochum": "Bochum"
}

# --- POPRAWIONE MAPOWANIE DLA LIGUE 1 ---
LIGUE1_MAP = {
    "Paris Saint-Germain": "Paris SG",
    "PSG": "Paris SG",
    "Paris SG": "Paris SG",
    "Olympique de Marseille": "Marseille",
    "Marseille": "Marseille",
    "AS Monaco": "Monaco",
    "Monaco": "Monaco",
    "Olympique Lyonnais": "Lyon",
    "Lyon": "Lyon",
    "LOSC Lille": "Lille",
    "Lille": "Lille",
    "Stade Rennais": "Rennes",
    "Rennes": "Rennes",
    "OGC Nice": "Nice",
    "Nice": "Nice",
    "RC Lens": "Lens",
    "Lens": "Lens",
    "RC Strasbourg": "Strasbourg",
    "Strasbourg": "Strasbourg",
    "FC Nantes": "Nantes",
    "Nantes": "Nantes",
    "Montpellier HSC": "Montpellier",
    "Montpellier": "Montpellier",
    "FC Toulouse": "Toulouse",
    "Toulouse": "Toulouse",
    "Stade Reims": "Reims",
    "Reims": "Reims",
    "Stade Brestois": "Brest",
    "Brest": "Brest",
    "Clermont Foot": "Clermont",
    "Clermont": "Clermont",
    "AJ Auxerre": "Auxerre",
    "Auxerre": "Auxerre",
    "Le Havre": "Le Havre",
    "Havre": "Le Havre",
    "FC Metz": "Metz",
    "Metz": "Metz"
}
    # Bundesliga
    "Bayern Munich": "Bayern Monachium",
    "Bayern München": "Bayern Monachium",
    "Borussia Dortmund": "Borussia Dortmund",
    "RB Leipzig": "RB Leipzig",
    "Bayer Leverkusen": "Bayer Leverkusen",
    "Eintracht Frankfurt": "Eintracht",
    "VfB Stuttgart": "Stuttgart",
    "VfL Wolfsburg": "Wolfsburg",
    "Borussia Mönchengladbach": "Gladbach",
    "1. FC Union Berlin": "Union Berlin",
    "Union Berlin": "Union Berlin",
    "SC Freiburg": "Freiburg",
    "1. FC Köln": "FC Koln",
    "FC Köln": "FC Koln",
    "FSV Mainz 05": "Mainz",
    "Mainz": "Mainz",
    "TSG Hoffenheim": "Hoffenheim",
    "Hoffenheim": "Hoffenheim",
    "Werder Bremen": "Werder Bremen",
    "VfL Bochum": "Bochum",
    "FC Augsburg": "Augsburg",
    "FC St. Pauli": "St Pauli",
    "St. Pauli": "St Pauli",
    "FC Heidenheim": "Heidenheim",
    "Heidenheim": "Heidenheim",
    
    # Serie A
    "AC Milan": "AC Milan",
    "Milan": "AC Milan",
    "Inter": "Inter",
    "Internazionale": "Inter",
    "Juventus": "Juventus",
    "AS Roma": "Roma",
    "Roma": "Roma",
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
    "Hellas Verona": "Verona",
    "Verona": "Verona",
    "Lecce": "Lecce",
    "Cagliari": "Cagliari",
    "Genoa": "Genoa",
    "Monza": "Monza",
    "Frosinone": "Frosinone",
    
    # Ligue 1
    "Paris Saint-Germain": "PSG",
    "PSG": "PSG",
    "Marseille": "Marseille",
    "Olympique Marseille": "Marseille",
    "Monaco": "Monaco",
    "AS Monaco": "Monaco",
    "Lyon": "Lyon",
    "Olympique Lyonnais": "Lyon",
    "Lille": "Lille",
    "LOSC": "Lille",
    "Rennes": "Rennes",
    "Stade Rennais": "Rennes",
    "Nice": "Nice",
    "OGC Nice": "Nice",
    "Lens": "Lens",
    "RC Lens": "Lens",
    "Strasbourg": "Strasbourg",
    "RC Strasbourg": "Strasbourg",
    "Nantes": "Nantes",
    "FC Nantes": "Nantes",
    "Montpellier": "Montpellier",
    "Montpellier HSC": "Montpellier",
    "Toulouse": "Toulouse",
    "FC Toulouse": "Toulouse",
    "Reims": "Reims",
    "Stade Reims": "Reims",
    "Brest": "Brest",
    "Stade Brestois": "Brest",
    "Clermont": "Clermont",
    "Clermont Foot": "Clermont",
    "Auxerre": "Auxerre",
    "AJ Auxerre": "Auxerre",
    "Le Havre": "Le Havre",
    "Havre": "Le Havre",
    "Metz": "Metz",
    "FC Metz": "Metz"
}

# --- FUNKCJE DANYCH ---
@st.cache_data(ttl=900)
def load_historical(league_code):
    try:
        url = f"https://www.football-data.co.uk/mmz4281/2526/{league_code}.csv"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        # Konwersja daty z obsługą błędów
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam'])
        df = df.sort_values('Date')
        df['total_gole'] = df['FTHG'] + df['FTAG']
        df['total_kartki'] = df['HY'] + df['AY'] + (df['HR'] + df['AR']) * 2
        df['total_rozne'] = df['HC'] + df['AC']
        return df
    except Exception as e:
        st.error(f"Błąd pobierania danych historycznych: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def load_schedule(filename):
    try:
        df = pd.read_csv(filename)
        
        # Kluczowa poprawka: obsługa daty z strefą czasową
        df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
        
        # Jeśli brak kolumny 'round', utwórz ją
        if 'round' not in df.columns:
            df = df.sort_values('date')
            unique_dates = df['date'].dt.date.unique()
            date_to_round = {date: i+1 for i, date in enumerate(unique_dates)}
            df['round'] = df['date'].dt.date.map(date_to_round)
        
        # Debug: pokaż pierwsze kilka wierszy w sidebarze
        with st.sidebar.expander(f"🔍 Debug {filename}"):
            st.write(f"Liczba meczów: {len(df)}")
            st.write(f"Kolumny: {list(df.columns)}")
            st.write(f"Data zakres: {df['date'].min()} do {df['date'].max()}")
            st.write(df[['date', 'home_team', 'away_team', 'round']].head())
        
        return df.dropna(subset=['date']).sort_values('date')
    except Exception as e:
        st.error(f"Problem z plikiem terminarza {filename}: {e}")
        return pd.DataFrame()

# Pobierz dane
historical = load_historical(LIGI[wybrana_liga]["csv_code"])
schedule = load_schedule(LIGI[wybrana_liga]["file"])

# --- OBLICZENIA ---
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

def koloruj(p):
    return "🟢" if p > 0.65 else "🟡" if p > 0.50 else "🔴"

# --- RENDEROWANIE ---
if not historical.empty:
    srednie_df = oblicz_wszystkie_statystyki(historical)
    forma_dict = oblicz_forme(historical)
    tabela = tabela_ligowa(historical)
    
    # Debug: pokaż dopasowanie nazw
    with st.sidebar.expander("🔍 Dopasowanie nazw drużyn"):
        if not schedule.empty:
            for _, mecz in schedule.iterrows():
                h = mecz['home_team']
                a = mecz['away_team']
                h_map = NAZWY_MAP.get(h, h)
                a_map = NAZWY_MAP.get(a, a)
                h_ok = h_map in srednie_df.index
                a_ok = a_map in srednie_df.index
                if not h_ok or not a_ok:
                    st.write(f"❌ {h} → {h_map}: {h_ok} | {a} → {a_map}: {a_ok}")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Bet Builder & Predykcje", "📊 Tabela i Forma", "📈 Statystyki Modelu"])
    
    with tab1:
        st.subheader("🎛️ Zbuduj własne combo")
        c1, c2, c3 = st.columns(3)
        with c1:
            linia_gole = st.selectbox("Linia goli", [1.5, 2.5, 3.5], index=1)
            typ_gole = st.selectbox("Typ goli", ["Over", "Under"], index=0)
        with c2:
            linia_rogi = st.selectbox("Linia rożnych", [7.5, 8.5, 9.5, 10.5], index=1)
            typ_rogi = st.selectbox("Typ rożnych", ["Over", "Under"], index=0)
        with c3:
            linia_kartki = st.selectbox("Linia kartek", [2.5, 3.5, 4.5, 5.5], index=1)
            typ_kartki = st.selectbox("Typ kartek", ["Over", "Under"], index=0)
        
        min_prob = st.slider("Minimalne prawdopodobieństwo combo", 0.0, 1.0, 0.40, 0.05)
        
        st.subheader("📅 Predykcje – najbliższa kolejka")
        
        # Poprawka: usuwamy porównanie z godzinami, tylko data
        dzisiaj = datetime.now().date()
        
        if not schedule.empty:
            # Konwertuj datetime na date dla porównania
            schedule['date_only'] = schedule['date'].dt.date
            
            # Filtrujemy mecze od dzisiaj wzwyż
            nadchodzace = schedule[schedule['date_only'] >= dzisiaj]
            
            st.caption(f"Znaleziono {len(nadchodzace)} nadchodzących meczów")
            
            if not nadchodzace.empty:
                min_round = nadchodzace['round'].min()
                mecze = nadchodzace[nadchodzace['round'] == min_round]
                
                st.caption(f"Kolejka {min_round} – {len(mecze)} meczów")
                
                col1, col2 = st.columns(2)
                
                # Combo Builder
                with col1:
                    st.write("**Combo Builder**")
                    combo_count = 0
                    for _, mecz in mecze.iterrows():
                        h = NAZWY_MAP.get(mecz['home_team'], mecz['home_team'])
                        a = NAZWY_MAP.get(mecz['away_team'], mecz['away_team'])
                        
                        if h in srednie_df.index and a in srednie_df.index:
                            lam_h = (srednie_df.loc[h, "Gole strzelone (dom)"] + srednie_df.loc[a, "Gole stracone (wyjazd)"]) / 2
                            lam_a = (srednie_df.loc[a, "Gole strzelone (wyjazd)"] + srednie_df.loc[h, "Gole stracone (dom)"]) / 2
                            lam_r = (srednie_df.loc[h, "Różne (dom)"] + srednie_df.loc[a, "Różne (wyjazd)"]) / 2
                            lam_k = (srednie_df.loc[h, "Kartki (dom)"] + srednie_df.loc[a, "Kartki (wyjazd)"]) / 2
                            
                            def oblicz_p(typ, linia, lam):
                                return 1 - poisson.cdf(linia, lam) if typ == "Over" else poisson.cdf(linia, lam)
                            
                            p_g = oblicz_p(typ_gole, linia_gole, lam_h + lam_a)
                            p_r = oblicz_p(typ_rogi, linia_rogi, lam_r)
                            p_k = oblicz_p(typ_kartki, linia_kartki, lam_k)
                            p_combo = p_g * p_r * p_k
                            
                            if p_combo >= min_prob:
                                combo_count += 1
                                with st.expander(f"{h} vs {a} ({p_combo:.1%})"):
                                    st.write(f"{koloruj(p_g)} Gole {typ_gole} {linia_gole}: {p_g:.1%}")
                                    st.write(f"{koloruj(p_r)} Rożne {typ_rogi} {linia_rogi}: {p_r:.1%}")
                                    st.write(f"{koloruj(p_k)} Kartki {typ_kartki} {linia_kartki}: {p_k:.1%}")
                    
                    if combo_count == 0:
                        st.info("Brak meczów spełniających kryteria. Zmniejsz próg.")
                
                # BTTS Ranking
                with col2:
                    st.write("**BTTS Ranking**")
                    btts_data = []
                    for _, mecz in mecze.iterrows():
                        h = NAZWY_MAP.get(mecz['home_team'], mecz['home_team'])
                        a = NAZWY_MAP.get(mecz['away_team'], mecz['away_team'])
                        if h in srednie_df.index and a in srednie_df.index:
                            lam_h = (srednie_df.loc[h, "Gole strzelone (dom)"] + srednie_df.loc[a, "Gole stracone (wyjazd)"]) / 2
                            lam_a = (srednie_df.loc[a, "Gole strzelone (wyjazd)"] + srednie_df.loc[h, "Gole stracone (dom)"]) / 2
                            p_btts = (1 - poisson.pmf(0, lam_h)) * (1 - poisson.pmf(0, lam_a))
                            btts_data.append((f"{h} - {a}", p_btts))
                    
                    if btts_data:
                        for mecz, p in sorted(btts_data, key=lambda x: x[1], reverse=True):
                            st.write(f"{koloruj(p)} **{mecz}**: {p:.1%}")
            else:
                st.info("Brak nadchodzących meczów w najbliższym terminie.")
        else:
            st.warning("Brak danych w pliku terminarza.")
    
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
        
        st.divider()
        st.caption(f"📊 Liczba meczów w bazie: {len(historical)}")
        st.caption(f"📅 Ostatnia aktualizacja: {historical['Date'].max().strftime('%d.%m.%Y') if not historical.empty else '—'}")
        
        if st.button("🔄 Odśwież dane"):
            st.cache_data.clear()
            st.rerun()