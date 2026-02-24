import streamlit as st
import pandas as pd
from scipy.stats import poisson
from datetime import datetime, timedelta
import requests
from io import StringIO
import numpy as np
import unicodedata

--- KONFIGURACJA LIG ---

LIGI = {
"Premier League": {"csv_code": "E0", "file": "terminarz_premier_2025.csv"},
"La Liga": {"csv_code": "SP1", "file": "terminarz_la_liga_2025.csv"},
"Bundesliga": {"csv_code": "D1", "file": "terminarz_bundesliga_2025.csv"},
"Serie A": {"csv_code": "I1", "file": "terminarz_serie_a_2025.csv"},
"Ligue 1": {"csv_code": "F1", "file": "terminarz_ligue_1_2025.csv"}
}

st.set_page_config(page_title="Predykcje Piłkarskie Top 5", layout="wide")

--- SIDEBAR - WYBÓR LIGI ---

st.sidebar.header("🌍 Wybór Rozgrywek")
wybrana_liga = st.sidebar.selectbox("Wybierz ligę", list(LIGI.keys()))

st.title(f"Predykcje {wybrana_liga} 2025/26")
st.markdown("Model Poissona + wagi formy")

--- FUNKCJA NORMALIZACJI NAZW ---

def normalize_name(name):
if not isinstance(name, str): return str(name)
# Usuwanie akcentów i ogonków (np. é -> e)
name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
# Usuwanie typowych skrótów
for suffix in [" FC", " CF", " UD", " CD", " RCD", " AS", " Stade", " 1."]:
name = name.replace(suffix, "")
return name.strip()

KOMPLETNE mapowanie nazw (nazwa z CSV → nazwa w football-data.co.uk)

NAZWY_MAP = {
# ===== PREMIER LEAGUE =====
"Brighton & Hove Albion": "Brighton",
"Brighton": "Brighton",
"West Ham United": "West Ham",
"West Ham": "West Ham",
"Newcastle United": "Newcastle",
"Newcastle": "Newcastle",
"Tottenham Hotspur": "Tottenham",
"Tottenham": "Tottenham",
"Leeds United": "Leeds",
"Leeds": "Leeds",
"Manchester United": "Man United",
"Man United": "Man United",
"Manchester City": "Man City",
"Man City": "Man City",
"Nottingham Forest": "Nott'm Forest",
"Nott'm Forest": "Nott'm Forest",
"Wolverhampton Wanderers": "Wolves",
"Wolverhampton": "Wolves",
"Wolves": "Wolves",
"Leicester City": "Leicester",
"Leicester": "Leicester",
"Southampton": "Southampton",
"Crystal Palace": "Crystal Palace",
"Everton": "Everton",
"Aston Villa": "Aston Villa",
"Fulham": "Fulham",
"Brentford": "Brentford",
"Burnley": "Burnley",
"Bournemouth": "Bournemouth",
"Sheffield United": "Sheffield Utd",
"Sheffield Utd": "Sheffield Utd",

# ===== LA LIGA =====  
"Girona FC": "Girona",  
"Girona": "Girona",  
"Rayo Vallecano": "Vallecano",  
"Vallecano": "Rayo Vallecano",  
"Villarreal": "Villarreal",  
"Real Oviedo": "Oviedo",  
"Oviedo": "Real Oviedo",  
"Mallorca": "Mallorca",  
"RCD Mallorca": "Mallorca",  
"Barcelona": "Barcelona",  
"FC Barcelona": "Barcelona",  
"Deportivo Alavés": "Alaves",  
"Deportivo Alaves": "Alaves",  
"Alaves": "Alaves",  
"Levante UD": "Levante",  
"Levante": "Levante",  
"Valencia": "Valencia",  
"Valencia CF": "Valencia",  
"Real Sociedad": "Sociedad",  
"Sociedad": "Real Sociedad",  
"Celta Vigo": "Celta",  
"Celta": "Celta",  
"Getafe": "Getafe",  
"Getafe CF": "Getafe",  
"Athletic Club": "Ath Bilbao",  
"Athletic Bilbao": "Ath Bilbao",  
"Ath Bilbao": "Ath Bilbao",  
"Sevilla": "Sevilla",  
"Sevilla FC": "Sevilla",  
"Espanyol": "Espanol",  
"Esponol": "Espanyol",  
"RCD Espanyol": "Espanol",  
"Atlético Madrid": "Ath Madrid",  
"Ath Madrid": "Atlético Madrid",  
"Atletico Madrid": "Ath Madrid",  
"Ath Madrid": "Ath Madrid",  
"Elche": "Elche",  
"Elche CF": "Elche",  
"Real Betis": "Betis",  
"Betis": "Betis",  
"Real Madrid": "Real Madrid",  
"Osasuna": "Osasuna",  
"CA Osasuna": "Osasuna",  
"Real Valladolid": "Valladolid",  
"Valladolid": "Valladolid",  
"UD Las Palmas": "Las Palmas",  
"Las Palmas": "Las Palmas",  
"Cadiz": "Cadiz",  
"Cadiz CF": "Cadiz",  
"Granada": "Granada",  
"Granada CF": "Granada",  
"Almeria": "Almeria",  
"UD Almeria": "Almeria",  
  
# ===== BUNDESLIGA =====  
"Bayern Munich": "Bayern Munich",  
"Bayern": "Bayern Munich",  
"FC Bayern München": "Bayern Munich",  
"Borussia Dortmund": "Dortmund",  
"Dortmund": "Dortmund",  
"RB Leipzig": "RB Leipzig",  
"Leipzig": "RB Leipzig",  
"Bayer 04 Leverkusen": "Leverkusen",  
"Leverkusen": "Leverkusen",  
"Eintracht Frankfurt": "Ein Frankfurt",  
"Eintracht": "Eintracht",  
"VfB Stuttgart": "Stuttgart",  
"Stuttgart": "Stuttgart",  
"VfL Wolfsburg": "Wolfsburg",  
"Wolfsburg": "Wolfsburg",  
"Borussia Mönchengladbach": "M'gladbach",  
"Borussia Monchengladbach": "M'gladbach",  
"Borussia M'gladbach": "M'gladbach",  
"M'gladbach": "M'gladbach",  
"1. FC Union Berlin": "Union Berlin",  
"Union Berlin": "Union Berlin",  
"SC Freiburg": "Freiburg",  
"Freiburg": "Freiburg",  
"1. FC Köln": "FC Koln",  
"FC Köln": "Koln",  
"Koln": "FC Köln",  
"Köln": "Köln",  
"1. FSV Mainz 05": "Mainz",  
"Mainz": "FSV Mainz 05",  
"Mainz": "Mainz",  
"TSG Hoffenheim": "Hoffenheim",  
"Hoffenheim": "Hoffenheim",  
"Werder Bremen": "Werder Bremen",  
"SV Werder Bremen": "Werder Bremen",  
"VfL Bochum": "Bochum",  
"Bochum": "Bochum",  
"FC Augsburg": "Augsburg",  
"Augsburg": "Augsburg",  
"FC St. Pauli": "St Pauli",  
"St. Pauli": "St Pauli",  
"St Pauli": "St Pauli",  
"1. FC Heidenheim": "Heidenheim",  
"Heidenheim": "Heidenheim",  
"Hamburger SV": "Hamburg",  
"Hamburg": "Hamburg",  
  
# ===== SERIE A =====  
"AC Milan": "AC Milan",  
"Milan": "AC Milan",  
"Milan": "Milan",  
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
  
# ===== LIGUE 1 =====  
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
"Stade de Reims": "Reims",  
"Reims": "Reims",  
"Stade Brestois 29": "Brest",  
"Stade Brestois": "Brest",  
"Brest": "Brest",  
"Clermont Foot": "Clermont",  
"Clermont": "Clermont",  
"AJ Auxerre": "Auxerre",  
"Auxerre": "Auxerre",  
"Le Havre AC": "Le Havre",  
"Le Havre": "Le Havre",  
"FC Metz": "Metz",  
"Metz": "Metz"

}

--- FUNKCJA DO MAPOWANIA Z FALLBACKIEM ---

def map_nazwa(nazwa_z_csv):
"""Próbuje różnych sposobów dopasowania nazwy drużyny"""
if not isinstance(nazwa_z_csv, str):
return str(nazwa_z_csv)

# 1. Bezpośrednie mapowanie  
if nazwa_z_csv in NAZWY_MAP:  
    return NAZWY_MAP[nazwa_z_csv]  
  
# 2. Sprawdź bez spacji i myślników  
uproszczona = nazwa_z_csv.replace(" ", "").replace("-", "").replace("'", "")  
for key, value in NAZWY_MAP.items():  
    if key.replace(" ", "").replace("-", "").replace("'", "") == uproszczona:  
        return value  
  
# 3. Znormalizowana wersja  
znormalizowana = normalize_name(nazwa_z_csv)  
if znormalizowana in NAZWY_MAP:  
    return NAZWY_MAP[znormalizowana]  
  
# 4. Jeśli nic nie pasuje, zwróć oryginał  
return nazwa_z_csv

--- FUNKCJE DANYCH ---

@st.cache_data(ttl=900)
def load_historical(league_code):
try:
url = f"https://www.football-data.co.uk/mmz4281/2526/{league_code}.csv"
r = requests.get(url, timeout=10)
r.raise_for_status()
df = pd.read_csv(StringIO(r.text))
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
df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)

# Jeśli brak kolumny 'round', utwórz ją na podstawie dat  
    if 'round' not in df.columns:  
        df = df.sort_values('date')  
        unique_dates = df['date'].dt.date.unique()  
        date_to_round = {date: i+1 for i, date in enumerate(unique_dates)}  
        df['round'] = df['date'].dt.date.map(date_to_round)  
      
    return df.dropna(subset=['date']).sort_values('date')  
except Exception as e:  
    st.error(f"Problem z plikiem terminarza {filename}: {e}")  
    return pd.DataFrame()

Pobierz dane

historical = load_historical(LIGI[wybrana_liga]["csv_code"])
schedule = load_schedule(LIGI[wybrana_liga]["file"])

--- OBLICZENIA ---

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

--- RENDEROWANIE ---

if not historical.empty:
# Oblicz statystyki
srednie_df = oblicz_wszystkie_statystyki(historical)
forma_dict = oblicz_forme(historical)
tabela = tabela_ligowa(historical)

# --- GŁÓWNY INTERFEJS ---  
tab1, tab2, tab3 = st.tabs(["🎯 Bet Builder & Predykcje", "📊 Tabela i Forma", "📈 Statystyki Modelu"])  
  
with tab1:  
    st.subheader("🎛️ Zbuduj własne combo")  
    c1, c2, c3 = st.columns(3)  
    with c1:  
        linia_gole = st.selectbox("Linia goli", [1.5, 2.5, 3.5, 4.5], index=1)  
        typ_gole = st.selectbox("Typ goli", ["Over", "Under"], index=0)  
    with c2:  
        linia_rogi = st.selectbox("Linia rożnych", [6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5], index=1)  
        typ_rogi = st.selectbox("Typ rożnych", ["Over", "Under"], index=0)  
    with c3:  
        linia_kartki = st.selectbox("Linia kartek", [1.5, 2.5, 3.5, 4.5, 5.5, 6.5], index=1)  
        typ_kartki = st.selectbox("Typ kartek", ["Over", "Under"], index=0)  
      
    min_prob = st.slider("Minimalne prawdopodobieństwo combo", 0.0, 1.0, 0.40, 0.05)  
      
    st.subheader("📅 Predykcje – najbliższa kolejka")  
      
    # Znajdź najbliższą kolejkę (pierwsza której mecze jeszcze się nie odbyły)  
    if not schedule.empty and not srednie_df.empty:  
        # Aktualna data (bez czasu)  
        dzisiaj = datetime.now().date()  
          
        # Filtruj mecze które jeszcze się nie odbyły (data >= dzisiaj)  
        przyszle_mecze = schedule[schedule['date'].dt.date >= dzisiaj]  
          
        if not przyszle_mecze.empty:  
            # Znajdź najniższy numer kolejki wśród przyszłych meczów  
            najblizsza_kolejka = przyszle_mecze['round'].min()  
            mecze = schedule[schedule['round'] == najblizsza_kolejka]  
              
            st.caption(f"Kolejka {int(najblizsza_kolejka)} – {len(mecze)} meczów")  
              
            # Funkcja do obliczania prawdopodobieństwa z uwzględnieniem typu  
            def oblicz_p(typ, linia, lam):  
                return 1 - poisson.cdf(linia, lam) if typ == "Over" else poisson.cdf(linia, lam)  
              
            col1, col2 = st.columns(2)  
              
            # Combo Builder  
            with col1:  
                st.write("**Combo Builder**")  
                combo_count = 0  
                for _, mecz in mecze.iterrows():  
                    h_raw = mecz['home_team']  
                    a_raw = mecz['away_team']  
                    h = map_nazwa(h_raw)  
                    a = map_nazwa(a_raw)  
                      
                    if h in srednie_df.index and a in srednie_df.index:  
                        lam_h = (srednie_df.loc[h, "Gole strzelone (dom)"] + srednie_df.loc[a, "Gole stracone (wyjazd)"]) / 2  
                        lam_a = (srednie_df.loc[a, "Gole strzelone (wyjazd)"] + srednie_df.loc[h, "Gole stracone (dom)"]) / 2  
                        lam_r = (srednie_df.loc[h, "Różne (dom)"] + srednie_df.loc[a, "Różne (wyjazd)"]) / 2  
                        lam_k = (srednie_df.loc[h, "Kartki (dom)"] + srednie_df.loc[a, "Kartki (wyjazd)"]) / 2  

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
                    h_raw = mecz['home_team']  
                    a_raw = mecz['away_team']  
                    h = map_nazwa(h_raw)  
                    a = map_nazwa(a_raw)  
                      
                    if h in srednie_df.index and a in srednie_df.index:  
                        lam_h = (srednie_df.loc[h, "Gole strzelone (dom)"] + srednie_df.loc[a, "Gole stracone (wyjazd)"]) / 2  
                        lam_a = (srednie_df.loc[a, "Gole strzelone (wyjazd)"] + srednie_df.loc[h, "Gole stracone (dom)"]) / 2  
                        p_btts = (1 - poisson.pmf(0, lam_h)) * (1 - poisson.pmf(0, lam_a))  
                        btts_data.append((f"{h} - {a}", p_btts))  
                  
                if btts_data:  
                    for mecz, p in sorted(btts_data, key=lambda x: x[1], reverse=True):  
                        st.write(f"{koloruj(p)} **{mecz}**: {p:.1%}")  
        else:  
            st.info("Brak nadchodzących meczów. Wszystkie mecze w terminarzu już się odbyły.")  
    else:  
        if schedule.empty:  
            st.warning("Brak danych terminarza.")  
        if srednie_df.empty:  
            st.warning("Brak statystyk drużyn (za mało danych historycznych).")  

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

else:
st.error("Nie udało się pobrać danych historycznych. Sprawdź połączenie z internetem lub spróbuj później.")