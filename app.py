import streamlit as st
import pandas as pd
from scipy.stats import poisson
from datetime import datetime
import requests
from io import StringIO
import os

st.set_page_config(page_title="Predykcje Premier League", layout="wide")

st.title("Predykcje Premier League 2025/26")
st.markdown("Dane automatycznie z football-data.co.uk + terminarz lokalny")

# Plik do zapisywania skuteczności
SKUTECZNOSC_FILE = "skutecznosc_predykcji.csv"

# Słownik mapowania nazw drużyn
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

# Automatyczne pobieranie E0.csv
@st.cache_data(ttl=3600)
def load_historical():
    url = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
    try:
        r = requests.get(url)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df = df.sort_values('Date')
        df['total_gole'] = df['FTHG'] + df['FTAG']
        df['total_kartki'] = df['HY'] + df['AY'] + df['HR']*2 + df['AR']*2
        df['total_rozne'] = df['HC'] + df['AC']
        return df
    except Exception as e:
        st.error(f"Błąd pobierania E0.csv: {e}")
        return pd.DataFrame()

# Ładujemy terminarz
@st.cache_data(ttl=86400)
def load_schedule():
    try:
        df = pd.read_csv("terminarz_premier_2025.csv")
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        return df
    except Exception as e:
        st.error(f"Błąd wczytywania terminarz_premier_2025.csv: {e}")
        return pd.DataFrame()

historical = load_historical()
schedule = load_schedule()

if historical.empty or schedule.empty:
    st.stop()

# Oblicz średnie
@st.cache_data
def oblicz_srednie():
    druzyny = pd.unique(historical[['HomeTeam', 'AwayTeam']].values.ravel())
    srednie = {}
    for druzyna in druzyny:
        mecze = historical[
            (historical['HomeTeam'] == druzyna) | (historical['AwayTeam'] == druzyna)
        ].tail(10)
        
        if len(mecze) < 3:
            continue
        
        strzelone = mecze.apply(lambda r: r['FTHG'] if r['HomeTeam'] == druzyna else r['FTAG'], axis=1).mean()
        stracone = mecze.apply(lambda r: r['FTAG'] if r['HomeTeam'] == druzyna else r['FTHG'], axis=1).mean()
        total_gole = strzelone + stracone
        kartki = mecze['total_kartki'].mean()
        rozne = mecze['total_rozne'].mean()
        celne_strzaly = mecze.apply(lambda r: r['HST'] if r['HomeTeam'] == druzyna else r['AST'], axis=1).mean()
        
        srednie[druzyna] = {
            'strzelone': round(strzelone, 2),
            'stracone': round(stracone, 2),
            'total_gole': round(total_gole, 2),
            'kartki': round(kartki, 2),
            'rozne': round(rozne, 2),
            'celne_strzaly': round(celne_strzaly, 2)
        }
    
    return pd.DataFrame(srednie).T.sort_index()

srednie_df = oblicz_srednie()

st.subheader("Średnie z ostatnich 10 meczów – wszystkie drużyny")
st.dataframe(srednie_df)

# Funkcja do wyszukiwania linii najbliższej 50% od góry (≥50%)
def find_najblizsza_linia(lambda_val, linie, min_prob=50):
    najlepsze = None
    najlepsze_roz = 1000
    for linia in linie:
        p = 1 - poisson.cdf(linia, lambda_val)
        if p >= min_prob / 100:
            roz = abs(p - 0.5)
            if roz < najlepsze_roz:
                najlepsze = f"Over {linia} ({p*100:.1f}%)"
                najlepsze_roz = roz
    return najlepsze if najlepsze else "Brak linii ≥50%"

# Funkcja do zapisu i weryfikacji skuteczności
def zapisz_i_weryfikuj_predykcje(mecze_kolejki, predykcje_df):
    if os.path.exists(SKUTECZNOSC_FILE):
        df_hist = pd.read_csv(SKUTECZNOSC_FILE)
    else:
        df_hist = pd.DataFrame(columns=['round', 'mecz', 'gole_typ', 'gole_liczba', 'rozne', 'rozne_liczba', 'kartki', 'kartki_liczba', 'trafiony_gole', 'trafiony_rozne', 'trafiony_kartki'])

    for _, mecz in mecze_kolejki.iterrows():
        home = mecz['home_team']
        away = mecz['away_team']
        mecz_str = f"{home} vs {away}"
        
        # Sprawdź, czy mecz już rozegrany i mamy wynik w historical
        wynik = historical[
            (historical['HomeTeam'] == home_map) & (historical['AwayTeam'] == away_map) &
            (historical['Date'].dt.date == mecz['date'].date())
        ]
        
        if not wynik.empty:
            rzeczywiste_gole = wynik.iloc[0]['total_gole']
            rzeczywiste_rozne = wynik.iloc[0]['total_rozne']
            rzeczywiste_kartki = wynik.iloc[0]['total_kartki']
            
            # Znajdź predykcję dla tego meczu
            pred = predykcje_df[predykcje_df['Mecz'] == mecz_str]
            if not pred.empty:
                gole_typ = pred.iloc[0]['Gole']
                rozne = pred.iloc[0]['Rozne']
                kartki = pred.iloc[0]['Kartki']
                
                # Trafienie goli
                if "Over 2.5" in gole_typ and rzeczywiste_gole > 2.5:
                    traf_gole = 1
                elif "Under 2.5" in gole_typ and rzeczywiste_gole <= 2.5:
                    traf_gole = 1
                else:
                    traf_gole = 0
                
                # Trafienie rożnych (przykład dla over 9.5 – dostosuj jeśli linia inna)
                if "Over" in rozne:
                    linia_rogi = float(rozne.split()[1])
                    if rzeczywiste_rozne > linia_rogi:
                        traf_rozne = 1
                    else:
                        traf_rozne = 0
                else:
                    traf_rozne = 0  # na razie tylko over
                
                # Trafienie kartek (podobnie)
                if "Over" in kartki:
                    linia_kartki = float(kartki.split()[1])
                    if rzeczywiste_kartki > linia_kartki:
                        traf_kartki = 1
                    else:
                        traf_kartki = 0
                else:
                    traf_kartki = 0
                
                # Zapisz do historii
                nowa_wiersz = {
                    'round': min_round,
                    'mecz': mecz_str,
                    'gole_typ': gole_typ,
                    'gole_liczba': rzeczywiste_gole,
                    'rozne': rozne,
                    'rozne_liczba': rzeczywiste_rozne,
                    'kartki': kartki,
                    'kartki_liczba': rzeczywiste_kartki,
                    'trafiony_gole': traf_gole,
                    'trafiony_rozne': traf_rozne,
                    'trafiony_kartki': traf_kartki
                }
                df_hist = pd.concat([df_hist, pd.DataFrame([nowa_wiersz])], ignore_index=True)
                df_hist.to_csv(SKUTECZNOSC_FILE, index=False)

# Weryfikacja skuteczności (pokazuje tabelę)
if os.path.exists(SKUTECZNOSC_FILE):
    df_skut = pd.read_csv(SKUTECZNOSC_FILE)
    st.subheader("Skuteczność predykcji z poprzednich kolejek")
    st.dataframe(df_skut)
    
    if not df_skut.empty:
        traf_gole = df_skut['trafiony_gole'].mean() * 100
        traf_rozne = df_skut['trafiony_rozne'].mean() * 100
        traf_kartki = df_skut['trafiony_kartki'].mean() * 100
        
        st.write(f"Skuteczność goli: {traf_gole:.1f}%")
        st.write(f"Skuteczność rożnych: {traf_rozne:.1f}%")
        st.write(f"Skuteczność kartek: {traf_kartki:.1f}%")

# Predykcje na najbliższą kolejkę
st.subheader("Predykcje na najbliższą kolejkę")

dzisiaj = datetime.now()
nadchodzace = schedule[schedule['date'] > dzisiaj].copy()

if nadchodzace.empty:
    st.warning("Brak nadchodzących meczów.")
else:
    min_round = nadchodzace['round'].min()
    mecze_kolejki = nadchodzace[nadchodzace['round'] == min_round]

    predykcje = []
    for _, mecz in mecze_kolejki.iterrows():
        home = mecz['home_team']
        away = mecz['away_team']
        
        home_map = NAZWY_MAP.get(home, home)
        away_map = NAZWY_MAP.get(away, away)
        
        if home_map in srednie_df.index and away_map in srednie_df.index:
            lambda_gole = (srednie_df.loc[home_map, 'total_gole'] + srednie_df.loc[away_map, 'total_gole']) / 2
            p_over_25 = 1 - poisson.cdf(2, lambda_gole)
            
            if p_over_25 >= 0.5:
                gole_typ = f"Over 2.5 ({p_over_25*100:.1f}%)"
            else:
                p_under_25 = poisson.cdf(2, lambda_gole)
                gole_typ = f"Under 2.5 ({p_under_25*100:.1f}%)"
            
            lambda_rogi = (srednie_df.loc[home_map, 'rozne'] + srednie_df.loc[away_map, 'rozne']) / 2
            lambda_kartki = (srednie_df.loc[home_map, 'kartki'] + srednie_df.loc[away_map, 'kartki']) / 2
            
            linia_rogi = find_najblizsza_linia(lambda_rogi, [5.5, 6.5, 7.5, 8.5, 9.5])
            linia_kartki = find_najblizsza_linia(lambda_kartki, [1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
            
            predykcje.append({
                'Mecz': f"{home} vs {away}",
                'Data': mecz['date'].date(),
                'Gole': gole_typ,
                'Rozne': linia_rogi,
                'Kartki': linia_kartki
            })
        else:
            predykcje.append({
                'Mecz': f"{home} vs {away}",
                'Data': mecz['date'].date(),
                'Gole': "Brak danych"
            })
    
    st.dataframe(pd.DataFrame(predykcje))

# Weryfikacja skuteczności po odświeżeniu
zapisz_i_weryfikuj_predykcje(mecze_kolejki, pd.DataFrame(predykcje))

if st.button("Odśwież dane"):
    st.cache_data.clear()
    st.rerun()