import streamlit as st
import pandas as pd
from scipy.stats import poisson
from datetime import datetime
import requests
from io import StringIO
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# KONFIGURACJA STRONY
# ==========================================
st.set_page_config(
    page_title="Premier League Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# STAŁE I MAPOWANIA
# ==========================================

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
    "Wolverhampton Wanderers": "Wolves"
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
# WCZYTYWANIE DANYCH
# ==========================================

@st.cache_data(ttl=900)
def load_historical():
    """Pobiera dane historyczne z football-data.co.uk"""
    url = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
    
    try:
        r = requests.get(url, timeout=10)
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
        st.error(f"Błąd pobierania danych: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def load_schedule():
    """Wczytuje terminarz z pliku CSV"""
    try:
        df = pd.read_csv("terminarz_premier_2025.csv")
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date')
    except FileNotFoundError:
        st.warning("Brak pliku terminarz_premier_2025.csv - pokazuję przykładowe dane")
        # Przykładowe dane na wypadek braku pliku
        data = {
            'date': [datetime.now() + pd.Timedelta(days=i) for i in range(1, 10)],
            'round': [27] * 9,
            'home_team': ['Chelsea', 'Brentford', 'Aston Villa', 'West Ham', 'Man City',
                         'Crystal Palace', 'Nott\'m Forest', 'Sunderland', 'Everton'],
            'away_team': ['Burnley', 'Brighton', 'Leeds', 'Bournemouth', 'Newcastle',
                         'Wolves', 'Liverpool', 'Fulham', 'Man United']
        }
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Błąd wczytywania terminarza: {e}")
        return pd.DataFrame()

# ==========================================
# OBLICZANIE STATYSTYK DRUŻYN
# ==========================================

@st.cache_data
def oblicz_srednie(df):
    """
    Oblicza średnie statystyki dla każdej drużyny
    z uwzględnieniem podziału na dom/wyjazd i wagą formy
    """
    druzyny = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel())
    dane = {}

    for d in druzyny:
        # Mecze domowe
        home = df[df['HomeTeam'] == d].tail(10)
        # Mecze wyjazdowe
        away = df[df['AwayTeam'] == d].tail(10)

        # Minimum 3 mecze dla wiarygodności
        if len(home) < 3 or len(away) < 3:
            continue

        dane[d] = {
            # Gole
            "Gole strzelone (dom)": weighted_mean(home['FTHG']),
            "Gole stracone (dom)": weighted_mean(home['FTAG']),
            "Gole strzelone (wyjazd)": weighted_mean(away['FTAG']),
            "Gole stracone (wyjazd)": weighted_mean(away['FTHG']),
            
            # Rzuty rożne
            "Różne (dom)": weighted_mean(home['HC']),
            "Różne (wyjazd)": weighted_mean(away['AC']),
            
            # Kartki
            "Kartki (dom)": weighted_mean(home['total_kartki']),
            "Kartki (wyjazd)": weighted_mean(away['total_kartki']),
            
            # Dodatkowe statystyki
            "BTTS% (dom)": weighted_mean(home['btts']),
            "Over2.5% (dom)": weighted_mean(home['over25']),
            "BTTS% (wyjazd)": weighted_mean(away['btts']),
            "Over2.5% (wyjazd)": weighted_mean(away['over25']),
        }

    return pd.DataFrame(dane).T.round(3)

@st.cache_data
def oblicz_forme(df):
    """Oblicza formę drużyny (ostatnie 5 meczów) jako ciąg W/D/L"""
    druzyny = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel())
    forma = {}

    for d in druzyny:
        mecze = df[
            (df['HomeTeam'] == d) | (df['AwayTeam'] == d)
        ].tail(5)

        wyniki = []
        for _, m in mecze.iterrows():
            if m['HomeTeam'] == d:
                if m['FTHG'] > m['FTAG']:
                    wyniki.append("W")
                elif m['FTHG'] < m['FTAG']:
                    wyniki.append("L")
                else:
                    wyniki.append("D")
            else:
                if m['FTAG'] > m['FTHG']:
                    wyniki.append("W")
                elif m['FTAG'] < m['FTHG']:
                    wyniki.append("L")
                else:
                    wyniki.append("D")

        forma[d] = "".join(wyniki) if wyniki else "—"

    return forma

@st.cache_data
def tabela_ligowa(df):
    """Tworzy aktualną tabelę ligową"""
    table = {}

    for _, m in df.iterrows():
        home, away = m['HomeTeam'], m['AwayTeam']
        hg, ag = m['FTHG'], m['FTAG']

        for team in [home, away]:
            if team not in table:
                table[team] = {"pts": 0, "gf": 0, "ga": 0, "played": 0, "w": 0, "d": 0, "l": 0}

        table[home]["gf"] += hg
        table[home]["ga"] += ag
        table[home]["played"] += 1

        table[away]["gf"] += ag
        table[away]["ga"] += hg
        table[away]["played"] += 1

        if hg > ag:
            table[home]["pts"] += 3
            table[home]["w"] += 1
            table[away]["l"] += 1
        elif hg < ag:
            table[away]["pts"] += 3
            table[away]["w"] += 1
            table[home]["l"] += 1
        else:
            table[home]["pts"] += 1
            table[away]["pts"] += 1
            table[home]["d"] += 1
            table[away]["d"] += 1

    df_table = pd.DataFrame(table).T
    df_table["diff"] = df_table["gf"] - df_table["ga"]
    df_table["ppg"] = (df_table["pts"] / df_table["played"]).round(2)
    
    return df_table.sort_values(["pts", "diff", "gf"], ascending=False).round(2)

# ==========================================
# WCZYTAJ DANE
# ==========================================
historical = load_historical()
schedule = load_schedule()

if historical.empty:
    st.stop()

# Oblicz statystyki
srednie_df = oblicz_srednie(historical)
forma_dict = oblicz_forme(historical)
tabela_df = tabela_ligowa(historical)

# ==========================================
# NAGŁÓWEK
# ==========================================
st.title("⚽ Premier League Predictor 2025/26")
st.markdown("""
    Model predykcyjny oparty na **rozkładzie Poissona** z uwzględnieniem:
    - 🏠 Przewagi własnego boiska
    - 📈 Formy z ostatnich meczów (wagi liniowe)
    - 📊 Historycznych statystyk
""")

# ==========================================
# SIDEBAR - KONTROLKI
# ==========================================
with st.sidebar:
    st.header("🎮 Panel sterowania")
    
    # Wybór kolejki
    if not schedule.empty:
        available_rounds = sorted(schedule['round'].unique())
        if available_rounds:
            selected_round = st.selectbox(
                "Kolejka:",
                available_rounds,
                format_func=lambda x: f"{x}. kolejka"
            )
            
            mecze = schedule[schedule['round'] == selected_round]
            st.caption(f"📅 {len(mecze)} meczów")
    
    st.divider()
    
    # Odświeżanie danych
    if st.button("🔄 Odśwież dane"):
        st.cache_data.clear()
        st.rerun()
    
    st.divider()
    
    # Informacje o modelu
    with st.expander("ℹ️ O modelu"):
        st.markdown("""
        **Model Poissona** zakłada, że liczba goli w meczu podlega rozkładowi Poissona.
        
        λ (lambda) = średnia liczba goli oczekiwana dla danej drużyny.
        
        **Wagi:** ostatnie mecze mają wagę 2× wyższą niż najstarsze z 10 analizowanych.
        
        **Źródło danych:** football-data.co.uk
        """)

# ==========================================
# GŁÓWNA SEKCJA - BET BUILDER
# ==========================================
st.divider()
st.header("🎛️ Bet Builder – zbuduj własne combo")

col1, col2, col3, col4 = st.columns([1,1,1,1])

with col1:
    linia_gole = st.selectbox(
        "⚽ Gole",
        [1.5, 2.5, 3.5, 4.5],
        index=1
    )
    typ_gole = st.selectbox(
        "Typ goli",
        ["Over", "Under"]
    )

with col2:
    linia_rogi = st.selectbox(
        "🚩 Rożne",
        [7.5, 8.5, 9.5, 10.5, 11.5, 12.5],
        index=2
    )
    typ_rogi = st.selectbox(
        "Typ rożnych",
        ["Over", "Under"]
    )

with col3:
    linia_kartki = st.selectbox(
        "🟨 Kartki",
        [3.5, 4.5, 5.5, 6.5, 7.5],
        index=1
    )
    typ_kartki = st.selectbox(
        "Typ kartek",
        ["Over", "Under"]
    )

with col4:
    min_prob = st.slider(
        "Min. prawdopodobieństwo",
        0.0, 1.0, 0.40, 0.05
    )
    
    pokaz_rozkład = st.checkbox(
        "📊 Pokaż rozkład",
        value=False
    )

# ==========================================
# OBLICZANIE PREDYKCJI
# ==========================================
if not schedule.empty and not mecze.empty:
    
    bet_builder_results = []

    for _, mecz in mecze.iterrows():
        home = NAZWY_MAP.get(mecz['home_team'], mecz['home_team'])
        away = NAZWY_MAP.get(mecz['away_team'], mecz['away_team'])

        if home in srednie_df.index and away in srednie_df.index:
            
            # Lambda dla goli
            lambda_home = (
                srednie_df.loc[home, "Gole strzelone (dom)"] +
                srednie_df.loc[away, "Gole stracone (wyjazd)"]
            ) / 2
            
            lambda_away = (
                srednie_df.loc[away, "Gole strzelone (wyjazd)"] +
                srednie_df.loc[home, "Gole stracone (dom)"]
            ) / 2
            
            lambda_gole = lambda_home + lambda_away
            lambda_rogi = (
                srednie_df.loc[home, "Różne (dom)"] +
                srednie_df.loc[away, "Różne (wyjazd)"]
            ) / 2
            lambda_kartki = (
                srednie_df.loc[home, "Kartki (dom)"] +
                srednie_df.loc[away, "Kartki (wyjazd)"]
            ) / 2
            
            # Oblicz prawdopodobieństwa
            p_gole = oblicz_prob_poisson(typ_gole, linia_gole, lambda_gole)
            p_rogi = oblicz_prob_poisson(typ_rogi, linia_rogi, lambda_rogi)
            p_kartki = oblicz_prob_poisson(typ_kartki, linia_kartki, lambda_kartki)
            p_combo = p_gole * p_rogi * p_kartki
            
            bet_builder_results.append({
                "home": home,
                "away": away,
                "data": mecz['date'].strftime("%d.%m.%Y"),
                "lambda_home": lambda_home,
                "lambda_away": lambda_away,
                "lambda_gole": lambda_gole,
                "lambda_rogi": lambda_rogi,
                "lambda_kartki": lambda_kartki,
                "p_gole": p_gole,
                "p_rogi": p_rogi,
                "p_kartki": p_kartki,
                "p_combo": p_combo
            })

    # Sortuj i filtruj
    bet_builder_results.sort(key=lambda x: x["p_combo"], reverse=True)
    bet_builder_results = [m for m in bet_builder_results if m["p_combo"] >= min_prob]

    # Wyświetl wyniki
    if bet_builder_results:
        for m in bet_builder_results:
            with st.expander(f"{m['home']} vs {m['away']} ({m['data']})"):
                
                col_left, col_right = st.columns([2, 1])
                
                with col_left:
                    st.write(f"{koloruj(m['p_gole'])} ⚽ {typ_gole} {linia_gole}: **{m['p_gole']*100:.1f}%**")
                    st.write(f"{koloruj(m['p_rogi'])} 🚩 {typ_rogi} {linia_rogi}: **{m['p_rogi']*100:.1f}%**")
                    st.write(f"{koloruj(m['p_kartki'])} 🟨 {typ_kartki} {linia_kartki}: **{m['p_kartki']*100:.1f}%**")
                    
                    st.markdown("---")
                    st.markdown(f"### 🎯 Combo: {m['p_combo']*100:.2f}%")
                    
                    st.caption(
                        f"λ gole: {m['home']} {m['lambda_home']:.2f} – {m['away']} {m['lambda_away']:.2f} | "
                        f"λ rogi: {m['lambda_rogi']:.2f} | λ kartki: {m['lambda_kartki']:.2f}"
                    )
                
                with col_right:
                    if pokaz_rozkład:
                        # Wykres rozkładu Poissona
                        fig = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=("Rozkład goli", "Rozkład rożnych/kartek"),
                            vertical_spacing=0.15
                        )
                        
                        # Rozkład goli
                        x_gole = list(range(0, 7))
                        y_home = [poisson.pmf(i, m['lambda_home']) for i in x_gole]
                        y_away = [poisson.pmf(i, m['lambda_away']) for i in x_gole]
                        
                        fig.add_trace(
                            go.Bar(x=x_gole, y=y_home, name=m['home'], 
                                  marker_color='#1f77b4', opacity=0.7),
                            row=1, col=1
                        )
                        fig.add_trace(
                            go.Bar(x=x_gole, y=y_away, name=m['away'], 
                                  marker_color='#ff7f0e', opacity=0.7),
                            row=1, col=1
                        )
                        
                        # Rozkład rożnych i kartek
                        x_extra = list(range(0, 21))
                        y_rogi = [poisson.pmf(i, m['lambda_rogi']) for i in x_extra]
                        y_kartki = [poisson.pmf(i, m['lambda_kartki']) for i in x_extra]
                        
                        fig.add_trace(
                            go.Scatter(x=x_extra, y=y_rogi, name="Rożne", 
                                      mode='lines+markers', line=dict(color='green')),
                            row=2, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=x_extra, y=y_kartki, name="Kartki", 
                                      mode='lines+markers', line=dict(color='red')),
                            row=2, col=1
                        )
                        
                        fig.update_layout(
                            height=450,
                            showlegend=True,
                            margin=dict(l=30, r=30, t=50, b=30)
                        )
                        
                        fig.update_xaxes(title_text="Liczba goli", row=1, col=1)
                        fig.update_yaxes(title_text="Prawdopodobieństwo", row=1, col=1)
                        fig.update_xaxes(title_text="Liczba", row=2, col=1)
                        fig.update_yaxes(title_text="Prawdopodobieństwo", row=2, col=1)
                        
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ℹ️ Brak meczów spełniających kryteria. Zmniejsz próg minimalnego prawdopodobieństwa.")
else:
    st.warning("Brak meczów w wybranej kolejce.")

# ==========================================
# HIGH PROBABILITY – TOP PREDYKCJE
# ==========================================
st.divider()
st.header("🔥 High Probability – top predykcje")

if not schedule.empty and not mecze.empty:
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⚽ Największe szanse na BTTS")
        btts_list = []
        for _, mecz in mecze.iterrows():
            home = NAZWY_MAP.get(mecz['home_team'], mecz['home_team'])
            away = NAZWY_MAP.get(mecz['away_team'], mecz['away_team'])
            
            if home not in srednie_df.index or away not in srednie_df.index:
                continue
                
            lambda_home = (
                srednie_df.loc[home, "Gole strzelone (dom)"] +
                srednie_df.loc[away, "Gole stracone (wyjazd)"]
            ) / 2
            lambda_away = (
                srednie_df.loc[away, "Gole strzelone (wyjazd)"] +
                srednie_df.loc[home, "Gole stracone (dom)"]
            ) / 2
            
            p_home_0 = poisson.pmf(0, lambda_home)
            p_away_0 = poisson.pmf(0, lambda_away)
            p_btts = 1 - p_home_0 - p_away_0 + (p_home_0 * p_away_0)
            
            btts_list.append({
                "Mecz": f"{home} vs {away}",
                "BTTS": round(p_btts * 100, 1),
                "λ_home": round(lambda_home, 2),
                "λ_away": round(lambda_away, 2)
            })
        
        if btts_list:
            btts_df = pd.DataFrame(btts_list).sort_values("BTTS", ascending=False)
            st.dataframe(btts_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("⚽ Największe szanse na Over 2.5")
        over_list = []
        for _, mecz in mecze.iterrows():
            home = NAZWY_MAP.get(mecz['home_team'], mecz['home_team'])
            away = NAZWY_MAP.get(mecz['away_team'], mecz['away_team'])
            
            if home not in srednie_df.index or away not in srednie_df.index:
                continue
                
            lambda_home = (
                srednie_df.loc[home, "Gole strzelone (dom)"] +
                srednie_df.loc[away, "Gole stracone (wyjazd)"]
            ) / 2
            lambda_away = (
                srednie_df.loc[away, "Gole strzelone (wyjazd)"] +
                srednie_df.loc[home, "Gole stracone (dom)"]
            ) / 2
            
            lambda_total = lambda_home + lambda_away
            p_over25 = 1 - poisson.cdf(2, lambda_total)
            
            over_list.append({
                "Mecz": f"{home} vs {away}",
                "Over 2.5": round(p_over25 * 100, 1),
                "λ_total": round(lambda_total, 2)
            })
        
        if over_list:
            over_df = pd.DataFrame(over_list).sort_values("Over 2.5", ascending=False)
            st.dataframe(over_df, use_container_width=True, hide_index=True)

# ==========================================
# STATYSTYKI DRUŻYN
# ==========================================
st.divider()
st.header("📊 Statystyki drużyn")

tab1, tab2, tab3 = st.tabs([
    "📈 Tabela ligowa",
    "📊 Średnie statystyki",
    "📉 Forma ostatnich meczów"
])

with tab1:
    st.subheader("Tabela ligowa")
    
    # Formatowanie tabeli
    display_table = tabela_df.copy()
    display_table.index.name = "Drużyna"
    
    # Dodaj kolumny z procentami
    display_table['W%'] = (display_table['w'] / display_table['played'] * 100).round(1).astype(str) + '%'
    display_table['ppg'] = display_table['ppg'].astype(str) + ' pkt/m'
    
    st.dataframe(display_table, use_container_width=True)

with tab2:
    st.subheader("Średnie drużyn (ostatnie 10 meczów)")
    
    # Wybór kolumn do wyświetlenia
    cols_to_show = [
        "Gole strzelone (dom)", "Gole stracone (dom)",
        "Gole strzelone (wyjazd)", "Gole stracone (wyjazd)",
        "Różne (dom)", "Różne (wyjazd)",
        "Kartki (dom)", "Kartki (wyjazd)",
        "BTTS% (dom)", "Over2.5% (dom)"
    ]
    
    display_df = srednie_df[cols_to_show].copy()
    display_df.index.name = "Drużyna"
    
    # Formatowanie procentów
    for col in ["BTTS% (dom)", "Over2.5% (dom)"]:
        if col in display_df.columns:
            display_df[col] = (display_df[col] * 100).round(1).astype(str) + '%'
    
    st.dataframe(display_df, use_container_width=True)

with tab3:
    st.subheader("Forma (ostatnie 5 meczów)")
    forma_df = pd.DataFrame.from_dict(forma_dict, orient="index", columns=["Forma"])
    forma_df.index.name = "Drużyna"
    
    # Dodaj kolumnę z wizualizacją
    def format_forma(f):
        if f == "—":
            return "—"
        return f.replace("W", "🟢 ").replace("D", "🟡 ").replace("L", "🔴 ")
    
    forma_df["Wizualizacja"] = forma_df["Forma"].apply(format_forma)
    
    # Dodaj kolorowanie dla ostatniego meczu
    def ostatni_mecz_kolor(f):
        if f == "—" or len(f) == 0:
            return "⚪"
        last = f[-1]
        if last == "W":
            return "🟢"
        elif last == "D":
            return "🟡"
        elif last == "L":
            return "🔴"
        return "⚪"
    
    forma_df["Ostatni"] = forma_df["Forma"].apply(ostatni_mecz_kolor)
    
    st.dataframe(forma_df, use_container_width=True)

# ==========================================
# STOPKA Z INFORMACJAMI
# ==========================================
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "📊 Liczba meczów w bazie",
        len(historical)
    )

with col2:
    st.metric(
        "🔄 Ostatnia aktualizacja",
        historical['Date'].max().strftime("%d.%m.%Y") if not historical.empty else "—"
    )

with col3:
    st.metric(
        "🏆 Drużyn w lidze",
        len(srednie_df)
    )

st.caption(
    f"⚽ Premier League Predictor 2025/26 | "
    f"Model Poisson z wagą formy | "
    f"Źródło: football-data.co.uk"
)

# ==========================================
# PRZYCISK ODŚWIEŻANIA (dolny)
# ==========================================
if st.button("🔄 Odśwież dane (dolny)", use_container_width=True):
    st.cache_data.clear()
    st.rerun()