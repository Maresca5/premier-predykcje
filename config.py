"""
config.py – stałe konfiguracyjne współdzielone przez wszystkie moduły.
"""
from datetime import datetime

LIGI = {
    # tau: dni time-decay; fw: form_weight krok per W/L
    # fd_org_id: ID ligi w football-data.org API
    # cup_ids: ID rozgrywek pucharowych dla fatigue (UCL/UEL/UECL/krajowe puchary)
    "Premier League": {"csv_code": "E0",  "fd_org_id": 2021, "cup_ids": [2001,2002,2139,2021], "file": "terminarz_premier_2025.csv",  "tau": 30.0, "fw": 0.03},
    "La Liga":        {"csv_code": "SP1", "fd_org_id": 2014, "cup_ids": [2001,2002,2079],       "file": "terminarz_la_liga_2025.csv",  "tau": 28.0, "fw": 0.04},
    "Bundesliga":     {"csv_code": "D1",  "fd_org_id": 2002, "cup_ids": [2001,2002,2011],       "file": "terminarz_bundesliga_2025.csv","tau": 28.0, "fw": 0.04},
    "Serie A":        {"csv_code": "I1",  "fd_org_id": 2019, "cup_ids": [2001,2002,2080],       "file": "terminarz_serie_a_2025.csv",  "tau": 28.0, "fw": 0.04},
    "Ligue 1":        {"csv_code": "F1",  "fd_org_id": 2015, "cup_ids": [2001,2002,2081],       "file": "terminarz_ligue_1_2025.csv",  "tau": 21.0, "fw": 0.05},
}

DB_FILE    = "predykcje.db"

# Bieżący sezon (kod football-data.co.uk: "2526" = sezon 2025/26)
# Europejskie ligi: sezon startuje w lipcu/sierpniu
# Jeśli miesiąc >= 7 → rok bieżący, inaczej rok-1
def _oblicz_biezacy_sezon() -> str:
    _y = datetime.now().year if datetime.now().month >= 7 else datetime.now().year - 1
    return f"{str(_y)[2:]}{str(_y+1)[2:]}"

BIEZACY_SEZON = _oblicz_biezacy_sezon()   # np. "2526" dla sezonu 2025/26


# Waga SOT w blendzie z golami – parametr globalny
SOT_BLEND_W = 0.30   # 0.0 = tylko gole, 0.30 = 70% gole + 30% SOT


# Shrinkage per liga – zoptymalizowany na backtestach 2 sezony x 5 lig (3098 meczow)
# E0:0.25 | I1:0.28 | SP1:0.38 | D1:0.40 | F1:0.45
KALIBRACJA_PER_LIGA = {"E0": 0.25, "SP1": 0.38, "D1": 0.40, "I1": 0.28, "F1": 0.45}
SHRINK_ALPHA  = 0.25
PROG_PEWNY    = 0.55
PROG_PODWOJNA = 0.55


# ── Market Noise & Kelly helpers ──────────────────────────────────────
MARKET_NOISE_MAX = 0.25
KELLY_FRACTION   = 0.125         # 1/8 Kelly – default
KELLY_BANKROLL_DEFAULT = 1000.0
# Conservative Kelly – shrinkage prawdopodobieństwa TYLKO do Kelly/EV
# Symulacja (5 lig × 2 sezony): -15% nadwyżki + KF/2 → MaxDD 55%→33%
# p_kelly = 0.5 + (p_model - 0.5) * KELLY_PROB_SCALE
KELLY_PROB_SCALE = 0.85          # -15% nadwyżki powyżej 50%
KELLY_FRAC_SCALE = 0.50          # Half-Kelly na wyjście stawki

# Per-rynek ułamki Kelly – obniżone o 50% względem poprzednich wartości
# Analiza backtestu E0 2425: model zawyża p przy faworytach (p=0.80 → actual 50%)
# → mniejsze frakcje chronią przed agresywnym Kelly na niepewnych predykcjach
KELLY_FRACTIONS = {
    "1X2":    0.125,  # 1/8 Kelly – backtest wykazał optymalny balans ROI/DrawDown
    "AH":     0.125,
    "Gole":   0.075,  # niezweryfikowane historycznie + losowość bramkowa
    "BTTS":   0.075,
    "Rożne":  0.000,  # WYŁĄCZONE – lambda bramkowa ≠ rozkład rożnych (neg. dwumianowy)
    "Kartki": 0.000,  # WYŁĄCZONE – brak kalibracji statystyk kartkowych
    "SOT":    0.050,
}
MAX_EXPOSURE_PCT = 0.05   # max 5% bankrollu per mecz (suma wszystkich rynków)
