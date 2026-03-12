"""
model.py – Dixon-Coles, lambdy, Kelly, predykcje, entropia.
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson

from config import (LIGI, BIEZACY_SEZON, SOT_BLEND_W, KALIBRACJA_PER_LIGA,
                    SHRINK_ALPHA, PROG_PEWNY, PROG_PODWOJNA,
                    MARKET_NOISE_MAX, KELLY_FRACTION, KELLY_BANKROLL_DEFAULT,
                    KELLY_PROB_SCALE, KELLY_FRAC_SCALE, KELLY_FRACTIONS,
                    MAX_EXPOSURE_PCT)

def oblicz_lambdy(h: str, a: str, srednie_df: pd.DataFrame,
                  srednie_lig: dict, forma_dict: dict,
                  sot_w: float = SOT_BLEND_W,
                  csv_code: str = "E0") -> tuple:
    """
    Zwraca (lam_h, lam_a, lam_r, lam_k, sot_aktywny, lam_sot).

    Asymetryczny shrink lambdy (potwierdzone backtestem na 3 datasetach):
      ALPHA_LAM_OFF = 0.10  – atak jest zmienny → shrinkujemy mocniej
      ALPHA_LAM_DEF = 0.20  – obrona jest stabilna → shrinkujemy słabiej
    Wynik: +1.5–2.6pp hit rate, +2–5pp ROI, Brier -0.002 na wszystkich 3 datasetach.
    """
    # ── Asymetryczne parametry shrinkage lambdy ──────────────────
    ALPHA_OFF = 0.10   # shrink składowej ofensywnej w kierunku avg_ligi
    ALPHA_DEF = 0.20   # shrink składowej defensywnej w kierunku avg_ligi

    avg_h = max(srednie_lig["avg_home"], 0.5)
    avg_a = max(srednie_lig["avg_away"], 0.5)
    atak_h   = srednie_df.loc[h, "Gole strzelone (dom)"]    / avg_h
    obrona_a = srednie_df.loc[a, "Gole stracone (wyjazd)"]  / avg_a
    atak_a   = srednie_df.loc[a, "Gole strzelone (wyjazd)"] / avg_a
    obrona_h = srednie_df.loc[h, "Gole stracone (dom)"]     / avg_h

    def form_weight(team: str) -> float:
        f = forma_dict.get(team, "")
        w = f.count("W"); l = f.count("L")
        return float(np.clip(1.0 + (w - l) * 0.03, 0.85, 1.15))

    lam_h_goals = avg_h * atak_h * obrona_a * form_weight(h)
    lam_a_goals = avg_a * atak_a * obrona_h * form_weight(a)

    # ── Asymetryczny shrink: lam_h = f(atak domu), lam_a = f(obrona domu)
    # lam_h zależy głównie od ataku drużyny domowej → alpha_off
    # lam_a zależy głównie od obrony drużyny domowej → alpha_def (mniejszy shrink)
    lam_h_goals = (1 - ALPHA_OFF) * lam_h_goals + ALPHA_OFF * avg_h
    lam_a_goals = (1 - ALPHA_DEF) * lam_a_goals + ALPHA_DEF * avg_a

    # SOT blend – tylko gdy oba mają dane i sot_w > 0
    sot_aktywny = False
    lam_h = lam_h_goals
    lam_a = lam_a_goals

    if sot_w > 0:
        sot_h = srednie_df.loc[h, "SOT (dom)"]   if "SOT (dom)"    in srednie_df.columns else None
        sot_a = srednie_df.loc[a, "SOT (wyjazd)"] if "SOT (wyjazd)" in srednie_df.columns else None
        avg_sot_h = srednie_lig.get("avg_sot_home", None)
        avg_sot_a = srednie_lig.get("avg_sot_away", None)
        if (sot_h is not None and sot_a is not None and
                avg_sot_h and avg_sot_a and
                not np.isnan(sot_h) and not np.isnan(sot_a)):
            conv_h = avg_h / avg_sot_h
            conv_a = avg_a / avg_sot_a
            lam_sot_h = sot_h * conv_h * obrona_a * form_weight(h)
            lam_sot_a = sot_a * conv_a * obrona_h * form_weight(a)
            # Asymetryczny shrink również dla składowej SOT
            lam_sot_h = (1 - ALPHA_OFF) * lam_sot_h + ALPHA_OFF * avg_h
            lam_sot_a = (1 - ALPHA_DEF) * lam_sot_a + ALPHA_DEF * avg_a
            lam_h = (1 - sot_w) * lam_h_goals + sot_w * lam_sot_h
            lam_a = (1 - sot_w) * lam_a_goals + sot_w * lam_sot_a
            sot_aktywny = True

    lam_r = (srednie_df.loc[h, "Różne (dom)"] + srednie_df.loc[a, "Różne (wyjazd)"]) / 2
    lam_k = (srednie_df.loc[h, "Kartki (dom)"] + srednie_df.loc[a, "Kartki (wyjazd)"]) / 2
    sot_h_raw = srednie_df.loc[h, "SOT (dom)"]   if "SOT (dom)"    in srednie_df.columns else None
    sot_a_raw = srednie_df.loc[a, "SOT (wyjazd)"] if "SOT (wyjazd)" in srednie_df.columns else None
    lam_sot_total = None
    if sot_h_raw is not None and sot_a_raw is not None:
        try:
            sh = float(sot_h_raw); sa = float(sot_a_raw)
            if not (np.isnan(sh) or np.isnan(sa)):
                lam_sot_total = sh + sa
        except (TypeError, ValueError):
            pass

    return (float(np.clip(lam_h, 0.3, 4.5)),
            float(np.clip(lam_a, 0.3, 4.5)),
            lam_r, lam_k, sot_aktywny, lam_sot_total)

def oblicz_forme(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    forma = {}
    for d in pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel()):
        mecze = df[(df["HomeTeam"] == d) | (df["AwayTeam"] == d)].tail(5)
        wyniki = []
        for _, m in mecze.iterrows():
            if m["HomeTeam"] == d:
                wyniki.append("W" if m["FTHG"] > m["FTAG"] else ("L" if m["FTHG"] < m["FTAG"] else "D"))
            else:
                wyniki.append("W" if m["FTAG"] > m["FTHG"] else ("L" if m["FTAG"] < m["FTHG"] else "D"))
        forma[d] = "".join(wyniki)
    return forma

def tabela_ligowa(df: pd.DataFrame) -> pd.DataFrame:
    if "_sezon" in df.columns:
        df = df[df["_sezon"] == "biezacy"]
    if df.empty:
        return pd.DataFrame()
    table: dict = {}
    for _, m in df.iterrows():
        home, away = m["HomeTeam"], m["AwayTeam"]
        hg, ag = m["FTHG"], m["FTAG"]
        for t in [home, away]:
            if t not in table:
                table[t] = {"Pkt": 0, "GZ": 0, "GS": 0, "M": 0}
        table[home]["GZ"] += hg; table[home]["GS"] += ag; table[home]["M"] += 1
        table[away]["GZ"] += ag; table[away]["GS"] += hg; table[away]["M"] += 1
        if   hg > ag: table[home]["Pkt"] += 3
        elif hg < ag: table[away]["Pkt"] += 3
        else:         table[home]["Pkt"] += 1; table[away]["Pkt"] += 1
    res = pd.DataFrame(table).T
    res["RG"] = res["GZ"] - res["GS"]
    return res.sort_values(["Pkt", "RG", "GZ"], ascending=False)

# ===========================================================================
# MODEL
# ===========================================================================
def dixon_coles_adj(M: np.ndarray, lam_h: float, lam_a: float,
                    rho: float = -0.13) -> np.ndarray:
    M = M.copy()
    tau = {(0,0): 1 - lam_h*lam_a*rho, (1,0): 1 + lam_a*rho,
           (0,1): 1 + lam_h*rho,        (1,1): 1 - rho}
    for (i, j), t in tau.items():
        if i < M.shape[0] and j < M.shape[1]:
            M[i, j] *= max(t, 0.001)
    M /= M.sum()
    return M


def _get_shrink(csv_code: str, n_train: int = 200) -> float:
    """
    Dynamiczny shrinkage: przy malej liczbie meczow w bazie shrinkujemy mocniej.
    Formula: shrink = base + bonus * (1 - min(n_train, N_FULL) / N_FULL)
    Przy n_train=0  -> base + 0.20 (max ostroznosc)
    Przy n_train=150 -> base + 0.00 (pelna kalibracja)
    """
    N_FULL   = 150   # po ilu meczach shrinkage osiaga wartosc docelowa
    BONUS    = 0.20  # maksymalny dodatek na poczatku sezonu
    base     = KALIBRACJA_PER_LIGA.get(csv_code, SHRINK_ALPHA)
    dynamic  = BONUS * max(0.0, 1.0 - min(n_train, N_FULL) / N_FULL)
    return float(np.clip(base + dynamic, 0.0, 0.85))

def kalibruj_prawdopodobienstwa(p_home: float, p_draw: float, p_away: float,
                                csv_code: str = "E0", n_train: int = 200) -> tuple:
    a = _get_shrink(csv_code, n_train)
    p_h = (1-a)*p_home + a/3
    p_d = (1-a)*p_draw + a/3
    p_a = (1-a)*p_away + a/3
    s = p_h+p_d+p_a
    return p_h/s, p_d/s, p_a/s

def wybierz_typ(p_home: float, p_draw: float, p_away: float,
                csv_code: str = "E0", n_train: int = 200) -> tuple:
    p_home, p_draw, p_away = kalibruj_prawdopodobienstwa(
        p_home, p_draw, p_away, csv_code, n_train)
    p_1x = p_home + p_draw; p_x2 = p_away + p_draw
    # Bundesliga (D1): wyłącz singiele – backtest 2324/2425:
    #   Typ=1 overconfidence +23-33pp vs tylko +4-18pp dla 1X/X2
    #   Hit rate singlei 47-59% vs 58-76% dla 1X/X2
    is_bundesliga = (csv_code == "D1")
    if not is_bundesliga:
        if p_home >= PROG_PEWNY: return "1",  p_home
        if p_away >= PROG_PEWNY: return "2",  p_away
    if p_1x >= PROG_PODWOJNA or p_x2 >= PROG_PODWOJNA:
        return ("1X", p_1x) if p_1x >= p_x2 else ("X2", p_x2)
    probs = {"1": p_home, "X": p_draw, "2": p_away}
    t = max(probs, key=probs.get)
    return t, probs[t]

# ── Market Noise & Kelly helpers ──────────────────────────────────────

def market_noise_check(p_model, p_impl):
    diff = abs(p_model - p_impl)
    noise = diff > MARKET_NOISE_MAX
    kierunek = "Model wyzej niz rynek" if p_model > p_impl else "Rynek wyzej niz model"
    kolor = "#F44336" if noise else ("#FF9800" if diff > 0.15 else "#4CAF50")
    zgodnosc = max(0.0, 1.0 - diff / MARKET_NOISE_MAX)
    return {"noise": noise, "diff": diff, "kierunek": kierunek,
            "kolor": kolor, "zgodnosc_pct": zgodnosc}

def kelly_stake(p_model, kurs_buk, bankroll=KELLY_BANKROLL_DEFAULT,
               fraction=None, rynek: str = "1X2",
               already_exposed: float = 0.0) -> dict:
    """
    Oblicza stawkę Kelly z:
    - Per-rynek ułamkiem (1X2=0.25, gole=0.15, kartki/rożne=0.10)
    - Max Exposure per Match: suma stawek na jeden mecz ≤ MAX_EXPOSURE_PCT * bankroll
    - already_exposed: ile już postawiono na ten mecz (z innych rynków)
    """
    try:
        # Wybierz ułamek per rynek (lub override)
        if fraction is None:
            fraction = KELLY_FRACTIONS.get(rynek, KELLY_FRACTION)

        # Kalibracja liniowa p_model przed Kelly
        # Conservative Kelly: p_kelly = 0.5 + (p_model - 0.5) * KELLY_PROB_SCALE
        # Symulacja 5 lig × 2 sezony: -15% nadwyżki + Half-Kelly → MaxDD 55%→33%
        # Stosujemy TYLKO do Kelly, nie do wyświetlanego p_model ani EV
        p_kelly = 0.5 + (float(p_model) - 0.5) * KELLY_PROB_SCALE
        p_kelly = max(0.01, min(0.99, p_kelly))
        fraction = fraction * KELLY_FRAC_SCALE  # Half-Kelly

        # Rynki z fraction=0 są wyłączone z Kelly (tylko informacyjne)
        if fraction == 0.0:
            return {"f_full":0,"f_frac":0,"stake_pln":0,"ev_per_unit":0,
                    "safe":False,"rynek":rynek,"capped":False,
                    "disabled":True,"fraction_used":0}

        b = kurs_buk - 1.0
        q = 1.0 - p_kelly
        if b <= 0 or p_kelly <= 0 or p_kelly >= 1:
            return {"f_full":0,"f_frac":0,"stake_pln":0,"ev_per_unit":0,
                    "safe":False,"rynek":rynek,"capped":False}
        f_full = max(0.0, (p_kelly * b - q) / b)
        f_frac = f_full * fraction
        stake  = bankroll * f_frac

        # Max Exposure per Match – cap jeśli przekracza limit
        max_allowed = bankroll * MAX_EXPOSURE_PCT - already_exposed
        capped = False
        if stake > max_allowed > 0:
            stake  = max_allowed
            f_frac = stake / bankroll
            capped = True
        elif max_allowed <= 0:
            stake  = 0.0
            f_frac = 0.0
            capped = True

        ev_puu = p_kelly * b - q
        # EV filter: nie stawiamy jeśli EV < 5% (backtest: EV 0-5% → ROI -64%)
        ev_ok = ev_puu >= 0.05
        return {"f_full":round(f_full,4),"f_frac":round(f_frac,4),
                "stake_pln":round(stake, 2),"ev_per_unit":round(ev_puu,4),
                "safe": f_frac > 0 and ev_puu > 0 and ev_ok,
                "ev_ok": ev_ok,
                "rynek":rynek,"capped":capped,
                "fraction_used":round(fraction,3),
                "p_kelly":round(p_kelly,4)}
    except Exception:
        return {"f_full":0,"f_frac":0,"stake_pln":0,"ev_per_unit":0,
                "safe":False,"rynek":rynek,"capped":False}

def due_to_score_flag(team, srednie_df, historical):
    try:
        if "SOT (dom)" not in srednie_df.columns or team not in srednie_df.index:
            return None
        sot_d = float(srednie_df.loc[team, "SOT (dom)"])
        sot_w = float(srednie_df.loc[team, "SOT (wyjazd)"])
        sot_avg = (sot_d + sot_w) / 2
        if np.isnan(sot_avg): return None
        mecze_t = historical[(historical["HomeTeam"]==team)|(historical["AwayTeam"]==team)].tail(3)
        if len(mecze_t) < 3: return None
        gole = [int(m["FTHG"]) if m["HomeTeam"]==team else int(m["FTAG"])
                for _, m in mecze_t.iterrows()]
        avg_g = sum(gole)/len(gole)
        exp_g = sot_avg * 0.30
        if avg_g < exp_g * 0.60:
            return {"active":True,"sot_avg":round(sot_avg,1),"gole_last3":gole,
                    "avg_gole":round(avg_g,2),"expected":round(exp_g,2),
                    "deficit":round(exp_g-avg_g,2),
                    "msg":f"SOT avg {sot_avg:.1f} → tylko {avg_g:.1f} gola/mecz (ost.3). Oczekiwane odbicie."}
    except Exception:
        pass
    return None

# ── Bankroll persistence helpers ─────────────────────────────────────────
def wybierz_wynik(M: np.ndarray, lam_h: float, lam_a: float) -> tuple:
    idx_max = np.unravel_index(M.argmax(), M.shape)
    p_max = float(M[idx_max])
    top3  = np.sort(M.flatten())[::-1][:3]
    if float(top3[0] - top3[2]) < 0.04:
        g_h = min(int(round(lam_h)), M.shape[0] - 1)
        g_a = min(int(round(lam_a)), M.shape[1] - 1)
        return g_h, g_a, float(M[g_h, g_a])
    return int(idx_max[0]), int(idx_max[1]), p_max

def entropy_meczu(p_home: float, p_draw: float, p_away: float) -> float:
    ps = [p for p in [p_home, p_draw, p_away] if p > 0]
    return float(-sum(p * np.log2(p) for p in ps))

def chaos_label(entropy: float) -> tuple:
    pct = entropy / np.log2(3)
    if pct < 0.55:   return "Pewny",    "🔒", pct
    elif pct < 0.75: return "Klarowny", "🎯", pct
    elif pct < 0.90: return "Otwarty",  "⚡", pct
    else:            return "Chaos",    "🌀", pct


def sot_momentum(team: str, historical: pd.DataFrame) -> float | None:
    """
    SOT Momentum = (avg SOT ostatnie 3 mecze) / (avg SOT sezon).
    > 1.3 → drużyna 'puka do bramki' silniej niż zwykle (under-rewarded jeśli goli mało).
    < 0.7 → drużyna wygasła.
    Zwraca ratio lub None jeśli brak danych.
    """
    if historical.empty or "HST" not in historical.columns:
        return None
    try:
        home_m = historical[historical["HomeTeam"] == team][["Date", "FTHG", "HST"]].rename(
            columns={"HST": "sot", "FTHG": "gole"})
        away_m = historical[historical["AwayTeam"] == team][["Date", "FTAG", "AST"]].rename(
            columns={"AST": "sot", "FTAG": "gole"})
        all_m = pd.concat([home_m, away_m]).dropna(subset=["sot"])
        all_m = all_m.sort_values("Date") if "Date" in all_m.columns else all_m
        if len(all_m) < 5:
            return None
        season_avg = float(all_m["sot"].mean())
        recent_avg = float(all_m.tail(3)["sot"].mean())
        if season_avg < 0.5:
            return None
        return round(recent_avg / season_avg, 2)
    except Exception:
        return None


def quality_score(p_typ: float, entropy: float,
                  sot_mom_h: float | None, sot_mom_a: float | None,
                  ev: float = 0.0) -> tuple[int, str, str]:
    """
    Quality Score 1-5 gwiazdek dla zakładu.
    Kryteria:
      • p_typ    → siła sygnału modelu
      • entropy  → pewność (niski chaos)
      • SOT mom  → atak pod presją
      • EV       → wartość vs rynek
    Zwraca (gwiazdki: int, label: str, tooltip: str)
    """
    score = 0.0
    reasons = []

    # 1. Prawdopodobieństwo (0-2 pkt)
    if p_typ >= 0.75:
        score += 2.0; reasons.append("p≥75%")
    elif p_typ >= 0.62:
        score += 1.2; reasons.append("p≥62%")
    elif p_typ >= 0.55:
        score += 0.6; reasons.append("p≥55%")

    # 2. Entropy (chaos) (0-1.5 pkt)
    ent_norm = entropy / np.log2(3)
    if ent_norm < 0.55:
        score += 1.5; reasons.append("Chaos Low")
    elif ent_norm < 0.75:
        score += 0.9; reasons.append("Chaos Med")
    elif ent_norm < 0.90:
        score += 0.3; reasons.append("Chaos High")
    # chaos pełny: 0 pkt

    # 3. SOT Momentum (0-1 pkt) — atak pod presją
    mom_h = sot_mom_h or 1.0
    mom_a = sot_mom_a or 1.0
    best_mom = max(mom_h, mom_a)
    if best_mom >= 1.35:
        score += 1.0; reasons.append("🔥 SOT Momentum")
    elif best_mom >= 1.15:
        score += 0.5; reasons.append("SOT ↑")

    # 4. EV vs rynek (0-0.5 pkt)
    if ev >= 0.12:
        score += 0.5; reasons.append("EV≥12%")
    elif ev >= 0.06:
        score += 0.25; reasons.append("EV≥6%")

    # Normalizuj do 1-5 gwiazdek (max raw ≈ 5.0)
    stars = int(np.clip(round(score), 1, 5))
    star_str = "★" * stars + "☆" * (5 - stars)
    label = {5: "Wzorcowy", 4: "Silny", 3: "Dobry", 2: "Umiarkowany", 1: "Ryzykowny"}[stars]
    tooltip = " · ".join(reasons) if reasons else "brak sygnałów"
    return stars, star_str, label, tooltip


def entropy_kelly_multiplier(entropy: float) -> float:
    """
    Płynny bezpiecznik Kelly na podstawie chaosu macierzy.
    entropy ~0.5 (pewny) → 1.0×   (pełna stawka)
    entropy ~1.0 (otwarty) → 0.70×
    entropy ~1.3 (chaos) → 0.45×
    entropy ≥1.5 → 0.30×
    """
    # Liniowa interpolacja w kawałkach
    if entropy <= 0.5:  return 1.00
    if entropy <= 1.0:  return 1.00 - 0.30 * (entropy - 0.5) / 0.5   # 1.00 → 0.70
    if entropy <= 1.3:  return 0.70 - 0.25 * (entropy - 1.0) / 0.3   # 0.70 → 0.45
    return max(0.30, 0.45 - 0.15 * (entropy - 1.3) / 0.2)            # 0.45 → 0.30

def confidence_score(p_home: float, p_draw: float, p_away: float) -> tuple:
    vals  = sorted([p_home, p_draw, p_away], reverse=True)
    spread = vals[0] - vals[2]
    edge   = vals[0] - vals[1]
    if edge > 0.18:   return "High",     "🟢", f"Wyraźny faworyt (+{edge:.0%} nad 2. opcją)"
    elif edge > 0.08: return "Medium",   "🟡", f"Umiarkowana przewaga (+{edge:.0%})"
    else:             return "Coinflip", "🔴", f"Mecz bardzo wyrównany (spread {spread:.0%})"

def fair_odds(p: float) -> float:
    return round(1 / p, 2) if 0 < p <= 1 else 999.0

def predykcja_meczu(lam_h: float, lam_a: float, rho: float = -0.13, csv_code: str = "E0", n_train: int = 200) -> dict:
    max_gole = int(np.clip(np.ceil(max(lam_h, lam_a) + 5), 8, 12))
    M = dixon_coles_adj(
        np.outer(poisson.pmf(range(max_gole), lam_h),
                 poisson.pmf(range(max_gole), lam_a)),
        lam_h, lam_a, rho=rho
    )
    p_home = float(np.tril(M, -1).sum())
    p_draw = float(np.trace(M))
    p_away = float(np.triu(M, 1).sum())
    wynik_h, wynik_a, p_exact = wybierz_wynik(M, lam_h, lam_a)
    p_home_cal, p_draw_cal, p_away_cal = kalibruj_prawdopodobienstwa(p_home, p_draw, p_away, csv_code, n_train)
    typ, p_typ = wybierz_typ(p_home, p_draw, p_away, csv_code, n_train)
    conf_level, conf_emoji, conf_opis = confidence_score(p_home_cal, p_draw_cal, p_away_cal)
    ent = entropy_meczu(p_home_cal, p_draw_cal, p_away_cal)
    ch_label, ch_emoji, ch_pct = chaos_label(ent)
    return {
        "lam_h": lam_h, "lam_a": lam_a,
        "p_home": p_home_cal, "p_draw": p_draw_cal, "p_away": p_away_cal,
        "p_home_raw": p_home, "p_draw_raw": p_draw, "p_away_raw": p_away,
        "wynik_h": wynik_h, "wynik_a": wynik_a, "p_exact": p_exact,
        "fo_home": fair_odds(p_home_cal), "fo_draw": fair_odds(p_draw_cal), "fo_away": fair_odds(p_away_cal),
        "typ": typ, "p_typ": p_typ, "fo_typ": fair_odds(p_typ),
        "conf_level": conf_level, "conf_emoji": conf_emoji, "conf_opis": conf_opis,
        "entropy": ent, "chaos_label": ch_label, "chaos_emoji": ch_emoji, "chaos_pct": ch_pct,
        "macierz": M,
        "shrink_uzyte": _get_shrink(csv_code, n_train),
        "n_train": n_train,
        "ci_half": _get_shrink(csv_code, n_train) * 0.5,
    }

# ===========================================================================
# ALTERNATYWNE ZDARZENIA
# ===========================================================================
def alternatywne_zdarzenia(lam_h: float, lam_a: float, lam_r: float,
                            lam_k: float, rho: float,
                            prog_min: float = 0.55,
                            lam_sot: float = None) -> list:
    zdarzenia = []
    mg = int(np.clip(np.ceil(max(lam_h, lam_a) + 5), 8, 12))
    M = dixon_coles_adj(
        np.outer(poisson.pmf(range(mg), lam_h), poisson.pmf(range(mg), lam_a)),
        lam_h, lam_a, rho=rho
    )

    for linia in [1.5, 2.5, 3.5]:
        p_over = float(sum(M[i,j] for i in range(mg) for j in range(mg) if i+j > linia))
        p_under = 1 - p_over
        if p_over >= prog_min:
            zdarzenia.append(("⚽", f"Over {linia} goli",  p_over,  fair_odds(p_over),  "Gole", linia))
        if p_under >= prog_min:
            zdarzenia.append(("⚽", f"Under {linia} goli", p_under, fair_odds(p_under), "Gole", linia))

    p_btts   = float(1 - M[0,:].sum() - M[:,0].sum() + M[0,0])
    p_nobtts = 1 - p_btts
    if p_btts   >= prog_min: zdarzenia.append(("⚽", "BTTS – Tak",  p_btts,   fair_odds(p_btts),   "BTTS", 0))
    if p_nobtts >= prog_min: zdarzenia.append(("⚽", "BTTS – Nie",  p_nobtts, fair_odds(p_nobtts), "BTTS", 0))

    for linia in [7.5, 8.5, 9.5, 10.5]:
        p_over = float(1 - poisson.cdf(int(linia), lam_r))
        if p_over >= prog_min:
            zdarzenia.append(("🚩", f"Over {linia} rożnych", p_over, fair_odds(p_over), "Rożne", linia))

    for linia in [2.5, 3.5, 4.5]:
        p_over = float(1 - poisson.cdf(int(linia), lam_k))
        if p_over >= prog_min:
            zdarzenia.append(("🟨", f"Over {linia} kartek", p_over, fair_odds(p_over), "Kartki", linia))

    # Celne strzały (HST+AST) – Poisson, lam_sot przekazywane opcjonalnie
    # Filtr odds >= 1.30 – poniżej tej granicy rynek nie ma wartości
    SOT_MIN_ODDS = 1.30
    if lam_sot is not None and lam_sot > 0:
        for linia in [3.5, 4.5, 5.5, 6.5]:
            p_over = float(1 - poisson.cdf(int(linia), lam_sot))
            fo_sot = fair_odds(p_over)
            if p_over >= prog_min and fo_sot >= SOT_MIN_ODDS:
                zdarzenia.append(("🎯", f"Over {linia} celnych", p_over,
                                  fo_sot, "SOT", linia))

    return sorted(zdarzenia, key=lambda x: -x[2])

# ===========================================================================
# ZAPIS ZDARZEŃ DO BAZY (tracking skuteczności)
# ===========================================================================