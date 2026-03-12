"""
render.py – funkcje renderujące HTML: forma, macierz, statystyki, sędzia.
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson
from datetime import datetime

from config import LIGI, BIEZACY_SEZON, PROG_PEWNY, KALIBRACJA_PER_LIGA

def _forma_badge(litera: str) -> str:
    """Zwraca span z kolorowym tłem dla W/D/L."""
    colors = {"W": ("#1b5e20","#4CAF50"), "D": ("#4a3800","#FFC107"), "L": ("#7f0000","#ef5350")}
    bg, fg = colors.get(litera, ("#1a1a1a","#888"))
    return (f"<span style='background:{bg};color:{fg};padding:2px 6px;"
            f"border-radius:4px;font-weight:700;font-size:0.82em;margin:1px'>{litera}</span>")

def _forma_html(forma_str: str) -> str:
    """Render formy jako kolorowe badge'e."""
    if not forma_str or forma_str == "?": return "<span style='color:#555'>brak danych</span>"
    return "".join(_forma_badge(c) for c in forma_str)

def _trend_forma(forma_str: str) -> tuple:
    """Zwraca (trend_opis, trend_color)."""
    if not forma_str or len(forma_str) < 3:
        return "za mało meczów", "#555"
    ostatnie2 = forma_str[-2:]
    w = forma_str.count("W"); l = forma_str.count("L")
    if ostatnie2 in ("WW",) or w >= 4:
        return "↗ rosnąca forma", "#4CAF50"
    if ostatnie2 in ("LL",) or l >= 3:
        return "↘ spadkowa forma", "#ef5350"
    if w == l:
        return "→ stabilna forma", "#FFC107"
    return "→ zmienna forma", "#888"

def _stat_bar(val, avg_liga, label, unit="", higher_better=True) -> str:
    """Mini pasek porównania z ligową średnią."""
    if val is None or avg_liga is None or avg_liga == 0:
        return f"<span style='color:#555;font-size:0.78em'>{label}: brak</span>"
    ratio = val / avg_liga
    pct = min(ratio * 50, 100)  # 50% = średnia ligi
    c = "#4CAF50" if (ratio > 1.1 and higher_better) or (ratio < 0.9 and not higher_better)         else ("#ef5350" if (ratio < 0.9 and higher_better) or (ratio > 1.1 and not higher_better)         else "#FFC107")
    bar_w = int(min(pct, 100))
    diff = val - avg_liga
    sign = "+" if diff >= 0 else ""
    return (f"<div style='margin:3px 0'>"
            f"<div style='display:flex;justify-content:space-between;margin-bottom:2px'>"
            f"<span style='font-size:0.75em;color:#666'>{label}</span>"
            f"<span style='font-size:0.75em;color:{c};font-weight:600'>"
            f"{val:.2f}{unit} <span style='color:#444'>({sign}{diff:.2f})</span></span></div>"
            f"<div style='background:var(--bg-card2);border-radius:3px;height:4px'>"
            f"<div style='background:{c};width:{bar_w}%;height:4px;border-radius:3px'></div>"
            f"</div></div>")

def render_forma_kontekst(home: str, away: str, pred: dict,
                           forma_dict: dict, srednie_df, srednie_lig: dict,
                           odds_buk: dict = None) -> str:
    """Generuje HTML panel z formą drużyn i kontekstem statystycznym."""
    fh = forma_dict.get(home, "?")
    fa = forma_dict.get(away, "?")
    lh = float(pred.get("lam_h", 1.2))
    la = float(pred.get("lam_a", 1.0))
    ph = float(pred.get("p_home", 0.33))
    pd_ = float(pred.get("p_draw", 0.33))
    pa = float(pred.get("p_away", 0.33))
    ent = float(pred.get("entropy", 0))
    avg_h_lig = srednie_lig.get("avg_home", 1.5)
    avg_a_lig = srednie_lig.get("avg_away", 1.2)

    # Statystyki per drużyna z srednie_df
    def _get(team, col, default=None):
        try:
            v = srednie_df.loc[team, col]
            return float(v) if v is not None and str(v) != "nan" else default
        except Exception: return default

    h_gs   = _get(home, "Gole strzelone (dom)")
    h_gc   = _get(home, "Gole stracone (dom)")
    a_gs   = _get(away, "Gole strzelone (wyjazd)")
    a_gc   = _get(away, "Gole stracone (wyjazd)")
    h_sot  = _get(home, "SOT (dom)")
    a_sot  = _get(away, "SOT (wyjazd)")
    h_konw = _get(home, "Konwersja (dom)")
    a_konw = _get(away, "Konwersja (wyjazd)")
    avg_sot_h_lig = srednie_lig.get("avg_sot_home")
    avg_sot_a_lig = srednie_lig.get("avg_sot_away")

    trend_h, tc_h = _trend_forma(fh)
    trend_a, tc_a = _trend_forma(fa)

    # Konflikt forma vs model
    konflikty = []
    if ph > 0.50 and fh.count("L") >= 3:
        konflikty.append(f"⚠️ {home}: słaba forma [{fh}] vs model faworyt (λ={lh:.2f}) – model widzi siłę statystyczną, nie serię")
    elif pa > 0.50 and fa.count("L") >= 3:
        konflikty.append(f"⚠️ {away}: słaba forma [{fa}] vs model faworyt (λ={la:.2f}) – model widzi siłę statystyczną, nie serię")
    if ph > 0.50 and fh.count("W") >= 4:
        konflikty.append(f"✅ {home}: forma [{fh}] zbieżna z modelem")
    if pa > 0.50 and fa.count("W") >= 4:
        konflikty.append(f"✅ {away}: forma [{fa}] zbieżna z modelem")

    # Deep value vs bukmacher
    dv_html = ""
    if odds_buk:
        try:
            typ = pred.get("typ","?")
            k_buk = {"1": float(odds_buk.get("odds_h",0) or 0),
                     "X": float(odds_buk.get("odds_d",0) or 0),
                     "2": float(odds_buk.get("odds_a",0) or 0)}.get(typ, 0)
            fo = float(pred.get("fo_typ", 0) or 0)
            if k_buk > 1 and fo > 1:
                rozb = (k_buk - fo) / fo
                if abs(rozb) >= 0.08:
                    dv_c = "#4CAF50" if rozb > 0 else "#ef5350"
                    dv_label = "⚡ Rynek płaci więcej niż fair" if rozb > 0 else "⚠️ Rynek poniżej fair"
                    dv_html = (f"<div style='margin-top:8px;background:var(--bg-card);border-left:3px solid {dv_c};"
                               f"padding:6px 10px;border-radius:0 4px 4px 0;font-size:0.8em'>"
                               f"<span style='color:{dv_c};font-weight:700'>{dv_label}:</span> "
                               f"fair={fo:.2f} · buk={k_buk:.2f} · różnica {rozb:+.0%}</div>")
        except Exception: pass

    # Entropia – jak przewidywalny mecz?
    if ent < 0.85:   chaos_label, chaos_c = "Łatwy do typowania", "#4CAF50"
    elif ent < 1.10: chaos_label, chaos_c = "Umiarkowanie otwarty", "#FFC107"
    else:            chaos_label, chaos_c = "Losowy mecz", "#ef5350"

    # Buduj HTML
    konflikty_html = ""
    if konflikty:
        konflikty_html = "".join(
            f"<div style='font-size:0.78em;color:#aaa;margin:3px 0;padding:4px 8px;"
            f"background:var(--bg-card);border-radius:4px'>{k}</div>"
            for k in konflikty)

    html = (
        f"<div style='background:var(--bg-app);border:1px solid var(--border);border-radius:8px;"
        f"padding:12px 14px;font-family:inherit'>"
        # Wiersz drużyn
        f"<div style='display:grid;grid-template-columns:1fr auto 1fr;gap:12px;align-items:start'>"
        # Lewa: dom
        f"<div>"
        f"<div style='font-size:0.8em;color:#555;margin-bottom:4px'>🏠 {home}</div>"
        f"<div style='margin-bottom:6px'>{_forma_html(fh)}</div>"
        f"<div style='font-size:0.75em;color:{tc_h}'>{trend_h}</div>"
        + _stat_bar(h_gs, avg_h_lig, "Gole strzel. (dom)")
        + _stat_bar(h_gc, avg_h_lig, "Gole strac. (dom)", higher_better=False)
        + ((_stat_bar(h_sot, avg_sot_h_lig, "SOT (dom)")) if h_sot and avg_sot_h_lig else "")
        + f"</div>"
        # Środek: szanse
        f"<div style='text-align:center;padding:0 8px'>"
        f"<div style='font-size:0.7em;color:#444;margin-bottom:4px'>MODEL</div>"
        f"<div style='font-size:1.1em;font-weight:800;color:#eee'>"
        f"<span style='color:#2196F3'>{ph:.0%}</span> · "
        f"<span style='color:#FF9800'>{pd_:.0%}</span> · "
        f"<span style='color:#E91E63'>{pa:.0%}</span></div>"
        f"<div style='font-size:0.65em;color:#444;margin-top:2px'>1 · X · 2</div>"
        f"<div style='margin-top:8px;font-size:0.7em;color:{chaos_c}'>{chaos_label}</div>"
        f"<div style='font-size:0.68em;color:#444'>H={ent:.2f}</div>"
        f"<div style='margin-top:8px'>"
        f"<div style='font-size:0.7em;color:#444'>λ domu</div>"
        f"<div style='font-size:0.95em;font-weight:700;color:#eee'>{lh:.2f}</div>"
        f"<div style='font-size:0.7em;color:#444;margin-top:4px'>λ gości</div>"
        f"<div style='font-size:0.95em;font-weight:700;color:#eee'>{la:.2f}</div>"
        f"</div></div>"
        # Prawa: wyjazd
        f"<div style='text-align:right'>"
        f"<div style='font-size:0.8em;color:#555;margin-bottom:4px'>{away} ✈️</div>"
        f"<div style='margin-bottom:6px;text-align:right'>{_forma_html(fa)}</div>"
        f"<div style='font-size:0.75em;color:{tc_a};text-align:right'>{trend_a}</div>"
        + _stat_bar(a_gs, avg_a_lig, "Gole strzel. (wyjazd)")
        + _stat_bar(a_gc, avg_a_lig, "Gole strac. (wyjazd)", higher_better=False)
        + ((_stat_bar(a_sot, avg_sot_a_lig, "SOT (wyjazd)")) if a_sot and avg_sot_a_lig else "")
        + f"</div>"
        f"</div>"
        # Dół: konflikty + deep value
        + (f"<div style='margin-top:8px;border-top:1px solid #1e2028;padding-top:8px'>{konflikty_html}</div>" if konflikty_html else "")
        + dv_html
        + f"</div>"
    )
    return html


# ===========================================================================
# HELPERS UI
# ===========================================================================
def koloruj(p: float) -> str:
    return "🟢" if p > 0.65 else ("🟡" if p > 0.50 else "🔴")

def oblicz_p(typ: str, linia: float, lam: float) -> float:
    return 1 - poisson.cdf(linia, lam) if typ == "Over" else poisson.cdf(linia, lam)

def badge_typ(typ: str) -> str:
    if   typ == "1X": styl = "background:linear-gradient(90deg,#2196F3,#FF9800)"
    elif typ == "X2": styl = "background:linear-gradient(90deg,#FF9800,#E91E63)"
    else:
        kolory = {"1": "#2196F3", "X": "#FF9800", "2": "#E91E63"}
        styl   = f"background:{kolory.get(typ, '#888')}"
    return (f"<span style='{styl};color:var(--text-primary);padding:2px 14px;"
            f"border-radius:12px;font-weight:bold;font-size:0.95em'>{typ}</span>")

def render_macierz_html(M: np.ndarray, home: str, away: str) -> str:
    max_val = M.max() if M.max() > 0 else 1
    best    = np.unravel_index(M.argmax(), M.shape)
    n       = M.shape[0]
    rows    = ["<table style='border-collapse:collapse;font-size:0.70em;width:100%;margin-top:6px'>",
               "<thead><tr>",
               f"<th style='padding:2px 5px;color:#999;font-size:0.85em'>{home[:7]}↓ {away[:7]}→</th>"]
    for j in range(n):
        rows.append(f"<th style='padding:2px 5px;text-align:center;color:#aaa'>{j}</th>")
    rows.append("</tr></thead><tbody>")
    for i in range(n):
        rows.append("<tr>")
        rows.append(f"<td style='padding:2px 5px;font-weight:bold;color:#aaa;text-align:center'>{i}</td>")
        for j in range(n):
            v = M[i,j]; ratio = v / max_val
            g = int(255 * (1 - ratio * 0.85))
            bold = "font-weight:bold;" if (i,j) == best else ""
            bord = "border:2px solid #E91E63;" if (i,j) == best else ""
            rows.append(f"<td style='padding:2px 5px;text-align:center;background:rgb(255,{g},{g});{bold}{bord}'>{v:.1%}</td>")
        rows.append("</tr>")
    rows.append("</tbody></table>")
    return "".join(rows)

def macierz_goli_p(lam_h, lam_a, rho, linia_int, typ_gole):
    mg = int(np.clip(np.ceil(max(lam_h, lam_a) + 5), 8, 12))
    M  = dixon_coles_adj(
        np.outer(poisson.pmf(range(mg), lam_h), poisson.pmf(range(mg), lam_a)),
        lam_h, lam_a, rho=rho
    )
    p_over = float(sum(M[i,j] for i in range(mg) for j in range(mg) if i+j > linia_int))
    return p_over if typ_gole == "Over" else 1 - p_over

# ===========================================================================
# NOWA FUNKCJA – OSTRZEŻENIA SĘDZIOWSKIE
# ===========================================================================
def ostrzezenie_sedziego(sedzia, sedziowie_df, prog_kartki=4.5):
    """Zwraca ostrzeżenie dotyczące sędziego lub None gdy brak danych.
    Dane sędziów dostępne tylko dla Premier League (kolumna Referee w CSV).
    """
    if sedziowie_df.empty or sedzia in ("Nieznany", "", None):
        return None
    if sedzia not in sedziowie_df["Sędzia"].values:
        return None
    sedz = sedziowie_df[sedziowie_df["Sędzia"] == sedzia].iloc[0]
    avg_cards = sedz["Total Kart/M ↓"]
    if avg_cards >= prog_kartki:
        return f"🚩 Wysokie ryzyko kartek – śr. {avg_cards:.1f}/mecz"
    elif avg_cards >= 3.5:
        return f"🟡 Umiarkowane ryzyko kartek ({avg_cards:.1f}/mecz)"
    else:
        return f"🟢 Niskie ryzyko kartek ({avg_cards:.1f}/mecz)"

# ===========================================================================
# DEEP DATA – Power Rankings + Sędziowie
# ===========================================================================
@st.cache_data
def deep_data_stats(df_json: str, druzyny_ligi: set = None) -> tuple:
    """Zwraca (power_df, sedziowie_df) dla tab Deep Data.
    druzyny_ligi: jeśli podany, pokazuje tylko drużyny bieżącego sezonu (bez spadkowiczów).
    """
    df = pd.read_json(df_json)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # ── Power Rankings ────────────────────────────────────────────────────
    druzyny_all = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
    druzyny = [d for d in druzyny_all if d in druzyny_ligi] if druzyny_ligi else druzyny_all
    power_rows = []
    for d in druzyny:
        h_df = df[df["HomeTeam"] == d]
        a_df = df[df["AwayTeam"] == d]
        all_m = pd.concat([
            h_df.assign(_gole_str=h_df["FTHG"], _gole_strac=h_df["FTAG"],
                        _sot=h_df.get("HST", np.nan), _kartki=h_df["total_kartki"],
                        _rozne=h_df["total_rozne"]),
            a_df.assign(_gole_str=a_df["FTAG"], _gole_strac=a_df["FTHG"],
                        _sot=a_df.get("AST", np.nan), _kartki=a_df["total_kartki"],
                        _rozne=a_df["total_rozne"])
        ])
        if len(all_m) < 3:
            continue
        gol_str  = all_m["_gole_str"].mean()
        gol_strac = all_m["_gole_strac"].mean()
        sot_sr   = all_m["_sot"].dropna().mean() if all_m["_sot"].notna().any() else None
        kart_sr  = all_m["_kartki"].mean()
        rozne_sr = all_m["_rozne"].mean()
        konv     = (gol_str / sot_sr) if (sot_sr and sot_sr > 0) else None
        # xG-proxy: SOT × liga_średnia_konwersji (szacunkowa)
        xg_proxy = (sot_sr * 0.11) if sot_sr else None  # ~11% konwersja PL
        # Forma ostatnie 5
        mecze5 = df[(df["HomeTeam"]==d)|(df["AwayTeam"]==d)].tail(5)
        form5_pts = 0
        for _, m5 in mecze5.iterrows():
            if m5["HomeTeam"] == d:
                if m5["FTHG"] > m5["FTAG"]: form5_pts += 3
                elif m5["FTHG"] == m5["FTAG"]: form5_pts += 1
            else:
                if m5["FTAG"] > m5["FTHG"]: form5_pts += 3
                elif m5["FTAG"] == m5["FTHG"]: form5_pts += 1
        power_rows.append({
            "Drużyna":         d,
            "M":               len(all_m),
            "Gole/M ↑":        round(gol_str, 2),
            "Strac./M ↓":      round(gol_strac, 2),
            "SOT/M":           round(sot_sr, 1) if sot_sr else "–",
            "Konwersja%":      f"{konv:.1%}" if konv else "–",
            "xG-proxy":        round(xg_proxy, 2) if xg_proxy else "–",
            "Kartki/M":        round(kart_sr, 1),
            "Rożne/M":         round(rozne_sr, 1),
            "Forma (pkt/5M)":  form5_pts,
            "_gol_str":        gol_str,
            "_gol_strac":      gol_strac,
            "_forma":          form5_pts,
        })
    power_df = pd.DataFrame(power_rows)

    # ── Sędziowie ────────────────────────────────────────────────────────
    sedzio_df = pd.DataFrame()
    if "Referee" in df.columns:
        ref_grp = df.groupby("Referee").agg(
            Meczów=("Referee", "count"),
            Kartki_Y_avg=("HY", lambda x: (x + df.loc[x.index, "AY"]).mean()),
            Kartki_R_avg=("HR", lambda x: (x + df.loc[x.index, "AR"]).mean()),
            Gole_avg=("total_gole", "mean"),
        ).reset_index()
        # Łączna suma kartek (Yellow + 2*Red) per mecz
        ref_grp2 = []
        for ref, grp in df.groupby("Referee"):
            n = len(grp)
            if n < 3:
                continue
            y_avg  = (grp["HY"] + grp["AY"]).mean()
            r_avg  = (grp["HR"] + grp["AR"]).mean()
            tot_k  = y_avg + r_avg * 2
            g_avg  = grp["total_gole"].mean()
            ref_grp2.append({
                "Sędzia": ref, "Meczów": n,
                "Kartki Y/M": round(y_avg, 1),
                "Kartki R/M": round(r_avg, 2),
                "Total Kart/M ↓": round(tot_k, 1),
                "Gole/M": round(g_avg, 1),
                "_tot_k": tot_k,
            })
        sedzio_df = pd.DataFrame(ref_grp2).sort_values("_tot_k", ascending=False) if ref_grp2 else pd.DataFrame()

    return power_df, sedzio_df

# ===========================================================================
# ŁADOWANIE DANYCH I SIDEBAR
# ===========================================================================
# ── SIDEBAR ─────────────────────────────────────────────────────────────────
_liga_list = list(LIGI.keys())
if "._liga_override" in str(st.session_state):
    pass
_liga_default_idx = _liga_list.index(st.session_state.get("_liga_override", _liga_list[0])) \
    if st.session_state.get("_liga_override") in _liga_list else 0
wybrana_liga = st.sidebar.selectbox("🌍 Liga", _liga_list, index=_liga_default_idx)
debug_mode   = False  # ustawiane w expander Parametry niżej

# ── Kursy live (The Odds API) ─────────────────────────────────────────────
_CSV_CODE  = LIGI[wybrana_liga]["csv_code"]
_OA_DB     = "predykcje.db"
_oa_cached = {}
