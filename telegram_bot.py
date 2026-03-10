"""
telegram_bot.py – moduł powiadomień Telegram dla aplikacji predykcji piłkarskich.

Funkcje:
  send_value_alert(mecz, pred, kurs, kelly)  – powiadomienie o value becie (EV>X%)
  send_morning_digest(liga, value_bets)       – poranny digest w dzień kolejki
  poll_commands(token, chat_id)               – obsługa komend /value, /status, /help
  should_send_morning(liga)                   – sprawdza czy pora na poranny digest

Komendy obsługiwane przez bota:
  /value   – lista aktualnych value betów (wszystkie ligi)
  /status  – statystyki modelu (hit rate, ROI per liga)
  /help    – pomoc

Logika anti-spam:
  - Alerty real-time: tylko przy EV >= progu (domyślnie 15%) – max 1/mecz/dzień
  - Poranny digest: raz dziennie, 8:00-9:00, tylko w dzień meczu kolejki
  - Komendy: zawsze na życzenie
"""

import requests
import json
import sqlite3
from datetime import datetime, timedelta
from collections import defaultdict
import streamlit as st

# ── Konfiguracja ──────────────────────────────────────────────────────────────
TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"
DIGEST_HOUR_FROM = 8    # poranny digest od tej godziny
DIGEST_HOUR_TO   = 10   # do tej godziny
MIN_EV_ALERT     = 0.12  # minimalny EV dla real-time alertu (12%)
MIN_EV_DIGEST    = 0.08  # minimalny EV dla porannego digestu (8%)


def _get_credentials():
    """Pobierz token i chat_id z Streamlit secrets."""
    try:
        token   = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "")
        return token, str(chat_id)
    except Exception:
        return "", ""


def _api(token: str, method: str, **kwargs) -> dict:
    """Wywołaj metodę Telegram Bot API. Zwraca dict z odpowiedzią."""
    try:
        url = TELEGRAM_API.format(token=token, method=method)
        r = requests.post(url, json=kwargs, timeout=8)
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}


def send_message(text: str, token: str = "", chat_id: str = "",
                 parse_mode: str = "HTML", disable_preview: bool = True) -> bool:
    """Wyślij wiadomość tekstową. Zwraca True jeśli sukces."""
    if not token or not chat_id:
        token, chat_id = _get_credentials()
    if not token or not chat_id:
        return False
    res = _api(token, "sendMessage",
               chat_id=chat_id,
               text=text,
               parse_mode=parse_mode,
               disable_web_page_preview=disable_preview)
    return res.get("ok", False)


# ── Anti-spam: SQLite log wysłanych alertów ──────────────────────────────────
_ALERT_DB = "telegram_alerts.db"

def _init_alert_db():
    con = sqlite3.connect(_ALERT_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS sent_alerts (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_key TEXT UNIQUE,
        sent_at   TEXT
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS digest_log (
        id      INTEGER PRIMARY KEY AUTOINCREMENT,
        liga    TEXT,
        date    TEXT,
        UNIQUE(liga, date)
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS bot_offset (
        id     INTEGER PRIMARY KEY,
        offset INTEGER DEFAULT 0
    )""")
    con.execute("INSERT OR IGNORE INTO bot_offset(id, offset) VALUES (1, 0)")
    con.commit()
    con.close()

_init_alert_db()


def _alert_already_sent(key: str) -> bool:
    """Sprawdź czy alert o danym kluczu (mecz+typ) był dziś wysłany."""
    today = datetime.now().strftime("%Y-%m-%d")
    full_key = f"{today}_{key}"
    con = sqlite3.connect(_ALERT_DB)
    row = con.execute("SELECT 1 FROM sent_alerts WHERE alert_key=?", (full_key,)).fetchone()
    con.close()
    return row is not None


def _mark_alert_sent(key: str):
    today = datetime.now().strftime("%Y-%m-%d")
    full_key = f"{today}_{key}"
    con = sqlite3.connect(_ALERT_DB)
    con.execute("INSERT OR IGNORE INTO sent_alerts(alert_key, sent_at) VALUES (?,?)",
                (full_key, datetime.now().isoformat()))
    con.commit()
    con.close()


def _digest_already_sent(liga: str) -> bool:
    today = datetime.now().strftime("%Y-%m-%d")
    con = sqlite3.connect(_ALERT_DB)
    row = con.execute("SELECT 1 FROM digest_log WHERE liga=? AND date=?",
                      (liga, today)).fetchone()
    con.close()
    return row is not None


def _mark_digest_sent(liga: str):
    today = datetime.now().strftime("%Y-%m-%d")
    con = sqlite3.connect(_ALERT_DB)
    con.execute("INSERT OR IGNORE INTO digest_log(liga, date) VALUES (?,?)",
                (liga, today))
    con.commit()
    con.close()


# ── Formatowanie wiadomości ───────────────────────────────────────────────────
_LIGA_EMOJI = {
    "Premier League": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "La Liga":        "🇪🇸",
    "Bundesliga":     "🇩🇪",
    "Serie A":        "🇮🇹",
    "Ligue 1":        "🇫🇷",
}

def _fmt_bet(home: str, away: str, typ: str, kurs: float,
             p_model: float, ev: float, stake_pln: float,
             kolejka: int = None) -> str:
    """Formatuj jeden value bet do wiadomości Telegram (HTML)."""
    ev_c  = "🟢" if ev >= 0.15 else "🟡"
    kol_s = f" · kolejka {kolejka}" if kolejka else ""
    return (
        f"{ev_c} <b>{home} vs {away}</b>{kol_s}\n"
        f"   Typ: <code>{typ}</code> · Kurs: <b>{kurs:.2f}</b>\n"
        f"   P model: {p_model:.1%} · EV: <b>{ev:+.1%}</b>\n"
        f"   Stawka Kelly: <b>{stake_pln:.0f} zł</b>"
    )


def send_value_alert(home: str, away: str, liga: str, typ: str,
                     kurs: float, p_model: float, ev: float,
                     stake_pln: float, kolejka: int = None) -> bool:
    """
    Wyślij real-time alert o value becie.
    Anti-spam: max 1 alert per mecz+typ na dzień, min EV=12%.
    """
    if ev < MIN_EV_ALERT:
        return False
    alert_key = f"{home}_{away}_{typ}".replace(" ", "_")
    if _alert_already_sent(alert_key):
        return False

    flag = _LIGA_EMOJI.get(liga, "⚽")
    kol_s = f" · kolejka {kolejka}" if kolejka else ""
    text = (
        f"🔔 <b>VALUE BET</b> {flag} {liga}{kol_s}\n\n"
        f"{_fmt_bet(home, away, typ, kurs, p_model, ev, stake_pln, kolejka)}"
    )
    ok = send_message(text)
    if ok:
        _mark_alert_sent(alert_key)
    return ok


def send_morning_digest(liga: str, value_bets: list, kolejka: int = None,
                        force: bool = False) -> bool:
    """
    Poranny digest z listą value betów w dniu kolejki.
    value_bets: lista dict {home, away, typ, kurs, p_model, ev, stake_pln}
    force=True: wyślij nawet poza oknem godzinowym (do testów / na życzenie)
    """
    if not force and _digest_already_sent(liga):
        return False

    now = datetime.now()
    in_window = DIGEST_HOUR_FROM <= now.hour < DIGEST_HOUR_TO
    if not force and not in_window:
        return False

    # Filtruj po MIN_EV_DIGEST
    vbs = [b for b in value_bets if b.get("ev", 0) >= MIN_EV_DIGEST]
    if not vbs and not force:
        return False

    flag = _LIGA_EMOJI.get(liga, "⚽")
    kol_s = f" · kolejka {kolejka}" if kolejka else ""
    date_s = now.strftime("%d.%m.%Y")

    if not vbs:
        text = (
            f"☀️ <b>Poranny raport</b> {flag} {liga}{kol_s} · {date_s}\n\n"
            f"Brak value betów (EV ≥ {MIN_EV_DIGEST:.0%}) w tej kolejce.\n"
            f"Model pracuje — sprawdź aplikację po odświeżeniu kursów."
        )
    else:
        bets_txt = "\n\n".join(
            _fmt_bet(b["home"], b["away"], b["typ"], b["kurs"],
                     b["p_model"], b["ev"], b["stake_pln"], kolejka)
            for b in sorted(vbs, key=lambda x: -x["ev"])
        )
        text = (
            f"☀️ <b>Poranny raport</b> {flag} {liga}{kol_s} · {date_s}\n\n"
            f"<b>{len(vbs)} value bet{'ów' if len(vbs)!=1 else ''}</b> "
            f"z EV ≥ {MIN_EV_DIGEST:.0%}:\n\n"
            f"{bets_txt}\n\n"
            f"<i>Fair Odds z modelu Dixon-Coles · Stawki Conservative Kelly 1/8</i>"
        )

    ok = send_message(text)
    if ok and not force:
        _mark_digest_sent(liga)
    return ok


def should_send_morning(liga: str) -> bool:
    """Czy powinienem teraz wysłać poranny digest dla danej ligi?"""
    now = datetime.now()
    in_window = DIGEST_HOUR_FROM <= now.hour < DIGEST_HOUR_TO
    return in_window and not _digest_already_sent(liga)


# ── Obsługa komend bota ───────────────────────────────────────────────────────
def _get_offset() -> int:
    con = sqlite3.connect(_ALERT_DB)
    row = con.execute("SELECT offset FROM bot_offset WHERE id=1").fetchone()
    con.close()
    return row[0] if row else 0


def _set_offset(offset: int):
    con = sqlite3.connect(_ALERT_DB)
    con.execute("UPDATE bot_offset SET offset=? WHERE id=1", (offset,))
    con.commit()
    con.close()


def poll_commands(all_value_bets_fn=None, stats_fn=None) -> list:
    """
    Sprawdź nowe wiadomości od użytkownika i odpowiedz na komendy.
    all_value_bets_fn: callable() → lista dictów z value betami ze wszystkich lig
    stats_fn:          callable() → dict z {'liga': {'hr':%, 'roi':%, 'n':int}}
    Zwraca listę przetworzonych komend.
    """
    token, chat_id = _get_credentials()
    if not token or not chat_id:
        return []

    offset = _get_offset()
    res = _api(token, "getUpdates", offset=offset, timeout=5)
    if not res.get("ok"):
        return []

    processed = []
    for update in res.get("result", []):
        uid = update["update_id"]
        _set_offset(uid + 1)

        msg = update.get("message", {})
        text = msg.get("text", "").strip().lower()
        from_id = str(msg.get("chat", {}).get("id", ""))

        # Odpowiadaj tylko na wiadomości z chat_id właściciela
        if from_id != chat_id:
            continue

        if text.startswith("/value"):
            _handle_value_command(token, chat_id, all_value_bets_fn)
            processed.append("value")

        elif text.startswith("/status"):
            _handle_status_command(token, chat_id, stats_fn)
            processed.append("status")

        elif text.startswith("/help"):
            _handle_help_command(token, chat_id)
            processed.append("help")

        elif text.startswith("/digest"):
            # Wymuś poranny digest na życzenie
            processed.append("digest_requested")

    return processed


def _handle_value_command(token: str, chat_id: str, all_value_bets_fn):
    """Odpowiedz na /value – lista aktualnych value betów ze wszystkich lig."""
    if all_value_bets_fn is None:
        send_message("⏳ Dane nie są dostępne – otwórz aplikację aby odświeżyć.", token, chat_id)
        return

    try:
        bets = all_value_bets_fn()
    except Exception as e:
        send_message(f"❌ Błąd pobierania danych: {e}", token, chat_id)
        return

    # Filtruj: p_model >= 0.58 (identyczny próg jak na stronie)
    # Bez live API kurs = fair odds → EV ≈ 0, więc filtrujemy po pewności modelu
    vbs = [b for b in bets if b.get("p_model", 0) >= 0.58]

    if not vbs:
        send_message(
            "🔍 Brak typów z pewnością modelu ≥ 58% w żadnej lidze.\n\n"
            "Sprawdź ponownie gdy kolejka zostanie zaplanowana.",
            token, chat_id)
        return

    # Czy mamy live kursy z API?
    has_live = any(b.get("live_odds") for b in vbs)
    live_note = "" if has_live else "\n<i>⚠️ Brak live kursów (The Odds API) – kurs = fair odds modelu</i>"

    # Grupuj per liga, sortuj po p_model malejąco
    by_liga = defaultdict(list)
    for b in vbs:
        by_liga[b.get("liga", "?")].append(b)

    now = datetime.now().strftime("%d.%m %H:%M")
    sections = [f"💰 <b>Value Bets – wszystkie ligi</b>\n🕐 {now} · {len(vbs)} typów{live_note}\n"]

    for liga, liga_bets in by_liga.items():
        flag = _LIGA_EMOJI.get(liga, "⚽")
        liga_bets_sorted = sorted(liga_bets, key=lambda x: -x["p_model"])[:5]  # max 5 per liga
        sections.append(f"\n{flag} <b>{liga}</b>")
        for b in liga_bets_sorted:
            _data = f" · {b['data']}" if b.get("data") else ""
            _kurs_tag = "🔴live" if b.get("live_odds") else "✦fair"
            _ev_str = f" · EV: {b['ev']:+.1%}" if abs(b.get("ev", 0)) > 0.001 else ""
            _stake = f"\n  🏦 Kelly: {b['stake_pln']:.0f} zł" if b.get("stake_pln", 0) > 5 else ""
            sections.append(
                f"  <b>{b['home']} – {b['away']}</b>{_data}\n"
                f"  Typ: <b>{b['typ']}</b> @ <b>{b['kurs']:.2f}</b> {_kurs_tag}"
                f" · p=<b>{b['p_model']:.0%}</b>{_ev_str}{_stake}"
            )

    text = "\n".join(sections)
    if len(text) > 4000:
        text = text[:3900] + "\n\n<i>…(skrócono)</i>"
    send_message(text, token, chat_id)


def _handle_status_command(token: str, chat_id: str, stats_fn):
    """Odpowiedz na /status – statystyki per liga."""
    if stats_fn is None:
        send_message("⏳ Statystyki niedostępne – otwórz aplikację.", token, chat_id)
        return

    try:
        stats = stats_fn()
    except Exception as e:
        send_message(f"❌ Błąd: {e}", token, chat_id)
        return

    if not stats:
        send_message("📭 Brak danych w bazie. Zacznij typować w aplikacji.", token, chat_id)
        return

    lines = []
    for liga, s in stats.items():
        flag = _LIGA_EMOJI.get(liga, "⚽")
        hr   = s.get("hr", 0)
        roi  = s.get("roi", 0)
        n    = s.get("n", 0)
        hr_c = "🟢" if hr >= 0.60 else ("🟡" if hr >= 0.50 else "🔴")
        roi_c = "🟢" if roi >= 0 else "🔴"
        lines.append(
            f"{flag} <b>{liga}</b>\n"
            f"   {hr_c} Hit: {hr:.1%} · {roi_c} ROI: {roi:+.1f}% · {n} typów"
        )

    now = datetime.now().strftime("%d.%m %H:%M")
    text = f"📊 <b>Status modelu</b> · {now}\n\n" + "\n\n".join(lines)
    send_message(text, token, chat_id)


def _handle_help_command(token: str, chat_id: str):
    """Odpowiedz na /help."""
    text = (
        "🤖 <b>Komendy bota predykcji</b>\n\n"
        "/value  – aktualne value bety (EV ≥ 8%)\n"
        "/status – hit rate i ROI per liga\n"
        "/help   – ta wiadomość\n\n"
        "<i>Powiadomienia automatyczne:</i>\n"
        "• ☀️ Poranny digest – w dzień kolejki (8-10 rano)\n"
        "• 🔔 Alert real-time – EV ≥ 12% (max 1/mecz/dzień)"
    )
    send_message(text, token, chat_id)
