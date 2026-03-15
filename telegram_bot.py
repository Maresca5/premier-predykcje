"""
telegram_bot.py – moduł powiadomień Telegram dla aplikacji predykcji piłkarskich.

Strategia powiadomień (bez spamu):
  1. DIGEST DZIENNY – jedno zbiorcze powiadomienie dziennie (8-10 rano)
     z wszystkimi value betami EV ≥ 8%. Max 1x na dobę per liga.
  2. KOMENDY – zawsze na życzenie:
     /value  – aktualne value bety (wszystkie ligi)
     /status – hit rate i ROI per liga
     /help   – pomoc

NIE MA real-time alertów per-mecz — powodowały spam przy każdym rerun Streamlit.
"""

import requests
import sqlite3
from datetime import datetime
from collections import defaultdict
import streamlit as st

# ── Konfiguracja ──────────────────────────────────────────────────────────────
TELEGRAM_API     = "https://api.telegram.org/bot{token}/{method}"
DIGEST_HOUR_FROM = 8     # poranny digest od tej godziny
DIGEST_HOUR_TO   = 10    # do tej godziny
MIN_EV_ALERT     = 0.12  # próg dla /value komendy (zostawiony dla kompatybilności)
MIN_EV_DIGEST    = 0.08  # minimalny EV dla porannego digestu (8%)

_LIGA_EMOJI = {
    "Premier League": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "La Liga":        "🇪🇸",
    "Bundesliga":     "🇩🇪",
    "Serie A":        "🇮🇹",
    "Ligue 1":        "🇫🇷",
}


# ── Credentials ───────────────────────────────────────────────────────────────
def _get_credentials():
    try:
        token   = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "")
        return token, str(chat_id)
    except Exception:
        return "", ""


def _api(token: str, method: str, **kwargs) -> dict:
    try:
        url = TELEGRAM_API.format(token=token, method=method)
        r = requests.post(url, json=kwargs, timeout=8)
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}


def send_message(text: str, token: str = "", chat_id: str = "",
                 parse_mode: str = "HTML", disable_preview: bool = True) -> bool:
    if not token or not chat_id:
        token, chat_id = _get_credentials()
    if not token or not chat_id:
        return False
    res = _api(token, "sendMessage",
               chat_id=chat_id, text=text,
               parse_mode=parse_mode,
               disable_web_page_preview=disable_preview)
    return res.get("ok", False)


# ── Anti-spam DB ──────────────────────────────────────────────────────────────
_ALERT_DB = "telegram_alerts.db"

def _init_alert_db():
    con = sqlite3.connect(_ALERT_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS digest_log (
        id   INTEGER PRIMARY KEY AUTOINCREMENT,
        liga TEXT, date TEXT,
        UNIQUE(liga, date)
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS bot_offset (
        id INTEGER PRIMARY KEY, offset INTEGER DEFAULT 0
    )""")
    con.execute("INSERT OR IGNORE INTO bot_offset(id, offset) VALUES (1, 0)")
    con.commit()
    con.close()

_init_alert_db()


def _digest_already_sent(liga: str) -> bool:
    today = datetime.now().strftime("%Y-%m-%d")
    con   = sqlite3.connect(_ALERT_DB)
    row   = con.execute(
        "SELECT 1 FROM digest_log WHERE liga=? AND date=?", (liga, today)
    ).fetchone()
    con.close()
    return row is not None


def _mark_digest_sent(liga: str):
    today = datetime.now().strftime("%Y-%m-%d")
    con   = sqlite3.connect(_ALERT_DB)
    con.execute("INSERT OR IGNORE INTO digest_log(liga, date) VALUES (?,?)",
                (liga, today))
    con.commit()
    con.close()


def should_send_morning(liga: str) -> bool:
    now = datetime.now()
    return (DIGEST_HOUR_FROM <= now.hour < DIGEST_HOUR_TO
            and not _digest_already_sent(liga))


# ── Formatowanie ──────────────────────────────────────────────────────────────
def _fmt_bet(home, away, typ, kurs, p_model, ev, stake_pln, kolejka=None):
    ev_c  = "🟢" if ev >= 0.15 else "🟡"
    kol_s = f" · kol. {kolejka}" if kolejka else ""
    return (
        f"{ev_c} <b>{home} vs {away}</b>{kol_s}\n"
        f"   Typ: <code>{typ}</code> · Kurs: <b>{kurs:.2f}</b>\n"
        f"   p={p_model:.1%} · EV: <b>{ev:+.1%}</b>\n"
        f"   Kelly: <b>{stake_pln:.0f} zł</b>"
    )


# ── Digest dzienny ────────────────────────────────────────────────────────────
def send_morning_digest(liga: str, value_bets: list, kolejka: int = None,
                        force: bool = False) -> bool:
    """
    Wysyła jeden zbiorczy digest per liga, max raz dziennie w oknie 8-10.
    value_bets: lista dict {home, away, typ, kurs, p_model, ev, stake_pln}
    force=True: wyślij natychmiast (np. po kliknięciu przycisku w UI)
    """
    if not force and _digest_already_sent(liga):
        return False
    now = datetime.now()
    if not force and not (DIGEST_HOUR_FROM <= now.hour < DIGEST_HOUR_TO):
        return False

    vbs = [b for b in value_bets if b.get("ev", 0) >= MIN_EV_DIGEST]

    flag  = _LIGA_EMOJI.get(liga, "⚽")
    kol_s = f" · kolejka {kolejka}" if kolejka else ""
    date_s = now.strftime("%d.%m.%Y %H:%M")

    if not vbs:
        text = (
            f"☀️ <b>Poranny raport</b> {flag} {liga}{kol_s} · {date_s}\n\n"
            f"Brak value betów (EV ≥ {MIN_EV_DIGEST:.0%}) dziś.\n"
            f"Sprawdź aplikację po odświeżeniu kursów."
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
            f"<i>Dixon-Coles · Kelly 1/8</i>"
        )

    ok = send_message(text)
    if ok and not force:
        _mark_digest_sent(liga)
    return ok


# ── Usunięta funkcja send_value_alert (powodowała spam) ──────────────────────
def send_value_alert(*args, **kwargs) -> bool:
    """
    Wyłączona — real-time alerty per-mecz powodowały spam przy każdym
    rerun Streamlit. Zamiast tego używaj send_morning_digest() lub
    komendy /value na Telegramie.
    """
    return False


# ── Obsługa komend ────────────────────────────────────────────────────────────
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
    Polling komend Telegram. Wywoływany przy każdym rerun Streamlit.
    Offset zapobiega podwójnemu przetwarzaniu tej samej wiadomości.
    """
    token, chat_id = _get_credentials()
    if not token or not chat_id:
        return []

    offset = _get_offset()
    res    = _api(token, "getUpdates", offset=offset, timeout=5)
    if not res.get("ok"):
        return []

    processed = []
    for update in res.get("result", []):
        uid = update["update_id"]
        _set_offset(uid + 1)

        msg     = update.get("message", {})
        text    = msg.get("text", "").strip()
        from_id = str(msg.get("chat", {}).get("id", ""))

        if from_id != chat_id:
            continue

        tl = text.lower()
        if tl.startswith("/value"):
            _handle_value_command(token, chat_id, all_value_bets_fn)
            processed.append("value")
        elif tl.startswith("/status"):
            _handle_status_command(token, chat_id, stats_fn)
            processed.append("status")
        elif tl.startswith("/help"):
            _handle_help_command(token, chat_id)
            processed.append("help")

    return processed


def _handle_value_command(token, chat_id, all_value_bets_fn):
    if all_value_bets_fn is None:
        send_message("⏳ Dane niedostępne – otwórz aplikację.", token, chat_id)
        return
    try:
        bets = all_value_bets_fn()
    except Exception as e:
        send_message(f"❌ Błąd: {e}", token, chat_id)
        return

    vbs = [b for b in bets if b.get("ev", 0) >= MIN_EV_DIGEST]

    if not vbs:
        send_message(
            f"🔍 Brak value betów (EV ≥ {MIN_EV_DIGEST:.0%}) dziś.\n"
            "Sprawdź ponownie gdy kolejka zostanie zaplanowana.",
            token, chat_id)
        return

    by_liga = defaultdict(list)
    for b in vbs:
        by_liga[b.get("liga", "?")].append(b)

    now      = datetime.now().strftime("%d.%m %H:%M")
    has_live = any(b.get("live_odds") for b in vbs)
    live_note = "" if has_live else "\n<i>⚠️ Brak live kursów – kurs = fair odds</i>"
    sections = [f"💰 <b>Value Bets</b> · {now} · {len(vbs)} typów{live_note}\n"]

    for liga, liga_bets in by_liga.items():
        flag = _LIGA_EMOJI.get(liga, "⚽")
        sections.append(f"\n{flag} <b>{liga}</b>")
        for b in sorted(liga_bets, key=lambda x: -x.get("ev", 0))[:5]:
            _kurs_tag  = "🔴live" if b.get("live_odds") else "✦fair"
            _ev_str    = f" · EV {b['ev']:+.1%}" if abs(b.get("ev", 0)) > 0.001 else ""
            _stake     = f"\n  🏦 {b['stake_pln']:.0f} zł" if b.get("stake_pln", 0) > 5 else ""
            _data      = f" · {b['data']}" if b.get("data") else ""
            sections.append(
                f"  <b>{b['home']} – {b['away']}</b>{_data}\n"
                f"  <code>{b['typ']}</code> @ <b>{b['kurs']:.2f}</b> {_kurs_tag}"
                f" · p={b['p_model']:.0%}{_ev_str}{_stake}"
            )

    text = "\n".join(sections)
    if len(text) > 4000:
        text = text[:3900] + "\n\n<i>…(skrócono)</i>"
    send_message(text, token, chat_id)


def _handle_status_command(token, chat_id, stats_fn):
    if stats_fn is None:
        send_message("⏳ Statystyki niedostępne.", token, chat_id)
        return
    try:
        stats = stats_fn()
    except Exception as e:
        send_message(f"❌ Błąd: {e}", token, chat_id)
        return
    if not stats:
        send_message("📭 Brak danych w bazie.", token, chat_id)
        return

    lines = []
    for liga, s in stats.items():
        flag  = _LIGA_EMOJI.get(liga, "⚽")
        hr    = s.get("hr", 0)
        roi   = s.get("roi", 0)
        n     = s.get("n", 0)
        hr_c  = "🟢" if hr >= 0.60 else ("🟡" if hr >= 0.50 else "🔴")
        roi_c = "🟢" if roi >= 0 else "🔴"
        lines.append(
            f"{flag} <b>{liga}</b>\n"
            f"   {hr_c} Hit: {hr:.1%} · {roi_c} ROI: {roi:+.1f}% · {n} typów"
        )

    now  = datetime.now().strftime("%d.%m %H:%M")
    text = f"📊 <b>Status modelu</b> · {now}\n\n" + "\n\n".join(lines)
    send_message(text, token, chat_id)


def _handle_help_command(token, chat_id):
    text = (
        "🤖 <b>Komendy bota ZipyBets</b>\n\n"
        "/value  – aktualne value bety (EV ≥ 8%)\n"
        "/status – hit rate i ROI per liga\n"
        "/help   – ta wiadomość\n\n"
        "<i>Powiadomienie automatyczne:</i>\n"
        "☀️ Poranny digest – raz dziennie (8-10 rano)\n"
        "   tylko gdy są mecze w tej kolejce"
    )
    send_message(text, token, chat_id)
