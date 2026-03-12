"""
fetch_xg_local.py
=================
Uruchom LOKALNIE (nie na Streamlit Cloud / Colab).
Pobiera dane xG z understat.com i zapisuje do xg_cache.db
który następnie commituj do repozytorium.

Użycie:
    python fetch_xg_local.py                  # wszystkie ligi, bieżący sezon
    python fetch_xg_local.py --liga E0        # tylko EPL
    python fetch_xg_local.py --force          # ignoruj cooldown

Po uruchomieniu:
    git add xg_cache.db
    git commit -m "update xG cache"
    git push
"""

import sys
import os

# Dodaj folder projektu do path
sys.path.insert(0, os.path.dirname(__file__))

try:
    import understat_fetcher as xgf
except ImportError:
    print("Błąd: understat_fetcher.py musi być w tym samym katalogu.")
    sys.exit(1)

import argparse

def main():
    parser = argparse.ArgumentParser(description="Lokalny fetcher xG — uruchamiaj na własnym komputerze")
    parser.add_argument("--liga",  default="all", help="E0/SP1/D1/I1/F1 lub 'all'")
    parser.add_argument("--force", action="store_true", help="Ignoruj cooldown 20h")
    parser.add_argument("--prev",  action="store_true", help="Pobierz też poprzedni sezon")
    args = parser.parse_args()

    print("=" * 55)
    print("  understat.com xG fetcher — tryb lokalny")
    print("=" * 55)
    print(f"  DB:    {xgf.XG_DB_FILE}")
    print(f"  Liga:  {args.liga}")
    print(f"  Sezon: {xgf._biezacy_sezon_understat()} {'+ prev' if args.prev else ''}")
    print()

    if args.liga == "all":
        results = xgf.fetch_all_ligas(force=args.force, include_prev=args.prev)
        for r in results:
            status = r.get("status", "?")
            liga   = r.get("liga", "?")
            sezon  = r.get("sezon", "?")
            n      = r.get("n_mecze", 0)
            msg    = r.get("msg", "")
            if status == "ok":
                print(f"  ✅  {liga}/{sezon}  →  {n} meczów")
            elif status == "skipped":
                print(f"  ⏭️   {liga}/{sezon}  →  cooldown (użyj --force)")
            else:
                print(f"  ❌  {liga}/{sezon}  →  {msg}")
    else:
        r = xgf.fetch_liga_xg(args.liga, force=args.force)
        if r.get("status") == "ok":
            print(f"  ✅  {r['liga']}/{r['sezon']}  →  {r['n_mecze']} meczów")
        else:
            print(f"  ❌  {r.get('msg')}")

    print()

    # Podsumowanie co jest w DB
    info = xgf.get_last_fetch_info()
    if info:
        print("Stan xg_cache.db:")
        for liga, d in info.items():
            print(f"  {liga:4s}  sezon={d['sezon']}  meczów={d['n_mecze']}  "
                  f"ostatni fetch={str(d['ts'])[:16]}")
        print()

    db_size = os.path.getsize(xgf.XG_DB_FILE) if os.path.exists(xgf.XG_DB_FILE) else 0
    print(f"Plik: {xgf.XG_DB_FILE}  ({db_size/1024:.0f} KB)")
    print()
    print("Następny krok — wgraj do repo:")
    print(f"  git add {xgf.XG_DB_FILE}")
    print(f'  git commit -m "update xG cache {xgf._biezacy_sezon_understat()}"')
    print(f"  git push")

if __name__ == "__main__":
    main()
