"""Reorganize trading_view data folder by category with clean filenames."""
import os
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8")

TV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "trading_view")

# ── Category mapping ──
CATEGORIES = {
    "equity_indices": ["SPX", "CAC40"],
    "equity_etfs":    ["SPY"],
    "volatility":     ["VIX", "VVIX", "VIX1D", "VIX9D", "VIX3M", "VIX6M", "VIX1Y"],
    "vol_etfs":       ["SVXY", "UVXY", "VIXY"],
    "vix_futures":    ["VX1!", "VX2!"],
    "sp_futures":     ["ES1!", "ES2!"],
    "rates":          ["US02Y", "US05Y", "US10Y", "US30Y"],
    "fx":             ["DXY"],
    "sentiment":      ["SKEW", "COR1M", "PC", "PCSPX"],
}

# Reverse lookup: ticker -> category
TICKER_TO_CAT = {}
for cat, tickers in CATEGORIES.items():
    for t in tickers:
        TICKER_TO_CAT[t] = cat

# ── Timeframe normalization ──
TF_MAP = {
    "1": "1m", "5": "5m", "10": "10m", "15": "15m",
    "30": "30m", "45": "45m", "60": "1h",
    "1D": "daily", "5S": "5s",
    "1min": "1m", "5min": "5m", "10min": "10m", "15min": "15m",
    "30min": "30m", "45min": "45m", "60min": "1h",
}

EXCHANGE_PREFIXES = ["CBOE_DLY_", "TVC_", "BATS_", "CME_MINI_DL_", "USI_", "SP_"]


def parse_files():
    """Parse all CSV files and return (moves, legacy, skipped) lists."""
    files = [f for f in os.listdir(TV) if f.endswith(".csv")]
    moves = []     # (src_filename, category, dst_filename)
    legacy = []    # old/duplicate files
    skipped = []

    for f in sorted(files):
        name = f[:-4]  # strip .csv

        # ── Old format: SP_SPX,30min.csv / TVC_VIX,15min.csv ──
        if "," in name and "min" in name.split(",")[-1]:
            parts = name.split(",")
            raw_ticker = parts[0].split("_", 1)[-1] if "_" in parts[0] else parts[0]
            raw_tf = parts[1].strip()
            tf = TF_MAP.get(raw_tf, raw_tf)
            ticker = raw_ticker.upper()
            legacy.append((f, f"{ticker.lower()}_{tf}.csv"))
            continue

        # ── New format: EXCHANGE_TICKER, TF.csv ──
        if ", " in name:
            parts = name.split(", ", 1)
            exchange_ticker = parts[0]
            tf_raw = parts[1].strip()

            # Handle duplicates: '15 (1)' -> '15'
            is_dup = "(" in tf_raw
            if is_dup:
                tf_raw = tf_raw.split("(")[0].strip()

            tf = TF_MAP.get(tf_raw, tf_raw)

            # Extract ticker
            ticker = exchange_ticker
            for p in EXCHANGE_PREFIXES:
                if exchange_ticker.startswith(p):
                    ticker = exchange_ticker[len(p):]
                    break

            cat = TICKER_TO_CAT.get(ticker)
            if cat is None:
                skipped.append(f)
                continue

            # TVC_VIX duplicates CBOE_DLY_VIX -> legacy
            source_exchange = exchange_ticker.split("_")[0]
            if ticker == "VIX" and source_exchange == "TVC":
                legacy.append((f, f"vix_{tf}_tvc.csv"))
                continue

            # Clean ticker for filename (remove !)
            clean = ticker.lower().replace("!", "")
            dst_name = f"{clean}_{tf}.csv"
            moves.append((f, cat, dst_name))
        else:
            skipped.append(f)

    return moves, legacy, skipped


def execute():
    moves, legacy, skipped = parse_files()

    # Create all needed category dirs
    cats_needed = set(m[1] for m in moves)
    cats_needed.add("_legacy")
    for c in cats_needed:
        os.makedirs(os.path.join(TV, c), exist_ok=True)

    done = 0

    # ── Move main files ──
    for src, cat, dst_name in moves:
        src_path = os.path.join(TV, src)
        dst_path = os.path.join(TV, cat, dst_name)
        if not os.path.exists(src_path):
            continue
        if os.path.exists(dst_path):
            # Keep larger file, move smaller to _legacy
            if os.path.getsize(src_path) > os.path.getsize(dst_path):
                shutil.move(dst_path, os.path.join(TV, "_legacy", f"dup_{dst_name}"))
                shutil.move(src_path, dst_path)
            else:
                shutil.move(src_path, os.path.join(TV, "_legacy", src))
        else:
            shutil.move(src_path, dst_path)
        done += 1
        print(f"  [MOVE] {src:55s} -> {cat}/{dst_name}")

    # ── Move legacy/old files ──
    for src, dst_name in legacy:
        src_path = os.path.join(TV, src)
        if not os.path.exists(src_path):
            continue
        dst_path = os.path.join(TV, "_legacy", dst_name)
        if os.path.exists(dst_path):
            base, ext = os.path.splitext(dst_name)
            dst_path = os.path.join(TV, "_legacy", f"{base}_dup{ext}")
        shutil.move(src_path, dst_path)
        done += 1
        print(f"  [LEGACY] {src:53s} -> _legacy/{os.path.basename(dst_path)}")

    print(f"\n{'='*70}")
    print(f"Total moved: {done}")
    if skipped:
        print(f"Skipped (unknown ticker): {skipped}")
    print()

    # ── Print final tree ──
    print("FINAL STRUCTURE:")
    print("=" * 70)
    for root, dirs, fnames in os.walk(TV):
        dirs.sort()
        level = root.replace(TV, "").count(os.sep)
        indent = "  " * level
        folder = os.path.basename(root) or "trading_view"
        if fnames:
            print(f"{indent}{folder}/  ({len(fnames)} files)")
        else:
            print(f"{indent}{folder}/")
        for fn in sorted(fnames):
            sz = os.path.getsize(os.path.join(root, fn)) / 1024
            print(f"{indent}  {fn:40s} {sz:>8.1f} KB")


if __name__ == "__main__":
    execute()
