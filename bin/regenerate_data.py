"""
Data Regeneration Entrypoint
============================
Rebuilds datasets under data/ from:
  - Yahoo Finance (^VIX, ^VVIX, ^GSPC, SOFR-like symbols)
  - TradingView local exports (data/trading_view/)
  - CBOE VX monthly CSV endpoint (for vix_futures_all.csv)

Usage:
  python bin/regenerate_data.py                        # Yahoo only (~60 days intraday)
  python bin/regenerate_data.py --mode tradingview     # TradingView exports (richer, years of data)
  python bin/regenerate_data.py --mode merge           # TradingView + Yahoo (best coverage)
  python bin/regenerate_data.py --force                # Force re-download/overwrite
"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_pipeline import DataRegenerator


def main():
    parser = argparse.ArgumentParser(description="Regenerate project datasets into data/")
    parser.add_argument("--mode", choices=["yahoo", "tradingview", "merge"], default="yahoo",
                        help="Data source: yahoo (download ~60d), tradingview (local exports, years), merge (both)")
    parser.add_argument("--force", action="store_true", help="Force re-download and overwrite existing files")
    parser.add_argument("--sofr-url", type=str, default="", help="Override NY Fed SOFR URL")
    parser.add_argument("--legacy-aliases", action="store_true",
                        help="Also write legacy root files (data/TVC_VIX,*.csv, data/SP_SPX,*.csv)")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Do not remove redundant legacy root files")
    args = parser.parse_args()

    regen = DataRegenerator(root=Path(__file__).parent.parent, force_download=args.force)
    result = regen.run(
        sofr_url=args.sofr_url or None,
        mode=args.mode,
        keep_legacy_aliases=args.legacy_aliases,
        cleanup_redundant=not args.no_cleanup,
    )

    print("\n" + "=" * 70)
    print("DATA REGENERATION COMPLETE")
    print("=" * 70)
    print(f"Generated: {len(result.generated)} files")
    for k, v in result.generated.items():
        print(f"  + {k}: {v}")

    if result.skipped:
        print(f"\nSkipped: {len(result.skipped)}")
        for k, v in result.skipped.items():
            print(f"  - {k}: {v}")

    if result.metadata:
        print("\nMetadata:")
        for k, v in result.metadata.items():
            print(f"  * {k}: {v}")


if __name__ == "__main__":
    main()
