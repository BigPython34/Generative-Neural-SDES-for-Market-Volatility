"""
Options Data Fetch Entrypoint
=============================
Fetches SPY options surface from Yahoo Finance and caches locally.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quant.options_cache import download_and_cache_options


def main():
    surface = download_and_cache_options()
    return surface


if __name__ == "__main__":
    main()
