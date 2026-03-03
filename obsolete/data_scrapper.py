"""Compatibility wrapper for legacy imports.

Use the single entrypoint: bin/regenerate_data.py
"""

from pathlib import Path
from utils.data_pipeline import DataRegenerator


def refresh_all(root: str | Path = ".", force: bool = False):
    regen = DataRegenerator(root=root, force_download=force)
    return regen.run()


if __name__ == "__main__":
    res = refresh_all(root=Path(__file__).parent.parent, force=False)
    print(f"Generated: {len(res.generated)}")
    print(f"Skipped: {len(res.skipped)}")
