"""Minimal launcher bootstrap for calibration scripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def bootstrap() -> Path:
    root = Path(__file__).resolve().parents[2]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    venv_site = root / ".venv" / "Lib" / "site-packages"
    if venv_site.exists():
        venv_str = str(venv_site)
        if venv_str not in sys.path:
            sys.path.insert(0, venv_str)

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    try:
        import sitecustomize  # noqa: F401
    except Exception:
        pass
    return root
