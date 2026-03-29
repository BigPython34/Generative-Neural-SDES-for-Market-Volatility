"""Startup bootstrap for trusted native extensions.

This module is imported automatically by Python if it is present on sys.path.
It copies the installed jaxlib package to a trusted location outside the user
profile so Windows application-control policies that block DLLs in the venv do
not prevent JAX from loading.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def _bootstrap_jaxlib() -> None:
    try:
        venv_root = Path(sys.prefix)
        source = venv_root / "Lib" / "site-packages" / "jaxlib"
        if not source.exists():
            return

        target_root = Path(os.environ.get("TEMP", r"C:\Temp")) / "jax_trusted"
        target = target_root / "jaxlib"
        target_root.mkdir(parents=True, exist_ok=True)

        source_dll = source / "jax_common.dll"
        target_dll = target / "jax_common.dll"
        needs_copy = (
            not target.exists()
            or not target_dll.exists()
            or (source_dll.exists() and target_dll.stat().st_mtime < source_dll.stat().st_mtime)
        )

        if needs_copy:
            shutil.copytree(source, target, dirs_exist_ok=True)

        trusted_site_packages = str(target_root)
        if trusted_site_packages not in sys.path:
            sys.path.insert(0, trusted_site_packages)
    except Exception:
        # Never break interpreter startup.
        return


_bootstrap_jaxlib()
