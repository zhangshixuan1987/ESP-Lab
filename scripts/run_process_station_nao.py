#!/usr/bin/env python3
"""Compatibility entry point for the generalized modes-of-variability runner."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from scripts.run_process_modes_of_variability import main
except ModuleNotFoundError:
    from run_process_modes_of_variability import main


if "--modes" not in sys.argv:
    sys.argv.extend(["--modes", "NAO"])
if "--legacy-nao-layout" not in sys.argv:
    sys.argv.append("--legacy-nao-layout")

if __name__ == "__main__":
    main()
