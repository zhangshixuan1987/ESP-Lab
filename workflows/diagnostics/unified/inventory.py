"""
inventory.py
============
Multi-source inventory & readiness reporting.

Checks availability for:
  1. daily_hindcast
  2. monthly_hindcast
  3. observations
  4. historical reference

Outputs:
  {S2D_DIAG_ROOT}/multimodel/unified_diagnostics/inventory/*.csv
  {S2D_DIAG_ROOT}/multimodel/unified_diagnostics/inventory/readiness_summary.json
  {S2D_DIAG_ROOT}/multimodel/unified_diagnostics/inventory/missing_data_report.txt
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from .config import DEFAULT_OUTPUT_ROOT, READINESS_FLAGS


def run_inventory(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    source_paths: dict[str, str | Path | list[str | Path]] | None = None,
    verbose: bool = True,
) -> dict:
    out_dir = Path(output_root) / "inventory"
    out_dir.mkdir(parents=True, exist_ok=True)

    source_paths = source_paths or {}
    readiness = {}
    checked_paths = {}
    for source in READINESS_FLAGS:
        configured = source_paths.get(source)
        if configured is None:
            readiness[source] = "NOT_CONFIGURED"
            checked_paths[source] = []
            continue
        paths = configured if isinstance(configured, list) else [configured]
        resolved = [Path(path).expanduser() for path in paths]
        checked_paths[source] = [str(path) for path in resolved]
        readiness[source] = "PASS" if resolved and all(path.exists() for path in resolved) else "MISSING"

    summary = {
        "readiness": readiness,
        "checked_paths": checked_paths,
        "all_required_ready": all(value == "PASS" for value in readiness.values()),
    }

    json_path = out_dir / "readiness_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    rpt_path = out_dir / "missing_data_report.txt"
    rpt_text = (
        "Unified S2D Multi-Source Inventory Report\n"
        "=========================================\n"
        f"daily_hindcast:          {readiness['daily_hindcast']}\n"
        f"monthly_hindcast:        {readiness['monthly_hindcast']}\n"
        f"observations:            {readiness['observations']}\n"
        f"e3sm_historical_daily:   {readiness['e3sm_historical_daily']}\n"
        f"e3sm_historical_monthly: {readiness['e3sm_historical_monthly']}\n"
    )
    rpt_path.write_text(rpt_text)

    if verbose:
        print("Multi-Source Readiness Summary:")
        print(rpt_text)

    return summary
