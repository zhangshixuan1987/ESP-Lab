"""
run_monthly.py
==============
Top-level orchestrator for the monthly S2D spatial bias and drift workflow.

Calls Steps 1–9 in order:
  1. Discover files          → discover.py
  2–4. Inventory + gate      → inventory.py
  5. Preprocess              → preprocess.py
  6–7. Bias + drift          → diagnostics.py
  8. Bootstrap               → bootstrap.py
  9. Figures + tables        → plotting.py

Usage
-----
    # Pilot (1980–1982, May, PRECT)
    python -m workflows.diagnostics.monthly_drift.run_monthly --pilot --variable PRECT --season may

    # Full 1980–1986 campaign, both seasons, PRECT
    python -m workflows.diagnostics.monthly_drift.run_monthly --variable PRECT --season both

    # Skip inventory (reuse existing CSV)
    python -m workflows.diagnostics.monthly_drift.run_monthly --pilot --skip-inventory

    # All variables
    python -m workflows.diagnostics.monthly_drift.run_monthly --pilot

Outputs
-------
    {S2D_DIAG_ROOT}/multimodel/leadtime_drift/atm/monthly_spatial/*.csv
    {S2D_DIAG_ROOT}/multimodel/leadtime_drift/atm/monthly_spatial/diagnostics/
    {S2D_DIAG_ROOT}/multimodel/leadtime_drift/atm/monthly_spatial/summary_tables/
    {FIGURE_OUTDIR}/leadtime_drift/atm/monthly_spatial/*.png
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .config import DEFAULT_FIGURE_OUTDIR, DEFAULT_OUTPUT_ROOT, build_monthly_config
from . import bootstrap as _bootstrap
from . import diagnostics as _diagnostics
from . import inventory as _inventory
from . import plotting as _plotting
from . import preprocess as _preprocess
from esp_lab.diagnostics.store import (
    diagnostic_store_is_valid,
    diagnostic_store_path,
    dataframe_fingerprint,
    file_fingerprint,
    load_diagnostic_store,
    save_diagnostic_store,
)


def run(
    pilot: bool = True,
    variable: Optional[str] = "PRECT",
    season: Optional[str] = None,
    skip_inventory: bool = False,
    gate_strict: bool = False,
    obs_path: Optional[str] = None,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    figure_outdir: str | Path = DEFAULT_FIGURE_OUTDIR,
    n_bootstrap: int = 1000,
    reuse_diagnostics: bool = True,
    force_recompute: bool = False,
    verbose: bool = True,
) -> None:
    """Run the full monthly S2D spatial analysis pipeline."""

    pm = {"may": [5], "nov": [11], "november": [11], "both": [5, 11]}.get(
        (season or "both").lower(), [5, 11]
    )

    config = build_monthly_config(
        pilot_only=pilot,
        pilot_months=pm if pilot else [5, 11],
        variable_filter=[variable] if variable else None,
        gate_strict=gate_strict,
        output_root=output_root,
        n_bootstrap=n_bootstrap,
    )

    if verbose:
        print("\n" + "=" * 70)
        print("Monthly S2D Spatial Analysis — Full Pipeline")
        print("=" * 70)
        print(f"  Pilot mode  : {pilot}")
        print(f"  Variable    : {variable or 'ALL'}")
        print(f"  Season(s)   : {pm}")
        print(f"  Init years  : {config.active_years}")
        print(f"  Gate strict : {gate_strict}")
        print(f"  Output root : {output_root}")
        print(f"  Figure dir  : {figure_outdir}")
        print()

    # Steps 2–4: Inventory and gate
    inventory_df, report = _inventory.run(
        pilot_only=pilot,
        variable=variable,
        season=season,
        skip_discovery=skip_inventory,
        verbose=verbose,
        output_root=output_root,
        config=config,
    )

    # Step 5: Preprocess approved data
    variables_to_run = [variable] if variable else [v.native_name for v in config.variables]

    all_summaries = []
    for var in variables_to_run:
        if verbose:
            print(f"\n{'='*70}\n  Variable: {var}\n{'='*70}")

        try:
            store_context = {
                "inventory": dataframe_fingerprint(inventory_df),
                "observations": file_fingerprint(obs_path),
            }
            store_path = diagnostic_store_path(
                Path(config.output_root), "monthly_drift", config, var,
                context=store_context,
            )
            use_saved = (
                reuse_diagnostics and not force_recompute
                and diagnostic_store_is_valid(
                    store_path, config, variable=var, context=store_context
                )
            )
            if use_saved:
                if verbose:
                    print(f"  Reusing diagnostics: {store_path}")
                results, boot_results = load_diagnostic_store(store_path)
            else:
                data, obs_da = _preprocess.run_preprocess(
                    config=config,
                    inventory_df=inventory_df,
                    report=report,
                    field=var,
                    obs_path=obs_path,
                    verbose=verbose,
                )

                results = _diagnostics.run_diagnostics(
                    data=data,
                    config=config,
                    obs_da=obs_da,
                    verbose=verbose,
                )
                boot_results = _bootstrap.run_bootstrap(
                    results=results,
                    config=config,
                    verbose=verbose,
                )
                save_diagnostic_store(
                    store_path, results, boot_results,
                    config=config, variable=var, workflow="monthly_drift",
                    context=store_context,
                )
                if verbose:
                    print(f"  Saved diagnostics: {store_path}")

            # Step 9: Figures + tables
            summary_df = _plotting.run_plotting(
                results=results,
                boot_results=boot_results,
                config=config,
                variable=var,
                figure_outdir=figure_outdir,
                verbose=verbose,
            )
            all_summaries.append(summary_df)

        except Exception as exc:
            warnings.warn(f"Pipeline failed for variable '{var}': {exc}", stacklevel=2)
            if gate_strict:
                raise

    if verbose:
        print("\n" + "=" * 70)
        print("Pipeline complete.")
        if all_summaries:
            import pandas as pd
            combined = pd.concat(all_summaries, ignore_index=True)
            print(combined.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monthly S2D spatial bias and drift analysis — full pipeline."
    )
    parser.add_argument(
        "--pilot", action="store_true", default=False,
        help="Pilot mode: 1980–1982, May only (override with --season).",
    )
    parser.add_argument(
        "--full-campaign", action="store_true",
        help="Process all init years (1980–1986, May + November).",
    )
    parser.add_argument("--variable", default="PRECT",
                        help="Variable native name (default: PRECT).")
    parser.add_argument("--season", default=None,
                        choices=["may", "nov", "both"],
                        help="Limit to one season.")
    parser.add_argument("--skip-inventory", action="store_true",
                        help="Reuse existing monthly_inventory_raw.csv.")
    parser.add_argument("--gate-strict", action="store_true",
                        help="Exit 1 on gate failure (batch mode).")
    parser.add_argument("--obs-path", default=None,
                        help="Path to observation NetCDF file.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--figure-outdir", default=str(DEFAULT_FIGURE_OUTDIR))
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--force-recompute", action="store_true",
                        help="Ignore a compatible saved diagnostic store.")
    args = parser.parse_args()

    pilot = args.pilot or not args.full_campaign

    run(
        pilot=pilot,
        variable=args.variable,
        season=args.season,
        skip_inventory=args.skip_inventory,
        gate_strict=args.gate_strict,
        obs_path=args.obs_path,
        output_root=args.output_root,
        figure_outdir=args.figure_outdir,
        n_bootstrap=args.n_bootstrap,
        force_recompute=args.force_recompute,
    )


if __name__ == "__main__":
    main()
