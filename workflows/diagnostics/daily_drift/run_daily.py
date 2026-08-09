"""
run_daily.py
============
Step 12: Top-level CLI orchestrator for the daily S2D spatial analysis pipeline.

Calls Steps 3–11 in sequence.
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

from .config import DEFAULT_FIGURE_OUTDIR, DEFAULT_OUTPUT_ROOT, build_daily_config
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
    variable: Optional[str] = "TREFHT",
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
    pm = {"may": [5], "nov": [11], "november": [11], "both": [5, 11]}.get(
        (season or "both").lower(), [5, 11]
    )

    config = build_daily_config(
        pilot_only=pilot,
        pilot_months=pm if pilot else [5, 11],
        variable_filter=[variable] if variable else None,
        gate_strict=gate_strict,
        output_root=output_root,
        n_bootstrap=n_bootstrap,
    )

    if verbose:
        print("\n" + "=" * 70)
        print("Daily S2D Spatial Bias & Drift Analysis — Full Pipeline")
        print("=" * 70)
        print(f"  Pilot mode  : {pilot}")
        print(f"  Variable    : {variable or 'ALL'}")
        print(f"  Season(s)   : {pm}")
        print(f"  Init years  : {config.active_years}")
        print(f"  Lead days   : 1..{max(config.lead_days)}")
        print(f"  Output root : {output_root}")
        print()

    # Steps 4–5: Inventory & Gate
    inventory_df, report = _inventory.run(
        pilot_only=pilot,
        variable=variable,
        season=season,
        skip_discovery=skip_inventory,
        verbose=verbose,
        output_root=output_root,
        config=config,
    )

    # Preprocess, diagnostics, bootstrap, plotting per variable
    vars_to_run = [variable] if variable else [v.native_field for v in config.variables]

    all_summaries = []
    for var in vars_to_run:
        if verbose:
            print(f"\n{'='*70}\n  Variable: {var}\n{'='*70}")

        try:
            store_context = {
                "inventory": dataframe_fingerprint(inventory_df),
                "observations": file_fingerprint(obs_path),
            }
            store_path = diagnostic_store_path(
                Path(config.output_root), "daily_drift", config, var,
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
                    config=config, variable=var, workflow="daily_drift",
                    context=store_context,
                )
                if verbose:
                    print(f"  Saved diagnostics: {store_path}")

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
            warnings.warn(f"Daily pipeline failed for variable '{var}': {exc}", stacklevel=2)
            if gate_strict:
                raise

    if verbose:
        print("\n" + "=" * 70)
        print("Daily Pipeline Complete.")
        if all_summaries:
            import pandas as pd
            combined = pd.concat(all_summaries, ignore_index=True)
            print(combined.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Daily S2D spatial analysis pipeline."
    )
    parser.add_argument("--pilot", action="store_true", default=False)
    parser.add_argument("--full-campaign", action="store_true")
    parser.add_argument("--variable", default="TREFHT")
    parser.add_argument("--season", default=None, choices=["may", "nov", "both"])
    parser.add_argument("--skip-inventory", action="store_true")
    parser.add_argument("--gate-strict", action="store_true")
    parser.add_argument("--obs-path", default=None)
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
