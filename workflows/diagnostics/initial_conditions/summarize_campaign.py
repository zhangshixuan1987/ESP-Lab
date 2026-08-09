#!/usr/bin/env python3
"""
06_summarize_campaign.py
=========================
Step 6: Aggregate IC difference statistics across all matched start dates
        and produce the campaign-level summary.

What this script does
---------------------
1. Load all per-date variable statistics CSVs from Step 3.
2. Aggregate separately for:
   - May starts
   - November starts
   - Individual years
   - Cross-year mean and spread
3. For every component and priority variable, compute:
   - Mean IC RMSE across start years
   - Interannual range (max − min RMSE)
   - Spatially averaged bias (mean of mean_diff)
   - Sign-agreement fraction (fraction of starts with positive mean_diff)
   - May vs November contrast
4. Write campaign_summary/ tables.

Usage
-----
    python 06_summarize_campaign.py
    python 06_summarize_campaign.py --full-campaign

Outputs
-------
    output/campaign_summary/campaign_stats.csv           (main table)
    output/campaign_summary/may_vs_nov_contrast.csv
    output/campaign_summary/top_variables_by_rmse.csv
    output/campaign_summary/campaign_summary.png         (heatmap figure)
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from esp_lab.diagnostics.ic_core import aggregate_campaign_stats

from .config import build_ic_config, load_config

FONTZ = 11
FIG_DPI = 150


def _load_all_stats(vs_dir: Path) -> pd.DataFrame:
    """Load and concatenate all per-date variable statistics CSVs."""
    combined_csv = vs_dir / "all_variable_stats.csv"
    if combined_csv.exists():
        return pd.read_csv(combined_csv)

    # Fallback: scan per-date per-component files
    dfs = []
    for csv_path in sorted(vs_dir.glob("*_stats.csv")):
        try:
            dfs.append(pd.read_csv(csv_path))
        except Exception as exc:
            warnings.warn(f"Could not read {csv_path}: {exc}", stacklevel=2)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _campaign_heatmap(pivot: pd.DataFrame, title: str, outpath: Path) -> None:
    """Heatmap of mean RMSE: rows = variables, columns = components."""
    if pivot.empty:
        return
    fig, ax = plt.subplots(
        figsize=(max(4, pivot.shape[1] * 1.2), max(4, pivot.shape[0] * 0.35)),
        dpi=FIG_DPI,
    )
    data = pivot.values.astype(float)
    # Normalise each row for visual clarity
    row_max = np.nanmax(data, axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0
    normed = data / row_max

    im = ax.imshow(normed, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, fontsize=FONTZ - 1, rotation=30, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=FONTZ - 2)
    ax.set_title(title, fontsize=FONTZ)

    # Annotate with raw RMSE values
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            val = data[r, c]
            if np.isfinite(val):
                ax.text(
                    c, r, f"{val:.2g}",
                    ha="center", va="center",
                    fontsize=max(6, FONTZ - 4),
                    color="white" if normed[r, c] > 0.65 else "black",
                )

    fig.colorbar(im, ax=ax, label="RMSE (row-normalised)", fraction=0.03, pad=0.02)
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def run(
    config_path: Path,
    pilot_only: bool | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    cfg = load_config(config_path)
    ic_cfg = build_ic_config(cfg, pilot_only=pilot_only)

    out_root = Path(config_path.parent) / ic_cfg.output_root
    subdirs = cfg.get("output", {}).get("subdirs", {})
    vs_dir = out_root / subdirs.get("variable_statistics", "variable_statistics")
    cs_dir = out_root / subdirs.get("campaign_summary", "campaign_summary")
    cs_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("IC Analysis — Step 6: Campaign Summary")
        print("=" * 70)

    all_stats = _load_all_stats(vs_dir)
    if all_stats.empty:
        warnings.warn(
            "No variable statistics found. Run Step 3 first.", stacklevel=2
        )
        return pd.DataFrame()

    # Ensure season column
    if "season" not in all_stats.columns:
        all_stats["season"] = all_stats["start_date"].apply(
            lambda d: {5: "May", 11: "November"}.get(int(d.split("-")[1]), "Unknown")
        )

    # Aggregate by component + variable + season
    records = all_stats.to_dict(orient="records")
    campaign_df = aggregate_campaign_stats(records, group_by="component,variable,season")
    campaign_df.to_csv(cs_dir / "campaign_stats.csv", index=False)

    if verbose:
        print(f"\n  Campaign stats → {cs_dir / 'campaign_stats.csv'}")
        print(f"  Rows: {len(campaign_df)}")

    # May vs November contrast
    may_df = campaign_df[campaign_df.get("season", "") == "May"] \
        if "season" in campaign_df.columns else pd.DataFrame()
    nov_df = campaign_df[campaign_df.get("season", "") == "November"] \
        if "season" in campaign_df.columns else pd.DataFrame()

    if not may_df.empty and not nov_df.empty:
        contrast = may_df.set_index(["component", "variable"])[["mean_rmse"]].rename(
            columns={"mean_rmse": "may_mean_rmse"}
        ).join(
            nov_df.set_index(["component", "variable"])[["mean_rmse"]].rename(
                columns={"mean_rmse": "nov_mean_rmse"}
            ),
            how="outer",
        ).reset_index()
        contrast["may_minus_nov_rmse"] = contrast["may_mean_rmse"] - contrast["nov_mean_rmse"]
        contrast.to_csv(cs_dir / "may_vs_nov_contrast.csv", index=False)
        if verbose:
            print(f"  May vs Nov contrast → {cs_dir / 'may_vs_nov_contrast.csv'}")

    # Top-20 variables by mean RMSE (all seasons combined)
    agg_all = aggregate_campaign_stats(records, group_by="component,variable")
    top_vars = agg_all.nlargest(20, "mean_rmse")
    top_vars.to_csv(cs_dir / "top_variables_by_rmse.csv", index=False)

    if verbose:
        print("\n--- Top 20 variables by mean IC RMSE ---")
        cols = ["component", "variable", "mean_rmse", "std_rmse", "sign_agreement_frac"]
        available = [c for c in cols if c in top_vars.columns]
        print(top_vars[available].to_string(index=False))

    # Campaign heatmap: rows = top variables, columns = components
    if not agg_all.empty and "mean_rmse" in agg_all.columns:
        try:
            pivot = agg_all.pivot_table(
                index="variable", columns="component", values="mean_rmse"
            )
            top_idx = pivot.max(axis=1).nlargest(30).index
            pivot = pivot.loc[top_idx]
            _campaign_heatmap(
                pivot,
                title=(
                    f"IC RMSE: {ic_cfg.experiment_pair.test_label} − "
                    f"{ic_cfg.experiment_pair.ref_label}\n"
                    f"({len(all_stats['start_date'].unique())} start dates)"
                ),
                outpath=cs_dir / "campaign_summary.png",
            )
            if verbose:
                print(f"\n  Campaign heatmap → {cs_dir / 'campaign_summary.png'}")
        except Exception as exc:
            warnings.warn(f"  Heatmap failed: {exc}", stacklevel=2)

    return campaign_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IC Analysis Step 6: Aggregate campaign-level summary."
    )
    parser.add_argument("--config", default=str(_SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--full-campaign", action="store_true")
    args = parser.parse_args()
    run(
        config_path=Path(args.config),
        pilot_only=False if args.full_campaign else None,
    )


if __name__ == "__main__":
    main()
