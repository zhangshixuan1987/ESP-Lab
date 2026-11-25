"""3a-style regional ACC/nRMSE lead-time skill figure."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import xarray as xr


ACC_FLOOR = -0.2
NRMSE_MIN_TOP = 1.25
NRMSE_CAP = 2.5


def acc_axis_limits(values) -> tuple[float, float]:
    """ACC axis: -0.2 to 1.0, widened downward in steps of 0.1 when needed."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    low = ACC_FLOOR if finite.size == 0 else min(ACC_FLOOR, np.floor(finite.min() * 10.0) / 10.0)
    return float(max(low, -1.0)), 1.0


def nrmse_axis_limits(values, *, cap: float = NRMSE_CAP) -> tuple[float, float]:
    """nRMSE axis: 0 to the largest value rounded up to 0.25, within [1.25, cap]."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    top = NRMSE_MIN_TOP if finite.size == 0 else np.ceil(finite.max() * 4.0) / 4.0
    return 0.0, float(min(max(top, NRMSE_MIN_TOP), cap))


def weak_observation_leads(data: xr.Dataset, region: str, fraction: float) -> xr.DataArray | None:
    """True at leads whose regional observed std is below ``fraction`` of its largest value."""
    if fraction <= 0 or "sig_obs" not in data:
        return None
    sig = data["sig_obs"].sel(region=region)
    largest = float(sig.max(skipna=True)) if sig.notnull().any() else np.nan
    if not np.isfinite(largest) or largest <= 0:
        return None
    return sig < fraction * largest


def _style(
    case: str, index: int, case_styles: Mapping[str, Mapping[str, object]] | None
) -> dict[str, object]:
    if case_styles is not None and case in case_styles:
        return dict(case_styles[case])
    colors = ("tab:red", "tab:green", "tab:brown", "tab:pink")
    return {"color": colors[index % len(colors)], "linestyle": "-", "marker": "o"}


def plot_regional_acc_skill_template(
    skills_by_case_month: Mapping[tuple[str, int], xr.Dataset],
    *,
    monthly_skills_by_case_month: Mapping[tuple[str, int], xr.Dataset] | None = None,
    region: str,
    title: str | None = None,
    init_months: tuple[int, ...] = (5, 11),
    figsize: tuple[float, float] | None = None,
    figwidth: float = 14.0,
    row_height: float = 2.9,
    acc_ylim: tuple[float, float] | str = "auto",
    nrmse_ylim: tuple[float, float] | str = "auto",
    nrmse_cap: float = NRMSE_CAP,
    weak_obs_fraction: float = 0.05,
    case_styles: Mapping[str, Mapping[str, object]] | None = None,
    season_labels: Mapping[int, tuple[str, ...]] | None = None,
    init_labels: Mapping[int, str] | None = None,
    lead_offset: float = 0.0,
    lead_xlabel: str = "Lead time (months)",
    line_width: float = 2.6,
    marker_size: float = 6.5,
    fontz: float = 13.0,
    panel_title_scale: float = 1.0,
    header_scale: float = 1.15,
    label_scale: float = 0.95,
    tick_scale: float = 0.85,
    legend_scale: float = 0.85,
    layout_rect: tuple[float, float, float, float] = (0.04, 0.10, 0.98, 0.93),
    legend_bbox: tuple[float, float] = (0.5, 0.018),
    legend_ncol: int | None = None,
    monthly_xlim: tuple[float, float] = (0.5, 24.5),
    monthly_tick_step: int = 2,
):
    """Return a six-panel 3a-style figure for a single region.

    The final row uses true monthly diagnostics when
    ``monthly_skills_by_case_month`` is supplied; otherwise it transparently
    falls back to the available seasonal leads.

    Each row is ``row_height`` inches tall whatever the number of start months
    (``figsize`` overrides it). ``"auto"`` axis limits follow the data: ACC from
    -0.2 (lower if needed) to 1; nRMSE from 0 to the largest value rounded up to
    0.25, between 1.25 and ``nrmse_cap``, with larger values drawn as markers on
    the top edge and labelled. Leads whose regional observed standard deviation
    is below ``weak_obs_fraction`` of its largest value are left blank.
    """
    import matplotlib.pyplot as plt

    panel_title_fontsize = fontz * panel_title_scale
    header_fontsize = fontz * header_scale
    label_fontsize = fontz * label_scale
    tick_fontsize = fontz * tick_scale
    legend_fontsize = fontz * legend_scale
    selected = [month for month in init_months if any(key[1] == month for key in skills_by_case_month)]
    if not selected:
        raise ValueError("No requested initialization months are available.")
    cases = list(dict.fromkeys(case for case, _ in skills_by_case_month))
    nrows = len(selected) + 1
    top_inches, bottom_inches = (1.0 if title else 0.6), 1.0
    if figsize is None:
        figsize = (figwidth, nrows * row_height + top_inches + bottom_inches)
    fig, axes = plt.subplots(nrows, 2, figsize=figsize, squeeze=False)
    layout_rect = (layout_rect[0], bottom_inches / figsize[1], layout_rect[2], 1.0 - top_inches / figsize[1])
    blanked_leads = 0
    plotted = {"corr": [], "rmse": []}
    pairs = []
    panel_letters = [f"({chr(97 + index)})" for index in range(nrows * 2)]

    def draw_pair(acc_ax, rmse_ax, datasets, label, panel_index, month=None,
                  current_lead_offset=lead_offset, monthly=False):
        acc_ax.set_title(f"{panel_letters[panel_index]} {label}", loc="left", fontweight="medium", fontsize=panel_title_fontsize)
        rmse_ax.set_title(f"{panel_letters[panel_index + 1]} {label}", loc="left", fontweight="medium", fontsize=panel_title_fontsize)
        valid = None
        for data in datasets.values():
            values = data.sel(region=region)
            present = values.corr.notnull() | values.rmse.notnull()
            valid = present if valid is None else valid | present
        valid_leads = valid.L.where(valid, drop=True).values
        nonlocal blanked_leads
        series = []
        for index, (case, data) in enumerate(datasets.items()):
            values = data.sel(region=region).sel(L=valid_leads)
            weak = weak_observation_leads(data.sel(L=valid_leads), region, weak_obs_fraction)
            if weak is not None:
                blanked_leads += int(weak.sum())
                values = values.where(~weak)
            style = _style(case, index, case_styles)
            display_leads = values.L.values + current_lead_offset
            plotted["corr"].append(values.corr.values)
            plotted["rmse"].append(values.rmse.values)
            series.append((case, style, display_leads, values))
        pairs.append((acc_ax, rmse_ax, series))
        available = next(iter(datasets.values())).sel(region=region).sel(L=valid_leads)
        stored_leads = available.L.values
        leads = stored_leads + current_lead_offset
        seasons = () if season_labels is None or month is None else season_labels.get(month, ())
        # Label each lead by its position in the full stored lead list (seasonal
        # caches store L=3, 6, ...; the first maps to seasons[0]), so leads
        # dropped for lack of verifiable data do not shift the labels.
        all_leads = [int(value) for value in valid.L.values]
        labels = [
            f"{int(lead)}:{seasons[all_leads.index(int(stored))]}"
            if all_leads.index(int(stored)) < len(seasons) else str(int(lead))
            for lead, stored in zip(leads, stored_leads)
        ]
        for axis, ylabel, reference in (
            (acc_ax, "ACC", 0.0),
            (rmse_ax, "nRMSE", 1.0),
        ):
            axis.set_ylabel(ylabel, fontsize=label_fontsize)
            if monthly:
                axis.set_xlim(monthly_xlim)
                axis.set_xticks(np.arange(1, monthly_xlim[1], monthly_tick_step))
            elif leads.size == 0:
                # e.g. snow water equivalent over the tropics: nothing to verify.
                axis.set_xticks([])
                axis.text(0.5, 0.5, "no verifiable data", transform=axis.transAxes,
                          ha="center", va="center", fontsize=label_fontsize, color="0.4")
            else:
                axis.set_xticks(leads)
                axis.set_xticklabels(labels)
                axis.set_xlim(float(np.min(leads)) - 0.5, float(np.max(leads)) + 0.5)
            axis.axhline(reference, color="0.25", linewidth=0.8)
            axis.tick_params(labelsize=tick_fontsize)
            axis.grid(True, which="major", color="0.85", linewidth=0.6)

    for row, month in enumerate(selected):
        datasets = {
            case: data for (case, cached_month), data in skills_by_case_month.items()
            if cached_month == month
        }
        draw_pair(axes[row, 0], axes[row, 1], datasets,
                  (init_labels or {}).get(month, f"{month:02d} init"),
                  row * 2, month)

    all_init_skills = monthly_skills_by_case_month or skills_by_case_month
    averaged = {}
    for case in cases:
        data = [all_init_skills[(case, month)] for month in selected if (case, month) in all_init_skills]
        if data:
            averaged[case] = xr.concat(data, dim="init_month").mean("init_month", skipna=True)
    draw_pair(
        axes[-1, 0], axes[-1, 1], averaged, "ALL init average", (nrows - 1) * 2,
        current_lead_offset=0.0 if monthly_skills_by_case_month else lead_offset,
        monthly=monthly_skills_by_case_month is not None,
    )
    acc_limits = acc_axis_limits(np.concatenate([np.ravel(v) for v in plotted["corr"]])) if acc_ylim == "auto" else tuple(acc_ylim)
    nrmse_limits = nrmse_axis_limits(np.concatenate([np.ravel(v) for v in plotted["rmse"]]), cap=nrmse_cap) if nrmse_ylim == "auto" else tuple(nrmse_ylim)
    off_scale_points = 0
    for acc_ax, rmse_ax, series in pairs:
        for case, style, display_leads, values in series:
            acc_ax.plot(display_leads, values.corr, linewidth=line_width, markersize=marker_size, label=case, **style)
            rmse = np.asarray(values.rmse.values, dtype=float)
            top = nrmse_limits[1]
            off_scale = np.isfinite(rmse) & (rmse > top)
            rmse_ax.plot(display_leads, np.where(off_scale, top, rmse), linewidth=line_width,
                         markersize=marker_size, label=case, clip_on=False, **style)
            if off_scale.any():
                off_scale_points += int(off_scale.sum())
                rmse_ax.plot(np.asarray(display_leads)[off_scale], np.full(off_scale.sum(), top), linestyle="none",
                             marker="^", color=style.get("color", "k"), markersize=marker_size * 1.3,
                             clip_on=False, zorder=5)
        acc_ax.set_ylim(acc_limits)
        acc_ax.set_yticks(np.round(np.arange(acc_limits[0], acc_limits[1] + 1e-9, 0.2), 1))
        rmse_ax.set_ylim(nrmse_limits)
    axes[-1, 0].set_xlabel(lead_xlabel, fontsize=label_fontsize)
    axes[-1, 1].set_xlabel(lead_xlabel, fontsize=label_fontsize)

    fig.tight_layout(rect=layout_rect)
    left_position, right_position = axes[0, 0].get_position(), axes[0, 1].get_position()
    header_y = 1.0 - (0.68 if title else 0.3) / figsize[1]
    fig.text(left_position.x0 + left_position.width / 2, header_y, "Anomaly Correlation (ACC)", ha="center", va="center", fontsize=header_fontsize, fontweight="bold")
    fig.text(right_position.x0 + right_position.width / 2, header_y, "Normalized RMSE (nRMSE)", ha="center", va="center", fontsize=header_fontsize, fontweight="bold")
    handles, labels = axes[-1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=legend_ncol or max(1, len(labels)), bbox_to_anchor=legend_bbox, frameon=True, facecolor="#fcfcfc", edgecolor="#d0d0d0", fontsize=legend_fontsize)
    if title:
        fig.suptitle(title, fontsize=fontz * 1.3, fontweight="bold", y=1.0 - 0.22 / figsize[1])
    notes = []
    if off_scale_points:
        notes.append(f"\u25b2 nRMSE above {nrmse_limits[1]:g} (drawn at the top edge)")
    if blanked_leads:
        notes.append(f"blank leads: observed variability below {weak_obs_fraction:.0%} of its largest value in this region")
    if notes:
        fig.text(0.99, 0.005, ";  ".join(notes), ha="right", va="bottom", fontsize=legend_fontsize * 0.8, color="0.35")
    fig.blanked_leads = blanked_leads
    fig.off_scale_points = off_scale_points
    return fig, axes


__all__ = ["acc_axis_limits", "nrmse_axis_limits", "plot_regional_acc_skill_template", "weak_observation_leads"]
