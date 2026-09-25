"""3a-style regional ACC/nRMSE lead-time skill figure."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import xarray as xr


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
    figsize: tuple[float, float] = (14.0, 12.5),
    acc_ylim: tuple[float, float] = (-0.2, 1.0),
    nrmse_ylim: tuple[float, float] = (0.3, 1.3),
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
    fig, axes = plt.subplots(nrows, 2, figsize=figsize, squeeze=False)
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
        valid_leads = valid.L.where(valid, drop=True)
        for index, (case, data) in enumerate(datasets.items()):
            values = data.sel(region=region).sel(L=valid_leads)
            style = _style(case, index, case_styles)
            display_leads = values.L.values + current_lead_offset
            acc_ax.plot(display_leads, values.corr, linewidth=line_width, markersize=marker_size, label=case, **style)
            rmse_ax.plot(display_leads, values.rmse, linewidth=line_width, markersize=marker_size, label=case, **style)
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
        for axis, ylabel, ylim, reference in (
            (acc_ax, "ACC", acc_ylim, 0.0),
            (rmse_ax, "nRMSE", nrmse_ylim, 1.0),
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
            axis.set_ylim(ylim)
            axis.axhline(reference, color="0.25", linewidth=0.8)
            axis.tick_params(labelsize=tick_fontsize)

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
    axes[-1, 0].set_xlabel(lead_xlabel, fontsize=label_fontsize)
    axes[-1, 1].set_xlabel(lead_xlabel, fontsize=label_fontsize)

    fig.tight_layout(rect=layout_rect)
    left_position, right_position = axes[0, 0].get_position(), axes[0, 1].get_position()
    header_y = 0.945 if title else 0.96
    fig.text(left_position.x0 + left_position.width / 2, header_y, "Anomaly Correlation (ACC)", ha="center", va="center", fontsize=header_fontsize, fontweight="bold")
    fig.text(right_position.x0 + right_position.width / 2, header_y, "Normalized RMSE (nRMSE)", ha="center", va="center", fontsize=header_fontsize, fontweight="bold")
    handles, labels = axes[-1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=legend_ncol or max(1, len(labels)), bbox_to_anchor=legend_bbox, frameon=True, facecolor="#fcfcfc", edgecolor="#d0d0d0", fontsize=legend_fontsize)
    if title:
        fig.suptitle(title, fontsize=fontz * 1.3, fontweight="bold", y=0.985)
    return fig, axes


__all__ = ["plot_regional_acc_skill_template"]
