import numpy as np
import xarray as xr

from esp_lab.diagnostics.enso_frequency import (
    compute_enso_frequency_product,
    paired_frequency_contrasts,
    persistent_event_mask,
    summarize_enso_frequency,
)


def test_persistent_event_mask_marks_only_complete_runs():
    condition = xr.DataArray(
        [[False, True, True, True, False, True, True]],
        dims=("Y", "L"),
    )

    actual = persistent_event_mask(condition, min_run=3)

    expected = np.array([[False, True, True, True, False, False, False]])
    np.testing.assert_array_equal(actual.values, expected)


def test_frequency_product_and_summary_for_known_el_nino_starts():
    years = np.arange(2000, 2006)
    leads = np.arange(1, 9)
    members = np.arange(2)
    values = np.zeros((years.size, leads.size, members.size))
    values[3:, :, :] = 1.0
    model = xr.DataArray(
        values,
        dims=("Y", "L", "M"),
        coords={"Y": [f"case-{year}" for year in years], "L": leads, "M": members},
    )
    observations = xr.DataArray(
        values.mean(axis=2),
        dims=("Y", "L"),
        coords={"Y": model.Y, "L": leads},
    )

    product = compute_enso_frequency_product(
        model,
        observations,
        initialization_years=years,
        climatology_years=(2000, 2002),
        min_persistence=5,
    )
    summary = summarize_enso_frequency(
        product,
        experiment="test",
        init_month=5,
        n_bootstrap=100,
    ).set_index("category")

    assert product.sizes["L"] == 6
    assert product.forecast_el_nino.sel(Y="case-2003").all()
    assert not product.forecast_el_nino.sel(Y="case-2002").any()
    assert summary.loc["el_nino", "event_season_frequency"] == 50.0
    assert summary.loc["el_nino", "observed_event_season_frequency"] == 50.0
    assert summary.loc["el_nino", "starts_with_event_frequency"] == 50.0
    assert summary.loc["el_nino", "brier_score"] == 0.0

    renamed_product = product.assign_coords(Y=[f"other-{year}" for year in years])
    contrasts = paired_frequency_contrasts(
        {
            ("method_a", 5): product,
            ("method_b", 5): renamed_product,
            ("method_a", 11): product,
            ("method_b", 11): renamed_product,
        },
        experiments=("method_a", "method_b"),
        init_months=(5, 11),
        n_bootstrap=100,
    )
    np.testing.assert_allclose(contrasts.difference_percentage_points, 0.0)
