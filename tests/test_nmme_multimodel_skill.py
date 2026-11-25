"""NMME cohort selection and multi-model scoring (line = MME forecast, band = min-max)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from workflows.modes_of_variability.cross_source import (
    NMME_BAND_METHOD,
    nmme_multimodel_mean,
    nmme_skill,
    select_nmme_cohort,
)

YEARS = np.arange(1980, 2012)
LEADS = np.arange(1, 4)


def _index(spec, seed=0):
    """spec: {model: (n_members, first_year, last_year, n_leads)}"""
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=(YEARS.size, LEADS.size))
    pieces, labels = [], []
    for model, (members, year0, year1, nleads) in spec.items():
        for member in range(1, members + 1):
            values = truth + rng.normal(scale=1.0, size=truth.shape)
            values[(YEARS < year0) | (YEARS > year1), :] = np.nan
            values[:, nleads:] = np.nan
            pieces.append(values)
            labels.append(f"{model}:M{member:03d}")
    data = xr.DataArray(
        np.stack(pieces, axis=-1), dims=("Y", "L", "M"),
        coords={"Y": YEARS, "L": LEADS, "M": labels}, name="mode_index",
    )
    observation = xr.DataArray(truth, dims=("Y", "L"), coords={"Y": YEARS, "L": LEADS})
    return data, observation


def _skill_function(observation):
    def score(forecast):
        mean = forecast.mean("M", skipna=True)
        obs = observation.sel(Y=forecast.Y)
        return xr.Dataset({
            "corr": xr.corr(mean, obs, dim="Y").where(mean.notnull().any("Y")),
            "rmse": np.sqrt(((mean - obs) ** 2).mean("Y")),
        })
    return score


SPEC = {
    "A": (10, 1981, 2011, 3),
    "B": (10, 1982, 2010, 3),
    "C": (10, 1990, 2011, 3),  # starts late
    "D": (3, 1981, 2011, 3),   # too few members
    "E": (10, 1982, 2010, 2),  # shorter forecast
}


def test_auto_cohort_keeps_models_that_cover_every_start_with_enough_members():
    index, _ = _index(SPEC)
    subset, dropped = select_nmme_cohort(index, (1982, 2010))
    assert sorted({label.split(":")[0] for label in subset.M.values}) == ["A", "B", "E"]
    assert set(dropped) == {"C", "D"}
    assert list(subset.Y.values) == list(range(1982, 2011))
    assert subset.attrs["nmme_years"] == "1982-2010"
    assert subset.attrs["nmme_models"] == "A,B,E"


def test_changing_the_years_changes_the_automatic_model_set():
    index, _ = _index(SPEC)
    subset, dropped = select_nmme_cohort(index, (1991, 2009))
    assert subset.attrs["nmme_models"] == "A,B,C,E"
    assert list(dropped) == ["D"]


def test_explicit_models_must_cover_the_years():
    index, _ = _index(SPEC)
    subset, _ = select_nmme_cohort(index, (1982, 2010), models=["B"])
    assert subset.attrs["nmme_models"] == "B"
    with pytest.raises(ValueError, match="do not cover"):
        select_nmme_cohort(index, (1982, 2010), models=["C"])
    with pytest.raises(ValueError, match="not in the product"):
        select_nmme_cohort(index, (1982, 2010), models=["Z"])


def test_multimodel_mean_weights_models_equally_not_members():
    index = xr.DataArray(
        np.array([1.0, 1.0, 1.0, 1.0, 5.0]).reshape(1, 1, 5), dims=("Y", "L", "M"),
        coords={"Y": [2000], "L": [1], "M": ["A:1", "A:2", "A:3", "A:4", "B:1"]},
    )
    mean = nmme_multimodel_mean(index)
    assert float(mean.squeeze()) == pytest.approx(3.0)  # member pooling would give 1.8
    assert list(mean.M.values) == ["NMME"]


def test_line_is_the_multimodel_forecast_and_band_spans_the_models():
    index, observation = _index(SPEC)
    subset, _ = select_nmme_cohort(index, (1982, 2010))
    score = _skill_function(observation)
    skill = nmme_skill(subset, score)
    per_model = [score(subset.sel(M=[m for m in subset.M.values if m.startswith(name + ":")])) for name in "ABE"]
    expected_min = xr.concat([s.corr for s in per_model], "model").min("model")
    np.testing.assert_allclose(skill.corr_min.values, expected_min.values)
    np.testing.assert_allclose(skill.corr.values, score(nmme_multimodel_mean(subset)).corr.values)
    # Averaging forecasts cancels independent errors, so the line beats the typical model.
    assert float(skill.corr.isel(L=0)) > float(xr.concat([s.corr for s in per_model], "model").mean("model").isel(L=0))
    assert list(skill.model_count.values[:2]) == [3, 3]
    assert skill.model_count.values[2] == 2  # model E stops after two leads
    assert skill.attrs["nmme_band_method"] == NMME_BAND_METHOD
    assert skill.attrs["nmme_years"] == "1982-2010"


def test_single_model_line_uses_its_ensemble_mean_and_member_band():
    index, observation = _index(SPEC)
    subset, _ = select_nmme_cohort(index, (1982, 2010))
    skill = nmme_skill(subset, _skill_function(observation), model="A")
    assert skill.attrs["nmme_models"] == "A"
    assert int(skill.member_count.isel(L=0)) == 10
    # The ensemble mean scores at least as well as the weakest member.
    assert bool((skill.corr_min <= skill.corr).all())
