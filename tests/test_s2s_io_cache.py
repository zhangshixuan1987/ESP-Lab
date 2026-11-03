"""S2S weekly caches and figure names follow the s2d_skill conventions."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from esp_lab import env_paths
from esp_lab.diagnostics import s2s_io


def _fake_campaign(calls):
    def load(**kwargs):
        calls.append(kwargs)
        years, members, weeks = kwargs["years"], kwargs["members"], list(kwargs["weeks"])
        data = np.ones((len(years), len(members), len(weeks), 2, 3))
        return xr.DataArray(
            data,
            dims=("Y", "M", "L", "lat", "lon"),
            coords={"Y": years, "M": members, "L": weeks, "lat": [0.0, 1.0], "lon": [0.0, 1.0, 2.0]},
            name=kwargs["field"],
        )
    return load


def _ensure(tmp_path, **overrides):
    kwargs = dict(
        source="JRA55_FOSIRL",
        case_prefix="WCYCL20TR_ne30pg2_r05_IcoswISC30E3r5_JRA55_FOSIRL",
        years=[1980, 1981],
        init_month=5,
        members=["EN00", "EN01"],
        field="TREFHT",
        weeks=range(1, 9),
        root=tmp_path,
        verbose=False,
    )
    kwargs.update(overrides)
    return s2s_io.ensure_s2s_campaign_weekly(**kwargs)


def test_weekly_cache_is_source_first_and_reused(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(s2s_io, "load_s2s_campaign_weekly", _fake_campaign(calls))

    built = _ensure(tmp_path)
    path = (
        tmp_path / "JRA55_FOSIRL" / "s2s_weekly" / "inputs" / "atm" / "TREFHT"
        / "JRA55_FOSIRL_init05_TREFHT_y1980-1981_ny2_N02_W08_week.nc"
    )
    assert path.is_file() and len(calls) == 1
    assert built.attrs["source"] == "JRA55_FOSIRL"

    reused = _ensure(tmp_path)
    assert len(calls) == 1
    xr.testing.assert_identical(reused, built)


def test_stale_weekly_cache_is_rebuilt_in_place(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(s2s_io, "load_s2s_campaign_weekly", _fake_campaign(calls))
    _ensure(tmp_path)
    _ensure(tmp_path, case_prefix="another_case")
    assert len(calls) == 2
    files = list((tmp_path / "JRA55_FOSIRL" / "s2s_weekly" / "inputs" / "atm" / "TREFHT").glob("*.nc"))
    assert len(files) == 1
    with xr.open_dataarray(files[0]) as cached:
        assert cached.attrs["case_prefix"] == "another_case"


def test_require_mode_never_builds(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(s2s_io, "load_s2s_campaign_weekly", _fake_campaign(calls))
    with pytest.raises(FileNotFoundError):
        _ensure(tmp_path, mode="require")
    assert calls == []


def test_s2s_figure_names_and_root(monkeypatch):
    assert (
        s2s_io.s2s_figure_filename("atm", "acc", "TREFHT", "JRA55_FOSIRL", "init05", "w1_w8")
        == "fig_s2s_atm_acc_trefht_jra55_fosirl_init05_w1_w8.png"
    )
    monkeypatch.delenv("ESP_LAB_S2S_FIGURE_ROOT", raising=False)
    assert env_paths.s2s_figure_root() == Path("/global/cfs/cdirs/e3sm/www/zhan391/e3sm-s2s_diag")


def test_gallery_groups_s2s_figures():
    from esp_lab.diagnostics.web import _infer_shortname_and_type, _infer_workflow_group

    name = "fig_s2s_lnd_acc_soilwater_10cm_reanalysis_init11_w1_w8.png"
    assert _infer_workflow_group(name, "", "") == "S2S"
    assert _infer_shortname_and_type(name, "S2S") == ("LND · SOILWATER_10CM", "Weekly ACC · Reanalysis Nov")
