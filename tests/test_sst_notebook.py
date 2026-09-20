"""Behavioral checks for standalone SST notebook orchestration."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import xarray as xr

_notebook_matches = list((Path(__file__).resolve().parents[1] / 'jupyter').rglob('3a_sst_skill_ts.ipynb'))
NOTEBOOK = _notebook_matches[0] if _notebook_matches else Path(__file__).resolve().parents[1] / 'jupyter/3a_sst_skill_ts.ipynb'


def source(index):
    cell = json.loads(NOTEBOOK.read_text())['cells'][index]
    return ''.join(line for line in cell['source'] if not line.startswith('%'))


def test_seasonal_plot_coordinates_match_display_lead_ticks():
    plotting = source(17).split(
        "# 2. Monthly ALL-init average panels", 1
    )[0]

    assert "seasonal_lead_offset = 2" in plotting
    assert "plot_lead = tmp.L - seasonal_lead_offset" in plotting
    assert "plot_lead_psl = tmp_psl.L - seasonal_lead_offset" in plotting
    assert "plot_lead_smyle = tmp_smyle.L - seasonal_lead_offset" in plotting
    assert "plot_lead_smyle_psl = tmp_smyle_psl.L - seasonal_lead_offset" in plotting
    assert "plot_lead_nmme = tmp_nmme.L - seasonal_lead_offset" in plotting
    assert (
        "plot_lead_nmme_spread = "
        "tmp_nmme_spread.L.data - seasonal_lead_offset"
    ) in plotting
    assert "ax.plot(tmp.L," not in plotting
    assert "ax2.plot(tmp.L," not in plotting


def test_monthly_plot_starts_at_native_lead_one():
    plotting = source(17).split(
        "# 2. Monthly ALL-init average panels", 1
    )[1]

    assert "monthly_xticks = np.arange(12) * 2 + 1" in source(17)
    assert "monthly_xticks_minor = np.arange(2, 25, 2)" in source(17)
    assert "monthly_xlim = [0.5, 24.5]" in source(17)
    assert ".L - 1" not in plotting
    assert "ax.plot(tmp.L, tmp.corr" in plotting
    assert "ax2.plot(tmp.L, tmp.rmse" in plotting
    assert "ax.fill_between(tmp_nmme_spread.L.data," in plotting


def test_timeseries_axes_and_lines_use_matching_coordinates():
    plotting = source(20)

    assert "a.set_yticks(major_yticks)" in plotting
    assert "a.set_yticks(minor_yticks, minor=True)" in plotting
    assert "a.set_yticks(minor_yticks, minor=False)" not in plotting
    assert "a.plot(obs_djf.time.dt.year, obs_djf" in plotting
    assert "a.plot(psl_djf.time.dt.year, psl_djf" in plotting
    for prefix in ("case", "smyle", "nmme"):
        assert f"ax_panel.plot({prefix}_datatime_plot," in plotting
        assert f"ax_panel.fill_between({prefix}_datatime_plot," in plotting


@pytest.mark.parametrize('mode,exists,calls,fails', [
    ('auto', True, 0, False), ('auto', False, 1, False),
    ('require', False, 0, True), ('rebuild', True, 1, False),
])
def test_input_modes(tmp_path, mode, exists, calls, fails):
    paths = {f: tmp_path / f'{f}.nc' for f in ['mon', 'seas']}
    if exists:
        for path in paths.values():
            path.touch()
    def process(args):
        for path in paths.values():
            path.touch()
    processor = SimpleNamespace(process_obs=Mock(side_effect=process))
    ns = dict(requested_upstream_sources={'obs'}, E3SM_CASES={},
              _obs_sst_index_file=lambda freq: paths[freq],
              _sst_index_file_is_current=lambda path: (path.exists(), 'missing'),
              upstream_mode=mode, sst_index_processor=processor,
              _base_sst_index_args=lambda **kw: SimpleNamespace(**kw),
              file_inventory_digest=lambda paths: 'test-digest')
    body = source(5).split('required_sst_index_files = {}', 1)[1]
    body = 'required_sst_index_files = {}' + body
    if fails:
        with pytest.raises(RuntimeError, match='Required SST-index diagnostics'):
            exec(body, ns)
    else:
        exec(body, ns)
    assert processor.process_obs.call_count == calls
    if calls:
        assert processor.process_obs.call_args.args[0].force == (mode == 'rebuild')


@pytest.mark.parametrize('missing', [None, 'smyle_seas_skill_pval', 'smyle_skill_rmse', 'smyle_skill_ref_corr'])
def test_smyle_cache_rejects_missing_consumed_metrics(tmp_path, missing):
    names = {f'{prefix}_{metric}'
             for prefix in ['smyle_skill', 'smyle_seas_skill', 'smyle_skill_ref', 'smyle_seas_skill_ref']
             for metric in ['corr', 'pval', 'rmse', 'sample_count', 'target_year_start', 'target_year_end']}
    if missing:
        names.remove(missing)
    ds = xr.Dataset({name: xr.DataArray(1.) for name in names},
                    coords={'seasonal_L': [3, 6, 9, 12, 15, 18, 21], 'reference': ['HadISST2']},
                    attrs=dict(skill_cache_version=5, verification_start=1981, verification_end=2011,
                               climatology_start=1981, climatology_end=2010, region='AtlMDR',
                               input_inventory_identity='test', detrend=1, init_months='5,11'))
    path = tmp_path / 'skill.nc'
    ds.to_netcdf(path)
    node = next(node for node in ast.parse(source(14)).body
                if isinstance(node, ast.If) and 'SMYLE_SKILL_FILE.is_file()' in ast.unparse(node.test))
    ns = dict(xr=xr, SMYLE_SKILL_FILE=path, smyle_cfg={}, smyle_skill_cache_exists=False,
              SST_SKILL_CACHE_VERSION=5, skill_year0=1981, skill_year1=2011, climy0=1981, climy1=2010,
              cfg=SimpleNamespace(region_name='AtlMDR', init_months=[5, 11]),
              smyle_input_identity='test', WORKFLOW_SETTINGS={'skill': {'detrend': True}},
              psl_sst_index_reference_available=False)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'smyle_cache_cell', 'exec'), ns)
    assert ns['smyle_skill_cache_exists'] == (missing is None)
