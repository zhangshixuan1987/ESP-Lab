"""Synthetic checks for standalone drift input preparation."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr

from workflows.diagnostics import drift_inputs as inputs


class DriftInputsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.kw = dict(output_root=self.root / 'out', variables=['PSL'], init_months=[5],
                       sources=inputs.SOURCES, regions={'PSL': ('Nino3.4',)},
                       analysis_years=(1980, 1982), variable_years={})
        rng = np.random.default_rng(19)
        self.values = xr.DataArray(1000 + rng.normal(size=(3, 24, 2, 2, 3)),
                                  dims=('Y', 'L', 'M', 'lat', 'lon'),
                                  coords={'Y': [1980, 1981, 1982], 'L': np.arange(1, 25),
                                          'M': [0, 1], 'lat': [-5, 5], 'lon': [190, 210, 240]},
                                  attrs={'units': 'hPa'})
        ref = self.values.mean('M')
        self.hc_path = self.root / 'hindcast.nc'
        self.ref_path = self.root / 'reference.nc'
        time = xr.DataArray(np.tile(np.arange(24), (3, 1)), dims=('Y', 'L'),
                            coords={'Y': ref.Y, 'L': ref.L})
        xr.Dataset({'PSL': self.values, 'time': time}).to_netcdf(self.hc_path)
        xr.Dataset({'X_obs': ref + 0.5, 'X_att': ref - 0.3, 'sigma_att': xr.ones_like(ref),
                    'valid_time': time}).to_netcdf(self.ref_path)

    def build(self):
        with patch.object(inputs, 'default_variable_config', return_value={
            'PSL': {'component': 'atm', 'include_spread': True}}), \
             patch.object(inputs, 'ensure_reference', return_value=self.ref_path), \
             patch.object(inputs.workflow, 'discover_monthly_hindcast', return_value=self.hc_path), \
             patch.object(inputs.workflow, 'load_drift_references', side_effect=lambda *a, **kw: xr.open_dataset(self.ref_path)):
            return inputs.ensure_regional_products(**self.kw)

    def test_first_build_numeric_values_and_fast_reuse(self):
        rows = self.build()
        self.assertEqual(len(rows), 2)
        with xr.open_dataset(rows.iloc[0]['path']) as result:
            np.testing.assert_allclose(result.e_obs, -0.5, atol=1e-10)
            np.testing.assert_allclose(result.skill_rmse, 0.5, atol=1e-10)
            self.assertIn('skill_ensemble_spread', result)
            self.assertEqual(result.fraction_regime_1.dims, ('region', 'L'))
        with patch.object(inputs, 'ensure_reference', side_effect=AssertionError('must reuse')), \
             patch.object(inputs, 'default_variable_config', side_effect=AssertionError('must not scan raw archives')):
            again = inputs.ensure_regional_products(**self.kw)
        self.assertEqual(list(rows.path), list(again.path))
        self.assertEqual(len(list(self.kw['output_root'].glob('*.nc'))), 2)

    def test_require_missing_writes_nothing(self):
        with self.assertRaises(FileNotFoundError):
            inputs.ensure_regional_products(**self.kw, mode='require')
        self.assertFalse(self.kw['output_root'].exists())

    def test_settings_change_is_rejected_in_require_mode(self):
        self.build()
        with self.assertRaises(FileNotFoundError):
            inputs.ensure_regional_products(**self.kw, mode='require', baseline_lead=2)

    def test_missing_skill_field_invalidates_product(self):
        rows = self.build()
        path = Path(rows.iloc[0]['path'])
        with xr.open_dataset(path) as opened:
            ds = opened.load().drop_vars('skill_rmse')
        ds.to_netcdf(path)
        with self.assertRaises(FileNotFoundError):
            inputs.ensure_regional_products(**self.kw, mode='require')


if __name__ == '__main__':
    unittest.main()
