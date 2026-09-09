"""Unit tests for workflows.diagnostics.drift_summary."""
import unittest
import numpy as np
import xarray as xr

from workflows.diagnostics import drift_summary as ds


class DriftSummaryTests(unittest.TestCase):
    def test_figure_filename(self):
        f = ds.figure_filename("TREFHT", "drift_bias", "ERA5", "Nino3.4")
        self.assertEqual(f, "fig_trefht_drift_bias_era5_nino3_4.png")

    def test_safe_token(self):
        self.assertEqual(ds.safe_token("JRA55-FOSIRL / test"), "JRA55_FOSIRL_test")

    def test_unit_conversions(self):
        da_k = xr.DataArray([273.15, 300.0], attrs={"units": "K"})
        da_c = ds.convert_kelvin_to_celsius(da_k)
        np.testing.assert_allclose(da_c.values, [0.0, 26.85])
        self.assertEqual(da_c.attrs["units"], "degC")

        norm_c = ds.normalize_regional_temperature_units(da_k)
        np.testing.assert_allclose(norm_c.values, [0.0, 26.85])

        da_pa = xr.DataArray([100000.0], attrs={"units": "Pa"})
        da_hpa = ds.convert_pa_to_hpa(da_pa)
        self.assertEqual(da_hpa.item(), 1000.0)

        da_precip = xr.DataArray([1.0e-5], attrs={"units": "m/s"})
        da_mmday = ds.convert_precip_mps_to_mmday(da_precip)
        self.assertAlmostEqual(da_mmday.item(), 864.0)

    def test_calendar_and_lead_math(self):
        # May init (month 5): Lead 1 -> May (5), Lead 2 -> Jun (6), Lead 12 -> Apr (4), Lead 13 -> May (5)
        self.assertEqual(ds.lead_to_cal_month(5, 1), 5)
        self.assertEqual(ds.lead_to_cal_month(5, 2), 6)
        self.assertEqual(ds.lead_to_cal_month(5, 12), 4)
        self.assertEqual(ds.lead_to_cal_month(5, 13), 5)

        vtime = ds.lead_to_valid_time(1990, 5, 1)
        self.assertEqual(vtime.year, 1990)
        self.assertEqual(vtime.month, 5)
        self.assertEqual(vtime.day, 15)

        vtimes = ds.valid_times_for_leads(1990, 5, [1, 2, 3])
        self.assertEqual(len(vtimes), 3)

        ticks, labels = ds.lead_axis_ticks([1, 2, 3], 5)
        self.assertEqual(list(ticks), [0, 1, 2])
        self.assertIn("May", labels[0])
        self.assertIn("Jun", labels[1])

    def test_shock_index_and_drift_bias(self):
        bias = xr.DataArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                            dims=("L",), coords={"L": np.arange(1, 7)})
        shock = ds.shock_index_from_bias(bias, slice(1, 3))
        # sqrt((1^2 + 2^2 + 3^2) / 3) = sqrt(14/3) = 2.160246899
        self.assertAlmostEqual(shock, np.sqrt(14.0 / 3.0))

        rate = ds.season1_drift_from_bias(bias, slice(1, 6))
        # (bias[6] - bias[1]) / (6 - 1) = (6 - 1) / 5 = 1.0
        self.assertAlmostEqual(rate, 1.0)


if __name__ == "__main__":
    unittest.main()
