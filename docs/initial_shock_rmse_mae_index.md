# Initial-shock anomaly RMSE and MAE

Open [`jupyter/6b_initial_shock_rmse_mae_index.ipynb`](../jupyter/6b_initial_shock_rmse_mae_index.ipynb).

This workflow applies the metrics from
[`temp/Compute_RMSE_With_MAE_Index_share.ncl`](../temp/Compute_RMSE_With_MAE_Index_share.ncl)
to the E3SM and CESM-SMYLE monthly archives. It shares archive discovery, strict
ensemble checks, verification-time validation, unit conversion, conservative
regridding, area weighting, and compact global-index caches with the `6a` workflow.

For each model case and initialization month it:

1. averages ensemble members, requiring every configured member;
2. forms unweighted, complete monthly blocks;
3. computes area-weighted global model and observation indices;
4. subtracts the model's mean over the full requested `Y × block` cohort from
   the model index and independently subtracts the observation cohort mean;
5. computes RMSE and MAE over paired blocks for every initialization.

For anomalies \(m'_i\) and \(o'_i\), the metrics are

\[
\operatorname{RMSE}=\sqrt{\frac{1}{n}\sum_i(m'_i-o'_i)^2},\qquad
\operatorname{MAE}=\frac{1}{n}\sum_i|m'_i-o'_i|.
\]

The separate full-cohort baselines are essential: raw-temperature errors and
window-centered errors are different metrics. The NCL script uses 60 months and
five annual values per window. With the current 24-month archive configuration,
the notebook applies the same definitions to two consecutive 12-month blocks.
May and November cohorts are processed separately.

The default TREFHT panels preserve the NCL thresholds:

- RMSE: 0.15, 0.20, 0.25, 0.30, 0.35, and 0.40 °C
- MAE: 0.13, 0.16, 0.19, 0.22, 0.25, and 0.28 °C

Change these plotting thresholds when applying the metric to another variable;
they do not affect cached numerical results. Every derived cache records its
source block-index identity, algorithm version, settings, and source provenance.
