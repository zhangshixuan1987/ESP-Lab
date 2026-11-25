# NMME comparison Slurm template

`nmme_cross_source_test.slurm` runs the NMME comparison end to end with this
checkout. Run it with `bash` from a login node; it submits both steps:

```bash
export ESP_LAB_ROOT=/path/to/ESP-Lab          # checkout to test
export OUTPUT_ROOT=/path/to/your/test/output  # everything is written here
export SBATCH_ACCOUNT=<allocation>
bash scripts/slurm/nmme_cross_source_test.slurm
```

1. **Process** (array job, one task per NMME model):
   `scripts/run_process_modes_of_variability.py` builds each model's mode
   indices. Each model's lead-dependent drift is removed with that model's own
   climatology, and its anomalies are projected onto the fixed observed EOF
   (HadISST2 for SST modes, ERA5 for pressure modes).
2. **Render** (after step 1): `scripts/render_nmme_comparison.py` scores
   E3SM-FOSIRL, E3SM-Reanalysis and NMME over the NMME start years with the same
   skill function and draws ACC/nRMSE against lead.

E3SM products and the observed reference are read from `DIAG_ROOT`
(default `/global/cfs/cdirs/e3sm/S2S2D/s2d_diag`) and never written.

## NMME definition

* **Line:** skill of the equal-weight multi-model forecast. Each model's
  ensemble mean is formed first and the models are then averaged with one
  weight per model (as in Yeager et al.).
* **Band:** minimum to maximum of the individual models' skill. The legend
  gives the model count; leads where a model's forecast has ended are drawn
  lighter.
* **Years and models:** by default, May and November starts in 1982-2010 (the
  NMME hindcast period) and every model that has all of those starts with at
  least four members: 20 of the 23 archived models. The render step repeats the
  scores for 1991-2009, when all 23 models overlap, as a robustness check.

## Options

Export these before submitting to change the defaults:

| Variable | Default | Meaning |
|---|---|---|
| `MODES` | `PDO NPGO AMO` | modes to process and score |
| `INIT_MONTHS` | `5 11` | start months |
| `NMME_YEARS` | `1982 2010` | NMME start years for the main comparison |
| `NMME_MODELS` | `auto` | `auto`, or a list of model folder names |
| `SINGLE_MODEL` | empty | also draw one model alone (band = its members), e.g. `CMC1-CanCM3` |
| `CHECK_YEARS` | `1991 2009` | robustness-check years; empty skips it |
| `NMME_ROOT`, `DIAG_ROOT` | Perlmutter paths | archive and diagnostics roots |
| `PYTHON`, `E3SM_ENV_SCRIPT` | `python` on `PATH` | environment |

Pressure modes (NAO, PNA, ...) use NMME `prmsl`, which only the Canadian models
provide; models without it are skipped for those modes.

## Output

```
$OUTPUT_ROOT/nmme_models/<model>/NMME/modes_variability/...   per-model products
$OUTPUT_ROOT/figures/nmme_1982_2010/<mode>/                   main comparison
$OUTPUT_ROOT/figures/check_nmme_1991_2009/<mode>/             robustness check
$OUTPUT_ROOT/logs/                                            Slurm logs
```

Each comparison folder holds `<mode>_init<MM>_nmme<years>_skill.{nc,csv,png}`;
the NetCDF attributes list the models used and any model left out, with the
reason.
