# NMME modes-of-variability Slurm wrappers

These templates run the NMME source adapter and the P6/P7 cross-source
workflows on a Slurm system.  They deliberately retain each NMME model's
native lead availability: never pad leads that are absent from an archive.

Before submission, export these paths or edit the defaults in a local copy:

```bash
export ESP_LAB_ROOT=/path/to/ESP-Lab
export E3SM_ENV_SCRIPT=/path/to/site/environment.sh
export NMME_ROOT=/path/to/NMME-all
export OUTPUT_ROOT=/path/to/durable/output
```

`run_nmme_sst_mov_independent_models.slurm` is an array template.  It writes
one fixed-HadISST2 projected-index product per model; that separation is
required before forming a selected-model or all-model P6 comparison.
`render_p6_common_cohort.slurm` then builds the selected-CMC1 and equal-model
all-NMME P6 products from those independent inputs.  It also needs
`P6_SOURCE_ROOT`, an existing E3SM/SMYLE/HadISST2 source-product root.

The P7 plotting utilities are library functions in
`workflows.modes_of_variability.cross_source`; site-specific jobs should use
the same source-native, no-difference, member/model-spread conventions.

## P7.4 cross-source map conditions

The baseline is climatological: each source retains its own complete native
record. A sensitivity experiment restricted to overlapping years is a later
analysis, not a replacement for the baseline. In the current products,
`E3SM` denotes only
`WCYCL20TR_ne30pg2_r05_IcoswISC30E3r5_JRA55_FOSIRL` (members EN00--EN09); it
does not combine the three available E3SM experiments.

Render source products side by side and do not create source-difference maps.
For skill curves, E3SM and CESM-SMYLE bands are the standard deviation of
member skill curves, a selected NMME model uses its member-skill standard
deviation, and the all-NMME band is the standard deviation across equally
weighted model-mean skill curves. Extend map production only after a
source-consistent Nino3.4 one-field, one-lead pilot has been validated; then
expand to TREFHT, precipitation, PSL, and SST and to both selected-NMME and
all-NMME products.
