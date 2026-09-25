# Install ESP-Lab

ESP-Lab can be installed in several ways depending on your computing environment:

## Development Version from Source (Recommended)

To install the latest development version supporting E3SM S2D diagnostics:

```bash
git clone -b e3sm-esp https://github.com/zhangshixuan1987/ESP-Lab.git
cd ESP-Lab
```

### Option A: Create a Dedicated Conda Environment

```bash
conda env create --file environment.yml
conda activate esp-lab
pip install -e .
```

### Option B: Install into an Existing Conda Environment

If you already maintain an analysis environment (such as `e3sm_analysis` on NERSC Perlmutter):

```bash
conda activate e3sm_analysis
cd ESP-Lab
pip install -e .
```

Use an editable install (`pip install -e .`): the notebooks import the `workflows`
package and read the reference index files under `external/` from the repository
checkout. `7a_tc_method_analysis.ipynb` also needs `global-land-mask`, which
`environment.yml` installs (otherwise `pip install global-land-mask`).

Data and output roots default to the NERSC E3SM paths. Override them per account with
`ESP_LAB_S2D_DIAG_ROOT`, `ESP_LAB_FIGURE_ROOT`, `ESP_LAB_RAW_MODEL_ROOT`,
`ESP_LAB_OBS_ROOT`, and `ESP_LAB_DATA_ROOT`.

## PyPI Package

The `esp-lab` package on PyPI is the upstream CESM ESP-Lab release (1.1.x). It does not
include the E3SM S2D extensions, workflows, or notebooks, so install from source as above.
