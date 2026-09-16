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

## Pip Installation

ESP-Lab can also be installed from PyPI:

```bash
python -m pip install esp-lab
```

## Conda Installation

When released to conda-forge:

```bash
conda install esp-lab --channel conda-forge
```
