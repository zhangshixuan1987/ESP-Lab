# How to use preprocessors

Preprocessors are applied to individual time series files in order to return mean model fields with centered time coordinates aligned along standardized lead-time dimensions.

The main steps in preprocessing are:
1. Extract a time slice of size `nlead`.
2. Create a lead time coordinate `L` as a 1-based integer sequence.
3. Swap `time` with `L` so that `L` becomes a shared coordinate to aggregate data over, while `time` becomes a data variable indexed by initialization year (`Y`).
4. Extract the requested field variable from the dataset.
5. Chunk to include all `L` values (which all come from a single NetCDF file).

Preprocessors are tailored for specific model configurations:
- **E3SM S2D**: `esp_lab.data_access_e3sm.preprocessor()`
- **CESM SMYLE**: `esp_lab.data_access_cesm_smyle.preprocessor()`
- **Legacy SMYLE**: `esp_lab.data_access_smyle.preprocessor()`

Workflow examples demonstrating how preprocessors are used in distributed evaluation pipelines can be found in the [Analysis Notebooks](../tutorials/index.md) and [`jupyter/`](https://github.com/zhangshixuan1987/ESP-Lab/tree/e3sm-esp/jupyter) directory.
