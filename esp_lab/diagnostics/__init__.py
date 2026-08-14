from .config import (
    DEFAULT_CLIMATOLOGY_END_YEAR,
    DEFAULT_CLIMATOLOGY_START_YEAR,
    DEFAULT_CLIMATOLOGY_YEARS,
    S2DConfig,
)
from .s2d import S2DDiagnostics
from .web import discover_workflow_figures, generate_diagnostics_webpage

# IC analysis — pure core (no I/O, fully unit-testable)
from .ic_core import (
    # Configuration
    COMPONENT_NAMES,
    DEFAULT_COMPONENTS,
    ComponentSpec,
    ExperimentPair,
    ICConfig,
    # File audit
    FileRecord,
    AuditResult,
    compare_file_records,
    # Schema comparison
    StructureDiff,
    compare_nc_schema,
    # Variable statistics
    classify_variable,
    ic_variable_stats,
    # Campaign aggregation
    aggregate_campaign_stats,
    # Physical consistency
    ConsistencyReport,
    check_atm_surface_vs_land,
    check_ocean_ice_consistency,
    # Provenance
    write_manifest_json,
)

# IC analysis — filesystem I/O bridge
from .ic_io import (
    AUDIT_MANIFEST_COLUMNS,
    MatchedPair,
    discover_start_dates,
    match_start_dates,
    compute_sha256,
    read_rpointer,
    open_restart_file,
    read_nc_dims,
    read_nc_global_attrs,
    read_sim_timestamp,
    build_file_record,
    build_file_records_for_pair,
    build_file_manifest,
    write_audit_csv,
    load_audit_csv,
)

# Drift analysis pipeline (single unified module)
from .drift import (
    # Configuration dataclasses
    DriftConfig,
    ExperimentSpec,
    VariableSpec,
    # Data contract
    CaseArray,
    # Calendar utilities
    lead_to_cal_month,
    valid_year_month,
    # Observation lookup
    build_obs_lookup,
    # Core drift diagnostics
    absolute_bias,
    adjustment,
    paired_difference,
    paired_difference_by_start,
    # Skill diagnostics
    compute_skill,
    # Paired bootstrap
    bootstrap_paired_ci,
    # Window summary and manifest
    window_summary,
    write_manifest,
    # Unit conversions
    convert_kelvin_to_celsius,
    convert_kelvin_to_celsius_if_needed,
    convert_precip_mps_to_mmday,
    convert_pa_to_hpa,
    no_conversion,
    # ESP-Lab I/O
    open_regional_cache,
    regional_cache_path,
    cache_to_case_array,
    build_obs_regional,
    # Pipeline runner
    run_pipeline,
)

# Two-reference drift calculations (Y is retained throughout).
from .two_reference_drift import (
    REGIME_DEFINITIONS,
    area_weighted_mean,
    area_weighted_rmse,
    bootstrap_paired_mean_ci,
    build_attractor_lookup,
    build_reference_lookup,
    classify_drift_regime,
    compare_drift_skill_relationship,
    compare_initializations,
    compute_diagnostics as compute_two_reference_diagnostics,
    compute_distance_change,
    compute_early_drift_late_error,
    compute_ensemble_mean,
    compute_initialization_adjustment,
    compute_lead_window_mean,
    compute_reference_departure,
    compute_regime_fraction,
    compute_prediction_skill,
    compute_spatial_drift_summary,
    run_pipeline as run_two_reference_pipeline,
)

__all__ = [
    # Legacy S2D diagnostics
    "DEFAULT_CLIMATOLOGY_END_YEAR",
    "DEFAULT_CLIMATOLOGY_START_YEAR",
    "DEFAULT_CLIMATOLOGY_YEARS",
    "S2DConfig",
    "S2DDiagnostics",
    "discover_workflow_figures",
    "generate_diagnostics_webpage",
    # Drift — configuration
    "DriftConfig",
    "ExperimentSpec",
    "VariableSpec",
    # Drift — data contract
    "CaseArray",
    # Drift — calendar
    "lead_to_cal_month",
    "valid_year_month",
    # Drift — obs lookup
    "build_obs_lookup",
    # Drift — diagnostics
    "absolute_bias",
    "adjustment",
    "paired_difference",
    "paired_difference_by_start",
    "compute_skill",
    "bootstrap_paired_ci",
    "window_summary",
    "write_manifest",
    # Drift — unit conversions
    "convert_kelvin_to_celsius",
    "convert_kelvin_to_celsius_if_needed",
    "convert_precip_mps_to_mmday",
    "convert_pa_to_hpa",
    "no_conversion",
    # Drift — I/O
    "open_regional_cache",
    "regional_cache_path",
    "cache_to_case_array",
    "build_obs_regional",
    "run_pipeline",
    # Two-reference drift
    "REGIME_DEFINITIONS",
    "area_weighted_mean",
    "area_weighted_rmse",
    "bootstrap_paired_mean_ci",
    "build_attractor_lookup",
    "build_reference_lookup",
    "classify_drift_regime",
    "compare_drift_skill_relationship",
    "compare_initializations",
    "compute_two_reference_diagnostics",
    "compute_distance_change",
    "compute_early_drift_late_error",
    "compute_ensemble_mean",
    "compute_initialization_adjustment",
    "compute_lead_window_mean",
    "compute_reference_departure",
    "compute_regime_fraction",
    "compute_prediction_skill",
    "compute_spatial_drift_summary",
    "run_two_reference_pipeline",
    # IC analysis — core
    "COMPONENT_NAMES",
    "DEFAULT_COMPONENTS",
    "ComponentSpec",
    "ExperimentPair",
    "ICConfig",
    "FileRecord",
    "AuditResult",
    "compare_file_records",
    "StructureDiff",
    "compare_nc_schema",
    "classify_variable",
    "ic_variable_stats",
    "aggregate_campaign_stats",
    "ConsistencyReport",
    "check_atm_surface_vs_land",
    "check_ocean_ice_consistency",
    "write_manifest_json",
    # IC analysis — I/O
    "AUDIT_MANIFEST_COLUMNS",
    "MatchedPair",
    "discover_start_dates",
    "match_start_dates",
    "compute_sha256",
    "read_rpointer",
    "open_restart_file",
    "read_nc_dims",
    "read_nc_global_attrs",
    "read_sim_timestamp",
    "build_file_record",
    "build_file_records_for_pair",
    "build_file_manifest",
    "write_audit_csv",
    "load_audit_csv",
]

# Monthly spatial analysis — pure core (no I/O, fully unit-testable)
from .monthly_core import (
    # Configuration
    ExperimentSpec as MonthlyExperimentSpec,
    WindowDef,
    VariableConversionSpec,
    MonthlyConfig,
    DEFAULT_EXPERIMENT_SPECS,
    DEFAULT_WINDOW_DEFS,
    # Inventory types
    InventoryStatus,
    GateStatus,
    LeadCoverageResult,
    PairedReadinessReport,
    # Inventory validation
    check_lead_coverage,
    classify_analysis_status,
    classify_archive_status,
    build_paired_readiness,
    # Spatial diagnostics
    valid_year_month_spatial,
    spatial_bias,
    spatial_adjustment,
    spatial_paired_diff,
    paired_effect_size,
    paired_fdr_significance,
    window_average,
    # Bootstrap
    bootstrap_spatial_ci,
    significance_mask,
    # Helpers
    apply_mask,
    init_tag_from_year_month,
    # Unit conversions
    convert_prect_to_mmday,
    VARIABLE_CONVERSIONS,
    # Inventory output
    InventoryRecord,
    inventory_records_to_df,
    write_inventory_csv,
    write_missing_report,
    write_inventory_json,
)

# Monthly spatial analysis — filesystem I/O bridge
from .monthly_io import (
    build_expected_file_path,
    glob_field_files,
    probe_file,
    discover_all_files,
    load_spatial_field,
    load_all_members,
    load_campaign_field,
    load_obs_spatial,
)

__all__ += [
    # Monthly core — config
    "MonthlyExperimentSpec",
    "WindowDef",
    "VariableConversionSpec",
    "MonthlyConfig",
    "DEFAULT_EXPERIMENT_SPECS",
    "DEFAULT_WINDOW_DEFS",
    # Monthly core — inventory types
    "InventoryStatus",
    "GateStatus",
    "LeadCoverageResult",
    "PairedReadinessReport",
    # Monthly core — inventory validation
    "check_lead_coverage",
    "classify_analysis_status",
    "classify_archive_status",
    "build_paired_readiness",
    # Monthly core — spatial diagnostics
    "valid_year_month_spatial",
    "spatial_bias",
    "spatial_adjustment",
    "spatial_paired_diff",
    "paired_effect_size",
    "paired_fdr_significance",
    "window_average",
    # Monthly core — bootstrap
    "bootstrap_spatial_ci",
    "significance_mask",
    # Monthly core — helpers
    "apply_mask",
    "init_tag_from_year_month",
    # Monthly core — conversions
    "convert_prect_to_mmday",
    "VARIABLE_CONVERSIONS",
    # Monthly core — inventory output
    "InventoryRecord",
    "inventory_records_to_df",
    "write_inventory_csv",
    "write_missing_report",
    "write_inventory_json",
    # Monthly IO
    "build_expected_file_path",
    "glob_field_files",
    "probe_file",
    "discover_all_files",
    "load_spatial_field",
    "load_all_members",
    "load_campaign_field",
    "load_obs_spatial",
]

# Daily spatial analysis — pure core (no I/O, fully unit-testable)
from .daily_core import (
    ExperimentSpec as DailyExperimentSpec,
    DailyWindowDef,
    DailyVariableSpec,
    DailyDriftConfig,
    DEFAULT_DAILY_EXPERIMENT_SPECS,
    DEFAULT_DAILY_WINDOW_DEFS,
    RECOMMENDED_DAILY_WINDOW_DEFS,
    DailyInventoryStatus,
    DailyGateStatus,
    DailyLeadCoverageResult,
    DailyPairedReadinessReport,
    check_daily_lead_coverage,
    classify_daily_window_status,
    classify_daily_analysis_status,
    classify_daily_archive_status,
    build_daily_paired_readiness,
    valid_date_from_init_and_lead,
    daily_spatial_bias,
    daily_spatial_adjustment,
    daily_spatial_paired_diff,
    rapid_adjustment_metrics,
    daily_window_average,
    bootstrap_daily_spatial_ci,
    daily_significance_mask,
    daily_regional_timeseries,
    convert_kelvin_to_celsius_if_needed as daily_convert_kelvin_to_celsius,
    DAILY_VARIABLE_CONVERSIONS,
    DailyInventoryRecord,
    daily_inventory_records_to_df,
    write_daily_inventory_csv,
    write_daily_missing_report,
    write_daily_inventory_json,
)

# Daily spatial analysis — filesystem I/O bridge
from .daily_io import (
    daily_data_dir,
    discover_daily_files,
    probe_daily_file,
    discover_all_daily_files,
    load_daily_spatial_field,
    load_daily_campaign_field,
    load_daily_obs_spatial,
)

__all__ += [
    # Daily core
    "DailyExperimentSpec",
    "DailyWindowDef",
    "DailyVariableSpec",
    "DailyDriftConfig",
    "DEFAULT_DAILY_EXPERIMENT_SPECS",
    "DEFAULT_DAILY_WINDOW_DEFS",
    "RECOMMENDED_DAILY_WINDOW_DEFS",
    "DailyInventoryStatus",
    "DailyGateStatus",
    "DailyLeadCoverageResult",
    "DailyPairedReadinessReport",
    "check_daily_lead_coverage",
    "classify_daily_window_status",
    "classify_daily_analysis_status",
    "classify_daily_archive_status",
    "build_daily_paired_readiness",
    "valid_date_from_init_and_lead",
    "daily_spatial_bias",
    "daily_spatial_adjustment",
    "daily_spatial_paired_diff",
    "rapid_adjustment_metrics",
    "daily_window_average",
    "bootstrap_daily_spatial_ci",
    "daily_significance_mask",
    "daily_regional_timeseries",
    "daily_convert_kelvin_to_celsius",
    "DAILY_VARIABLE_CONVERSIONS",
    "DailyInventoryRecord",
    "daily_inventory_records_to_df",
    "write_daily_inventory_csv",
    "write_daily_missing_report",
    "write_daily_inventory_json",
    # Daily IO
    "daily_data_dir",
    "discover_daily_files",
    "probe_daily_file",
    "discover_all_daily_files",
    "load_daily_spatial_field",
    "load_daily_campaign_field",
    "load_daily_obs_spatial",
]

# Physical consistency diagnostics — pure core & I/O bridge
from .physical_core import (
    compute_evaporative_fraction,
    compute_bowen_ratio,
    compute_sst_trefht_contrast,
    integrate_soil_moisture,
    compute_coupling_slope,
    compute_precip_sm_lag_response,
    compute_apparent_energy_residual,
    bootstrap_physical_metric_ci,
)

from .physical_io import (
    load_physical_variables_daily,
    load_physical_variables_monthly,
)

__all__ += [
    # Physical core
    "compute_evaporative_fraction",
    "compute_bowen_ratio",
    "compute_sst_trefht_contrast",
    "integrate_soil_moisture",
    "compute_coupling_slope",
    "compute_precip_sm_lag_response",
    "compute_apparent_energy_residual",
    "bootstrap_physical_metric_ci",
    # Physical IO
    "load_physical_variables_daily",
    "load_physical_variables_monthly",
]

# Model-Attractor diagnostics — pure core & reference bridge
from .attractor_core import (
    MovementCategory,
    spatial_rmsd,
    compute_attractor_distances,
    compute_relative_movement,
    classify_movement_category,
    classify_spatial_movement,
    TrajectoryPoint2Axis,
    build_2axis_trajectory,
)

from .references_io import (
    build_observation_reference,
    build_e3sm_historical_climatology,
)

__all__ += [
    # Attractor core
    "MovementCategory",
    "spatial_rmsd",
    "compute_attractor_distances",
    "compute_relative_movement",
    "classify_movement_category",
    "classify_spatial_movement",
    "TrajectoryPoint2Axis",
    "build_2axis_trajectory",
    # References IO
    "build_observation_reference",
    "build_e3sm_historical_climatology",
]

# Compute-once/read-many diagnostic stores
from .store import (
    config_fingerprint,
    dataframe_fingerprint,
    diagnostic_store_is_valid,
    diagnostic_store_path,
    file_fingerprint,
    load_diagnostic_store,
    save_diagnostic_store,
)

__all__ += [
    "config_fingerprint",
    "dataframe_fingerprint",
    "diagnostic_store_is_valid",
    "diagnostic_store_path",
    "file_fingerprint",
    "load_diagnostic_store",
    "save_diagnostic_store",
]

# Shared 5a--5f product contract and IC-to-drift attribution
from .products import (
    PAIRED_SIGN_CONVENTION,
    PRODUCT_COLUMNS,
    PRODUCT_SCHEMA_VERSION,
    combine_product_bundles,
    read_product_bundle,
    resolve_experiment_roles,
    standardize_product_table,
    validate_product_table,
    write_product_bundle,
)
from .attribution_core import across_start_attribution, weighted_ic_alignment

__all__ += [
    "PAIRED_SIGN_CONVENTION", "PRODUCT_COLUMNS", "PRODUCT_SCHEMA_VERSION",
    "combine_product_bundles", "read_product_bundle", "standardize_product_table",
    "resolve_experiment_roles",
    "validate_product_table", "write_product_bundle", "across_start_attribution",
    "weighted_ic_alignment",
]
