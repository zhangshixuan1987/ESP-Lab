"""Standalone input orchestration for modes-of-variability analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from scripts import run_process_cesm_smyle_benchmark as smyle_benchmark
from scripts import run_process_modes_of_variability as mode_processor
from workflows.modes_of_variability import analysis
from workflows.modes_of_variability import teleconnections


VALID_ENSURE_MODES = {"auto", "require", "rebuild"}


def _boolean_flag(name: str, enabled: bool) -> str:
    return f"--{name}" if enabled else f"--no-{name}"


def _base_argv(settings: Mapping[str, object], sources: list[str]) -> list[str]:
    year0, year1 = settings["years"]
    clim0, clim1 = settings["climatology_years"]
    obs0, obs1 = settings["observation_years"]
    eof0, eof1 = settings["eof_reference_years"]
    argv = [
        "--outdir", str(settings["outdir"]),
        "--modes", str(settings["mode"]),
        "--sources", *sources,
        "--init-months", *(str(value) for value in settings["init_months"]),
        "--start-year", str(year0), "--end-year", str(year1),
        "--clim-start", str(clim0), "--clim-end", str(clim1),
        "--obs-start-year", str(obs0), "--obs-end-year", str(obs1),
        "--eof-reference-start-year", str(eof0),
        "--eof-reference-end-year", str(eof1),
        "--monthly-nlead", str(settings["monthly_nlead"]),
        "--target-dlat", str(settings["target_dlat"]),
        "--target-dlon", str(settings["target_dlon"]),
        "--regrid-method", str(settings.get("regrid_method", "conservative")),
        "--dask-workers", str(settings.get("workers", 4)),
        "--smyle-benchmark-dir", str(settings["smyle_benchmark_dir"]),
        "--smyle-nens", str(settings.get("smyle_nens", 20)),
        "--obs-dir", str(settings["obs_dir"]),
        "--psl-obs-product", str(settings.get("psl_obs_product", "ERA5")),
        "--psl-obs-var", str(settings.get("psl_obs_var", "psl")),
        "--ts-obs-product", str(settings.get("ts_obs_product", "HadISST2")),
        "--ts-obs-var", str(settings.get("ts_obs_var", "sst")),
        "--eof-strategy", "fixed_obs_projection",
        "--eof-bootstrap-iterations", str(settings.get("eof_bootstrap_iterations", 0)),
        "--merge-manifest",
        _boolean_flag("model-sst-land-mask", bool(settings.get("model_sst_land_mask", True))),
    ]
    if bool(settings.get("force", False)):
        argv.append("--force")
    return argv


def _processor_args(
    settings: Mapping[str, object],
    sources: list[str],
    *,
    case: Mapping[str, object] | None = None,
    nmme_models: list[str] | None = None,
):
    argv = _base_argv(settings, sources)
    if case is not None:
        argv.extend([
            "--e3sm-data-dir", str(case["data_dir"]),
            "--e3sm-case-prefix", str(case["case_prefix"]),
            "--e3sm-cache-tag", str(case["cache_tag"]),
            "--e3sm-display-name", str(case.get("display_name", case["cache_tag"])),
            "--e3sm-nens", str(case.get("nens", settings.get("e3sm_nens", 10))),
        ])
    if nmme_models is not None:
        argv.extend([
            "--nmme-root", str(settings["nmme_root"]),
            "--nmme-fixed-dir", str(settings["nmme_fixed_dir"]),
            "--nmme-models", *nmme_models,
            "--nmme-field", str(settings.get("nmme_field", "auto")),
            "--nmme-chunks", str(settings.get("nmme_chunks", "")),
            _boolean_flag("nmme-sst-land-mask", bool(settings.get("nmme_sst_land_mask", True))),
        ])
    return analysis.parse_args(argv)


def _smyle_benchmark_issues(
    settings: Mapping[str, object],
) -> tuple[str, str, dict[int, list[str]]]:
    mode_settings = analysis.mode_settings(str(settings["mode"]))
    field = str(mode_settings.get(
        "archive_field", "TS" if mode_settings["field"] == "SST" else mode_settings["field"]
    ))
    frequency = "seas" if mode_settings["frequency"] == "seasonal" else "mon"
    root = Path(settings["smyle_benchmark_dir"])
    years = list(range(int(settings["years"][0]), int(settings["years"][1]) + 1))
    members = [f"EN{index:02d}" for index in range(1, int(settings.get("smyle_nens", 20)) + 1)]
    issues: dict[int, list[str]] = {}
    for month in settings["init_months"]:
        try:
            path = analysis.smyle_access.benchmark_path(
                field, int(month), root,
                nens=len(members), nlead=int(settings["monthly_nlead"]), freq=frequency,
            )
            reasons = smyle_benchmark.existing_benchmark_issues(
                path, field=field, init_month=int(month), years=years,
                members=members, nlead=int(settings["monthly_nlead"]), freq=frequency,
            )
        except (OSError, ValueError, KeyError) as error:
            reasons = [str(error)]
        if reasons:
            issues[int(month)] = reasons
    return field, frequency, issues


def _ensure_smyle_benchmarks(settings: Mapping[str, object], ensure_mode: str) -> None:
    field, frequency, issues = _smyle_benchmark_issues(settings)
    formatted = [
        f"init {month:02d} {field} {frequency}: {'; '.join(reasons)}"
        for month, reasons in issues.items()
    ]
    if ensure_mode == "require":
        if issues:
            raise RuntimeError("Required CESM-SMYLE benchmarks are unavailable:\n  - " + "\n  - ".join(formatted))
        return
    if not issues and ensure_mode != "rebuild":
        return
    years = list(range(int(settings["years"][0]), int(settings["years"][1]) + 1))
    members = [f"EN{index:02d}" for index in range(1, int(settings.get("smyle_nens", 20)) + 1)]
    months_to_build = (
        list(settings["init_months"]) if ensure_mode == "rebuild" else list(issues)
    )
    for month in months_to_build:
        status = smyle_benchmark.process_one(
            field=field, init_month=int(month), data_dir=str(settings["smyle_data_dir"]),
            outdir=str(settings["smyle_benchmark_dir"]), years=years, members=members,
            nlead=int(settings["monthly_nlead"]), require_all_members=True,
            verify_coverage=True, force=True, dry_run=False, run_verify=False,
            freqs=[frequency], open_parallel=False,
        )
        if status not in {"ok", "skipped"}:
            raise RuntimeError(f"CESM-SMYLE benchmark processing failed for init {month}: {status}")
    _, _, remaining = _smyle_benchmark_issues(settings)
    if remaining:
        remaining_text = [
            f"init {month:02d} {field} {frequency}: {'; '.join(reasons)}"
            for month, reasons in remaining.items()
        ]
        raise RuntimeError("CESM-SMYLE benchmarks remain invalid:\n  - " + "\n  - ".join(remaining_text))


def _discover_nmme_models(settings: Mapping[str, object]) -> list[str]:
    configured = [str(value).strip() for value in settings.get("nmme_models", []) if str(value).strip()]
    if configured:
        return configured
    field = "sst" if analysis.mode_settings(str(settings["mode"]))["field"] == "SST" else "prmsl"
    root = Path(settings["nmme_root"])
    if not root.is_dir():
        raise FileNotFoundError(f"NMME root does not exist: {root}")
    models = sorted(
        path.name for path in root.iterdir()
        if path.is_dir() and any((path / field).glob("M*"))
    )
    if not models:
        raise ValueError(f"No NMME models with {field} member directories under {root}")
    return models


def ensure_mode_products(settings: Mapping[str, object]) -> dict[str, object]:
    """Ensure and validate products for one downstream MOV analysis mode."""
    ensure_mode = str(settings.get("ensure_mode", "auto"))
    if ensure_mode not in VALID_ENSURE_MODES:
        raise ValueError(f"ensure_mode must be one of {sorted(VALID_ENSURE_MODES)}")
    selected_mode = str(settings["mode"]).upper().strip()
    if selected_mode not in analysis.SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode: {selected_mode}")
    effective = dict(settings, mode=selected_mode, force=ensure_mode == "rebuild")

    _ensure_smyle_benchmarks(effective, ensure_mode)
    calls = [
        _processor_args(effective, ["obs", "smyle"]),
        *(
            _processor_args(effective, ["obs", "e3sm"], case=case)
            for case in effective["e3sm_cases"].values()
        ),
    ]
    nmme_models: list[str] = []
    if bool(effective.get("include_nmme", False)):
        nmme_models = _discover_nmme_models(effective)
        calls.append(_processor_args(effective, ["obs", "nmme"], nmme_models=nmme_models))

    if ensure_mode != "require":
        for args in calls:
            mode_processor.run(args)
    issues = [issue for args in calls for issue in mode_processor.expected_product_issues(args)]
    if issues:
        raise RuntimeError("Modes-of-variability products are unavailable:\n  - " + "\n  - ".join(issues))
    products: dict[str, dict[str, str]] = {}
    for args in calls:
        products.update(mode_processor.expected_products(args))
    teleconnection_status = {}
    if bool(effective.get("ensure_global_teleconnections", True)):
        teleconnection_status = teleconnections.ensure_products(
            products,
            ensure_mode=ensure_mode,
            alpha=float(effective.get("regression_alpha", 0.05)),
            standardize_index=bool(
                effective.get("regression_standardize_index", True)
            ),
            fdr=bool(effective.get("regression_fdr", True)),
        )
    return {
        "mode": selected_mode,
        "ensure_mode": ensure_mode,
        "processor_calls": len(calls),
        "nmme_models": nmme_models,
        "teleconnection_products": teleconnection_status,
        "manifest": str(Path(effective["outdir"]) / "_manifests" / "modes_manifest.json"),
    }
