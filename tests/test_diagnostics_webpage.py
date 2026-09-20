import json
import stat
import pytest
from pathlib import Path
from esp_lab.diagnostics.web import (
    discover_workflow_figures,
    generate_diagnostics_webpage,
    _infer_shortname_and_type,
)


def test_generate_webpage_success(tmp_path):
    # Create mock figures.json
    mock_data = {
        "updated": "2026-06-29T10:00:00Z",
        "figures": [
            {
                "file": "fig_nao_skill.png",
                "mode": "NAO",
                "metric": "skill",
                "title": "Seasonal NAO Skill",
                "caption": "Verification details here",
            }
        ],
    }
    
    manifest_file = tmp_path / "figures.json"
    with open(manifest_file, "w", encoding="utf-8") as fh:
        json.dump(mock_data, fh, indent=2)

    # Generate webpage
    output_html = generate_diagnostics_webpage(tmp_path)
    
    assert output_html.exists()
    assert output_html.name == "index.html"
    
    # Read generated HTML and verify content
    with open(output_html, "r", encoding="utf-8") as fh:
        html_content = fh.read()
        
    assert "ESP-Lab Diagnostic Viewer" in html_content
    # Check if mock figures.json was correctly embedded
    assert "fig_nao_skill.png" in html_content
    assert "Seasonal NAO Skill" in html_content
    assert "All figure types" in html_content
    assert "classifyFigureType" in html_content
    assert "All Metrics" not in html_content


def test_generate_webpage_missing_directory():
    non_existent_dir = Path("/non/existent/path/for/diagnostics/webpage")
    with pytest.raises(FileNotFoundError):
        generate_diagnostics_webpage(non_existent_dir)


def test_generate_webpage_missing_manifest(tmp_path):
    # No figures.json created in tmp_path
    with pytest.raises(FileNotFoundError):
        generate_diagnostics_webpage(tmp_path)


def test_discover_workflow_figures_reconciles_manifest(tmp_path):
    (tmp_path / "fig_nao_skill.png").touch()
    (tmp_path / "fig_nmme_nino34_f03_skill.png").touch()
    (tmp_path / "notes.png").touch()
    manifest_file = tmp_path / "figures.json"
    manifest_file.write_text(
        json.dumps(
            {
                "figures": [
                    {
                        "file": "fig_nao_skill.png",
                        "mode": "NAO",
                        "metric": "skill",
                        "title": "Saved NAO title",
                        "caption": "Saved caption",
                    },
                    {"file": "fig_stale.png", "title": "Stale figure"},
                ]
            }
        ),
        encoding="utf-8",
    )

    manifest = discover_workflow_figures(tmp_path, write_manifest=True)

    assert [figure["file"] for figure in manifest["figures"]] == [
        "fig_nao_skill.png",
        "fig_nmme_nino34_f03_skill.png",
    ]
    assert manifest["figures"][0]["title"] == "Saved NAO title"
    assert manifest["figures"][0]["group"] == "MOV"
    assert manifest["figures"][1]["mode"] == "NMME"
    assert manifest["figures"][1]["group"] == "SST_INDEX"
    assert manifest["figures"][1]["title"] == "NMME Niño3.4 F03 Skill"
    assert json.loads(manifest_file.read_text(encoding="utf-8")) == manifest


def test_discover_workflow_figures_groups_leadtime_drift(tmp_path):
    figure = tmp_path / "fig_leadtime_drift_two_reference_sst_05_skill.png"
    figure.touch()

    manifest = discover_workflow_figures(tmp_path)

    assert len(manifest["figures"]) == 1
    assert manifest["figures"][0]["metric"] == "leadtime_drift"
    assert manifest["figures"][0]["group"] == "LEAD_DRIFT"


def test_generate_webpage_with_discovery_needs_no_existing_manifest(tmp_path):
    figure_path = tmp_path / "fig_eli_time_series.svg"
    figure_path.touch()
    figure_path.chmod(0o600)
    tmp_path.chmod(0o700)

    output_html = generate_diagnostics_webpage(tmp_path, discover_figures=True)
    html_content = output_html.read_text(encoding="utf-8")

    assert "fig_eli_time_series.svg" in html_content
    assert "ELI Time Series" in html_content
    assert (tmp_path / "figures.json").is_file()
    assert figure_path.stat().st_mode & stat.S_IROTH
    assert tmp_path.stat().st_mode & stat.S_IXOTH
    assert stat.S_IMODE(output_html.stat().st_mode) == 0o644
    assert stat.S_IMODE((tmp_path / "figures.json").stat().st_mode) == 0o644


def test_generate_webpage_with_discovery_requires_figures(tmp_path):
    with pytest.raises(FileNotFoundError, match="No workflow figures"):
        generate_diagnostics_webpage(tmp_path, discover_figures=True)


def test_discover_workflow_figures_recognizes_split_mov_figures(tmp_path):
    (tmp_path / "fig_nam_eof_patterns_year1.png").touch()
    (tmp_path / "fig_nam_eof_patterns_year2.png").touch()
    (tmp_path / "fig_pna_global_teleconnection_patterns_init05.png").touch()
    (tmp_path / "fig_pna_global_teleconnection_patterns_init11.png").touch()

    manifest = discover_workflow_figures(tmp_path)
    by_file = {fig["file"]: fig for fig in manifest["figures"]}

    eof_y1 = by_file["fig_nam_eof_patterns_year1.png"]
    assert eof_y1["mode"] == "NAM"
    assert eof_y1["metric"] == "eof_patterns"
    assert eof_y1["group"] == "MOV"
    assert eof_y1["title"] == "NAM EOF Patterns Year 1"

    eof_y2 = by_file["fig_nam_eof_patterns_year2.png"]
    assert eof_y2["mode"] == "NAM"
    assert eof_y2["metric"] == "eof_patterns"
    assert eof_y2["group"] == "MOV"
    assert eof_y2["title"] == "NAM EOF Patterns Year 2"

    tele_may = by_file["fig_pna_global_teleconnection_patterns_init05.png"]
    assert tele_may["mode"] == "PNA"
    assert tele_may["metric"] == "global_teleconnection_patterns"
    assert tele_may["group"] == "MOV"
    assert tele_may["title"] == "PNA Global Teleconnection Patterns May Init"

    tele_nov = by_file["fig_pna_global_teleconnection_patterns_init11.png"]
    assert tele_nov["mode"] == "PNA"
    assert tele_nov["metric"] == "global_teleconnection_patterns"
    assert tele_nov["group"] == "MOV"
    assert tele_nov["title"] == "PNA Global Teleconnection Patterns Nov Init"


def test_infer_shortname_and_type():
    # LEAD_ACC
    s, t = _infer_shortname_and_type("fig_leadtime_acc_skill_map_prect.png", "LEAD_ACC")
    assert s == "PRECT"
    assert t == "ACC Skill Map"

    s, t = _infer_shortname_and_type("fig_atm_acc_prect_acc_distribution.png", "LEAD_ACC")
    assert s == "PRECT"
    assert t == "Sample Period Sensitivity"

    # LEAD_RMSE
    s, t = _infer_shortname_and_type("fig_leadtime_rmse_compare_conus_prect.png", "LEAD_RMSE")
    assert s == "PRECT"
    assert t == "Compare (CONUS)"

    # SST_INDEX
    s, t = _infer_shortname_and_type("fig_sst_index_skill_nino34.png", "SST_INDEX")
    assert s == "Niño3.4"
    assert t == "ACC Skill"

    # MOV
    s, t = _infer_shortname_and_type("fig_nam_eof_patterns_year1.png", "MOV", mode="NAM")
    assert s == "NAM"
    assert t == "EOF Year 1"

    # ELI
    s, t = _infer_shortname_and_type("fig_eli_acc_nrmse_skill.png", "ELI")
    assert s == "ELI Diagnostics"
    assert t == "ACC / nRMSE Skill"

    # INITIAL_SHOCK
    s, t = _infer_shortname_and_type("fig_shock_normalized_change_scatter_seasonal_prect.png", "INITIAL_SHOCK")
    assert s == "PRECT"
    assert t == "Seasonal Scatter"

    # TELECONNECTIONS
    s, t = _infer_shortname_and_type("teleconnections/teleconnection_nino34_prect_corr_map.png", "TELECONNECTIONS")
    assert s == "Niño3.4 · PRECT"
    assert t == "Correlation Map"


def test_discover_workflow_figures_assigns_shortname_and_type(tmp_path):
    (tmp_path / "fig_leadtime_acc_skill_map_sst.png").touch()
    (tmp_path / "fig_sst_index_skill_iod.png").touch()

    manifest = discover_workflow_figures(tmp_path)
    by_file = {fig["file"]: fig for fig in manifest["figures"]}

    sst_acc = by_file["fig_leadtime_acc_skill_map_sst.png"]
    assert sst_acc["shortname"] == "SST"
    assert sst_acc["btn_type"] == "ACC Skill Map"

    iod_fig = by_file["fig_sst_index_skill_iod.png"]
    assert iod_fig["shortname"] == "IOD"
    assert iod_fig["btn_type"] == "ACC Skill"


def test_generate_webpage_contains_matrix_dashboard(tmp_path):
    (tmp_path / "fig_leadtime_acc_skill_map_prect.png").touch()
    output_html = generate_diagnostics_webpage(tmp_path, discover_figures=True)
    content = output_html.read_text(encoding="utf-8")

    # Check matrix view markup and scripts
    assert "matrixDashboard" in content
    assert "Quick Buttons View" in content
    assert "bumpOutModal" in content
    assert "renderMatrixView" in content
    assert "renderCardsView" in content
    assert "setViewMode" in content
    assert "openLightboxForFigureByFile" in content


def test_canonical_prefixes_classification(tmp_path):
    # Test canonical prefixes across workflow steps
    test_files = [
        ("fig_1a_prect_acc_compare.png", "LEAD_ACC", "PRECT", "Model Compare"),
        ("fig_atm_acc_prect_compare.png", "LEAD_ACC", "PRECT", "Model Compare"),
        ("fig_1b_h2osoi_acc_difference.png", "LEAD_ACC", "H2OSOI", "Difference"),
        ("fig_lnd_acc_h2osoi_difference.png", "LEAD_ACC", "H2OSOI", "Difference"),
        ("fig_2a_prect_rmse_conus.png", "LEAD_RMSE", "PRECT", "CONUS RMSE"),
        ("fig_atm_rmse_prect_conus.png", "LEAD_RMSE", "PRECT", "CONUS RMSE"),
        ("fig_2b_prect_rmse_difference_compare_init05.png", "LEAD_RMSE", "PRECT", "Diff Compare (May)"),
        ("fig_rmse_compare_prect_difference_compare_init05.png", "LEAD_RMSE", "PRECT", "Diff Compare (May)"),
        ("fig_rmse_compare_prect_rmse_difference_compare_init11.png", "LEAD_RMSE", "PRECT", "Diff Compare (Nov)"),
        ("fig_rmse_compare_prect_rmse_difference_global_init05.png", "LEAD_RMSE", "PRECT", "Diff Global (May)"),
        ("fig_rmse_compare_prect_rmse_difference_global_init11.png", "LEAD_RMSE", "PRECT", "Diff Global (Nov)"),
        ("fig_rmse_compare_prect_conus.png", "LEAD_RMSE", "PRECT", "Compare (CONUS)"),
        ("fig_rmse_compare_prect_rmse_compare_global.png", "LEAD_RMSE", "PRECT", "Compare (Global)"),
        ("fig_rmse_compare_psl_rmse_difference_compare_init05.png", "LEAD_RMSE", "PSL", "Diff Compare (May)"),
        ("fig_rmse_compare_psl_rmse_difference_compare_init11.png", "LEAD_RMSE", "PSL", "Diff Compare (Nov)"),
        ("fig_rmse_compare_psl_rmse_difference_global_init05.png", "LEAD_RMSE", "PSL", "Diff Global (May)"),
        ("fig_rmse_compare_psl_rmse_difference_global_init11.png", "LEAD_RMSE", "PSL", "Diff Global (Nov)"),
        ("fig_rmse_compare_trefht_rmse_difference_compare_init05.png", "LEAD_RMSE", "TREFHT", "Diff Compare (May)"),
        ("fig_rmse_compare_trefht_rmse_difference_compare_init11.png", "LEAD_RMSE", "TREFHT", "Diff Compare (Nov)"),
        ("fig_rmse_compare_trefht_rmse_difference_global_init05.png", "LEAD_RMSE", "TREFHT", "Diff Global (May)"),
        ("fig_rmse_compare_trefht_rmse_difference_global_init11.png", "LEAD_RMSE", "TREFHT", "Diff Global (Nov)"),
        ("fig_3a_nino34_acc_skill.png", "SST_INDEX", "Niño3.4", "ACC Skill"),
        ("fig_sst_index_nino34_acc_skill.png", "SST_INDEX", "Niño3.4", "ACC Skill"),
        ("fig_3a_roni_time_series.png", "SST_INDEX", "RONI", "Time Series"),
        ("fig_4a_pdo_eof_patterns_year1.png", "MOV", "PDO", "EOF Year 1"),
        ("fig_mov_pdo_eof_patterns_year1.png", "MOV", "PDO", "EOF Year 1"),
        ("fig_4a_nao_global_teleconnection_patterns_init05.png", "MOV", "NAO", "Telecon May"),
        ("fig_mov_nao_global_teleconnection_patterns_init05.png", "MOV", "NAO", "Telecon May"),
        ("fig_5a_eli_multimodel_acc_nrmse_skill.png", "ELI", "ELI Diagnostics", "ACC / nRMSE Skill"),
        ("fig_eli_multimodel_acc_nrmse_skill.png", "ELI", "ELI Diagnostics", "ACC / nRMSE Skill"),
        ("fig_3b_teleconnection_nino34_prect_corr_map.png", "TELECONNECTIONS", "Niño3.4 · PRECT", "Correlation Map"),
        ("fig_teleconnection_nino34_prect_corr_map.png", "TELECONNECTIONS", "Niño3.4 · PRECT", "Correlation Map"),
        ("fig_4b_teleconnection_pdo_prect_corr_map.png", "TELECONNECTIONS", "PDO · PRECT", "Correlation Map"),
        ("fig_teleconnection_pdo_prect_corr_map.png", "TELECONNECTIONS", "PDO · PRECT", "Correlation Map"),
        ("fig_5b_eli_drift_climatology.png", "ELI", "ELI Diagnostics", "Leadtime Climatology"),
        ("fig_eli_drift_climatology.png", "ELI", "ELI Diagnostics", "Leadtime Climatology"),
        ("fig_5c_teleconnection_eli_prect_corr_map.png", "TELECONNECTIONS", "ELI · PRECT", "Correlation Map"),
        ("fig_teleconnection_eli_prect_corr_map.png", "TELECONNECTIONS", "ELI · PRECT", "Correlation Map"),
        ("fig_6a_prect_absolute_normalized_change_seasonal.png", "INITIAL_SHOCK", "PRECT", "Seasonal Abs Change"),
        ("fig_6b_shock_metrics_heatmap_trefht_lead-year-1.png", "INITIAL_SHOCK", "TREFHT", "Lead Y1 Heatmap"),
        ("fig_shock_error_trefht_lead-year-1.png", "INITIAL_SHOCK", "TREFHT", "Lead Y1 Heatmap"),
        ("fig_shock_error_ts_lead-year-2.png", "INITIAL_SHOCK", "TS", "Lead Y2 Heatmap"),
        ("fig_shock_error_ts_monthly.png", "INITIAL_SHOCK", "TS", "Monthly Heatmap"),
        ("fig_7a_tc_tracks_density_sanity_compare.png", "TC", "Tropical Cyclones", "Sanity Compare"),
        ("fig_tc_tracks_density_sanity_compare.png", "TC", "Tropical Cyclones", "Sanity Compare"),
        ("fig_7a_tc_genesis_density_method_compare.png", "TC", "Tropical Cyclones", "Method Compare"),
        ("fig_tc_genesis_density_method_compare.png", "TC", "Tropical Cyclones", "Method Compare"),
        ("fig_7a_tc_trajectory_compare_set3.png", "TC", "Tropical Cyclones", "Trajectory Compare"),
        ("fig_tc_trajectory_compare_set3.png", "TC", "Tropical Cyclones", "Trajectory Compare"),
        ("fig_7b_tc_leadtime_track_density_compare.png", "TC", "Tropical Cyclones", "Leadtime Compare"),
        ("fig_tc_leadtime_track_density_compare.png", "TC", "Tropical Cyclones", "Leadtime Compare"),
        ("fig_7b_tc_track_density_enso_regression_fosirl_reanalysis.png", "TC", "Tropical Cyclones", "Regression Map"),
        ("fig_tc_track_density_enso_regression_fosirl_reanalysis.png", "TC", "Tropical Cyclones", "Regression Map"),
    ]
    for fn, expected_grp, expected_shortname, expected_btn_type in test_files:
        (tmp_path / fn).touch()

    manifest = discover_workflow_figures(tmp_path)
    by_file = {fig["file"]: fig for fig in manifest["figures"]}

    for fn, expected_grp, expected_shortname, expected_btn_type in test_files:
        fig_entry = by_file[fn]
        assert fig_entry["group"] == expected_grp, f"{fn} group mismatch"
        assert fig_entry["shortname"] == expected_shortname, f"{fn} shortname mismatch"
        assert fig_entry["btn_type"] == expected_btn_type, f"{fn} btn_type mismatch"

