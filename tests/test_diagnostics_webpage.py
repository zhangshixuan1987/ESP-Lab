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

