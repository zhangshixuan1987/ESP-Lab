"""Tests for centralized modes-of-variability figure configuration."""

import numpy as np

from workflows.modes_of_variability import figure_config


EXPECTED_MODES = {
    "NAM", "NAO", "SAM", "PNA", "NPO", "PDO", "NPGO", "EA", "SCA",
    "AMO", "PSA1", "PSA2",
}


def test_every_supported_plot_mode_has_one_profile():
    assert set(figure_config.MODE_FIGURE_PROFILES) == EXPECTED_MODES


def test_resolved_settings_do_not_mutate_shared_presets():
    first = figure_config.get_eof_pattern_settings("PNA")
    first["preferred_init_by_season"]["DJF"] = 5
    first["levels"][0] = 999

    second = figure_config.get_eof_pattern_settings("PNA")
    assert second["preferred_init_by_season"]["DJF"] == 11
    assert second["levels"][0] == -3.0


def test_pressure_and_temperature_profiles_use_expected_contours():
    np.testing.assert_allclose(
        figure_config.get_eof_pattern_settings("PNA")["levels"],
        np.arange(-3.0, 3.01, 0.5),
    )
    np.testing.assert_allclose(
        figure_config.get_eof_pattern_settings("PDO")["levels"],
        np.arange(-1.0, 1.01, 0.2),
    )


def test_global_teleconnection_overrides_are_shared():
    for mode in EXPECTED_MODES:
        settings = figure_config.get_teleconnection_settings(mode)
        assert settings["figsize_width"] == 16.0
        assert settings["longitude_ticks"].tolist() == [-180, -120, -60, 0, 60, 120, 180]


def test_atlantic_eof_layouts_are_wide_and_npo_profile_stays_unchanged():
    npo = figure_config.get_eof_pattern_settings("NPO")

    for mode in ("NAO", "EA", "SCA"):
        settings = figure_config.get_eof_pattern_settings(mode)
        assert settings["figsize_width"] == 15.5
        assert settings["figsize_row_height"] == 2.0
        assert settings["subplot_wspace"] == 0.08
        assert settings["subplot_hspace"] == 0.18

    amo = figure_config.get_eof_pattern_settings("AMO")
    assert amo["figsize_width"] == 11.0
    assert amo["figsize_row_height"] == 2.0
    assert amo["subplot_wspace"] == 0.08
    assert amo["subplot_hspace"] == 0.18
    assert amo["column_header_scale"] == 0.60
    assert amo["transform_first"] is True
    assert "transform_first" not in figure_config.get_eof_pattern_settings("NAO")
    assert figure_config.get_teleconnection_settings("AMO")["transform_first"] is True
    assert "transform_first" not in figure_config.get_teleconnection_settings("NAO")

    assert npo["figsize_width"] == 15.5
    assert npo["figsize_row_height"] == 2.0
    assert npo["subplot_wspace"] == 0.08
    assert npo["subplot_hspace"] == 0.22
    assert npo["panel_title_two_lines"] is False


def test_pdo_eof_layout_is_wide_and_compact():
    settings = figure_config.get_eof_pattern_settings("PDO")

    assert settings["figsize_width"] == 15.5
    assert settings["figsize_row_height"] == 2.0
    assert settings["subplot_wspace"] == 0.08
    assert settings["subplot_hspace"] == 0.22
    assert settings["panel_title_scale"] == 0.44
    assert settings["column_header_scale"] == 0.60


def test_psa1_and_psa2_restore_per_panel_coordinate_labels():
    psa1 = figure_config.get_eof_pattern_settings("PSA1")
    psa2 = figure_config.get_eof_pattern_settings("PSA2")
    sam = figure_config.get_eof_pattern_settings("SAM")

    for mode in ("PSA1", "PSA2"):
        overrides = figure_config.MODE_FIGURE_PROFILES[mode]["eof_overrides"]
        assert overrides == {
            "polar_longitude_labels_each_panel": True,
            "polar_latitude_labels_each_panel": True,
        }

    for settings in (psa1, psa2):
        assert "polar_column_width" not in settings
        assert "polar_colorbar_width" not in settings
        assert settings["figsize_row_height"] == 3.8
        assert settings["subplot_top"] == 0.88
        assert settings["subplot_wspace"] == 0.50
        assert settings["subplot_hspace"] == 0.15
        assert settings["polar_metric_box_x"] == 0.90
        assert settings["polar_metric_box_y"] == 0.020
        assert settings["map_label_scale"] == 0.70
        assert settings["polar_longitude_label_radius"] == 0.625
        assert settings["polar_longitude_labels_each_panel"] is True
        assert settings["polar_latitude_labels_each_panel"] is True
        assert settings["column_header_y_shift"] == 0.16

    assert sam["subplot_wspace"] == 0.70
    assert sam["subplot_hspace"] == 0.22
    assert sam["polar_longitude_labels_each_panel"] is True
    assert sam["polar_latitude_labels_each_panel"] is True


def test_npgo_eof_layout_matches_compact_north_pacific_sst_geometry():
    settings = figure_config.get_eof_pattern_settings("NPGO")

    assert settings["figsize_width"] == 15.5
    assert settings["figsize_row_height"] == 2.0
    assert settings["subplot_wspace"] == 0.08
    assert settings["subplot_hspace"] == 0.22


def test_complete_setup_preserves_public_notebook_sections():
    setup = figure_config.build_figure_setup("PNA", include_nmme=True)
    assert set(setup) == {"common", "eof_patterns", "teleconnections", "skill", "pc_time_series"}
    assert setup["pc_time_series"]["annotation_band_height"] == 4.2
    assert figure_config.metric_specs("PNA")[1][2] == (0.6, 1.3)
