import pytest

from esp_lab.utils.filename_utils import figure_filename, safe_token


def test_figure_filename_preserves_existing_readable_convention():
    assert figure_filename("PRECT", "ACC skill map") == "fig_prect_acc_skill_map.png"


def test_figure_filename_normalizes_unicode_and_ignores_empty_parts():
    assert safe_token("  Café / Niño  ") == "cafe_nino"
    assert figure_filename(None, "", "Café / Niño", ext=".SVG") == "fig_cafe_nino.svg"


@pytest.mark.parametrize("ext", ["", ".", "png/../../nc", "tar.gz", None])
def test_figure_filename_rejects_unsafe_extensions(ext):
    error = TypeError if ext is None else ValueError
    with pytest.raises(error):
        figure_filename("skill", ext=ext)


def test_figure_filename_requires_a_usable_part():
    with pytest.raises(ValueError, match="at least one"):
        figure_filename(None, "", "雪")


def test_figure_filename_bounds_long_names_deterministically():
    first = figure_filename("a" * 400, ext="png", max_length=80)
    second = figure_filename("a" * 400, ext="png", max_length=80)
    different = figure_filename("a" * 399 + "b", ext="png", max_length=80)

    assert first == second
    assert first != different
    assert len(first) == 80
    assert first.endswith(".png")


@pytest.mark.parametrize("max_length", [True, 1.5, 10, 256])
def test_figure_filename_validates_max_length(max_length):
    error = TypeError if isinstance(max_length, (bool, float)) else ValueError
    with pytest.raises(error):
        figure_filename("skill", max_length=max_length)
