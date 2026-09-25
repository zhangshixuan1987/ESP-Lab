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


def test_figure_filename_rejects_names_over_max_length():
    assert figure_filename("a" * 72, ext="png", max_length=80) == f"fig_{'a' * 72}.png"
    with pytest.raises(ValueError, match="max_length=80"):
        figure_filename("a" * 73, ext="png", max_length=80)


@pytest.mark.parametrize("max_length", [True, 1.5, 10, 256])
def test_figure_filename_validates_max_length(max_length):
    error = TypeError if isinstance(max_length, (bool, float)) else ValueError
    with pytest.raises(error):
        figure_filename("skill", max_length=max_length)
