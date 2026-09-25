"""Guard the s2d_diag naming convention across every name builder."""

import json
import re
from pathlib import Path

import pytest

from esp_lab.utils.filename_utils import source_init_prefix

REPO = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("esp_lab", "workflows", "scripts", "jupyter")

# A source (braced expression or literal) followed directly by a two-digit
# month, optionally with a bare underscore: ``{tag}{init_month:02d}_`` or
# ``CESM-SMYLE_{m:02d}_``.  The month must be written ``_init{MM}_`` instead.
GLUED_MONTH = re.compile(
    r"(\{\{?[A-Za-z_\[\]'\"]+\}?\}"
    r"|CESM-SMYLE|NMME|JRA55_FOSIRL|Reanalysis|4DEnVarOcn|E3SM|E3SMLE)_?"
    r"\{\{?(?:int\()?[a-z_.]*(?:month|mm|\bm)\)?:02d\}\}?_"
)


def _code_sources():
    for folder in SOURCE_DIRS:
        for path in sorted((REPO / folder).rglob("*")):
            if ".ipynb_checkpoints" in path.parts:
                continue
            if path.suffix == ".py":
                yield path, path.read_text()
            elif path.suffix == ".ipynb":
                cells = json.loads(path.read_text())["cells"]
                yield path, "\n".join(
                    "".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"
                )


def test_source_init_prefix_keeps_folder_name_and_separate_month():
    assert source_init_prefix("CESM-SMYLE", 5) == "CESM-SMYLE_init05"
    assert source_init_prefix("JRA55_FOSIRL", "11") == "JRA55_FOSIRL_init11"


@pytest.mark.parametrize("source, month", [("", 5), ("a/b", 5), ("NMME", 0), ("NMME", 13)])
def test_source_init_prefix_rejects_bad_input(source, month):
    with pytest.raises(ValueError):
        source_init_prefix(source, month)


def test_no_name_builder_glues_the_month_to_the_source():
    offenders = [
        f"{path.relative_to(REPO)}: {match.group(0)}"
        for path, text in _code_sources()
        for match in GLUED_MONTH.finditer(text)
        if "_init{" not in match.group(0)
    ]
    assert not offenders, "use source_init_prefix(): " + "; ".join(offenders)


def test_sst_index_filename_carries_the_processed_year_span():
    from esp_lab.utils.filename_utils import sst_index_filename

    assert (
        sst_index_filename("JRA55_FOSIRL", 5, (1980, 2018), 10, 24, native=True, seasonal=True)
        == "JRA55_FOSIRL_init05_ELI_native_y1980-2018_N10_M24_seas.nc"
    )
    assert (
        sst_index_filename("CESM-SMYLE", 11, (1980, 2018), 20, 24, index="Nino3.4", field="TS")
        == "CESM-SMYLE_init11_TS_y1980-2018_N20_M24_Nino3.4SST_mon.nc"
    )


def test_resolve_year_span_path_prefers_exact_then_narrowest_cover(tmp_path):
    from esp_lab.utils.filename_utils import resolve_year_span_path

    def name(span):
        return tmp_path / f"X_init05_ELI_native_{span}_N10_M24.nc"

    request = name("y1980-2011")
    # Nothing covers the request: the requested path comes back for building.
    name("y1990-2018").touch()
    assert resolve_year_span_path(request) == request
    # The narrowest covering span wins; a sibling product (seasonal) never matches.
    name("y1980-2018").touch()
    name("y1970-2020").touch()
    (tmp_path / "X_init05_ELI_native_y1980-2011_N10_M24_seas.nc").touch()
    assert resolve_year_span_path(request) == name("y1980-2018")
    # An exact file is always preferred.
    request.touch()
    assert resolve_year_span_path(request) == request
