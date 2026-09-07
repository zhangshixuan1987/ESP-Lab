import pytest


def test_xcdat_importable():
    xcdat = pytest.importorskip("xcdat", reason="xcdat not installed")
    assert xcdat is not None
