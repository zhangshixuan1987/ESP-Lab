"""Function-level cache fingerprints follow called code, not whole files."""

import importlib
import sys
import textwrap

import pytest

from esp_lab.utils.code_identity import code_digest


PROJECT_NAME = "esp_lab._codeid_test.mod"


@pytest.fixture
def digest_of(tmp_path, monkeypatch):
    """Return ``digest_of(source)``: the digest of ``compute`` in a module version.

    Each version is imported from its own directory but registered under one
    project module name, as one module would be before and after an edit.
    """
    monkeypatch.syspath_prepend(str(tmp_path))
    counter = {"n": 0}

    def digest(source):
        counter["n"] += 1
        package = f"codeid_pkg{counter['n']}"
        (tmp_path / package).mkdir()
        (tmp_path / package / "__init__.py").write_text("")
        (tmp_path / package / "mod.py").write_text(textwrap.dedent(source))
        importlib.invalidate_caches()
        module = importlib.import_module(f"{package}.mod")
        module.__name__ = PROJECT_NAME
        for value in vars(module).values():
            if callable(value) and getattr(value, "__module__", "") == f"{package}.mod":
                value.__module__ = PROJECT_NAME
        sys.modules[PROJECT_NAME] = module
        return code_digest([module.compute])

    yield digest
    for key in [k for k in sys.modules if k.startswith(("esp_lab._codeid_test", "codeid_pkg"))]:
        sys.modules.pop(key)


BASE = """
SCALE = 2

def helper(x):
    return x * SCALE

def compute(x):
    '''Docstring.'''
    return helper(x) + 1

def unrelated():
    return "not called"
"""


def test_digest_ignores_comments_docstrings_and_unrelated_code(digest_of):
    assert digest_of(BASE) == digest_of(
        BASE.replace("'''Docstring.'''", "'''A different docstring.'''")
        .replace("return helper(x) + 1", "return helper(x) + 1  # comment")
        .replace('return "not called"', 'return "changed, but never called"')
        + "\ndef new_helper():\n    return 3\n"
    )


@pytest.mark.parametrize(
    "change",
    [
        ("return x * SCALE", "return x * SCALE * 3"),  # a called helper
        ("SCALE = 2", "SCALE = 5"),  # a module constant it reads
        ("return helper(x) + 1", "return helper(x) + 2"),  # the function itself
    ],
)
def test_digest_changes_when_reached_code_changes(digest_of, change):
    assert digest_of(BASE) != digest_of(BASE.replace(*change))
