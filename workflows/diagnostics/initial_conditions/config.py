"""
config.py
=========
Configuration helper shared by the initial-condition workflow modules.

Provides ``load_config`` and ``build_ic_config`` so that config loading
logic is not duplicated across 01–06.

It lives with the workflow because it handles workflow-specific YAML.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Optional

import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from esp_lab.diagnostics.ic_core import (
    ComponentSpec,
    DEFAULT_COMPONENTS,
    ExperimentPair,
    ICConfig,
)


def _expand_env(value: str) -> str:
    """Expand ``${VAR:default}`` env-var references in a string."""
    def _replace(m):
        var, _, default = m.group(1).partition(":")
        return os.environ.get(var, default)
    return re.sub(r"\$\{([^}]+)\}", _replace, str(value))


def load_config(config_path: Path) -> dict:
    """Load and post-process config.yaml, expanding env-var references."""
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    cfg["experiments"]["ref_root"] = _expand_env(cfg["experiments"]["ref_root"])
    cfg["experiments"]["test_root"] = _expand_env(cfg["experiments"]["test_root"])
    output = cfg.setdefault("output", {})
    if "output_root" in output:
        output["output_root"] = _expand_env(output["output_root"])
    if "figure_outdir" in output:
        output["figure_outdir"] = _expand_env(output["figure_outdir"])
    return cfg


def build_ic_config(cfg: dict, pilot_only: Optional[bool] = None) -> ICConfig:
    """Construct an ICConfig from the parsed config dict."""
    exp = cfg["experiments"]
    pilot = cfg.get("pilot", {})
    hashing = cfg.get("hashing", {})
    output = cfg.get("output", {})

    all_dates = sorted(
        set(cfg.get("may_start_dates", []) + cfg.get("nov_start_dates", []))
    )

    comp_cfgs = cfg.get("components", {})
    components = [
        ComponentSpec(
            name=name,
            file_glob=ccfg["file_glob"],
            per_member=ccfg.get("per_member", False),
            member_pattern=ccfg.get("member_pattern", r"EN\d{2}"),
        )
        for name, ccfg in comp_cfgs.items()
    ] if comp_cfgs else DEFAULT_COMPONENTS

    use_pilot = pilot_only if pilot_only is not None else pilot.get("pilot_only", True)

    return ICConfig(
        experiment_pair=ExperimentPair(
            ref_label=exp["ref_label"],
            test_label=exp["test_label"],
            ref_root=exp["ref_root"],
            test_root=exp["test_root"],
        ),
        start_dates=all_dates,
        seasons=cfg.get("seasons", ["May", "November"]),
        members=cfg.get("members", []),
        components=components,
        pilot_date=pilot.get("pilot_date", "1980-05-01-00000"),
        pilot_only=use_pilot,
        output_root=output.get("output_root", "output"),
        atm_hash_all_members=hashing.get("atm_hash_all_members", True),
        non_atm_hash_per_member=hashing.get("non_atm_hash_per_member", False),
    )
