"""Ensure TempestExtremes inputs without executing the tracking runner notebook."""
from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path


WARM_CORE = {
    "set1": {"wc1": "Z200", "wc2": "Z500", "wc_mag": -16.0},
    "set2": {"wc1": "T200", "wc2": "T500", "wc_mag": -1.0},
    "set3": {"wc1": "Z300", "wc2": "Z500", "wc_mag": -6.0},
    "set4": {"wc1": "T300", "wc2": "T500", "wc_mag": -0.6},
    "set5": {"wc1": "T400", "wc2": "", "wc_mag": -0.4},
}


def track_paths(root, case, member, parset):
    base = Path(root) / case / member / "post" / "atm" / "tc-analysis"
    stem = f"{case}_{member}_{parset}_TCS"
    return base / f"{stem}_track.txt", base / f"{stem}_hist.nc"


def ensure_tracks(*, repo_root, track_root, cases, members, parsets, settings,
                  mode="auto", runner=None):
    """Reuse complete legacy pairs, or build missing/stale pairs with the CLI.

    Newly built pairs get a settings receipt. Legacy runner products have no
    receipt and are reported separately: their thresholds cannot be verified.
    Raw simulation files and executables are needed only for processing.
    """
    if mode not in {"auto", "require", "rebuild"}:
        raise ValueError("TC input mode must be auto, require, or rebuild")
    cases, members, parsets = list(cases), list(members), list(parsets)
    if not cases or not members or not parsets:
        raise ValueError("TC cases, members, and parsets must be nonempty")
    if set(parsets) - WARM_CORE.keys():
        raise ValueError(f"Unknown tracking methods: {set(parsets) - WARM_CORE.keys()}")
    runner = runner or subprocess.run
    report = {"reused": 0, "legacy_reused": 0, "built": 0}
    pending = []
    for parset in parsets:
        options = dict(settings)
        overrides = options.pop("warm_core_overrides", {})
        options.update(WARM_CORE[parset])
        options.update(overrides.get(parset, {}))
        identity = json.loads(json.dumps(options, default=str))
        for case in cases:
            for member in members:
                track, hist = track_paths(track_root, case, member, parset)
                receipt = track.with_suffix(".inputs.json")
                complete = track.is_file() and hist.is_file() and hist.stat().st_size > 0
                # An empty track can be a valid zero-storm result.
                compatible = True
                if receipt.exists():
                    try:
                        compatible = json.loads(receipt.read_text()) == identity
                    except (ValueError, OSError):
                        compatible = False
                if mode != "rebuild" and complete and compatible:
                    report["reused" if receipt.exists() else "legacy_reused"] += 1
                else:
                    pending.append((case, member, parset, options, identity, receipt))
    if mode == "require" and pending:
        preview = ", ".join(f"{c}/{m}/{p}" for c, m, p, *_ in pending[:5])
        raise FileNotFoundError(f"{len(pending)} missing or stale TC input pairs: {preview}")
    script = Path(repo_root) / "scripts" / "run_process_tc_track_e3sm.py"
    if pending and not script.is_file():
        raise FileNotFoundError(script)
    groups = {}
    for item in pending:
        groups.setdefault((item[0], item[2]), []).append(item)
    for (case, parset), items in groups.items():
        options = items[0][3]
        selected_members = [item[1] for item in items]
        cmd = [sys.executable, str(script), "--outdir", str(track_root),
               "--cases", case, "--members", *selected_members,
               "--parset", parset, "--force"]
        for key, value in options.items():
            cmd.extend(["--" + key.replace("_", "-"), str(value)])
        print(f"[TC inputs] Processing {case}/{parset}: {len(items)} members", flush=True)
        before = {}
        for _, member, *_ in items:
            for path in track_paths(track_root, case, member, parset):
                before[path] = (path.stat().st_mtime_ns, path.stat().st_size) if path.exists() else None
        runner(cmd, check=True, cwd=str(repo_root))
        for _, member, _, _, identity, receipt in items:
            track, hist = track_paths(track_root, case, member, parset)
            if not track.is_file() or not hist.is_file() or hist.stat().st_size == 0:
                raise RuntimeError(f"Tracking did not produce a complete output pair: {track}")
            if all(before[path] == (path.stat().st_mtime_ns, path.stat().st_size)
                   for path in (track, hist)):
                raise RuntimeError(f"Tracking left stale outputs unchanged: {track}")
            temporary = receipt.with_name(f".{receipt.name}.{uuid.uuid4().hex}.tmp")
            try:
                temporary.write_text(json.dumps(identity, sort_keys=True, indent=2) + "\n")
                temporary.replace(receipt)
            finally:
                temporary.unlink(missing_ok=True)
            report["built"] += 1
    if report["legacy_reused"]:
        print(f"[TC inputs] Reused {report['legacy_reused']} legacy pairs without settings receipts; "
              "use rebuild to enforce the configured tracking thresholds.")
    return report
