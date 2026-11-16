#!/usr/bin/env python
"""Fair-comparison diagnostics centered on E3SMv3-4DEnVarOcn for several start-year periods.

For every period (May starts, inclusive years) this builds copies of the three
fair-comparison notebooks in this folder and executes them:

* fair_4DEnVarLnd_comparison.ipynb  -> 1a-style ACC maps, one run per field;
* fair_4DEnVarLnd_sst_skill.ipynb   -> 3a-style Nino3.4 skill and time series;
* fair_4DEnVarLnd_scorecard.ipynb   -> area-mean/SST-index skill and the scorecard,
                                       with E3SMv3-4DEnVarOcn as the focus experiment.

Each copy sets the period (start years and climatology), drops experiments that
do not cover every start year of the period (4DEnVarLnd ends in 1989), and
writes its figures to <figure root>/fair_4DEnVarOcn_y<start>-<end>/. Executed
notebooks and logs go to <test diag root>/logs/fair_4DEnVarOcn/y<start>-<end>/.
The notebooks in this folder are not modified.

Run on a compute node, e.g. an interactive allocation:

    salloc -A gen0010 -q interactive -C cpu -N 1 -t 04:00:00 \\
        srun -n 1 -c 256 python run_fair_4DEnVarOcn.py

Options: --periods 1980-1984 1980-1989 1980-1995 (default), --fields ...,
--max-parallel N (notebooks executed at once, default 6), --prepare-only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
PYTHON_BIN = Path(sys.executable).parent
FIGURE_BASE = Path("/global/cfs/cdirs/e3sm/www/zhan391/e3sm-s2d_test")
DIAG_ROOT = Path("/global/cfs/cdirs/e3sm/S2S2D/s2d_diag/tmp/e3sm-s2d_test")
RUN_BASE = DIAG_ROOT / "logs" / "fair_4DEnVarOcn"

FOCUS_CASE = "E3SM-4DEnVarOcn"
DEFAULT_PERIODS = ["1980-1984", "1980-1989", "1980-1995"]
FIELDS = ["PRECT", "TS", "TREFHT", "PSL"]
# Last May start year of the experiments that do not cover the full record.
LAST_START_YEAR = {"E3SM-4DEnVarLnd": 1989, "E3SM-4DEnVarOcn": 1995, "E3SM-4DEnVarOcn-OLD": 2011}

SOURCE_FIGURE_ROOT = '"ESP_LAB_FIGURE_ROOT"] = "/global/cfs/cdirs/e3sm/www/zhan391/e3sm-s2d_test/sample_figures"'
SCORECARD_FIGURE_ROOT = '"ESP_LAB_FIGURE_ROOT"] = "/global/cfs/cdirs/e3sm/www/zhan391/e3sm-s2d_test"'
LND_BLOCKS = {
    "comparison": (
        '    "E3SM-4DEnVarLnd": {\n'
        '        "case_prefix": "WCYCL20TR_ne30pg2_r05_IcoswISC30E3r5_4DEnVarLnd",\n'
        '        "cache_tag": "4DEnVarLnd",\n'
        '        "display_name": "E3SMv3-4DEnVarLnd",\n'
        '        "source_revision": "post_process_v1",\n'
        '        # May starts exist only for 1980-1989.\n'
        '        "init_years_by_month": {5: range(1980, 1990)},  # all 10 starts used here\n'
        '    },\n'
    ),
    "sst_skill": (
        '    "E3SM-4DEnVarLnd": {\n'
        '        "case_prefix": "WCYCL20TR_ne30pg2_r05_IcoswISC30E3r5_4DEnVarLnd",\n'
        '        "cache_tag":   "4DEnVarLnd",\n'
        '        "display_name": "E3SMv3-4DEnVarLnd",\n'
        '        "color":       "tab:green",\n'
        '        "linestyle":   "-",\n'
        '        # May starts exist only for 1980-1989.\n'
        '        "init_years_by_month": {5: range(1980, 1990)},\n'
        '    },\n'
    ),
    "scorecard": (
        '    "E3SM-4DEnVarLnd": dict(case_prefix="WCYCL20TR_ne30pg2_r05_IcoswISC30E3r5_4DEnVarLnd",\n'
        '                            cache_tag="4DEnVarLnd", label="E3SMv3-4DEnVarLnd", color="tab:green", ls="-", marker="o"),\n'
    ),
}


def load(name: str) -> dict:
    nb = json.loads((HERE / f"{name}.ipynb").read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"], cell["execution_count"] = [], None
    return nb


def patch(nb: dict, replacements: list[tuple[str, str]], label: str) -> None:
    """Apply exact text replacements; each must match exactly once in the notebook."""
    for old, new in replacements:
        hits = 0
        for cell in nb["cells"]:
            source = "".join(cell["source"])
            if old in source:
                hits += source.count(old)
                cell["source"] = source.replace(old, new).splitlines(keepends=True)
        if hits != 1:
            raise RuntimeError(
                f"{label}: expected one match, found {hits}, for {old[:80]!r}. "
                "The source notebook changed; update run_fair_4DEnVarOcn.py."
            )


def period_edits(start: int, end: int) -> list[tuple[str, str]]:
    return [
        ("YEAR_END = 1989   # common 4DEnVarLnd period (production: 2018)",
         f"YEAR_END = {end}   # fair_4DEnVarOcn period {start}-{end} (production: 2018)"),
        ("climatology_years = (1980, 1989)   # production: (1981, 2010)",
         f"climatology_years = ({start}, {end})   # production: (1981, 2010)"),
    ]


def build_period(start: int, end: int, fields: list[str]) -> tuple[Path, list[Path], list[Path]]:
    """Write the period's notebooks; return (run dir, parallel notebooks, sequential chain)."""
    if start != 1980:
        raise ValueError("The source notebooks start in 1980; periods must start in 1980.")
    tag = f"y{start}-{end}"
    run_dir = RUN_BASE / tag
    figure_dir = FIGURE_BASE / f"fair_4DEnVarOcn_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    dropped = [key for key, last in LAST_START_YEAR.items() if last < end]
    figure_root = (SOURCE_FIGURE_ROOT.split(" = ")[0] + f' = "{figure_dir}"')

    def drop(kind: str) -> list[tuple[str, str]]:
        return [(LND_BLOCKS[kind], "")] if "E3SM-4DEnVarLnd" in dropped else []

    comparisons = []
    for field in fields:
        nb = load("fair_4DEnVarLnd_comparison")
        edits = [(SOURCE_FIGURE_ROOT, figure_root), *period_edits(start, end), *drop("comparison")]
        if field != "PRECT":
            edits += [('field = "PRECT"\n', '#field = "PRECT"\n'), (f'#field = "{field}"\n', f'field = "{field}"\n')]
        patch(nb, edits, f"comparison {field} {tag}")
        path = run_dir / f"fair_4DEnVarOcn_comparison_{field}.ipynb"
        path.write_text(json.dumps(nb, indent=1))
        comparisons.append(path)

    nb = load("fair_4DEnVarLnd_sst_skill")
    patch(nb, [
        (SOURCE_FIGURE_ROOT, figure_root), *period_edits(start, end), *drop("sst_skill"),
        ("YEARS = (1980, 1989)   # production: (1980, 2018)", f"YEARS = ({start}, {end})   # production: (1980, 2018)"),
        ('"plot_xmax": 1992.5,   # cohort 1980-1989 (+ two lead years)',
         f'"plot_xmax": {end + 3.5},   # cohort {start}-{end} (+ two lead years)'),
        ('"major_years": [1980, 1982, 1984, 1986, 1988, 1990, 1992],',
         f'"major_years": {list(range(start, end + 4, 2))},'),
        ('"minor_years": list(np.arange(1980, 1993, 1)),', f'"minor_years": list(np.arange({start}, {end + 4}, 1)),'),
        # Copies run outside this folder; point the repository search at the checkout.
        ("_repo_candidates = [Path.cwd().resolve(),", f'_repo_candidates = [Path("{REPO_ROOT}"), Path.cwd().resolve(),'),
    ], f"sst_skill {tag}")
    sst = run_dir / "fair_4DEnVarOcn_sst_skill.ipynb"
    sst.write_text(json.dumps(nb, indent=1))

    nb = load("fair_4DEnVarLnd_scorecard")
    patch(nb, [
        (SCORECARD_FIGURE_ROOT, figure_root),
        ('FIGURE_DIR = env_paths.figure_root() / "sample_figures"', "FIGURE_DIR = env_paths.figure_root()"),
        ("YEARS = (1980, 1989)                 # common initialization cohort (4DEnVarLnd coverage)",
         f"YEARS = ({start}, {end})                 # common initialization cohort"),
        ('FOCUS_CASE = "E3SM-4DEnVarLnd"', f'FOCUS_CASE = "{FOCUS_CASE}"'),
        ('("ΔACC (4DEnVarLnd − other)", "−ΔnRMSE (other − 4DEnVarLnd)")',
         '(f"ΔACC ({FOCUS_TAG} − other)", f"−ΔnRMSE (other − {FOCUS_TAG})")'),
        ('label="lead-mean difference\\n(red = 4DEnVarLnd better)")',
         'label=f"lead-mean difference\\n(red = {FOCUS_TAG} better)")'),
        *drop("scorecard"),
    ], f"scorecard {tag}")
    card = run_dir / "fair_4DEnVarOcn_scorecard.ipynb"
    card.write_text(json.dumps(nb, indent=1))

    print(f"[{tag}] notebooks in {run_dir}; figures -> {figure_dir}; "
          f"dropped (incomplete coverage): {dropped or 'none'}", flush=True)
    return run_dir, comparisons, [sst, card]


def execute(notebook: Path) -> tuple[Path, int, float]:
    started = time.time()
    log = notebook.with_suffix(".log")
    with log.open("w") as handle:
        rc = subprocess.run(
            [str(PYTHON_BIN / "jupyter"), "nbconvert", "--to", "notebook", "--execute", "--inplace",
             "--ExecutePreprocessor.timeout=-1", "--ExecutePreprocessor.kernel_name=e3sm_analysis", str(notebook)],
            cwd=notebook.parent, stdout=handle, stderr=subprocess.STDOUT,
        ).returncode
    minutes = (time.time() - started) / 60
    print(f"  {'ok  ' if rc == 0 else 'FAIL'} {notebook.parent.name}/{notebook.name} ({minutes:.1f} min; log {log.name})",
          flush=True)
    return notebook, rc, minutes


def run_chain(chain: list[Path]) -> list[tuple[Path, int, float]]:
    """Run notebooks in order, stopping at the first failure (the scorecard reuses SST-index inputs)."""
    results = []
    for notebook in chain:
        results.append(execute(notebook))
        if results[-1][1] != 0:
            break
    return results


def notebook_errors(notebook: Path) -> list[str]:
    nb = json.loads(notebook.read_text())
    return [f"{out.get('ename')}: {out.get('evalue', '')[:200]}"
            for cell in nb["cells"] if cell["cell_type"] == "code"
            for out in cell.get("outputs", []) if out.get("output_type") == "error"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--periods", nargs="+", default=DEFAULT_PERIODS, help="inclusive May start-year spans, e.g. 1980-1989")
    parser.add_argument("--fields", nargs="+", default=FIELDS, choices=FIELDS)
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--prepare-only", action="store_true", help="write the notebooks without executing them")
    args = parser.parse_args()

    jobs = []
    for period in args.periods:
        start, end = map(int, period.split("-"))
        jobs.append(build_period(start, end, args.fields))
    if args.prepare_only:
        return 0

    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futures = [pool.submit(run_chain, chain) for _, _, chain in jobs]
        futures += [pool.submit(execute, nb) for _, comparisons, _ in jobs for nb in comparisons]
        results = []
        for future in futures:
            outcome = future.result()
            results.extend(outcome if isinstance(outcome, list) else [outcome])

    failed = [(nb, rc) for nb, rc, _ in results if rc != 0 or notebook_errors(nb)]
    expected = sum(len(c) + len(chain) for _, c, chain in jobs)
    print(f"\n{len(results) - len(failed)}/{expected} notebooks completed without errors.")
    for nb, rc in failed:
        print(f"  FAILED {nb} (exit {rc}): {notebook_errors(nb)[:1]}")
    for period in args.periods:
        figure_dir = FIGURE_BASE / f"fair_4DEnVarOcn_y{period}"
        print(f"  {figure_dir}: {len(list(figure_dir.glob('*.png')))} figures")
    return 1 if failed or len(results) != expected else 0


if __name__ == "__main__":
    sys.exit(main())
