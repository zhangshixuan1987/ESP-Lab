import json
import re
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Union


FIGURE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".webp"}
CLIMATE_MODES = {
    "AMO",
    "EA",
    "NAM",
    "NAO",
    "NPGO",
    "NPO",
    "PDO",
    "PNA",
    "PSA1",
    "PSA2",
    "SAM",
    "SCA",
}


def _humanize_figure_name(filename: str) -> str:
    """Return a readable title for a workflow figure filename."""
    stem = Path(filename).stem
    if stem.lower().startswith("fig_"):
        stem = stem[4:]
    replacements = {
        "acc": "ACC",
        "eli": "ELI",
        "enso": "ENSO",
        "eof": "EOF",
        "iod": "IOD",
        "jja": "JJA",
        "djf": "DJF",
        "nmme": "NMME",
        "nrmse": "nRMSE",
        "prect": "PRECT",
        "psl": "PSL",
        "rmse": "RMSE",
        "sst": "SST",
        "tc": "TC",
        "trefht": "TREFHT",
    }
    words = []
    for token in re.split(r"[_\-]+", stem):
        lower = token.lower()
        if lower in {"nino34", "nino3.4"}:
            words.append("Niño3.4")
        elif lower == "nino3" and words and words[-1] == "Niño":
            words[-1] = "Niño3"
        elif lower == "4" and words and words[-1] == "Niño3":
            words[-1] = "Niño3.4"
        elif lower.startswith("nino") and lower[4:].isdigit():
            words.append(f"Niño{lower[4:]}")
        else:
            words.append(replacements.get(lower, token.capitalize()))
    return " ".join(words)


def _infer_metric(filename: str) -> str:
    """Infer a compact metric key for a figure without manifest metadata."""
    stem = Path(filename).stem.lower()
    candidates = (
        "leadtime_drift",
        "global_teleconnection_patterns",
        "multi_e3sm_rmse_skill_map_conus",
        "multi_e3sm_rmse_skill_diff",
        "multi_e3sm_rmse_skill_map",
        "multi_e3sm_acc_skill_map",
        "rmse_diff_compare",
        "rmse_compare_global",
        "rmse_compare_conus",
        "time_series",
        "eof_patterns",
        "acc_skill",
        "lead_time_benchmark",
        "drift_climatology",
        "skill",
    )
    return next((metric for metric in candidates if metric in stem), "workflow_figure")


def _infer_mode(filename: str) -> str:
    """Infer a broad diagnostic group for a figure without metadata."""
    stem = Path(filename).stem.lower()
    for token, label in (
        ("nmme", "NMME"),
        ("nino", "ENSO"),
        ("enso", "ENSO"),
        ("eli", "ELI"),
        ("tc_", "TC"),
        ("tropical_cyclone", "TC"),
        ("iod", "IOD"),
        ("nao", "NAO"),
        ("sst", "SST"),
        ("psl", "PSL"),
        ("prect", "PRECT"),
        ("trefht", "TREFHT"),
    ):
        if token in stem:
            return label
    return "Other"


def _infer_workflow_group(filename: str, metric: str, mode: str) -> str:
    """Map a figure to one of the major workflow diagnostic sections."""
    stem = Path(filename).stem.lower()
    metric_lower = metric.lower()
    if "eli" in stem or "eli" in metric_lower:
        return "ELI"
    if stem.startswith("fig_tc_") or "track_density" in stem:
        return "TC"
    if mode.upper() in CLIMATE_MODES or metric_lower in {
        "eof_patterns",
        "global_teleconnection_patterns",
        "pc_time_series",
    }:
        return "MOV"
    if metric_lower.startswith("leadtime_acc"):
        return "LEAD_ACC"
    if metric_lower.startswith("leadtime_drift"):
        return "LEAD_DRIFT"
    if metric_lower.startswith("leadtime_rmse") or metric_lower.startswith(
        "rmse_compare"
    ):
        return "LEAD_RMSE"
    return "SST_INDEX"


def discover_workflow_figures(
    diag_dir: Union[str, Path],
    *,
    pattern: str = "fig_*",
    write_manifest: bool = False,
) -> dict:
    """Catalog actual workflow figures in a directory.

    Only files matching ``pattern`` with a supported image extension are
    returned. Existing manifest metadata is retained for files that still
    exist; stale entries are removed and missing entries are synthesized.
    """
    diag_dir = Path(diag_dir)
    if not diag_dir.is_dir():
        raise FileNotFoundError(f"Diagnostics directory does not exist: {diag_dir}")

    manifest_path = diag_dir / "figures.json"
    existing_manifest = {"figures": []}
    if manifest_path.is_file():
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                existing_manifest = json.load(fh)
        except Exception as exc:
            raise ValueError(f"Failed to parse manifest JSON file: {exc}") from exc

    metadata_by_file = {
        entry.get("file"): entry
        for entry in existing_manifest.get("figures", [])
        if entry.get("file")
    }
    figure_paths = sorted(
        path
        for path in diag_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in FIGURE_EXTENSIONS
    )

    figures = []
    for path in figure_paths:
        entry = dict(metadata_by_file.get(path.name, {}))
        entry.update({"file": path.name})
        entry.setdefault("mode", _infer_mode(path.name))
        entry.setdefault("metric", _infer_metric(path.name))
        entry.setdefault("title", _humanize_figure_name(path.name))
        entry.setdefault("caption", "Workflow-generated diagnostic figure.")
        entry["group"] = _infer_workflow_group(
            path.name, entry["metric"], entry["mode"]
        )
        figures.append(entry)

    manifest = {
        key: value
        for key, value in existing_manifest.items()
        if key not in {"figures", "updated"}
    }
    manifest["figures"] = figures
    manifest["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if write_manifest:
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)
            fh.write("\n")

    return manifest


def _make_gallery_web_readable(diag_dir: Path, manifest: dict) -> None:
    """Ensure the portal can traverse the gallery and read its files."""
    directory_bits = (
        stat.S_IRUSR
        | stat.S_IWUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH
    )
    diag_dir.chmod(diag_dir.stat().st_mode | directory_bits)

    for entry in manifest.get("figures", []):
        filename = entry.get("file")
        if not filename:
            continue
        figure_path = diag_dir / filename
        if figure_path.is_file():
            readable_bits = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
            figure_path.chmod(figure_path.stat().st_mode | readable_bits)


def generate_diagnostics_webpage(
    diag_dir: Union[str, Path],
    *,
    discover_figures: bool = False,
    figure_pattern: str = "fig_*",
    make_web_readable: bool = True,
) -> Path:
    """Generate an interactive HTML webpage for viewing diagnostics figures.

    This function embeds figure metadata inside a modern, responsive dark-mode
    viewer and writes the output as `index.html` in the same directory. When
    ``discover_figures`` is true, the manifest is first synchronized with the
    actual workflow figure files in the directory.

    The generated webpage is 100% self-contained (no external remote CSS/JS dependencies)
    and uses direct JSON embedding to avoid browser CORS errors when loaded via file://.

    Parameters
    ----------
    diag_dir : str or pathlib.Path
        Path to the directory containing the figures and `figures.json`.
    discover_figures : bool, optional
        Discover actual workflow figures and remove stale manifest entries.
    figure_pattern : str, optional
        Filename glob used when discovering figures. Defaults to ``fig_*``.
    make_web_readable : bool, optional
        Make cataloged figures publicly readable and the gallery directory
        publicly traversable. Defaults to true.

    Returns
    -------
    pathlib.Path
        The path to the generated `index.html` file.

    Raises
    ------
    FileNotFoundError
        If the output directory or `figures.json` does not exist.
    """
    diag_dir = Path(diag_dir)
    if not diag_dir.exists():
        raise FileNotFoundError(f"Diagnostics directory does not exist: {diag_dir}")

    manifest_path = diag_dir / "figures.json"
    if discover_figures:
        manifest_data = discover_workflow_figures(
            diag_dir, pattern=figure_pattern, write_manifest=True
        )
        if not manifest_data["figures"]:
            raise FileNotFoundError(
                f"No workflow figures matching {figure_pattern!r} found in {diag_dir}"
            )
    else:
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Figures manifest not found at {manifest_path}. "
                "Please ensure you have generated figures and their entries have been saved."
            )
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest_data = json.load(fh)
        except Exception as e:
            raise ValueError(f"Failed to parse manifest JSON file: {e}")

    # Build the HTML template
    html_content = _build_html_template(manifest_data)

    output_path = diag_dir / "index.html"
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html_content)

    if make_web_readable:
        _make_gallery_web_readable(diag_dir, manifest_data)
        manifest_path.chmod(0o644)
        output_path.chmod(0o644)

    return output_path


def _build_html_template(manifest: dict) -> str:
    """Build the raw string content of the index.html page."""
    # Serialize the manifest to embed directly in the script tags
    manifest_json_str = json.dumps(manifest, indent=2)

    template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <title>ESP-Lab Diagnostic Viewer</title>
    <!-- Modern font from Google Fonts. Standard system-ui fallback is included for offline use. -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700;800&display=swap" rel="stylesheet">
    <style>
        /* Modern Reset and Design System */
        :root {
            --bg-base: #020617;
            --bg-surface: #0f172a;
            --bg-card: rgba(30, 41, 59, 0.45);
            --bg-card-hover: rgba(30, 41, 59, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --border-hover: rgba(255, 255, 255, 0.18);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-primary: #6366f1; /* Indigo */
            --accent-glow: rgba(99, 102, 241, 0.35);
            --accent-success: #10b981;
            --accent-danger: #ef4444;
            --sidebar-width: 320px;
            --font-display: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            --font-body: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
            --transition-speed: 0.25s;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-base);
            color: var(--text-primary);
            font-family: var(--font-body);
            display: flex;
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* App Layout */
        .sidebar {
            width: var(--sidebar-width);
            background-color: var(--bg-surface);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            position: fixed;
            top: 0;
            bottom: 0;
            left: 0;
            z-index: 100;
        }

        .main-content {
            margin-left: var(--sidebar-width);
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            min-height: 100vh;
            padding: 2.5rem;
            position: relative;
            background: radial-gradient(circle at top right, rgba(99, 102, 241, 0.05), transparent 60%);
            min-width: 0; /* Prevents flex items from expanding beyond viewport width */
        }

        /* Mobile Topbar styling */
        .mobile-topbar {
            display: none;
            height: 60px;
            background-color: var(--bg-surface);
            border-bottom: 1px solid var(--border-color);
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 150;
            align-items: center;
            padding: 0 1.25rem;
            gap: 1rem;
        }

        .menu-toggle-btn {
            background: transparent;
            border: none;
            color: var(--text-primary);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0.25rem;
        }

        .mobile-brand-title {
            font-family: var(--font-display);
            font-weight: 800;
            font-size: 1.2rem;
            background: linear-gradient(135deg, #a5b4fc, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .sidebar-backdrop {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(2, 6, 23, 0.7);
            backdrop-filter: blur(4px);
            z-index: 95;
        }

        /* Sidebar Styling */
        .sidebar-header {
            padding: 2rem 1.5rem;
            border-bottom: 1px solid var(--border-color);
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), transparent);
        }

        .brand-title {
            font-family: var(--font-display);
            font-weight: 800;
            font-size: 1.5rem;
            letter-spacing: -0.025em;
            background: linear-gradient(135deg, #a5b4fc, #6366f1, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .brand-subtitle {
            font-size: 0.8rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
        }

        .search-container {
            padding: 1.25rem 1.5rem;
            border-bottom: 1px solid var(--border-color);
        }

        .search-box {
            position: relative;
            width: 100%;
        }

        .search-input {
            width: 100%;
            padding: 0.75rem 1rem 0.75rem 2.5rem;
            background-color: rgba(2, 6, 23, 0.6);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-primary);
            font-family: var(--font-body);
            font-size: 0.9rem;
            transition: all var(--transition-speed) ease;
        }

        .search-input:focus {
            outline: none;
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }

        .search-icon {
            position: absolute;
            left: 0.85rem;
            top: 50%;
            transform: translateY(-50%);
            fill: var(--text-muted);
            pointer-events: none;
            width: 16px;
            height: 16px;
        }

        .group-list {
            flex-grow: 1;
            overflow-y: auto;
            padding: 1rem 0.75rem;
            list-style: none;
        }

        .group-item {
            margin-bottom: 0.25rem;
        }

        .group-btn {
            width: 100%;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.75rem 1rem;
            background: transparent;
            border: none;
            border-radius: 8px;
            color: var(--text-secondary);
            font-family: var(--font-body);
            font-size: 0.925rem;
            font-weight: 500;
            text-align: left;
            cursor: pointer;
            transition: all var(--transition-speed) ease;
        }

        .group-btn:hover {
            background-color: rgba(255, 255, 255, 0.04);
            color: var(--text-primary);
        }

        .group-btn.active {
            background-color: rgba(99, 102, 241, 0.15);
            color: var(--text-primary);
            border-left: 3px solid var(--accent-primary);
            padding-left: calc(1rem - 3px);
        }

        .group-badge {
            background-color: rgba(255, 255, 255, 0.08);
            color: var(--text-secondary);
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.2rem 0.5rem;
            border-radius: 9999px;
            transition: all var(--transition-speed) ease;
        }

        .group-btn.active .group-badge {
            background-color: var(--accent-primary);
            color: #ffffff;
        }

        .sidebar-footer {
            padding: 1.25rem 1.5rem;
            border-top: 1px solid var(--border-color);
            background-color: rgba(2, 6, 23, 0.3);
        }

        .updated-time {
            font-size: 0.75rem;
            color: var(--text-muted);
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }

        /* Header in main area */
        .main-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            margin-bottom: 2rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1.5rem;
        }

        .header-title-area {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .active-group-title {
            font-family: var(--font-display);
            font-weight: 700;
            font-size: 2.25rem;
            color: var(--text-primary);
            letter-spacing: -0.02em;
        }

        .active-group-desc {
            font-size: 0.95rem;
            color: var(--text-secondary);
        }

        /* Compact figure-type filters within the selected workflow group */
        .metrics-tabs {
            display: none;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 2rem;
            padding: 0.75rem;
            background: rgba(15, 23, 42, 0.45);
            border: 1px solid var(--border-color);
            border-radius: 12px;
        }

        .metric-tab-btn {
            background-color: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 9999px;
            color: var(--text-secondary);
            padding: 0.45rem 1rem;
            font-family: var(--font-body);
            font-size: 0.85rem;
            font-weight: 500;
            white-space: nowrap;
            cursor: pointer;
            transition: all var(--transition-speed) ease;
        }

        .metric-tab-btn:hover {
            border-color: var(--border-hover);
            color: var(--text-primary);
        }

        .metric-tab-btn.active {
            background-color: var(--accent-primary);
            border-color: var(--accent-primary);
            color: #ffffff;
            box-shadow: 0 0 12px rgba(99, 102, 241, 0.3);
        }

        /* Figures Grid */
        .figures-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
            gap: 2rem;
            flex-grow: 1;
        }

        /* Figure Cards */
        .figure-card {
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            transition: all var(--transition-speed) cubic-bezier(0.4, 0, 0.2, 1);
            backdrop-filter: blur(8px);
            box-shadow: var(--shadow-md);
        }

        .figure-card:hover {
            transform: translateY(-4px);
            border-color: var(--accent-primary);
            background-color: var(--bg-card-hover);
            box-shadow: 0 12px 24px -10px rgba(0, 0, 0, 0.5), 0 0 16px rgba(99, 102, 241, 0.1);
        }

        .card-img-wrapper {
            position: relative;
            aspect-ratio: 4/3;
            background-color: rgba(2, 6, 23, 0.4);
            overflow: hidden;
            border-bottom: 1px solid var(--border-color);
            cursor: pointer;
        }

        .card-img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            transition: transform 0.5s ease;
        }

        .figure-card:hover .card-img {
            transform: scale(1.02);
        }

        .card-actions-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(2, 6, 23, 0.6);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 1rem;
            opacity: 0;
            transition: opacity var(--transition-speed) ease;
        }

        .card-img-wrapper:hover .card-actions-overlay {
            opacity: 1;
        }

        .action-btn {
            background-color: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #ffffff;
            padding: 0.6rem 1rem;
            border-radius: 8px;
            font-size: 0.85rem;
            font-family: var(--font-body);
            font-weight: 500;
            cursor: pointer;
            transition: all var(--transition-speed) ease;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        .action-btn:hover {
            background-color: #ffffff;
            color: var(--bg-base);
            border-color: #ffffff;
            transform: translateY(-2px);
        }

        .compare-label-btn {
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.85rem;
            font-weight: 500;
            background-color: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 0.6rem 1rem;
            border-radius: 8px;
            color: #ffffff;
            transition: all var(--transition-speed) ease;
        }

        .compare-label-btn:hover {
            background-color: var(--accent-primary);
            border-color: var(--accent-primary);
            transform: translateY(-2px);
        }

        .compare-checkbox {
            appearance: none;
            width: 16px;
            height: 16px;
            border: 1.5px solid rgba(255, 255, 255, 0.7);
            border-radius: 4px;
            background-color: transparent;
            cursor: pointer;
            display: inline-grid;
            place-content: center;
            position: relative;
        }

        .compare-checkbox:checked {
            background-color: #ffffff;
            border-color: #ffffff;
        }

        .compare-checkbox:checked::before {
            content: "";
            width: 8px;
            height: 8px;
            background-color: var(--accent-primary);
            border-radius: 1px;
        }

        .card-body {
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            flex-grow: 1;
        }

        .card-title {
            font-family: var(--font-display);
            font-weight: 600;
            font-size: 1.1rem;
            line-height: 1.4;
            color: var(--text-primary);
        }

        .card-caption {
            font-size: 0.85rem;
            color: var(--text-secondary);
            line-height: 1.5;
            flex-grow: 1;
        }

        .card-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: auto;
            border-top: 1px solid rgba(255, 255, 255, 0.04);
            padding-top: 0.75rem;
        }

        .meta-tag {
            font-size: 0.7rem;
            font-weight: 600;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            background-color: rgba(255, 255, 255, 0.04);
            color: var(--text-muted);
            border: 1px solid var(--border-color);
            text-transform: uppercase;
        }

        .meta-tag.metric {
            color: var(--accent-primary);
            border-color: rgba(99, 102, 241, 0.2);
            background-color: rgba(99, 102, 241, 0.04);
        }

        /* Fullscreen Lightbox */
        .lightbox {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(2, 6, 23, 0.95);
            z-index: 1000;
            display: none;
            opacity: 0;
            transition: opacity var(--transition-speed) ease;
        }

        .lightbox.active {
            display: flex;
            opacity: 1;
        }

        .lightbox-close {
            position: absolute;
            top: 1.5rem;
            right: 1.5rem;
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #ffffff;
            font-size: 1.5rem;
            width: 44px;
            height: 44px;
            border-radius: 50%;
            display: grid;
            place-content: center;
            cursor: pointer;
            z-index: 1010;
            transition: all var(--transition-speed) ease;
        }

        .lightbox-close:hover {
            background-color: var(--accent-danger);
            border-color: var(--accent-danger);
            transform: rotate(90deg);
        }

        .lightbox-container {
            flex-grow: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
            padding: 2rem;
            user-select: none;
        }

        .lightbox-img-wrapper {
            position: relative;
            max-width: 85%;
            max-height: 85%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: grab;
        }

        .lightbox-img-wrapper:active {
            cursor: grabbing;
        }

        .lightbox-img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
            border-radius: 8px;
            transition: transform 0.1s cubic-bezier(0.1, 0.8, 0.3, 1);
            transform-origin: center;
        }

        .lightbox-nav {
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #ffffff;
            font-size: 1.5rem;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            display: grid;
            place-content: center;
            cursor: pointer;
            z-index: 1005;
            transition: all var(--transition-speed) ease;
        }

        .lightbox-nav:hover {
            background-color: var(--accent-primary);
            border-color: var(--accent-primary);
            transform: translateY(-50%) scale(1.1);
        }

        .lightbox-nav.prev {
            left: 2rem;
        }

        .lightbox-nav.next {
            right: 2rem;
        }

        /* Lightbox Sidebar Info Panel */
        .lightbox-info {
            width: 380px;
            background-color: var(--bg-surface);
            border-left: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            padding: 2.5rem;
            overflow-y: auto;
            z-index: 1005;
            box-shadow: -10px 0 30px rgba(0, 0, 0, 0.5);
        }

        .lightbox-info-title {
            font-family: var(--font-display);
            font-weight: 700;
            font-size: 1.5rem;
            margin-bottom: 1rem;
            line-height: 1.3;
        }

        .lightbox-info-caption {
            font-size: 0.95rem;
            color: var(--text-secondary);
            line-height: 1.6;
            margin-bottom: 2rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--border-color);
        }

        .lightbox-details-grid {
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }

        .detail-row {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }

        .detail-label {
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .detail-val {
            font-size: 0.9rem;
            color: var(--text-primary);
            word-break: break-all;
        }

        .lightbox-controls {
            margin-top: auto;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            padding-top: 2rem;
        }

        .ctrl-btn {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            background-color: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 0.75rem 1rem;
            border-radius: 8px;
            font-family: var(--font-body);
            font-size: 0.9rem;
            font-weight: 500;
            cursor: pointer;
            transition: all var(--transition-speed) ease;
        }

        .ctrl-btn:hover {
            background-color: var(--accent-primary);
            border-color: var(--accent-primary);
            box-shadow: 0 0 10px rgba(99, 102, 241, 0.2);
        }

        /* Zoom Control Overlay */
        .zoom-controls-overlay {
            position: absolute;
            bottom: 2rem;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 0.5rem;
            background-color: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(12px);
            padding: 0.5rem;
            border-radius: 9999px;
            z-index: 1005;
        }

        .zoom-btn {
            background: transparent;
            border: none;
            color: #ffffff;
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: grid;
            place-content: center;
            cursor: pointer;
            transition: background-color var(--transition-speed) ease;
        }

        .zoom-btn:hover {
            background-color: rgba(255, 255, 255, 0.15);
        }

        .zoom-btn.active {
            background-color: var(--accent-primary);
            color: #ffffff;
        }

        /* Bottom Compare Drawer */
        .compare-drawer {
            position: fixed;
            bottom: 0;
            left: var(--sidebar-width);
            right: 0;
            background-color: rgba(15, 23, 42, 0.9);
            backdrop-filter: blur(16px);
            border-top: 1px solid var(--border-color);
            box-shadow: 0 -10px 30px rgba(0, 0, 0, 0.4);
            z-index: 500;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1.25rem 2.5rem;
            transform: translateY(100%);
            transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .compare-drawer.active {
            transform: translateY(0);
        }

        .compare-drawer-left {
            display: flex;
            align-items: center;
            gap: 1.5rem;
        }

        .compare-count-txt {
            font-size: 0.95rem;
            font-weight: 500;
            color: var(--text-primary);
        }

        .compare-thumbnails {
            display: flex;
            gap: 0.75rem;
        }

        .compare-thumb {
            width: 60px;
            height: 45px;
            object-fit: cover;
            border-radius: 6px;
            border: 2px solid var(--accent-primary);
            background-color: #000;
        }

        .compare-drawer-right {
            display: flex;
            gap: 1rem;
        }

        .btn-secondary {
            background: transparent;
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 0.6rem 1.25rem;
            border-radius: 8px;
            font-family: var(--font-body);
            font-size: 0.85rem;
            font-weight: 500;
            cursor: pointer;
            transition: all var(--transition-speed) ease;
        }

        .btn-secondary:hover {
            border-color: var(--border-hover);
            color: var(--text-primary);
        }

        .btn-primary {
            background-color: var(--accent-primary);
            border: 1px solid var(--accent-primary);
            color: #ffffff;
            padding: 0.6rem 1.5rem;
            border-radius: 8px;
            font-family: var(--font-body);
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
            transition: all var(--transition-speed) ease;
        }

        .btn-primary:hover {
            background-color: #4f46e5;
            border-color: #4f46e5;
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(99, 102, 241, 0.4);
        }

        /* Comparison Portal Overlay */
        .comparison-portal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: var(--bg-base);
            z-index: 2000;
            display: none;
            flex-direction: column;
        }

        .comparison-portal.active {
            display: flex;
        }

        .comp-header {
            height: 70px;
            background-color: var(--bg-surface);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 2rem;
        }

        .comp-title {
            font-family: var(--font-display);
            font-weight: 700;
            font-size: 1.3rem;
            letter-spacing: -0.01em;
            background: linear-gradient(135deg, #a5b4fc, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .comp-grid {
            flex-grow: 1;
            display: grid;
            background-color: #020617;
            padding: 1.5rem;
            gap: 1.5rem;
            overflow: hidden;
        }

        /* Responsive Comparison Panel Layouts */
        .comp-grid.panels-1 { grid-template-columns: 1fr; }
        .comp-grid.panels-2 { grid-template-columns: 1fr 1fr; }
        .comp-grid.panels-3 { grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr; }
        .comp-grid.panels-4 { grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr; }

        /* Make 3rd panel stretch or display nicely in grid */
        .comp-grid.panels-3 > div:nth-child(3) {
            grid-column: span 2;
        }

        .comp-panel {
            background-color: rgba(15, 23, 42, 0.5);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            position: relative;
        }

        .comp-panel-header {
            background-color: rgba(15, 23, 42, 0.8);
            border-bottom: 1px solid var(--border-color);
            padding: 0.75rem 1rem;
            font-size: 0.85rem;
            font-weight: 600;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 10;
        }

        .comp-panel-body {
            flex-grow: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            position: relative;
            cursor: grab;
            user-select: none;
        }

        .comp-panel-body:active {
            cursor: grabbing;
        }

        .comp-panel-img {
            max-width: 95%;
            max-height: 95%;
            object-fit: contain;
            border-radius: 4px;
            transform-origin: center;
            transition: transform 0.1s cubic-bezier(0.1, 0.8, 0.3, 1);
        }

        .comp-panel-zoom-overlay {
            position: absolute;
            bottom: 1rem;
            right: 1rem;
            display: flex;
            gap: 0.25rem;
            background-color: rgba(2, 6, 23, 0.75);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 0.35rem;
            border-radius: 9999px;
            z-index: 15;
        }

        .comp-btn {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            border: none;
            background: transparent;
            color: #ffffff;
            cursor: pointer;
            display: grid;
            place-content: center;
            transition: background-color var(--transition-speed) ease;
        }

        .comp-btn:hover {
            background-color: rgba(255, 255, 255, 0.15);
        }

        /* Empty State */
        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 1rem;
            padding: 5rem 2rem;
            color: var(--text-secondary);
            text-align: center;
            grid-column: 1 / -1;
            background-color: var(--bg-card);
            border: 1px dashed var(--border-color);
            border-radius: 12px;
        }

        .empty-state svg {
            width: 64px;
            height: 64px;
            fill: var(--text-muted);
        }

        .empty-state-title {
            font-family: var(--font-display);
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        /* Scrollbar custom styles */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: var(--bg-base);
        }

        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.12);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.25);
        }

        /* Toggle Info class styles for Lightbox */
        .lightbox.info-collapsed .lightbox-info {
            display: none;
        }

        .lightbox.info-collapsed .lightbox-container {
            width: 100%;
        }

        /* Tablet & Desktop Large Responsiveness */
        @media (max-width: 992px) {
            .sidebar {
                transform: translateX(-100%);
                transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                box-shadow: 10px 0 30px rgba(0, 0, 0, 0.5);
            }

            .sidebar.open {
                transform: translateX(0);
            }

            .sidebar-backdrop.active {
                display: block;
            }

            .mobile-topbar {
                display: flex;
            }

            .main-content {
                margin-left: 0;
                padding: 5rem 1.5rem 2.5rem 1.5rem; /* Top padding to clear mobile topbar */
            }

            .active-group-title {
                font-size: 1.75rem;
            }

            .figures-grid {
                grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                gap: 1.25rem;
            }

            .compare-drawer {
                left: 0;
                padding: 1rem 1.5rem;
                flex-direction: column;
                gap: 1rem;
                align-items: stretch;
            }

            .compare-drawer-left {
                flex-direction: column;
                align-items: flex-start;
                gap: 0.5rem;
            }

            .compare-drawer-right {
                justify-content: flex-end;
            }

            /* Lightbox Responsive Layout */
            .lightbox {
                flex-direction: column;
            }

            .lightbox-container {
                padding: 1.5rem;
                height: 60vh;
                flex-grow: 1;
            }

            .lightbox-img-wrapper {
                max-width: 95%;
                max-height: 95%;
            }

            .lightbox-info {
                width: 100%;
                height: 40vh;
                border-left: none;
                border-top: 1px solid var(--border-color);
                padding: 1.5rem;
                box-shadow: 0 -10px 30px rgba(0, 0, 0, 0.5);
            }

            .lightbox.info-collapsed .lightbox-container {
                height: 100vh;
            }

            /* Comparison Grid on small screens */
            .comp-grid.panels-2 { grid-template-columns: 1fr; overflow-y: auto; }
            .comp-grid.panels-3 { grid-template-columns: 1fr; grid-template-rows: repeat(3, 400px); overflow-y: auto; }
            .comp-grid.panels-4 { grid-template-columns: 1fr; grid-template-rows: repeat(4, 400px); overflow-y: auto; }
            .comp-grid.panels-3 > div:nth-child(3) { grid-column: span 1; }
        }

        /* Mobile specific adjustments */
        @media (max-width: 576px) {
            .figures-grid {
                grid-template-columns: 1fr;
            }
            .lightbox-info {
                padding: 1.25rem;
            }
            .lightbox-info-title {
                font-size: 1.25rem;
            }
            .active-group-title {
                font-size: 1.5rem;
            }
        }
    </style>
</head>
<body>

    <!-- Mobile Top Header -->
    <div class="mobile-topbar">
        <button class="menu-toggle-btn" id="menuToggleBtn">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                <path d="M3 18h18v-2H3v2zm0-5h18v-2H3v2zm0-7v2h18V6H3z"/>
            </svg>
        </button>
        <span class="mobile-brand-title">ESP-Lab Diagnostics</span>
    </div>

    <div class="sidebar-backdrop" id="sidebarBackdrop"></div>

    <!-- Sidebar Navigation -->
    <aside class="sidebar">
        <div class="sidebar-header">
            <h1 class="brand-title">ESP-Lab</h1>
            <div class="brand-subtitle">Diagnostics Viewer</div>
        </div>

        <div class="search-container">
            <div class="search-box">
                <svg class="search-icon" viewBox="0 0 24 24">
                    <path d="M15.5 14h-.79l-.28-.27A6.471 6.471 0 0 0 16 9.5 6.5 6.5 0 1 0 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/>
                </svg>
                <input type="text" id="searchBar" class="search-input" placeholder="Search figures...">
            </div>
        </div>

        <ul class="group-list" id="groupList">
            <!-- Dynamically populated -->
        </ul>

        <div class="sidebar-footer">
            <div class="updated-time">
                <span>MANIFEST UPDATED:</span>
                <strong id="manifestUpdated">-</strong>
            </div>
        </div>
    </aside>

    <!-- Main Content Area -->
    <main class="main-content">
        <header class="main-header">
            <div class="header-title-area">
                <h2 class="active-group-title" id="activeGroupTitle">Loading...</h2>
                <div class="active-group-desc" id="activeGroupDesc">-</div>
            </div>
        </header>

        <!-- Metrics Filter Sub-Tabs -->
        <nav class="metrics-tabs" id="metricsTabs">
            <!-- Dynamically populated -->
        </nav>

        <!-- Figures Grid -->
        <section class="figures-grid" id="figuresGrid">
            <!-- Dynamically populated -->
        </section>
    </main>

    <!-- Bottom Compare Drawer -->
    <div class="compare-drawer" id="compareDrawer">
        <div class="compare-drawer-left">
            <span class="compare-count-txt" id="compareCountTxt">0 items selected</span>
            <div class="compare-thumbnails" id="compareThumbnails"></div>
        </div>
        <div class="compare-drawer-right">
            <button class="btn-secondary" id="clearCompareBtn">Clear All</button>
            <button class="btn-primary" id="launchCompareBtn">Compare Selected</button>
        </div>
    </div>

    <!-- Fullscreen Lightbox Modal -->
    <div class="lightbox" id="lightbox">
        <button class="lightbox-close" id="lightboxClose">&times;</button>
        
        <button class="lightbox-nav prev" id="lightboxPrev">&#10094;</button>
        <button class="lightbox-nav next" id="lightboxNext">&#10095;</button>

        <div class="lightbox-container" id="lightboxContainer">
            <div class="lightbox-img-wrapper" id="lightboxImgWrapper">
                <img class="lightbox-img" id="lightboxImg" src="" alt="" draggable="false">
            </div>

            <!-- Overlay Zoom Controls -->
            <div class="zoom-controls-overlay">
                <button class="zoom-btn" id="toggleInfoBtn" title="Toggle Info Panel">
                    <!-- SVG Info Icon -->
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/>
                    </svg>
                </button>
                <button class="zoom-btn" id="scrollModeBtn" title="Toggle Scroll Mode (Actual Size)">
                    <!-- SVG Scroll/Move Icon -->
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M4 6H2v14c0 1.1.9 2 2 2h14v-2H4V6zm16-4H8c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h12c1.1 0-2-.9-2-2V4c0-1.1-.9-2-2-2zm0 14H8V4h12v12z"/>
                    </svg>
                </button>
                <button class="zoom-btn" id="zoomOutBtn" title="Zoom Out">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M19 13H5v-2h14v2z"/>
                    </svg>
                </button>
                <button class="zoom-btn" id="zoomResetBtn" title="Reset Zoom">
                    <!-- SVG circular reset arrow icon -->
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 6v3l4-4-4-4v3c-4.42 0-8 3.58-8 8 0 1.57.46 3.03 1.24 4.26L4.7 14.74C3.61 13.9 3 12.58 3 11c0-4.97 4.03-9 9-9zm0 18c4.42 0 8-3.58 8-8 0-1.57-.46-3.03-1.24-4.26l-1.46 1.46c.78 1.23 1.2 2.69 1.2 4.26 0 4.97-4.03 9-9 9v-3l-4 4 4 4v-3z"/>
                    </svg>
                </button>
                <button class="zoom-btn" id="zoomInBtn" title="Zoom In">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/>
                    </svg>
                </button>
            </div>
        </div>

        <div class="lightbox-info">
            <h3 class="lightbox-info-title" id="lightboxTitle">Figure Title</h3>
            <p class="lightbox-info-caption" id="lightboxCaption">Figure caption detail carries information here.</p>
            
            <div class="lightbox-details-grid">
                <div class="detail-row">
                    <div class="detail-label">File Name</div>
                    <div class="detail-val" id="lightboxFilename">-</div>
                </div>
                <div class="detail-row">
                    <div class="detail-label">Category / Mode</div>
                    <div class="detail-val" id="lightboxMode">-</div>
                </div>
                <div class="detail-row">
                    <div class="detail-label">Diagnostic Metric</div>
                    <div class="detail-val" id="lightboxMetric">-</div>
                </div>
                <div class="detail-row" id="lightboxRefRow">
                    <div class="detail-label">Reference Baseline</div>
                    <div class="detail-val" id="lightboxReference">-</div>
                </div>
            </div>

            <div class="lightbox-controls">
                <a class="ctrl-btn" id="lightboxDownloadBtn" href="" download>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM17 13l-5 5-5-5h3V9h4v4h3z"/>
                    </svg>
                    Download Figure
                </a>
            </div>
        </div>
    </div>

    <!-- Side-by-Side Comparison Portal -->
    <div class="comparison-portal" id="comparisonPortal">
        <header class="comp-header">
            <h3 class="comp-title">Diagnostic Multi-Figure Comparison</h3>
            <button class="btn-secondary" id="closeComparePortalBtn">Exit Comparison</button>
        </header>
        <section class="comp-grid" id="compGrid">
            <!-- Panels dynamically inserted -->
        </section>
    </div>

    <!-- Embedded Figures Manifest Data -->
    <script>
        const manifest = __MANIFEST_JSON__;
    </script>

    <!-- Main Logic Script -->
    <script>
        // Store user state
        let activeGroup = "ALL";
        let activeMetric = "ALL";
        let searchQuery = "";
        let selectedForCompare = [];
        let filteredFigures = [];
        let activeLightboxIndex = 0;
        let isScrollMode = false;
        let isInfoCollapsed = false;

        // Group mapping metadata
        const GROUP_LABELS = {
            "ALL": "All Figures",
            "LEAD_ACC": "Lead-time ACC",
            "LEAD_DRIFT": "Lead-time Drift",
            "LEAD_RMSE": "Lead-time RMSE",
            "SST_INDEX": "SST Indices",
            "MOV": "Modes of Variability",
            "TC": "Tropical Cyclones",
            "ELI": "ELI Diagnostics",
            "OTHER": "Other"
        };

        const FIGURE_TYPE_LABELS = {
            "ALL": "All figure types",
            "SKILL": "Skill",
            "TIME_SERIES": "Time series",
            "SKILL_MAP": "Skill maps",
            "DIFFERENCE": "Differences",
            "MODEL_COMPARISON": "Model comparison",
            "EOF_PATTERNS": "EOF patterns",
            "TELECONNECTIONS": "Teleconnections",
            "PC_TIME_SERIES": "PC time series",
            "METHOD_COMPARISON": "Method comparison",
            "LEAD_TIME": "Lead-time comparison",
            "ENSO_REGRESSION": "ENSO regression",
            "TRACK_DENSITY": "Track density",
            "TRAJECTORIES": "Trajectories",
            "DRIFT": "Drift climatology",
            "NMME_BENCHMARK": "NMME benchmark",
            "ELI_NINO34": "ELI vs Niño3.4",
            "OTHER": "Other"
        };
        const FIGURE_TYPE_ORDER = [
            "SKILL", "SKILL_MAP", "TIME_SERIES", "MODEL_COMPARISON",
            "DIFFERENCE", "EOF_PATTERNS", "TELECONNECTIONS", "PC_TIME_SERIES",
            "METHOD_COMPARISON", "LEAD_TIME", "ENSO_REGRESSION", "TRACK_DENSITY",
            "TRAJECTORIES", "DRIFT", "NMME_BENCHMARK", "ELI_NINO34", "OTHER"
        ];

        // Bust browser image cache when figures are regenerated with the same filename.
        const PAGE_CACHE_BUSTER = Date.now().toString();

        function figureUrl(fig) {
            const base = fig.url || fig.file;
            if (!base) return "";
            const sep = base.includes("?") ? "&" : "?";
            return `${base}${sep}v=${encodeURIComponent(PAGE_CACHE_BUSTER)}`;
        }

        // Parse group names from filenames or properties
        function parseGroup(fig) {
            if (fig.group && fig.group.trim() !== "") {
                return fig.group.trim().toUpperCase();
            }
            if (fig.mode && fig.mode.trim() !== "") {
                const cleanMode = fig.mode.trim().toUpperCase();
                if (cleanMode === "NINO3_4" || cleanMode === "NINO34") return "NINO3.4";
                return cleanMode;
            }
            const fileLower = fig.file.toLowerCase();
            if (fileLower.includes("nino3_4") || fileLower.includes("nino34")) return "NINO3.4";
            if (fileLower.includes("nino12")) return "NINO12";
            if (fileLower.includes("nino3")) return "NINO3";
            if (fileLower.includes("nino4")) return "NINO4";
            if (fileLower.includes("prect")) return "PRECT";
            if (fileLower.includes("psl")) return "PSL";
            if (fileLower.includes("trefht")) return "TREFHT";
            if (fileLower.includes("atlnino")) return "ATLNINO";
            if (fileLower.includes("iod")) return "IOD";
            if (fileLower.includes("tna")) return "TNA";
            if (fileLower.includes("tc_") || fileLower.includes("track_density")) return "TC";

            // Splitting tokens as fallback
            const tokens = fig.file.split('_');
            if (tokens.length > 1) {
                return tokens[1].toUpperCase();
            }
            return "OTHER";
        }

        // Format metric name
        function formatMetricName(metric) {
            if (!metric) return "General";
            return metric
                .split('_')
                .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                .join(' ');
        }

        // Reduce internal metric keys to a few readable figure types.
        function classifyFigureType(fig) {
            const metric = (fig.metric || "").toLowerCase();
            const file = (fig.file || "").toLowerCase();

            if (fig.group === "LEAD_ACC") {
                if (metric.includes("compare")) return "MODEL_COMPARISON";
                if (metric.includes("diff")) return "DIFFERENCE";
                return "SKILL_MAP";
            }
            if (fig.group === "LEAD_DRIFT") {
                if (file.includes("spatial_maps")) return "SKILL_MAP";
                if (file.includes("regime_fraction")) return "DRIFT";
                if (file.includes("drift")) return "DRIFT";
                return "SKILL";
            }
            if (fig.group === "LEAD_RMSE") {
                if (metric.startsWith("rmse_compare")) return "MODEL_COMPARISON";
                if (metric.includes("diff")) return "DIFFERENCE";
                return "SKILL_MAP";
            }
            if (fig.group === "SST_INDEX") {
                return metric.includes("time_series") || file.includes("timeseries")
                    ? "TIME_SERIES" : "SKILL";
            }
            if (fig.group === "MOV") {
                if (metric === "eof_patterns") return "EOF_PATTERNS";
                if (metric === "global_teleconnection_patterns") return "TELECONNECTIONS";
                if (metric === "pc_time_series") return "PC_TIME_SERIES";
                return "SKILL";
            }
            if (fig.group === "TC") {
                if (metric.includes("method")) return "METHOD_COMPARISON";
                if (metric.includes("leadtime")) return "LEAD_TIME";
                if (metric.includes("enso_regression")) return "ENSO_REGRESSION";
                if (metric.includes("density")) return "TRACK_DENSITY";
                if (metric.includes("trajectory")) return "TRAJECTORIES";
            }
            if (fig.group === "ELI") {
                if (metric.includes("dual_axis")) return "ELI_NINO34";
                if (metric.includes("drift")) return "DRIFT";
                if (metric.includes("benchmark")) return "NMME_BENCHMARK";
                if (metric.includes("time_series")) return "TIME_SERIES";
                return "SKILL";
            }
            return "OTHER";
        }

        // Initialize Web Application
        document.addEventListener("DOMContentLoaded", () => {
            // Display Update Timestamp
            document.getElementById("manifestUpdated").innerText = manifest.updated || "Unknown";

            // Process groups and figure properties
            manifest.figures.forEach(fig => {
                fig.group = parseGroup(fig);
                fig.figureType = classifyFigureType(fig);
            });

            renderSidebar();
            updateActiveView();

            // Set up search listener
            document.getElementById("searchBar").addEventListener("input", (e) => {
                searchQuery = e.target.value.toLowerCase();
                updateActiveView();
            });

            // Lightbox and comparison events
            setUpLightboxEvents();
            setUpComparisonEvents();

            // Sidebar toggle logic for mobile
            const sidebar = document.querySelector(".sidebar");
            const backdrop = document.getElementById("sidebarBackdrop");
            const toggleBtn = document.getElementById("menuToggleBtn");

            if (toggleBtn) {
                toggleBtn.addEventListener("click", () => {
                    sidebar.classList.toggle("open");
                    backdrop.classList.toggle("active");
                });
            }
            if (backdrop) {
                backdrop.addEventListener("click", () => {
                    sidebar.classList.remove("open");
                    backdrop.classList.remove("active");
                });
            }

        });

        // Render groups lists in sidebar
        function renderSidebar() {
            const groupList = document.getElementById("groupList");
            groupList.innerHTML = "";

            // Calculate counts
            const counts = { "ALL": manifest.figures.length };
            manifest.figures.forEach(fig => {
                counts[fig.group] = (counts[fig.group] || 0) + 1;
            });

            // Follow the major section order used by the diagnostics workflow.
            const workflowOrder = [
                "LEAD_ACC", "LEAD_RMSE", "LEAD_DRIFT", "SST_INDEX", "MOV", "TC", "ELI", "OTHER"
            ];
            const groups = Object.keys(counts)
                .filter(g => g !== "ALL")
                .sort((a, b) => {
                    const ai = workflowOrder.indexOf(a);
                    const bi = workflowOrder.indexOf(b);
                    if (ai === -1 && bi === -1) return a.localeCompare(b);
                    if (ai === -1) return 1;
                    if (bi === -1) return -1;
                    return ai - bi;
                });
            const allGroups = ["ALL", ...groups];

            allGroups.forEach(grp => {
                const li = document.createElement("li");
                li.className = "group-item";
                
                const label = GROUP_LABELS[grp] || grp;
                
                li.innerHTML = `
                    <button class="group-btn ${grp === activeGroup ? 'active' : ''}" onclick="selectGroup('${grp}')">
                        <span>${label}</span>
                        <span class="group-badge">${counts[grp]}</span>
                    </button>
                `;
                groupList.appendChild(li);
            });
        }

        // Select active group
        window.selectGroup = function(grp) {
            activeGroup = grp;
            activeMetric = "ALL"; // Reset sub-metric when switching groups
            
            // Re-render sidebar to update active class
            renderSidebar();
            updateActiveView();

            // Close sidebar on mobile/tablet view
            const sidebar = document.querySelector(".sidebar");
            const backdrop = document.getElementById("sidebarBackdrop");
            if (window.innerWidth <= 992) {
                sidebar.classList.remove("open");
                backdrop.classList.remove("active");
            }
        };

        // Update main content grid
        function updateActiveView() {
            const grid = document.getElementById("figuresGrid");
            grid.innerHTML = "";

            // 1. Filter figures by Group & Search
            let list = manifest.figures.filter(fig => {
                const matchesGroup = (activeGroup === "ALL" || fig.group === activeGroup);
                const matchesSearch = searchQuery === "" || 
                    (fig.title && fig.title.toLowerCase().includes(searchQuery)) ||
                    (fig.caption && fig.caption.toLowerCase().includes(searchQuery)) ||
                    fig.file.toLowerCase().includes(searchQuery);
                return matchesGroup && matchesSearch;
            });

            // 2. Show concise figure-type filters only within a major group.
            const uniqueMetrics = new Set();
            list.forEach(fig => uniqueMetrics.add(fig.figureType));

            // Render sub-metric tab buttons
            const metricsTabs = document.getElementById("metricsTabs");
            metricsTabs.innerHTML = "";

            if (activeGroup !== "ALL" && uniqueMetrics.size > 1) {
                const allBtn = document.createElement("button");
                allBtn.className = `metric-tab-btn ${activeMetric === "ALL" ? "active" : ""}`;
                allBtn.innerText = FIGURE_TYPE_LABELS.ALL;
                allBtn.onclick = () => selectMetric("ALL");
                metricsTabs.appendChild(allBtn);

                Array.from(uniqueMetrics).sort((a, b) => {
                    const ai = FIGURE_TYPE_ORDER.indexOf(a);
                    const bi = FIGURE_TYPE_ORDER.indexOf(b);
                    return (ai === -1 ? 999 : ai) - (bi === -1 ? 999 : bi);
                }).forEach(met => {
                    const btn = document.createElement("button");
                    btn.className = `metric-tab-btn ${activeMetric === met ? "active" : ""}`;
                    btn.innerText = FIGURE_TYPE_LABELS[met] || formatMetricName(met);
                    btn.onclick = () => selectMetric(met);
                    metricsTabs.appendChild(btn);
                });
                metricsTabs.style.display = "flex";
            } else {
                metricsTabs.style.display = "none";
                activeMetric = "ALL";
            }

            // 3. Filter by sub-metric
            if (activeMetric !== "ALL") {
                list = list.filter(fig => fig.figureType === activeMetric);
            }

            filteredFigures = list; // Cache for lightbox arrows

            // Update main title info
            const label = GROUP_LABELS[activeGroup] || activeGroup;
            document.getElementById("activeGroupTitle").innerText = label;
            document.getElementById("activeGroupDesc").innerText =
                `${list.length} ${list.length === 1 ? "figure" : "figures"}`;

            // 4. Render Grid
            if (list.length === 0) {
                grid.innerHTML = `
                    <div class="empty-state">
                        <svg viewBox="0 0 24 24">
                            <path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z"/>
                        </svg>
                        <div class="empty-state-title">No figures found</div>
                        <p>Adjust your search query or select another category.</p>
                    </div>
                `;
                return;
            }

            list.forEach((fig, index) => {
                const isSelected = selectedForCompare.some(item => item.file === fig.file);
                const card = document.createElement("div");
                card.className = "figure-card";

                card.innerHTML = `
                    <div class="card-img-wrapper" onclick="openLightbox(${index})">
                        <img class="card-img" src="${figureUrl(fig)}" alt="${fig.title}" loading="lazy">
                        <div class="card-actions-overlay">
                            <button class="action-btn" onclick="event.stopPropagation(); openLightbox(${index})">
                                🔍 Fullscreen
                            </button>
                            <label class="compare-label-btn" onclick="event.stopPropagation();">
                                <input type="checkbox" class="compare-checkbox" ${isSelected ? 'checked' : ''} 
                                       onchange="toggleCompare(event, ${JSON.stringify(fig).replace(/"/g, '&quot;')})">
                                Compare
                            </label>
                        </div>
                    </div>
                    <div class="card-body">
                        <h4 class="card-title">${fig.title || fig.file}</h4>
                        ${fig.caption ? `<p class="card-caption">${fig.caption}</p>` : ''}
                        <div class="card-meta">
                            <span class="meta-tag">${fig.group}</span>
                            <span class="meta-tag metric">${FIGURE_TYPE_LABELS[fig.figureType] || formatMetricName(fig.metric)}</span>
                            ${fig.reference ? `<span class="meta-tag">${fig.reference}</span>` : ''}
                        </div>
                    </div>
                `;
                grid.appendChild(card);
            });
        }

        function selectMetric(met) {
            activeMetric = met;
            updateActiveView();
        }

        // Lightbox Functionality (Zoom, Pan, Arrows)
        let zoomScale = 1;
        let isDragging = false;
        let startX, startY, translateX = 0, translateY = 0;

        function openLightbox(index) {
            activeLightboxIndex = index;
            const fig = filteredFigures[index];
            if (!fig) return;

            // Reset scroll mode to fit mode for new image
            if (isScrollMode) {
                toggleScrollMode();
            }

            // Reset zoom & pan values
            zoomScale = 1;
            translateX = 0;
            translateY = 0;
            const img = document.getElementById("lightboxImg");
            img.style.transform = `translate(0px, 0px) scale(1)`;

            // Update DOM elements
            img.src = figureUrl(fig);
            img.alt = fig.title;
            document.getElementById("lightboxTitle").innerText = fig.title || fig.file;
            document.getElementById("lightboxCaption").innerText = fig.caption || "No caption details provided.";
            document.getElementById("lightboxFilename").innerText = fig.file;
            document.getElementById("lightboxMode").innerText = fig.group;
            document.getElementById("lightboxMetric").innerText = formatMetricName(fig.metric);
            
            const refRow = document.getElementById("lightboxRefRow");
            if (fig.reference) {
                refRow.style.display = "flex";
                document.getElementById("lightboxReference").innerText = fig.reference;
            } else {
                refRow.style.display = "none";
            }

            document.getElementById("lightboxDownloadBtn").href = figureUrl(fig);

            // Toggle active class
            document.getElementById("lightbox").classList.add("active");
            document.body.style.overflow = "hidden"; // Prevent body scrolling
        }

        function closeLightbox() {
            document.getElementById("lightbox").classList.remove("active");
            document.body.style.overflow = "";
        }

        function navigateLightbox(dir) {
            let nextIndex = activeLightboxIndex + dir;
            if (nextIndex < 0) nextIndex = filteredFigures.length - 1;
            if (nextIndex >= filteredFigures.length) nextIndex = 0;
            openLightbox(nextIndex);
        }

        function toggleScrollMode() {
            isScrollMode = !isScrollMode;
            const container = document.getElementById("lightboxContainer");
            const img = document.getElementById("lightboxImg");
            const wrapper = document.getElementById("lightboxImgWrapper");
            const scrollBtn = document.getElementById("scrollModeBtn");

            if (isScrollMode) {
                container.style.overflow = "auto";
                container.style.alignItems = "flex-start";
                container.style.justifyContent = "flex-start";
                
                wrapper.style.maxWidth = "none";
                wrapper.style.maxHeight = "none";
                wrapper.style.width = "auto";
                wrapper.style.height = "auto";
                wrapper.style.cursor = "default";

                img.style.maxWidth = "none";
                img.style.maxHeight = "none";
                img.style.width = "auto";
                img.style.height = "auto";
                img.style.transform = "none";
                
                scrollBtn.classList.add("active");
                // Hide other zoom buttons
                document.getElementById("zoomInBtn").style.display = "none";
                document.getElementById("zoomOutBtn").style.display = "none";
                document.getElementById("zoomResetBtn").style.display = "none";
            } else {
                container.style.overflow = "hidden";
                container.style.alignItems = "center";
                container.style.justifyContent = "center";

                wrapper.style.maxWidth = "85%";
                wrapper.style.maxHeight = "85%";
                wrapper.style.width = "auto";
                wrapper.style.height = "auto";
                wrapper.style.cursor = "grab";

                img.style.maxWidth = "100%";
                img.style.maxHeight = "100%";
                img.style.width = "auto";
                img.style.height = "auto";

                zoomScale = 1;
                translateX = 0;
                translateY = 0;
                applyImageTransform();

                scrollBtn.classList.remove("active");
                // Show other zoom buttons
                document.getElementById("zoomInBtn").style.display = "inline-flex";
                document.getElementById("zoomOutBtn").style.display = "inline-flex";
                document.getElementById("zoomResetBtn").style.display = "inline-flex";
            }
        }

        function toggleInfoPanel() {
            isInfoCollapsed = !isInfoCollapsed;
            const lb = document.getElementById("lightbox");
            const infoBtn = document.getElementById("toggleInfoBtn");

            if (isInfoCollapsed) {
                lb.classList.add("info-collapsed");
                infoBtn.classList.add("active");
            } else {
                lb.classList.remove("info-collapsed");
                infoBtn.classList.remove("active");
            }
        }

        function setUpLightboxEvents() {
            document.getElementById("lightboxClose").addEventListener("click", closeLightbox);
            document.getElementById("lightboxPrev").addEventListener("click", () => navigateLightbox(-1));
            document.getElementById("lightboxNext").addEventListener("click", () => navigateLightbox(1));
            document.getElementById("scrollModeBtn").addEventListener("click", toggleScrollMode);
            document.getElementById("toggleInfoBtn").addEventListener("click", toggleInfoPanel);

            // Keyboard navigation
            document.addEventListener("keydown", (e) => {
                const lb = document.getElementById("lightbox");
                if (!lb.classList.contains("active")) return;

                if (e.key === "Escape") closeLightbox();
                if (e.key === "ArrowLeft") navigateLightbox(-1);
                if (e.key === "ArrowRight") navigateLightbox(1);
            });

            // Close when clicking backdrop container, but not elements
            document.getElementById("lightboxContainer").addEventListener("click", (e) => {
                if (e.target === document.getElementById("lightboxContainer")) {
                    closeLightbox();
                }
            });

            // Zoom In / Out / Reset
            const img = document.getElementById("lightboxImg");

            function applyImageTransform() {
                img.style.transform = `translate(${translateX}px, ${translateY}px) scale(${zoomScale})`;
            }

            document.getElementById("zoomInBtn").addEventListener("click", () => {
                if (isScrollMode) return;
                zoomScale = Math.min(zoomScale + 0.25, 4);
                applyImageTransform();
            });

            document.getElementById("zoomOutBtn").addEventListener("click", () => {
                if (isScrollMode) return;
                zoomScale = Math.max(zoomScale - 0.25, 0.5);
                if (zoomScale <= 1) {
                    translateX = 0;
                    translateY = 0;
                }
                applyImageTransform();
            });

            document.getElementById("zoomResetBtn").addEventListener("click", () => {
                if (isScrollMode) return;
                zoomScale = 1;
                translateX = 0;
                translateY = 0;
                applyImageTransform();
            });

            // Mouse wheel zoom
            document.getElementById("lightboxContainer").addEventListener("wheel", (e) => {
                if (isScrollMode) return; // Native scroll takes over
                e.preventDefault();
                const delta = e.deltaY > 0 ? -0.15 : 0.15;
                zoomScale = Math.max(0.5, Math.min(zoomScale + delta, 4));
                if (zoomScale <= 1) {
                    translateX = 0;
                    translateY = 0;
                }
                applyImageTransform();
            }, { passive: false });

            // Pan/Drag Implementation
            const wrapper = document.getElementById("lightboxImgWrapper");
            wrapper.addEventListener("mousedown", (e) => {
                if (isScrollMode || zoomScale <= 1) return;
                isDragging = true;
                startX = e.clientX - translateX;
                startY = e.clientY - translateY;
            });

            document.addEventListener("mousemove", (e) => {
                if (isScrollMode || !isDragging) return;
                translateX = e.clientX - startX;
                translateY = e.clientY - startY;
                applyImageTransform();
            });

            document.addEventListener("mouseup", () => {
                isDragging = false;
            });

            // Double click to toggle zoom
            wrapper.addEventListener("dblclick", () => {
                if (isScrollMode) return;
                if (zoomScale > 1) {
                    zoomScale = 1;
                    translateX = 0;
                    translateY = 0;
                } else {
                    zoomScale = 2;
                }
                applyImageTransform();
            });
        }

        // Side-by-Side Comparison Logic
        function toggleCompare(event, fig) {
            const checked = event.target.checked;
            
            if (checked) {
                if (selectedForCompare.length >= 4) {
                    alert("You can compare a maximum of 4 figures at once.");
                    event.target.checked = false;
                    return;
                }
                selectedForCompare.push(fig);
            } else {
                selectedForCompare = selectedForCompare.filter(item => item.file !== fig.file);
            }

            updateCompareDrawer();
        }

        function updateCompareDrawer() {
            const drawer = document.getElementById("compareDrawer");
            const countTxt = document.getElementById("compareCountTxt");
            const thumbsContainer = document.getElementById("compareThumbnails");

            countTxt.innerText = `${selectedForCompare.length} item(s) selected`;
            thumbsContainer.innerHTML = "";

            if (selectedForCompare.length > 0) {
                selectedForCompare.forEach(fig => {
                    const img = document.createElement("img");
                    img.className = "compare-thumb";
                    img.src = figureUrl(fig);
                    thumbsContainer.appendChild(img);
                });
                drawer.classList.add("active");
            } else {
                drawer.classList.remove("active");
            }
        }

        // Comparison Portal Events
        function setUpComparisonEvents() {
            document.getElementById("clearCompareBtn").addEventListener("click", () => {
                selectedForCompare = [];
                updateCompareDrawer();
                // Refresh checkboxes
                updateActiveView();
            });

            document.getElementById("launchCompareBtn").addEventListener("click", launchComparison);
            document.getElementById("closeComparePortalBtn").addEventListener("click", () => {
                document.getElementById("comparisonPortal").classList.remove("active");
            });
        }

        function launchComparison() {
            if (selectedForCompare.length === 0) return;

            const grid = document.getElementById("compGrid");
            grid.innerHTML = "";
            grid.className = `comp-grid panels-${selectedForCompare.length}`;

            selectedForCompare.forEach((fig, index) => {
                const panel = document.createElement("div");
                panel.className = "comp-panel";
                
                panel.innerHTML = `
                    <div class="comp-panel-header">
                        <span>${fig.title || fig.file}</span>
                        <span class="meta-tag">${fig.group} | ${formatMetricName(fig.metric)}</span>
                    </div>
                    <div class="comp-panel-body" id="compBody-${index}">
                        <img class="comp-panel-img" id="compImg-${index}" src="${figureUrl(fig)}" alt="" draggable="false">
                        
                        <div class="comp-panel-zoom-overlay">
                            <button class="comp-btn zoom-out" onclick="zoomCompPanel(${index}, -0.2)">
                                ➖
                            </button>
                            <button class="comp-btn reset" onclick="zoomCompPanel(${index}, 0)">
                                🔄
                            </button>
                            <button class="comp-btn zoom-in" onclick="zoomCompPanel(${index}, 0.2)">
                                ➕
                            </button>
                        </div>
                    </div>
                `;
                grid.appendChild(panel);
                
                // Set up panning/zooming on each panel separately
                setUpPanelZoomPan(index);
            });

            document.getElementById("comparisonPortal").classList.add("active");
        }

        // Track zoom and pan for each active comparison panel
        let panelTransforms = {};

        function setUpPanelZoomPan(index) {
            panelTransforms[index] = { scale: 1, x: 0, y: 0, activeDrag: false, startX: 0, startY: 0 };
            const body = document.getElementById(`compBody-${index}`);
            const img = document.getElementById(`compImg-${index}`);

            function updatePanel() {
                const tf = panelTransforms[index];
                img.style.transform = `translate(${tf.x}px, ${tf.y}px) scale(${tf.scale})`;
            }

            // Mouse wheel zoom
            body.addEventListener("wheel", (e) => {
                e.preventDefault();
                const tf = panelTransforms[index];
                const delta = e.deltaY > 0 ? -0.15 : 0.15;
                tf.scale = Math.max(0.5, Math.min(tf.scale + delta, 4));
                if (tf.scale <= 1) {
                    tf.x = 0;
                    tf.y = 0;
                }
                updatePanel();
            }, { passive: false });

            // Drag Pan
            body.addEventListener("mousedown", (e) => {
                const tf = panelTransforms[index];
                if (tf.scale <= 1) return;
                tf.activeDrag = true;
                tf.startX = e.clientX - tf.x;
                tf.startY = e.clientY - tf.y;
            });

            document.addEventListener("mousemove", (e) => {
                const tf = panelTransforms[index];
                if (!tf || !tf.activeDrag) return;
                tf.x = e.clientX - tf.startX;
                tf.y = e.clientY - tf.startY;
                updatePanel();
            });

            document.addEventListener("mouseup", () => {
                const tf = panelTransforms[index];
                if (tf) tf.activeDrag = false;
            });

            // Double click to reset
            body.addEventListener("dblclick", () => {
                const tf = panelTransforms[index];
                if (tf.scale > 1) {
                    tf.scale = 1;
                    tf.x = 0;
                    tf.y = 0;
                } else {
                    tf.scale = 1.8;
                }
                updatePanel();
            });
        }

        // Controls on zoom overlay
        window.zoomCompPanel = function(index, delta) {
            const tf = panelTransforms[index];
            if (!tf) return;

            const img = document.getElementById(`compImg-${index}`);
            if (delta === 0) {
                tf.scale = 1;
                tf.x = 0;
                tf.y = 0;
            } else {
                tf.scale = Math.max(0.5, Math.min(tf.scale + delta, 4));
                if (tf.scale <= 1) {
                    tf.x = 0;
                    tf.y = 0;
                }
            }
            img.style.transform = `translate(${tf.x}px, ${tf.y}px) scale(${tf.scale})`;
        };
    </script>
</body>
</html>
"""

    # Embed manifest JSON string into placeholder
    return template.replace("__MANIFEST_JSON__", manifest_json_str)
