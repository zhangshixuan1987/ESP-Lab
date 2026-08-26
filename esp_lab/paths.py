"""Shared filesystem paths for ESP-Lab diagnostics.

Path-building helpers require an explicit root so machine-specific
configuration stays at workflow and script entry points.
"""

from __future__ import annotations

from pathlib import Path


def normalize_root(root: str | Path, *, name: str = "root") -> Path:
    """Return a usable root path and reject missing or blank configuration."""
    if root is None:
        raise ValueError(f"{name} must be explicitly configured")
    if isinstance(root, str) and not root.strip():
        raise ValueError(f"{name} must not be blank")
    return Path(root).expanduser()


def join_below(root: str | Path, *parts: str | Path) -> Path:
    """Join relative components below ``root`` without allowing path escape."""
    path = normalize_root(root)
    for part in parts:
        if isinstance(part, str) and not part.strip():
            raise ValueError("Path components must not be blank")
        component = Path(part)
        if component.is_absolute() or ".." in component.parts:
            raise ValueError(f"Unsafe path component: {str(part)!r}")
        path = path / component
    return path

def leadtime_acc_dir(
    source: str,
    *parts: str | Path,
    root: str | Path,
) -> Path:
    """Return a source-first lead-time ACC diagnostic directory.

    Examples are ``<root>/<case>/leadtime_acc/inputs/land/H2OSOI`` and
    ``<root>/<case>/leadtime_acc/skill/atm/TREFHT``. Passing ``root`` keeps
    notebook and command-line configurations explicit and machine portable.
    """
    return join_below(root, source, "leadtime_acc", *parts)


def diagnostic_dir(
    source: str,
    diagnostic: str,
    *parts: str | Path,
    root: str | Path,
) -> Path:
    """Return ``<root>/<source>/<diagnostic>/<parts...>``."""
    return join_below(root, source, diagnostic, *parts)


def multimodel_diagnostic_dir(
    diagnostic: str,
    *parts: str | Path,
    root: str | Path,
) -> Path:
    """Return a canonical directory for cross-experiment diagnostics."""
    return diagnostic_dir("multimodel", diagnostic, *parts, root=root)


def figure_output_dir(
    diagnostic: str,
    *parts: str | Path,
    root: str | Path,
) -> Path:
    """Return a diagnostic-specific directory below the public figure root."""
    return join_below(root, diagnostic, *parts)


__all__ = [
    "normalize_root",
    "join_below",
    "leadtime_acc_dir",
    "diagnostic_dir",
    "multimodel_diagnostic_dir",
    "figure_output_dir",
]
