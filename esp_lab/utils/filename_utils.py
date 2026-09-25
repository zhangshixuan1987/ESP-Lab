"""Portable filename helpers shared by notebooks and diagnostics."""

import re
import unicodedata


def safe_token(value):
    """Return a normalized ASCII token safe for use in a filename."""
    if value is None:
        return ""

    text = unicodedata.normalize("NFKD", str(value).strip())
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def source_init_prefix(source, init_month):
    """Return the ``<source>_init<MM>`` prefix that starts per-initialization files.

    ``source`` is the s2d_diag folder name kept exactly as written (hyphens
    included, e.g. ``CESM-SMYLE``), so every file starts with its folder name
    and the month is its own ``init<MM>`` token rather than glued to the source.
    """
    source = str(source).strip()
    if not source or re.search(r"[^A-Za-z0-9_.-]", source):
        raise ValueError(f"source must be a filename-safe folder name, got {source!r}")
    month = int(init_month)
    if not 1 <= month <= 12:
        raise ValueError(f"init_month must be between 1 and 12, got {init_month!r}")
    return f"{source}_init{month:02d}"


_YEAR_SPAN = re.compile(r"_y(\d{4})-(\d{4})_")


def year_span_token(years):
    """Return the ``y<start>-<end>`` token for an inclusive ``(start, end)`` span."""
    start, end = (int(year) for year in years)
    if start > end:
        raise ValueError(f"year span must satisfy start <= end, got {years!r}")
    return f"y{start}-{end}"


def sst_index_filename(
    source, init_month, years, nens, nlead, *,
    index="ELI", field=None, native=False, seasonal=False,
):
    """Return the fixed name of one hindcast SST-index time series.

    The processed initialization-year span is part of the name, so products
    built for different spans sit side by side instead of overwriting each
    other.  ``index="ELI"`` gives ``<src>_init<MM>_ELI[_native]_y<a>-<b>_N<n>_M<l>[_seas].nc``;
    any other index is a regional SST index read from ``field`` (``TS``/``SST``):
    ``<src>_init<MM>_<field>_y<a>-<b>_N<n>_M<l>_<index>SST_<mon|seas>.nc``.
    """
    stem = f"{source_init_prefix(source, init_month)}"
    counts = f"N{int(nens):02d}_M{int(nlead):02d}"
    span = year_span_token(years)
    if index == "ELI":
        native_tag = "_native" if native else ""
        return f"{stem}_ELI{native_tag}_{span}_{counts}{'_seas' if seasonal else ''}.nc"
    if not field:
        raise ValueError("field is required for regional SST indices")
    return f"{stem}_{field}_{span}_{counts}_{index}SST_{'seas' if seasonal else 'mon'}.nc"


def resolve_year_span_path(path, years=None):
    """Return ``path`` or the narrowest sibling whose year span covers ``years``.

    ``path`` carries a ``_y<start>-<end>_`` token.  An exact file wins; otherwise
    a file that is identical except for a wider span covering ``years`` (default:
    the span in ``path``) is returned, since consumers select their years from
    the longer series.  When nothing covers the request, ``path`` itself is
    returned so the caller can build it.
    """
    from pathlib import Path

    path = Path(path)
    if path.is_file():
        return path
    match = _YEAR_SPAN.search(path.name)
    if match is None:
        return path
    start, end = (
        (int(match.group(1)), int(match.group(2))) if years is None
        else (int(years[0]), int(years[-1]))
    )
    head, tail = path.name[: match.start()], path.name[match.end():]
    candidates = []
    for sibling in path.parent.glob(f"{head}_y*-*_{tail}"):
        found = _YEAR_SPAN.search(sibling.name)
        if (
            found and sibling.name[: found.start()] == head
            and sibling.name[found.end():] == tail
            and int(found.group(1)) <= start and int(found.group(2)) >= end
        ):
            candidates.append((int(found.group(2)) - int(found.group(1)), sibling.name, sibling))
    return min(candidates)[2] if candidates else path


def figure_filename(*parts, ext="png", max_length=240):
    """Build a portable, deterministic figure filename.

    Empty parts are ignored, while at least one usable part is required. The
    extension must be a single alphanumeric suffix. Names longer than
    ``max_length`` are rejected rather than truncated or hashed, so every
    figure keeps a fixed, readable name.
    """
    if not isinstance(ext, str):
        raise TypeError("ext must be a string")
    suffix = ext.strip().lstrip(".").lower()
    if not re.fullmatch(r"[a-z0-9]{1,10}", suffix):
        raise ValueError(
            "ext must contain 1-10 ASCII letters or digits without path separators"
        )
    if isinstance(max_length, bool) or not isinstance(max_length, int):
        raise TypeError("max_length must be an integer")

    minimum_length = len("fig_") + 1 + 1 + len(suffix)
    if not minimum_length <= max_length <= 255:
        raise ValueError(
            f"max_length must be between {minimum_length} and 255 for .{suffix}"
        )

    tokens = []
    for part in parts:
        if part is None or not str(part).strip():
            continue
        token = safe_token(part)
        if token:
            tokens.append(token)

    if not tokens:
        raise ValueError("at least one filename part must contain ASCII letters or digits")

    filename = f"{'_'.join(['fig', *tokens])}.{suffix}"
    if len(filename) > max_length:
        raise ValueError(
            f"figure filename is {len(filename)} characters, over max_length="
            f"{max_length}; use shorter name parts: {filename}"
        )
    return filename
