"""Portable filename helpers shared by notebooks and diagnostics."""

import hashlib
import re
import unicodedata


def safe_token(value):
    """Return a normalized ASCII token safe for use in a filename."""
    if value is None:
        return ""

    text = unicodedata.normalize("NFKD", str(value).strip())
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def figure_filename(*parts, ext="png", max_length=240):
    """Build a portable, deterministic figure filename.

    Empty parts are ignored, while at least one usable part is required. The
    extension must be a single alphanumeric suffix. Long names are truncated
    and receive a digest of the complete stem so distinct inputs remain
    distinct after truncation.
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

    minimum_length = len("fig_") + 12 + 1 + len(suffix)
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

    stem = "_".join(["fig", *tokens])
    available = max_length - len(suffix) - 1
    if len(stem) > available:
        digest = hashlib.sha256(stem.encode("ascii")).hexdigest()[:12]
        stem = f"{stem[:available - len(digest) - 1].rstrip('_')}_{digest}"

    return f"{stem}.{suffix}"
