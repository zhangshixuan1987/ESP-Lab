"""Exclusive file locks that also work on Perlmutter CFS compute nodes.

``fcntl.flock`` works on CFS from login nodes but fails on compute nodes with
errno 524 (ENOTSUPP). When flock is refused, the lock falls back to creating
a directory next to the lock file: ``mkdir`` is atomic on GPFS, so only one
process holds the lock at a time. A lock directory older than ``stale_after``
seconds is treated as left over from a killed job and removed.
"""

from __future__ import annotations

import fcntl
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def _flock_or_none(handle) -> bool:
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
    except OSError:
        return False
    return True


@contextmanager
def exclusive_lock(
    lock_path: str | Path,
    *,
    timeout: float = 1800.0,
    stale_after: float = 3600.0,
    poll_interval: float = 0.5,
) -> Iterator[None]:
    """Hold an exclusive lock on ``lock_path`` for the duration of the block."""
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as handle:
        if _flock_or_none(handle):
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
            return

    lock_dir = lock_path.with_name(lock_path.name + ".d")
    deadline = time.monotonic() + timeout
    while True:
        try:
            lock_dir.mkdir()
            break
        except FileExistsError:
            try:
                age = time.time() - lock_dir.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > stale_after:
                try:
                    (lock_dir / "owner").unlink(missing_ok=True)
                    lock_dir.rmdir()
                except OSError:
                    pass
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Timed out after {timeout:.0f}s waiting for lock {lock_dir}"
                ) from None
            time.sleep(poll_interval)
    try:
        (lock_dir / "owner").write_text(f"{os.uname().nodename} pid={os.getpid()}\n")
        yield
    finally:
        (lock_dir / "owner").unlink(missing_ok=True)
        try:
            lock_dir.rmdir()
        except OSError:
            pass
