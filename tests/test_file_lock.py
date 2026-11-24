import errno
import multiprocessing
import os
import time

import pytest

from esp_lab.utils import file_lock
from esp_lab.utils.file_lock import exclusive_lock


def _refuse_flock(monkeypatch):
    def refuse(handle, operation):
        raise OSError(524, "Unknown error 524")

    monkeypatch.setattr(file_lock.fcntl, "flock", refuse)


def test_flock_path_releases(tmp_path):
    lock = tmp_path / "a.lock"
    with exclusive_lock(lock):
        assert lock.exists()
    with exclusive_lock(lock, timeout=1):
        pass
    assert not (tmp_path / "a.lock.d").exists()


def test_fallback_creates_and_removes_lock_dir(tmp_path, monkeypatch):
    _refuse_flock(monkeypatch)
    lock = tmp_path / "a.lock"
    with exclusive_lock(lock):
        assert (tmp_path / "a.lock.d" / "owner").is_file()
    assert not (tmp_path / "a.lock.d").exists()


def test_fallback_times_out_while_held(tmp_path, monkeypatch):
    _refuse_flock(monkeypatch)
    lock = tmp_path / "a.lock"
    with exclusive_lock(lock):
        with pytest.raises(TimeoutError):
            with exclusive_lock(lock, timeout=0.3, poll_interval=0.05):
                pass


def test_fallback_removes_stale_lock_dir(tmp_path, monkeypatch):
    _refuse_flock(monkeypatch)
    lock_dir = tmp_path / "a.lock.d"
    lock_dir.mkdir()
    (lock_dir / "owner").write_text("dead job\n")
    old = time.time() - 7200
    os.utime(lock_dir, (old, old))
    with exclusive_lock(tmp_path / "a.lock", timeout=1, stale_after=3600):
        pass
    assert not lock_dir.exists()


def test_fallback_releases_after_exception(tmp_path, monkeypatch):
    _refuse_flock(monkeypatch)
    lock = tmp_path / "a.lock"
    with pytest.raises(RuntimeError):
        with exclusive_lock(lock):
            raise RuntimeError("boom")
    assert not (tmp_path / "a.lock.d").exists()


def _increment(path, lock, refuse):
    if refuse:
        def raise_524(handle, operation):
            raise OSError(errno.ENOTSUP, "refused")

        file_lock.fcntl.flock = raise_524
    for _ in range(20):
        with exclusive_lock(lock, poll_interval=0.001):
            value = int(path.read_text())
            time.sleep(0.001)
            path.write_text(str(value + 1))


@pytest.mark.parametrize("refuse", [False, True])
def test_read_modify_write_is_serialized(tmp_path, refuse):
    counter = tmp_path / "counter"
    counter.write_text("0")
    ctx = multiprocessing.get_context("fork")
    workers = [
        ctx.Process(target=_increment, args=(counter, tmp_path / "c.lock", refuse))
        for _ in range(4)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(60)
        assert worker.exitcode == 0
    assert counter.read_text() == "80"
