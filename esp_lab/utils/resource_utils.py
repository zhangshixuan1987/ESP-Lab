"""Lifecycle management for closeable workflow resources."""

from __future__ import annotations

import atexit
import warnings


class ResourceTracker:
    """Track closeable objects and release them once in reverse order."""

    def __init__(self, *resources):
        self._resources = []
        self._closed = False
        self._registered = False
        self._atexit_callback = self.close
        for resource in resources:
            self.track(resource)

    def track(self, resource):
        """Track a resource and return it for convenient inline use."""
        if resource is not None:
            if self._closed:
                raise RuntimeError("cannot track a resource after the tracker is closed")
            self._resources.append(resource)
        return resource

    def register_atexit(self):
        """Register cleanup at interpreter exit and return this tracker."""
        if not self._registered and not self._closed:
            atexit.register(self._atexit_callback)
            self._registered = True
        return self

    def close(self):
        """Close each unique resource once; repeated calls are harmless."""
        if self._registered:
            atexit.unregister(self._atexit_callback)
            self._registered = False
        if self._closed:
            return

        self._closed = True
        seen = set()
        for resource in reversed(self._resources):
            identity = id(resource)
            if identity in seen:
                continue
            seen.add(identity)
            close = getattr(resource, "close", None)
            if close is None:
                continue
            try:
                close()
            except Exception as exc:
                warnings.warn(
                    f"failed to close {type(resource).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        self._resources.clear()


__all__ = ["ResourceTracker"]
