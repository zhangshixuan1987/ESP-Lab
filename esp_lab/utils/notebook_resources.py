"""Kernel-local Dask lifecycle for restartable diagnostic notebooks."""
from __future__ import annotations

import dask
from dask.distributed import get_client

from esp_lab.utils.resource_utils import ResourceTracker


def close_notebook_resources(namespace, *, timeout=30):
    """Close tracked datasets, named clients and leftover defaults in this kernel.

    Close only cluster objects owned by these clients, never shut down a remote
    scheduler reached through a client address. Failure is surfaced rather than
    creating a second cluster after incomplete cleanup. No other kernels are touched.
    """
    clients = [namespace.get('client')]
    clusters = [namespace.get('cluster')]
    try:
        clients.append(get_client())
    except ValueError:
        pass
    for client in clients:
        clusters.append(getattr(client, 'cluster', None))
    tracker = namespace.get('workflow_resources')
    if tracker is not None:
        tracker.close()
    seen = set()
    while clients:
        client = clients.pop()
        if client is None or id(client) in seen:
            continue
        seen.add(id(client))
        clusters.append(getattr(client, 'cluster', None))
        client.close(timeout=timeout)
        try:
            remaining = get_client()
        except ValueError:
            remaining = None
        if remaining is not None:
            if id(remaining) in seen:
                raise RuntimeError('Closed Dask client remains active; restart the notebook kernel.')
            clients.append(remaining)
    seen_clusters = set()
    for cluster in clusters:
        if cluster is not None and id(cluster) not in seen_clusters:
            cluster.close(timeout=timeout)
            seen_clusters.add(id(cluster))
    # Remove a scheduler callable bound to any former client, including disabled-Dask runs.
    dask.config.set(scheduler='threads')
    namespace.update(client=None, cluster=None, workflow_resources=None)


def restart_notebook_cluster(namespace, factory):
    """Clean old resources, call factory -> (cluster, client), and route new work.

    Pass ``lambda: (None, None)`` for non-distributed execution. The factory keeps
    each notebook's existing cluster type, worker and memory settings explicit.
    Returns cluster, client, ResourceTracker. Call cleanup at notebook completion.
    """
    close_notebook_resources(namespace)
    cluster, client = factory()
    tracker = ResourceTracker(cluster, client).register_atexit()
    namespace.update(cluster=cluster, client=client, workflow_resources=tracker)
    try:
        dask.config.set(scheduler=client.get if client is not None else 'threads')
        if client is not None:
            info = client.scheduler_info()
            print('Scheduler:', info['address'])
            print('Connected workers:', len(info['workers']))
            print('Dashboard:', client.dashboard_link)
        else:
            print('Dask scheduler: local threads')
    except Exception:
        close_notebook_resources(namespace)
        raise
    return cluster, client, tracker
