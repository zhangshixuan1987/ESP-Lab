import pytest

from esp_lab.utils import dask_util


def test_client_failure_closes_partially_started_local_cluster(monkeypatch):
    clusters = []

    class Cluster:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            clusters.append(self)

        def close(self):
            self.closed = True

    def fail_client(cluster):
        raise RuntimeError("client startup failed")

    monkeypatch.setattr("dask.distributed.LocalCluster", Cluster)
    monkeypatch.setattr(dask_util, "Client", fail_client)

    with pytest.raises(RuntimeError, match="client startup failed"):
        dask_util.get_cluster_client(
            dask_util.DaskConfig(cluster_type="local", workers=2, memory_limit="1GB")
        )

    assert len(clusters) == 1
    assert clusters[0].closed
    assert clusters[0].kwargs["memory_limit"] == "1GB"
