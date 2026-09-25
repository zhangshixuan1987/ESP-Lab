import pytest
import dask
from esp_lab.utils import notebook_resources as lifecycle


@pytest.fixture
def fake_kernel(monkeypatch):
    active=[]
    clusters=[]
    class Cluster:
        def __init__(self): self.closed=False; clusters.append(self)
        def close(self, timeout=30): self.closed=True
    class Client:
        def __init__(self, owned=True):
            self.cluster=Cluster() if owned else None
            active.append(self)
        def close(self,timeout=30):
            if self in active:active.remove(self)
        def get(self,*args,**kwargs): pass
        def scheduler_info(self):return {'address':'new', 'workers':{'w':{}}}
        dashboard_link='new-dashboard'
    def current():
        if not active:raise ValueError('no client')
        return active[-1]
    monkeypatch.setattr(lifecycle,'get_client',current)
    with dask.config.set(scheduler='threads'):
        yield active,clusters,Client


@pytest.mark.parametrize('count',[0,1,2])
def test_restarts_close_all_prior_clusters(fake_kernel,count):
    active,clusters,Client=fake_kernel
    ns={}
    for _ in range(count):
        c=Client();ns.update(client=c,cluster=c.cluster)
    tracker=lifecycle.ResourceTracker();tracker.close()
    ns['workflow_resources']=tracker  # already-closed tracker must not bypass cleanup
    old_clusters=list(clusters)
    def create():
        assert not active and all(c.closed for c in old_clusters)
        c=Client();return c.cluster,c
    cluster,client,tracker=lifecycle.restart_notebook_cluster(ns,create)
    assert active==[client] and dask.config.get('scheduler')==client.get
    lifecycle.close_notebook_resources(ns)
    assert cluster.closed and not active
    assert dask.config.get('scheduler')=='threads'


def test_disabled_distributed_clears_stale_scheduler(fake_kernel):
    active,clusters,Client=fake_kernel
    c=Client();dask.config.set(scheduler=c.get)
    ns={'client':c,'cluster':c.cluster}
    cluster,client,tracker=lifecycle.restart_notebook_cluster(ns,lambda:(None,None))
    assert cluster is client is None and not active
    assert dask.config.get('scheduler')=='threads'
    lifecycle.close_notebook_resources(ns)


def test_cleanup_failure_does_not_start_cluster(fake_kernel):
    active,clusters,Client=fake_kernel
    c=Client()
    def fail(**kw):raise RuntimeError('close failed')
    c.close=fail
    with pytest.raises(RuntimeError,match='close failed'):
        lifecycle.restart_notebook_cluster({'client':c},lambda:pytest.fail('must not start'))


def test_remote_client_does_not_shutdown_scheduler(fake_kernel):
    active,clusters,Client=fake_kernel
    c=Client(owned=False)
    lifecycle.close_notebook_resources({'client':c})
    assert not active and not clusters
