from dataclasses import dataclass
from typing import Optional, Tuple
import dask
from dask.distributed import Client

@dataclass
class DaskConfig:
    cluster_type: str = "local"   # local, slurm, casper_pbs, none
    workers: int = 4
    cores: int = 4
    memory: str = "16GB"
    memory_limit: str = "4GB"     # memory limit per local worker process
    walltime: str = "02:00:00"
    queue: Optional[str] = None
    project: Optional[str] = None
    interface: Optional[str] = None


def get_cluster_client(cfg: DaskConfig) -> Tuple[object | None, object | None]:
    """
    Create a Dask cluster + client based on config.
    """

    dask.config.set({'array.slicing.split_large_chunks': True})

    if cfg.cluster_type in (None, "none"):
        return None, None

    # -----------------------
    # Local (laptop / debug)
    # -----------------------
    if cfg.cluster_type == "local":
        from dask.distributed import LocalCluster
        import socket
        import os

        # Check if running on login or shared Jupyter node
        is_login = False
        if "SLURM_JOB_ID" not in os.environ and "PBS_JOBID" not in os.environ:
            hostname = socket.gethostname().lower()
            if "login" in hostname or os.environ.get("NERSC_HOST") is not None:
                is_login = True

        local_workers = cfg.workers
        if is_login:
            print("\n" + "!" * 80)
            print("WARNING: You are running a Local Dask Cluster on a shared login/Jupyter node.")
            print("This can easily exceed memory/CPU limits and crash your Jupyter kernel.")
            print("We highly recommend launching Jupyter on an Exclusive Perlmutter Compute Node.")
            print("!" * 80)
            if local_workers > 4:
                print(
                    "WARNING: Scaling down local workers "
                    f"from {local_workers} to 4 for kernel stability."
                )
                local_workers = 4

        cluster = LocalCluster(
            n_workers=local_workers,
            threads_per_worker=1,
            memory_limit=cfg.memory_limit,
        )
        client = Client(cluster)
        return cluster, client

    # -----------------------
    # SLURM (Perlmutter)
    # -----------------------
    if cfg.cluster_type == "slurm":
        from dask_jobqueue import SLURMCluster
        import subprocess

        # Auto-detect project group if not set
        project = cfg.project
        if not project:
            try:
                groups_output = subprocess.check_output(
                    ["groups"], text=True
                ).split()
                if "e3sm" in groups_output:
                    project = "e3sm"
                elif "e3smdata" in groups_output:
                    project = "e3smdata"
                else:
                    m_groups = [
                        group
                        for group in groups_output
                        if group.startswith("m") and len(group) > 1
                    ]
                    if m_groups:
                        project = m_groups[0]
            except Exception as exc:
                print(
                    "Warning: Could not auto-detect NERSC billing project group: "
                    f"{exc}"
                )

        # Auto-detect queue if not set
        queue = cfg.queue or "regular"

        print(f"Initializing SLURMCluster on NERSC: queue={queue}, account={project}")

        cluster = SLURMCluster(
            cores=cfg.cores,
            processes=1,
            memory=cfg.memory,
            walltime=cfg.walltime,
            queue=queue,       # e.g., "regular", "shared"
            account=project,   # e3sm
        )
        cluster.scale(cfg.workers)

        client = Client(cluster)
        return cluster, client

    # -----------------------
    # Casper (NCAR)
    # -----------------------
    if cfg.cluster_type == "casper_pbs":
        from dask_jobqueue import PBSCluster

        cluster = PBSCluster(
            cores=1,
            memory="20GB",
            processes=1,
            queue="casper",
            resource_spec="select=1:ncpus=1:mem=20GB",
            project=cfg.project or "NCGD0011",
            walltime=cfg.walltime,
            interface=cfg.interface or "ib0",
        )
        cluster.scale(cfg.workers)

        client = Client(cluster)
        return cluster, client

    raise ValueError(f"Unknown cluster_type: {cfg.cluster_type}")


def close_cluster(cluster=None, client=None):
    """Safely close Dask resources."""
    try:
        if client is not None:
            client.close()
    except Exception:
        pass

    try:
        if cluster is not None:
            cluster.close()
    except Exception:
        pass


# -----------------------
# Optional helpers
# -----------------------

def maybe_persist(obj, do_persist=True):
    return obj.persist() if do_persist else obj


def maybe_load(obj, do_load=True):
    return obj.load() if do_load else obj
