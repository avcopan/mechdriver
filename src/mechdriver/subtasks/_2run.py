"""Standalone script to run AutoMech subtasks in parallel on an Ad Hoc SSH Cluster"""

import itertools
import math
import os
import shutil
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

import networkx as nx
import pint
import yaml
from hyperqueue import Client, Job
from hyperqueue.ffi.protocol import ResourceRequest

from ..base import Status
from ._0setup import INFO_FILE, SUBTASK_DIR, SubtasksInfo, Task
from ._1status import log_paths_with_check_results, parse_subtask_status

SCRIPT_DIR = Path(__file__).parent / "scripts"
RUN_SCRIPT = str(SCRIPT_DIR / "run_adhoc.sh")
HOME = Path(os.environ.get("HOME"))
HQ_PATH = HOME / ".hq-server" / "hq-current"


def run_multiple(
    paths: Sequence[str | Path] = (".",),
    dir_name: str = SUBTASK_DIR,
    statuses: Sequence[Status] = (Status.TBD,),
    auto_config_flags: str | None = None,
) -> None:
    """Run multiple sets of subtasks in parallel using HyperQueue.

    Assumes the subtasks were set up at this path using `automech subtasks setup`

    :param paths: The paths where the AutoMech subtasks were set up
    :param dir_name: The subtask directory name
    :param hyperqueue_path: The path to the HyperQueue server directory
    :param statuses: A comma-separated list of status to run or re-run
    :param auto_config: Automatically configure HyperQueue with these sbatch/qsub flags
    """
    if auto_config_flags is not None:
        start_hyperqueue_server()

    # Set up the HyperQueue client
    client = Client(HQ_PATH)

    # Set up the HyperQueue job workflow
    job = Job()

    for path in paths:
        job = setup_job(
            path=path,
            dir_name=dir_name,
            statuses=statuses,
            auto_config_flags=auto_config_flags,
            job=job,
        )

    submitted_job = client.submit(job)
    client.wait_for_jobs([submitted_job])


def setup_job(
    path: str = ".",
    dir_name: str = SUBTASK_DIR,
    statuses: Sequence[Status] = (Status.TBD,),
    auto_config_flags: str | None = None,
    job: Job | None = None,
) -> None:
    """Run subtasks in parallel using HyperQueue.

    Assumes the subtasks were set up at this path using `automech subtasks setup`

    :param path: The path where the AutoMech subtasks were set up
    :param dir_name: The subtask directory name
    :param statuses: A comma-separated list of status to run or re-run
    :param auto_config: Automatically configure HyperQueue with these sbatch/qsub flags
    :param job: Append to an existing job
    """
    path = Path(path).resolve()
    dir_path = path / dir_name
    info_file = dir_path / INFO_FILE
    info = SubtasksInfo.model_validate(yaml.safe_load(info_file.read_text()))

    if auto_config_flags is not None:
        all_tasks = list(itertools.chain.from_iterable(info.task_groups))
        mem = max(t.mem for t in all_tasks)
        cpus = max(t.nprocs for t in all_tasks)
        add_hyperqueue_allocation(mem=mem, cpus=cpus, flags=auto_config_flags)

    # Make sure the run and save directories exist
    info.run_path.mkdir(exist_ok=True)
    info.save_path.mkdir(exist_ok=True)

    # Set up the HyperQueue job workflow
    job = job or Job()

    # For now, just do this for the first task group
    job_dct = {}
    dep_graph = dependency_graph(info.task_groups)
    for group_idx, task_group in enumerate(info.task_groups):
        for task_idx, task in enumerate(task_group):
            for subtask in task.subtasks:
                # Determine dependencies from dependency graph
                job_key = (group_idx, task_idx, subtask.key)
                dep_job_keys = dep_graph.predecessors(job_key)
                deps = list(map(job_dct.get, dep_job_keys))

                # Create the job
                subtask_path = dir_path / subtask.path
                stem = "out"
                subtask_job = job.program(
                    ["automech", "run", "-p", str(subtask_path), "-r", stem],
                    cwd=subtask_path,
                    stdout=subtask_path / f"{stem}.log",
                    stderr=subtask_path / f"{stem}.log",
                    deps=deps,
                    resources=ResourceRequest(
                        cpus=task.nprocs, resources={"mem": memory_mib(task.mem)}
                    ),
                )

                # Add the job to the job dictionary
                job_dct[job_key] = subtask_job

    return job


def dependency_graph(task_groups: Sequence[Sequence[Task]]) -> nx.DiGraph:
    """Create a subtask dependency graph from task groups."""
    # Store the task count and the subtasks keys for each group, for determining
    # group-level dependencies
    group_dct = {
        group_idx: (len(tasks) - 1, [s.key for s in tasks[0].subtasks])
        for group_idx, tasks in enumerate(task_groups)
    }
    group_idx0_dct = {
        group_idx: next(
            (i for i in reversed(range(group_idx)) if all(group_dct[i])), None
        )
        for group_idx, _ in enumerate(task_groups)
    }

    # Build the dependency graph
    dep_graph = nx.DiGraph()
    for group_idx, tasks in enumerate(task_groups):
        # Add dependencies within the group
        for task_idx, task in enumerate(tasks):
            dep_task_idx = task_idx - 1 if task_idx > 0 else None
            for subtask in task.subtasks:
                dep_graph.add_node((group_idx, task_idx, subtask.key))
                if dep_task_idx is not None:
                    dep_graph.add_edge(
                        (group_idx, dep_task_idx, subtask.key),
                        (group_idx, task_idx, subtask.key),
                    )

        # Add dependencies between groups
        group_idx0 = group_idx0_dct.get(group_idx)
        if group_idx0 is not None:
            task_idx0, subtask_keys0 = group_dct.get(group_idx0)
            _, subtask_keys = group_dct.get(group_idx)
            task_idx = 0
            for key0, key in itertools.product(subtask_keys0, subtask_keys):
                dep_graph.add_edge(
                    (group_idx0, task_idx0, key0), (group_idx, task_idx, key)
                )

    assert nx.is_weakly_connected(dep_graph), (
        "Dependency graph must not be disconnected:\n"
        f"group_idx0_dct = {group_idx0_dct}\ngroup_dct={group_dct}"
    )

    return dep_graph


def memory_mib(mem: int) -> int:
    """Convert memory in GB to MiB.

    :param mem: Memory (GB)
    :return: Memory (MiB)
    """
    return math.ceil(pint.Quantity(mem, "GB").m_as("MiB"))


def start_hyperqueue_server() -> None:
    """Re-start HyperQueue server."""
    print("Re-starting HyperQueue server...")
    subprocess.Popen(
        ["hq", "server", "start"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )
    # Wait up to 1 second for the file to appear
    for _ in range(10):
        time.sleep(0.1)
        if os.path.exists(HQ_PATH):
            break
    assert os.path.exists(HQ_PATH), f"Could not start server at {HQ_PATH}"


def add_hyperqueue_allocation(mem: int, cpus: int, flags: str) -> None:
    """Create a HyperQueue allocation.

    :param mem: Memory (GB)
    :param nprocs: Number of processers
    :param flags: Additional flags for sbatch/qsub
    """
    print(f"Adding HyperQueue allocation with mem={mem}GB, cpus={cpus}, flags={flags}")
    alloc_args = [
        "--time-limit",
        "1h",
        f"--cpus={cpus}",
        f"--resource=mem=sum({memory_mib(mem)})",
    ]

    if shutil.which("sbatch"):
        print("Detected SLURM on system. HyperQueue allocation command:")
        args = [
            "hq",
            "alloc",
            "add",
            "slurm",
            *alloc_args,
            "--",
            f"--mem={mem}G",
            "--ntasks=1",
            *flags.split(),
        ]
        print(" ".join(args))
        subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    elif shutil.which("qsub"):
        print("Detected PBS on system. HyperQueue allocation command:")
        raise NotImplementedError("PBS auto-configuration not yet implemented.")
