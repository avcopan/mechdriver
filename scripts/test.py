#!/usr/bin/env python
"""Local testing CLI."""

import contextlib
import os
import socket
import subprocess
import warnings
from pathlib import Path

import click

import mechdriver
import test_utils as tu
from test_utils import Test


@click.group()
def main():
    """Route to test subcommand."""
    pass


@main.command("status")
def status():
    """Check the status of local tests."""
    mechdriver.subtasks.status_multiple(Test.paths())


@main.command("local")
@click.option(
    "-f",
    "--auto-config-flags",
    default=None,
    required=False,
    help="Automatically configure HyperQueue with these sbatch/qsub flags.",
)
def local(auto_config_flags: str | None = None):
    """Run local tests on one or more nodes.

    Runs hidden local_

    :param nodes: A list of nodes
    """
    print("Process ID:", os.getpid())
    print("Host name:", socket.gethostname())

    if auto_config_flags is None:
        msg = (
            "\nWARNING: Running without -f requires manual HyperQueue configuration."
            "\nMake sure you have a server running with the appropriate workers."
        )
        warnings.warn(msg, stacklevel=1)

    test_paths = tu.setup_tests()
    mechdriver.subtasks.setup_multiple(test_paths)
    mechdriver.subtasks.run_multiple(test_paths, auto_config_flags=auto_config_flags)
    tu.wrap_up_tests(from_archive=False, allow_override=False)


@main.command("sign")
def sign():
    """Sign off on local tests."""
    tu.wrap_up_tests(from_archive=True, allow_override=True)


# Helper function for manually creating workers on Bebop/CSED
# (Not intended for use on other clusters, but could be generalized.)
WORKER_DIR = Path(".workers")
WORKER_NAME = "worker-{mem:d}g-{cpus:d}p-{index:02d}"
WORKER_SCRIPT = """
#!/bin/bash
#PBS -l select=1:host={host}
#PBS -N {name}
#PBS -l walltime=01:00:00
#PBS -q {queue} -A {account}

hq worker start \\
    --idle-timeout "10m" \\
    --manager "pbs" \\
    --cpus "{cpus:d}" \\
    --resource "mem=sum({mem_mib})" \\
    --on-server-lost "finish-running" \\
    --time-limit "1h"
"""


@main.command("create-workers")
@click.argument("host")
@click.option("-n", "--number", default=1, help="Number of workers to create")
@click.option("-m", "--mem", default=20, help="Memory (GB)")
@click.option("-c", "--cpus", default=1, help="CPU count (#)")
@click.option("-q", "--queue", default="csed", help="Queue name")
@click.option("-A", "--account", default="g-CSE", help="Account name")
def create_workers(
    host: str,
    number: int = 1,
    mem: int = 20,
    cpus: int = 1,
    queue: str = "csed",
    account: str = "g-CSE",
):
    """Create workers running on specific nodes.

    This can be used for manual configuration of the HyperQueue server.
    """
    WORKER_DIR.mkdir(exist_ok=True)
    with contextlib.chdir(WORKER_DIR):
        for index in range(number):
            name = WORKER_NAME.format(mem=mem, cpus=cpus, index=index)
            script = Path(f"{name}.sh")
            script_text = WORKER_SCRIPT.format(
                host=host,
                name=name,
                cpus=cpus,
                mem_mib=mechdriver.subtasks.memory_mib(mem),
                queue=queue,
                account=account,
            )
            script.write_text(script_text)
            subprocess.run(["qsub", str(script)])


if __name__ == "__main__":
    main()
