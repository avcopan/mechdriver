import subprocess

import click

from . import subtasks
from .base import Status, check_log, run


@click.group()
def main():
    """AutoMech CLI"""
    pass


@main.command("run")
@click.option(
    "-p", "--path", default=".", show_default=True, help="The job run directory"
)
@click.option("-S", "--safemode-off", is_flag=True, help="Turn off safemode?")
def run_(path: str = ".", safemode_off: bool = False):
    """Run central workflow

    Central Execution script to launch a MechDriver process which will
    parse all of the user-supplied input files in a specified directory, then
    launches all of the requested electronic structure, transport,
    thermochemistry and kinetics calculations via their associated
    sub-drivers.

    The AutoMech directory must contain an `inp/` subdirectory with the following
    required files: run.dat, theory.dat, models.dat, species.csv, mechanism.dat
    """
    run(path=path, safemode_off=safemode_off)


@main.command("check-log")
@click.option(
    "-p", "--path", default=".", show_default=True, help="The path to the log file"
)
def check_log_(path: str = "."):
    """Check an AutoMech log file to see if it succeeded

    The path must point either directly to the log file, or to a directory where the log
    file is named "out.log"
    """
    check_log(path=path, log=True)


@main.group("subtasks")
def subtasks_():
    """Run AutoMech subtasks in parallel"""
    pass


@subtasks_.command("setup")
@click.option(
    "-p", "--path", default=".", show_default=True, help="The job run directory"
)
@click.option(
    "-o",
    "--out-path",
    default=subtasks.SUBTASK_DIR,
    show_default=True,
    help="The output path of the subtask directories",
)
@click.option(
    "-s",
    "--save-path",
    default=None,
    show_default=True,
    help="The save filesystem prefix",
)
@click.option(
    "-r",
    "--run-path",
    default=None,
    show_default=True,
    help="The run filesystem prefix",
)
@click.option(
    "-g",
    "--task-groups",
    default="els,thermo,ktp",
    show_default=True,
    help=(
        "The task groups to set up, as a comma-separated list.\n"
        "Options: els(=els-spc,els-pes), thermo, ktp"
    ),
)
def subtasks_setup_(
    path: str = ".",
    out_path: str = subtasks.SUBTASK_DIR,
    save_path: str | None = None,
    run_path: str | None = None,
    task_groups: str = "els,thermo,ktp",
):
    """Set-up subtasks from a user-supplied AutoMech directory

    The AutoMech directory must contain an `inp/` subdirectory with the following
    required files: run.dat, theory.dat, models.dat, species.csv, mechanism.dat
    """
    subtasks.setup(
        path=path,
        out_path=out_path,
        save_path=save_path,
        run_path=run_path,
        task_groups=task_groups.split(","),
    )


@subtasks_.command("run")
@click.argument("nodes", nargs=-1)
@click.option(
    "-p",
    "--path",
    default=subtasks.SUBTASK_DIR,
    show_default=True,
    help="The subtask directory",
)
@click.option(
    "-a",
    "--activation-hook",
    default=None,
    show_default=True,
    help="An activation hook, to be called using `eval`",
)
@click.option(
    "-s",
    "--statuses",
    default=f"{Status.TBD.value}",
    show_default=True,
    help="A comma-separated list of statuses to run or re-run",
)
@click.option(
    "-t",
    "--tar",
    default=False,
    is_flag=True,
    help="Tar the subtask data and save filesystem after running?",
)
def subtasks_run_(
    nodes: tuple[str, ...],
    path: str = subtasks.SUBTASK_DIR,
    activation_hook: str | None = None,
    statuses: str = f"{Status.TBD.value}",
    tar: bool = False,
):
    """Run subtasks in parallel on an Ad Hoc SSH Cluster

    Use a space-separated list of nodes:
        csed-0008 csed-0009 csed-0010
    or
        csed-00{08..10}

    """
    # For convenience, grab the Pixi activation hook automatically, if using Pixi
    # environment and activation hook is `None`
    result = subprocess.run(["pixi", "shell-hook"], capture_output=True, text=True)
    if activation_hook is None and result.stdout:
        activation_hook = result.stdout

    subtasks.run(
        path=path,
        nodes=nodes,
        activation_hook=activation_hook,
        statuses=list(map(Status, statuses.split(","))),
        tar=tar,
    )


@subtasks_.command("status")
@click.option(
    "-p",
    "--path",
    default=subtasks.SUBTASK_DIR,
    show_default=True,
    help="The subtask directory",
)
@click.option(
    "-c",
    "--check-file",
    default="check.log",
    show_default=True,
    help="Log file for writing paths to be checked",
)
@click.option(
    "-w",
    "--wrap",
    default=18,
    show_default=True,
    help="Wrap to included this many subtask columns per row",
)
def subtasks_status_(
    path: str = subtasks.SUBTASK_DIR, check_file: str = "check.log", wrap: int = 18
):
    """Check the status of running subtasks"""
    subtasks.status(path=path, check_file=check_file, wrap=wrap)
