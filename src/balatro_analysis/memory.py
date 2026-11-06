"""How much memory the dashboard may spend on loaded runs, and what a run costs.

A loaded run takes about 0.8 times the size of its logs once read (more while
it is read), so holding every run of a large runs folder at once (28 runs of
~870 MB) overran a 16 GB Slurm job.  The dashboard keeps loaded runs
within a budget: ``BALATRO_ANALYSIS_RUN_MEMORY`` (in GB) if set, else
``RUN_SHARE`` of the memory this process may use (the cgroup limit, as set by
Slurm or a container, else the machine's RAM).  The rest is left for the copies
that filters make, frames the tabs build and the peak while a run is read.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV = "BALATRO_ANALYSIS_RUN_MEMORY"
RUN_SHARE = 0.6  # of the memory limit, for loaded runs
LOG_FACTOR = 1.0  # a loaded run's memory per byte of its logs (measured: ~0.83), with headroom
_UNLIMITED = 1 << 60  # cgroup files say "max" or a number about this big for no limit


def memory_limit() -> int | None:
    """Bytes this process may use: the smallest of its cgroup limits, Slurm's
    ``--mem`` and the machine's RAM.  None if none of these can be read."""
    limits = [*_cgroup_limits(), *_slurm_limit(), *_physical_memory()]
    return min(limits) if limits else None


def run_budget() -> int | None:
    """Bytes loaded runs may take in all, or None for no limit (unknown memory)."""
    if value := os.environ.get(ENV, "").strip():
        try:
            return int(float(value) * 1e9)
        except ValueError:
            pass
    limit = memory_limit()
    return int(limit * RUN_SHARE) if limit else None


def estimate(log_bytes: int) -> int:
    """A run's memory once loaded, guessed from the size of its logs."""
    return int(log_bytes * LOG_FACTOR)


def process_rss() -> int | None:
    """This process's resident memory in bytes (Linux), else None."""
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def _read_limit(file: Path) -> int | None:
    try:
        text = file.read_text().strip()
    except OSError:
        return None
    if not text.isdigit():  # "max": no limit at this level
        return None
    value = int(text)
    return value if 0 < value < _UNLIMITED else None


def _cgroup_limits() -> list[int]:
    """Memory limits of this process's cgroup and every cgroup above it (Slurm
    may set the limit on the job and leave the step's own cgroup unlimited)."""
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return []
    found = []
    for line in lines:
        _, controllers, rel = line.split(":", 2)
        if controllers == "":  # cgroup v2
            base, name = Path("/sys/fs/cgroup"), "memory.max"
        elif "memory" in controllers.split(","):  # cgroup v1
            base, name = Path("/sys/fs/cgroup/memory"), "memory.limit_in_bytes"
        else:
            continue
        d = base / rel.lstrip("/")
        while True:
            if (value := _read_limit(d / name)) is not None:
                found.append(value)
            if d == base or base not in d.parents:
                break
            d = d.parent
    return found


def _slurm_limit() -> list[int]:
    """Slurm's ``--mem`` (or ``--mem-per-cpu`` times the CPUs), in bytes."""
    env = os.environ
    try:
        if "SLURM_MEM_PER_NODE" in env:
            return [int(env["SLURM_MEM_PER_NODE"]) << 20]
        if "SLURM_MEM_PER_CPU" in env:
            cpus = int(env.get("SLURM_CPUS_ON_NODE") or env.get("SLURM_CPUS_PER_TASK") or 1)
            return [(int(env["SLURM_MEM_PER_CPU"]) << 20) * cpus]
    except ValueError:
        pass
    return []


def _physical_memory() -> list[int]:
    try:
        return [os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")]
    except (ValueError, OSError, AttributeError):
        return []
