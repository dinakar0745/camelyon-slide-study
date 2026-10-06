"""Keeps the project folder under a fixed size.

Every script that downloads or writes something large calls reserve() first. If the write
would take the project past config.BUDGET_GB, or leave the drive nearly full, reserve()
raises BudgetError and nothing is written.

Run `python budget.py` to see what is using the space.
"""
import os
import shutil
from pathlib import Path

import config

GB = 1024 ** 3


class BudgetError(RuntimeError):
    """The requested write does not fit in the disk budget."""


def folder_bytes(path: Path) -> int:
    """Total size of the files under `path`. Missing folders count as zero."""
    total = 0
    stack = [str(path)]
    while stack:
        try:
            entries = list(os.scandir(stack.pop()))
        except (FileNotFoundError, NotADirectoryError):
            continue
        for e in entries:
            try:
                if e.is_dir(follow_symlinks=False):
                    stack.append(e.path)
                elif e.is_file(follow_symlinks=False):
                    total += e.stat(follow_symlinks=False).st_size
            except FileNotFoundError:  # deleted while we were looking
                pass
    return total


def used_bytes() -> int:
    return folder_bytes(config.ROOT)


def reserve(n_bytes: int, what: str) -> None:
    """Raise BudgetError unless `n_bytes` more can be written safely."""
    cap = int(config.BUDGET_GB * GB)
    used = used_bytes()
    if used + n_bytes > cap:
        raise BudgetError(
            f"{what} needs {n_bytes / GB:.2f} GB but the project is at {used / GB:.2f} GB "
            f"of its {config.BUDGET_GB} GB budget."
        )
    free = shutil.disk_usage(config.ROOT).free
    if free - n_bytes < config.DISK_FLOOR_GB * GB:
        raise BudgetError(
            f"{what} needs {n_bytes / GB:.2f} GB but the drive has only {free / GB:.2f} GB free "
            f"(floor: {config.DISK_FLOOR_GB} GB)."
        )


def report() -> str:
    lines = []
    parts = [config.SLIDES, config.CACHE, config.TILES, config.EMBEDDINGS, config.ANNOTATIONS]
    listed = 0
    for p in parts:
        b = folder_bytes(p)
        listed += b
        lines.append(f"  {p.relative_to(config.ROOT).as_posix():<20} {b / GB:8.2f} GB")
    total = used_bytes()
    lines.append(f"  {'everything else':<20} {(total - listed) / GB:8.2f} GB")
    lines.append(f"  {'total':<20} {total / GB:8.2f} GB of {config.BUDGET_GB} GB")
    lines.append(f"  drive free           {shutil.disk_usage(config.ROOT).free / GB:8.2f} GB")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
