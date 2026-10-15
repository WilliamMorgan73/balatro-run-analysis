"""Native folder picker.

The dashboard runs locally, so the server can open an OS dialog on the same
desktop. Tries KDE's kdialog, then GTK's zenity, then Tk.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

_TK_SCRIPT = """
import sys, tkinter as tk
from tkinter import filedialog
root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
print(filedialog.askdirectory(initialdir=sys.argv[1], title=sys.argv[2]) or "")
"""


class PickerUnavailable(RuntimeError):
    pass


def _commands(initial: str, title: str) -> list[list[str]]:
    """Dialog commands to try, best match for the desktop first."""
    native = {
        "kdialog": ["--title", title, "--getexistingdirectory", initial],
        "zenity": ["--file-selection", "--directory", "--title", title, f"--filename={initial}/"],
    }
    order = ["kdialog", "zenity"]
    if "KDE" not in os.environ.get("XDG_CURRENT_DESKTOP", ""):
        order.reverse()
    cmds = [[exe, *native[tool]] for tool in order if (exe := shutil.which(tool))]
    cmds.append([sys.executable, "-c", _TK_SCRIPT, initial, title])
    return cmds


def pick_directory(initial: str = "~", title: str = "Choose runs folder") -> str | None:
    """Open a folder dialog. Returns the chosen path, or None if cancelled.

    Raises PickerUnavailable if no dialog could be shown at all.
    """
    initial = os.path.expanduser(initial)
    if not os.path.isdir(initial):
        initial = os.path.expanduser("~")
    for cmd in _commands(initial, title):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            path = result.stdout.strip()
            return path or None
        if result.returncode == 1 and not result.stderr.strip():
            return None  # user pressed Cancel
    raise PickerUnavailable("No folder dialog available (tried kdialog, zenity, tkinter)")
