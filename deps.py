# SPDX-License-Identifier: GPL-3.0-or-later
"""NVIDIA Warp (GPU computation) as an optional dependency.

The library (~150 MB, includes the CUDA runtime) is too large to ship inside
the add-on, so it is installed on demand with pip from PyPI into the
add-on's user folder (no admin rights needed, Blender's own packages are not
touched: --no-deps keeps Blender's NumPy)."""

import importlib
import importlib.util
import os
import subprocess
import sys
import threading
import time

import bpy

WARP_SPEC = "warp-lang==1.17.0"

state = {"running": False, "done": False, "ok": False, "log": "", "error": "", "t0": 0.0}


def site_dir(create=False):
    try:
        return bpy.utils.extension_path_user(__package__, path="site-packages", create=create)
    except Exception:
        return None


def add_to_path():
    d = site_dir()
    if d and os.path.isdir(d) and d not in sys.path:
        sys.path.append(d)
        importlib.invalidate_caches()


_cache = {"available": None}


def warp_available(refresh=False):
    if refresh or _cache["available"] is None:
        add_to_path()
        _cache["available"] = importlib.util.find_spec("warp") is not None
    return _cache["available"]


def online_allowed():
    return bool(getattr(bpy.app, "online_access", True))


def _run(cmd, timeout):
    kw = {}
    if sys.platform == "win32":
        kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    env = dict(os.environ)
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, **kw)


def install_warp(target):
    """Blocking install into `target` (may run in a thread: no bpy calls)."""
    state.update(running=True, done=False, ok=False, log="", error="", t0=time.time())
    try:
        if not target:
            raise RuntimeError("cannot get the add-on folder")
        py = sys.executable
        r = _run([py, "-m", "pip", "--version"], 120)
        if r.returncode != 0:
            state["log"] = "Installing pip…"
            r = _run([py, "-m", "ensurepip", "--upgrade"], 600)
            if r.returncode != 0:
                r = _run([py, "-m", "ensurepip", "--upgrade", "--user"], 600)
            if r.returncode != 0:
                raise RuntimeError("pip is not available: " + (r.stderr or r.stdout).strip()[-300:])
        state["log"] = "Downloading NVIDIA Warp (~150 MB)…"
        r = _run([py, "-m", "pip", "install", "--upgrade", "--no-deps", "--only-binary=:all:",
                  "--target", target, WARP_SPEC], 3600)
        if r.returncode != 0:
            raise RuntimeError((r.stderr or r.stdout).strip()[-400:])
        if target not in sys.path:
            sys.path.append(target)
        importlib.invalidate_caches()
        _cache["available"] = importlib.util.find_spec("warp") is not None
        if not _cache["available"]:
            raise RuntimeError("the package is installed but cannot be imported")
        state["ok"] = True
    except Exception as exc:
        state["error"] = str(exc)
    finally:
        state["running"] = False
        state["done"] = True


def start_install():
    state.update(running=True, done=False, log="", error="", t0=time.time())
    t = threading.Thread(target=install_warp, args=(site_dir(create=True),),
                         name="sand_warp_install", daemon=True)
    t.start()
    return t
