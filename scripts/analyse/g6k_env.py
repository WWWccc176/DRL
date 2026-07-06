#!/usr/bin/env python3
"""
G6K loads its spherical-coding tables via a RELATIVE path:
    ./spherical_coding/sc_<beta>_256.def
so the process CWD must be the g6k root that contains that folder, otherwise
initialize_local() throws std::runtime_error -> terminate/abort.

Import this module FIRST (before `import g6k`) in every process that touches
the sieve.  It chdir()s to the correct root and remembers the original CWD so
you can still resolve dataset paths.
"""

import os
from pathlib import Path

ORIG_CWD = Path.cwd()  # remember where we launched from (for data paths)
G6K_ROOT = None  # set to the dir containing spherical_coding/


def _candidates():
    env = os.environ.get("G6K_ROOT")
    if env:
        yield Path(env).expanduser()
    yield Path.home() / "workspace" / "builds" / "g6k"
    # near the installed g6k package
    try:
        import importlib.util

        spec = importlib.util.find_spec("g6k")
        if spec and spec.origin:
            pkg = Path(spec.origin).resolve().parent  # .../g6k/g6k
            yield pkg
            yield pkg.parent  # .../g6k
            yield pkg.parent.parent
    except Exception:
        pass
    # a couple of common build spots
    yield Path.home() / "g6k"
    yield ORIG_CWD


def ensure_g6k_cwd(verbose=True):
    """chdir to a directory containing spherical_coding/. Returns that Path or None."""
    global G6K_ROOT
    if G6K_ROOT is not None:
        return G6K_ROOT

    seen = set()
    for c in _candidates():
        try:
            c = c.resolve()
        except Exception:
            continue
        if c in seen:
            continue
        seen.add(c)
        if (c / "spherical_coding").is_dir():
            os.chdir(c)
            G6K_ROOT = c
            if verbose:
                print(f"[g6k_env] chdir -> {c}")
            return c

    # last resort: shallow search under the likely build root
    base = Path(
        os.environ.get("G6K_ROOT", Path.home() / "workspace" / "builds" / "g6k")
    ).expanduser()
    if base.is_dir():
        for sub in base.rglob("spherical_coding"):
            if sub.is_dir():
                os.chdir(sub.parent)
                G6K_ROOT = sub.parent
                if verbose:
                    print(f"[g6k_env] chdir -> {sub.parent}")
                return sub.parent

    if verbose:
        print(
            "[g6k_env] WARNING: spherical_coding/ not found; "
            "set G6K_ROOT to the g6k build dir"
        )
    return None


# do it on import
ensure_g6k_cwd()
