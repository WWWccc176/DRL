#!/usr/bin/env python3
"""Locate and import the compiled my_project_backend .so from anywhere."""

import sys, importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # .../DRL


def import_backend():
    for cand in (ROOT, ROOT / "src", ROOT / "build", ROOT / "scripts"):
        sp = str(cand)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    return importlib.import_module("my_project_backend")
