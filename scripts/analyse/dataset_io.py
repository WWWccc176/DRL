#!/usr/bin/env python3
"""Read SVP-challenge bases from DRL/dataset by DIMENSION (seed ignored)."""

import re
from pathlib import Path
from lattice_metrics import parse_fplll

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset"
_PAT = re.compile(r"svpchallengedim(\d+)seed([A-Za-z0-9]+)\.txt$")


def list_dims():
    dims = set()
    for p in DATASET.glob("svpchallengedim*seed*.txt"):
        m = _PAT.search(p.name)
        if m:
            dims.add(int(m.group(1)))
    return sorted(dims)


def files_for_dim(dim, limit=None):
    """All files whose name contains dim<dim>, sorted; seed name ignored."""
    out = []
    for p in sorted(DATASET.glob(f"svpchallengedim{dim}seed*.txt")):
        m = _PAT.search(p.name)
        if m and int(m.group(1)) == dim:
            out.append((m.group(2), p))  # (seed_name, path)
    return out[:limit] if limit else out


def read_text(path):
    return Path(path).read_text()


def parse_fplll_file(path):
    return parse_fplll(Path(path).read_text())
