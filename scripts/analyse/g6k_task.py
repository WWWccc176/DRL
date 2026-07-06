#!/usr/bin/env python3
"""Run ONE ENUM_SIEVE file-task in an isolated process.

Prints exactly one line:
    RESULT {json}
argv: dim seed path beta steps_mult save_cos(0|1)
"""

import sys
import json
import time
import faulthandler

faulthandler.enable()

from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# MUST be first (before backend/reducers pull in g6k): chdir to g6k root.
import g6k_env

import numpy as np


def main():
    dim, seed, path, beta, steps_mult, save_cos = sys.argv[1:7]
    dim, beta, steps_mult = int(dim), int(beta), int(steps_mult)
    save_cos = save_cos == "1"

    # we chdir'd into the g6k root, so resolve the dataset path against the
    # original launch dir if it isn't already absolute.
    p = Path(path)
    if not p.is_absolute():
        p = g6k_env.ORIG_CWD / p
    p = p.resolve()

    from backend_loader import import_backend
    from dataset_io import read_text
    from lattice_metrics import parse_fplll, profile_metrics, cos_from_basis
    from reducers import reduce_full

    t0 = time.time()
    be = import_backend()
    text = read_text(str(p))
    n = len(parse_fplll(text))
    mid = be.create_matrix(text)

    rs = reduce_full(be, mid, n, "ENUM_SIEVE", beta, steps_mult * n)

    gs = np.asarray(be.evaluate_matrix(mid)["gs_log_norms"], float)
    pm = profile_metrics(gs, n)
    red = parse_fplll(be.dump_matrix(mid))
    cosL, maxc, meanc = cos_from_basis(red)
    be.free_matrix(mid)

    rec = dict(
        dim=dim,
        seed=seed,
        method="ENUM_SIEVE",
        beta=beta,
        steps_budget=steps_mult * n,
        steps_done=rs.get("steps", 0),
        time_s=round(time.time() - t0, 3),
        max_cos=maxc,
        mean_cos=meanc,
        accepted=rs["accepted"],
        calls=rs["calls"],
        **pm,
    )
    if save_cos:
        rec["_cosL"] = np.asarray(cosL, float).tolist()

    print("RESULT " + json.dumps(rec))


if __name__ == "__main__":
    main()

