#!/usr/bin/env python3
"""Full-basis reduction with a fixed per-file step budget.

Total steps per file = steps_mult * dim  (default 4 * dim).

BKZ / ENUM: budget split across progressive block-size *scales* (decreasing
arithmetic progression, smaller scales get more steps).

ENUM_SIEVE (alias G6K): NO scales. It sweeps pos = 0,1,2,... over the whole
basis exactly like BKZ/ENUM, and repeats until the step budget
(steps_mult * dim, default 4*dim) is exhausted. This keeps the GPU sieve busy
across the whole sweep instead of running a single block once.
"""

import numpy as np
from lattice_metrics import potential


def _pot(be, mid, n):
    gs = np.asarray(be.evaluate_matrix(mid)["gs_log_norms"], float)
    return potential(gs, n)


def make_scales(beta, n, num_scales=4):
    """Increasing progressive block sizes lo..beta (the reduction 'scales')."""
    beta = int(min(beta, n))
    if beta < 6 or num_scales <= 1:
        return [max(2, beta)]
    lo = max(4, beta // 2)
    return sorted({int(round(x)) for x in np.linspace(lo, beta, num_scales)})


def allocate_steps(total, num_scales):
    """Decreasing arithmetic progression summing to `total`.
    weights = k, k-1, ..., 1  ->  smallest scale (index 0) gets the most."""
    if num_scales <= 1:
        return [total]
    weights = list(range(num_scales, 0, -1))  # e.g. [4,3,2,1]
    wsum = sum(weights)
    alloc = [total * w // wsum for w in weights]
    alloc[0] += total - sum(alloc)  # remainder -> smallest scale
    return alloc


def _make_callf(be, mid, method):
    if method == "BKZ":
        return lambda pos, ab: bool(
            be.reduce(mid, "LOCAL_BKZ", ab, pos).get("accepted")
        )
    if method == "ENUM":
        return lambda pos, ab: bool(
            be.reduce(mid, "ORACLE_ENUM_BLOCK", ab, pos).get("accepted")
        )
    if method in ("G6K", "ENUM_SIEVE"):
        from g6k_oracle import g6k_reduce_block

        return lambda pos, ab: g6k_reduce_block(be, mid, pos, ab, threads=1)
    raise ValueError(method)


def reduce_full(be, mid, n, method, beta, total_steps):
    """Run block reductions distributed over scales.

    ENUM_SIEVE/G6K: single block size (=beta), swept pos=0..n-beta, repeated
    until total_steps (= steps_mult*dim) is used up.
    """
    stats = {"accepted": 0, "calls": 0, "steps": 0}
    be.reduce(mid, "LLL", 0, 0)
    if method == "LLL":
        stats["steps"] = 1
        return stats

    # -------- ENUM_SIEVE: sweep like BKZ/ENUM, budget = steps_mult*dim -------
    if method in ("G6K", "ENUM_SIEVE"):
        callf = _make_callf(be, mid, method)
        b = min(int(beta), n)
        total_steps = max(1, int(total_steps))
        max_pos = max(1, n - b + 1)

        pos = 0
        for _ in range(total_steps):
            ab = min(b, n - pos)
            if ab >= 2:
                stats["accepted"] += int(callf(pos, ab))
                stats["calls"] += 1
                stats["steps"] += 1
            pos += 1
            if pos >= max_pos:  # finished one pass -> LLL tidy, restart
                pos = 0
                be.reduce(mid, "LLL", 0, 0)

        be.reduce(mid, "LLL", 0, 0)
        return stats

    # ---------------------------- BKZ / ENUM --------------------------------
    scales = make_scales(beta, n)
    alloc = allocate_steps(int(total_steps), len(scales))

    callf = _make_callf(be, mid, method)

    for b, budget in zip(scales, alloc):
        pos = 0
        for _ in range(budget):
            ab = min(b, n - pos)
            if ab < 2:
                pos = 0
                be.reduce(mid, "LLL", 0, 0)
                ab = min(b, n - pos)
            stats["accepted"] += int(callf(pos, ab))
            stats["calls"] += 1
            stats["steps"] += 1
            pos += 1
            if pos >= n - 1:
                pos = 0
                be.reduce(mid, "LLL", 0, 0)
        be.reduce(mid, "LLL", 0, 0)
    return stats
