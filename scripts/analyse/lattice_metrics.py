#!/usr/bin/env python3
"""Parsing + quality metrics (row-vector convention, matrix = [[..],[..]])."""

import math
import numpy as np

from math import lgamma, log, pi


def parse_fplll(s):
    out = []
    for line in s.strip().splitlines():
        line = line.strip().lstrip("[").rstrip("]").strip()
        if line:
            out.append([int(x) for x in line.split()])
    return out


def log_gaussian_heuristic(gs_log, n):
    """log of the Gaussian-Heuristic length:
    GH = vol^{1/n} * Gamma(n/2+1)^{1/n} / sqrt(pi),  vol = prod ||b_i*||."""
    mean_log = float(np.mean(gs_log))  # (1/n) * log vol
    return mean_log + lgamma(n / 2.0 + 1.0) / n - 0.5 * log(pi)


def profile_metrics(gs_log, n):
    """Basis-profile metrics. Replaces rhf with b1_gh = ||b1|| / GH."""
    gs_log = np.asarray(gs_log, float)
    log_b1 = float(gs_log[0])  # ||b1|| = ||b1*||
    log_gh = log_gaussian_heuristic(gs_log, n)

    b1_gh = float(np.exp(log_b1 - log_gh))  # <-- the ratio you want
    logdet = float(np.sum(gs_log))
    # GSA slope: linear fit of log GS norms vs index (more negative = flatter)
    slope = float(np.polyfit(np.arange(n), gs_log, 1)[0])

    return dict(
        b1_gh=b1_gh,  # ||b1|| / GH   (near 1 = as short as expected)
        slope=slope,
        logdet=logdet,
        log_b1=log_b1,
    )


def potential(gs, n):
    """BKZ potential sum_i (n-i) log||b_i*||  (monotone under reduction)."""
    gs = np.asarray(gs, dtype=float)
    w = np.arange(n, 0, -1, dtype=float)
    return float((w * gs).sum())


def cos_from_basis(mat):
    """Lower-triangle |cos| matrix (float32) + (max,mean). float64 Gram."""
    B = np.array(mat, dtype=np.float64)
    n = B.shape[0]
    G = B @ B.T
    d = np.sqrt(np.clip(np.diag(G), 1e-30, None))
    C = np.abs(G / (np.outer(d, d) + 1e-30))
    iu = np.tril_indices(n, -1)
    cosL = np.zeros((n, n), dtype=np.float32)
    cosL[iu] = C[iu].astype(np.float32)
    vals = C[iu]
    return cosL, float(vals.max()), float(vals.mean())
