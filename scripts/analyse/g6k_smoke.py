#!/usr/bin/env python3
"""Minimal, no-pool G6K-GPU probe. Prints the REAL error instead of abort spam.

    cd ~/DRL/scripts/analyse
    python g6k_smoke.py

GPU watch (second terminal):
    watch -n 0.3 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'
"""

import sys
import faulthandler

faulthandler.enable()

sys.path.insert(0, ".")

# MUST be first: chdir into the g6k root that holds ./spherical_coding/
import g6k_env
from dataset_io import files_for_dim
from lattice_metrics import parse_fplll

print("== import g6k ==")
import g6k

print("g6k file:", g6k.__file__)
print("g6k root (cwd):", g6k_env.G6K_ROOT)
from g6k import Siever, SieverParams

print("\n== load one dim70 block ==")
seed, path = files_for_dim(70, 1)[0]
# resolve against original launch dir (we chdir'd away)
data_path = (g6k_env.ORIG_CWD / path) if not path.is_absolute() else path
B0 = parse_fplll(data_path.read_text())[:40]  # first 40 rows -> beta=40 block
from fpylll import IntegerMatrix

A = IntegerMatrix.from_matrix([[int(x) for x in r] for r in B0])
print("block:", A.nrows, "x", A.ncols)

print("\n== build GPU Siever (raw, no try/except) ==")
params = SieverParams(
    threads=1,
    gpus=1,
    gpu_bucketer=b"bdgl",
    gpu_triple=False,
)
try:
    print("params:", params.dict())
except Exception:
    print("params: <repr>", repr(params))

g = Siever(A, params)
print("Siever OK. relevant methods:")
print(
    " ",
    [
        m
        for m in dir(g)
        if any(k in m.lower() for k in ("gpu", "sieve", "lift", "init", "best"))
    ],
)

print("\n== sieve ==")
g.initialize_local(0, 0, A.nrows)
if hasattr(g, "gpu_sieve"):
    print("calling gpu_sieve()")
    g.gpu_sieve()
else:
    print("no gpu_sieve(); calling g(alg='bgj1')")
    g(alg="bgj1")
print("sieve done")

print("\n== lifts ==")
if hasattr(g, "best_lifts"):
    L = g.best_lifts()
    print("best_lifts:", None if not L else (len(L), L[0][:2]))
else:
    print("no best_lifts() method")

print("\nSMOKE OK")

