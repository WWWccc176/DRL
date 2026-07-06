#!/usr/bin/env python3
"""
DRL lattice-reduction benchmark driver.

Coverage rules:
  * STATISTICS: every INTEGER dimension in the range, and every file/seed in
    each dimension unless --seeds-per-dim > 0.
  * HEATMAP: only the first file of each integer dimension contributes its
    cosine matrix.

Execution paths:
  * CPU methods: LLL / BKZ / ENUM  (ProcessPoolExecutor, CUDA disabled).
  * GPU method:  ENUM_SIEVE (alias: G6K)  -> ONE persistent g6k_server.py worker.

Results:
  * JSON records:   results/bench/raw/
  * Cosine CSV:      results/bench/cos/
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results" / "bench"
RAW = OUT / "raw"
COS = OUT / "cos"

GPU_METHODS = {"ENUM_SIEVE"}


# --------------------------------------------------------------------------
# dims parsing
# --------------------------------------------------------------------------
def parse_dims(spec):
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",") if x]


# --------------------------------------------------------------------------
# file listing
# --------------------------------------------------------------------------
def list_files(dim, seeds_per_dim):
    from dataset_io import files_for_dim

    k = seeds_per_dim if (seeds_per_dim and seeds_per_dim > 0) else 10**9
    return list(files_for_dim(dim, k))


# --------------------------------------------------------------------------
# task construction (CPU methods only)
# --------------------------------------------------------------------------
def build_tasks(dims, methods, seeds_per_dim, betas, steps_mult):
    tasks = []
    for dim in dims:
        files = list_files(dim, seeds_per_dim)
        if not files:
            print(f"[warn] dim{dim}: no files found, skipping")
            continue
        for i, (seed, path) in enumerate(files):
            save_cos = i == 0
            for method in methods:
                tasks.append(
                    dict(
                        dim=dim,
                        seed=seed,
                        path=str(path),
                        method=method,
                        beta=betas[method],
                        steps_mult=steps_mult,
                        save_cos=save_cos,
                    )
                )
    return tasks


# --------------------------------------------------------------------------
# quality metric print helper
# --------------------------------------------------------------------------
def quality_value(rec):
    if "b1_gh" in rec:
        return rec["b1_gh"]
    if "b1_over_gh" in rec:
        return rec["b1_over_gh"]
    return float("nan")


# --------------------------------------------------------------------------
# CPU worker
# --------------------------------------------------------------------------
def worker_init():
    os.environ["LATTICE_DISABLE_CUDA"] = "1"
    sp = str(HERE)
    if sp not in sys.path:
        sys.path.insert(0, sp)
    for v in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[v] = "1"


def run_task(task):
    from backend_loader import import_backend
    from dataset_io import read_text
    from lattice_metrics import parse_fplll, profile_metrics, cos_from_basis
    from reducers import reduce_full

    t0 = time.time()

    be = import_backend()
    text = read_text(task["path"])
    n = len(parse_fplll(text))
    mid = be.create_matrix(text)

    total_steps = task["steps_mult"] * n
    rstats = reduce_full(be, mid, n, task["method"], task["beta"], total_steps)

    gs = np.asarray(be.evaluate_matrix(mid)["gs_log_norms"], float)
    pm = profile_metrics(gs, n)

    red = parse_fplll(be.dump_matrix(mid))
    cosL, maxc, meanc = cos_from_basis(red)

    be.free_matrix(mid)

    rec = dict(
        dim=task["dim"],
        seed=task["seed"],
        method=task["method"],
        beta=task["beta"],
        steps_budget=total_steps,
        steps_done=rstats.get("steps", 0),
        time_s=round(time.time() - t0, 3),
        max_cos=maxc,
        mean_cos=meanc,
        accepted=rstats["accepted"],
        calls=rstats["calls"],
        **pm,
    )

    if task["save_cos"]:
        rec["_cosL"] = np.asarray(cosL, float).tolist()

    return rec


def save_record(rec):
    RAW.mkdir(parents=True, exist_ok=True)

    cosL = rec.pop("_cosL", None)
    if cosL is not None:
        COS.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            COS / f"{rec['dim']}_{rec['method']}_{rec['seed']}.csv",
            np.asarray(cosL, float),
            fmt="%.5f",
            delimiter=",",
        )

    (RAW / f"{rec['dim']}_{rec['method']}_{rec['seed']}.json").write_text(
        json.dumps(rec, indent=2)
    )


def run_cpu_pool(tasks, workers):
    RAW.mkdir(parents=True, exist_ok=True)
    done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as ex:
        futs = {ex.submit(run_task, t): t for t in tasks}
        for fut in as_completed(futs):
            task = futs[fut]
            try:
                rec = fut.result()
                save_record(rec)
                done += 1
                q = quality_value(rec)
                print(
                    f"[{done}/{len(tasks)}] dim{rec['dim']} {rec['method']} "
                    f"seed{rec['seed']} b1/GH={q:.4f} t={rec['time_s']}s"
                )
            except Exception as e:
                print(
                    f"[FAIL] dim{task['dim']} {task['method']} seed{task['seed']}: {e}"
                )


# --------------------------------------------------------------------------
# persistent GPU worker launcher (ENUM_SIEVE)
# --------------------------------------------------------------------------
def run_g6k_server(dims_spec, seeds_per_dim, beta, steps_mult, prefetch=2):
    cmd = [
        sys.executable,
        str(HERE / "g6k_server.py"),
        "--dims",
        str(dims_spec),
        "--seeds-per-dim",
        str(seeds_per_dim),
        "--beta",
        str(beta),
        "--steps-mult",
        str(steps_mult),
        "--prefetch",
        str(prefetch),
    ]
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dims",
        default="70-110",
        help="'70-110' = every integer dim; or explicit '70,80,90'",
    )
    ap.add_argument(
        "--seeds-per-dim",
        type=int,
        default=0,
        help="0 -> all files in each dim. >0 caps per dim.",
    )
    ap.add_argument(
        "--methods",
        default="LLL,BKZ,ENUM",
        help="CPU: LLL,BKZ,ENUM.  GPU: ENUM_SIEVE (alias: G6K).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="0 -> auto, about 80%% of CPU cores for CPU methods.",
    )
    ap.add_argument(
        "--steps-mult",
        type=int,
        default=4,
        help="steps per file = steps_mult * dim.",
    )
    ap.add_argument("--beta-bkz", type=int, default=40, help="BKZ block size")
    ap.add_argument("--beta-enum", type=int, default=50, help="ENUM block size")
    ap.add_argument(
        "--beta-enum-sieve", type=int, default=60, help="ENUM+SIEVE block size"
    )
    ap.add_argument(
        "--g6k-prefetch",
        type=int,
        default=2,
        help="prefetch depth passed to persistent ENUM_SIEVE server.",
    )

    args = ap.parse_args()

    dims = parse_dims(args.dims)
    methods = [m.strip().upper() for m in args.methods.split(",") if m.strip()]
    methods = ["ENUM_SIEVE" if m == "G6K" else m for m in methods]  # alias

    if not dims:
        raise SystemExit("No dimensions selected.")

    print(
        f"dims = {dims[0]}..{dims[-1]} ({len(dims)} integer dims), "
        f"seeds/dim = {'ALL' if args.seeds_per_dim <= 0 else args.seeds_per_dim}"
    )

    # ----------------------------------------------------------------------
    # GPU path: ENUM_SIEVE must run alone (GPU serialized)
    # ----------------------------------------------------------------------
    if any(m in GPU_METHODS for m in methods) and len(methods) > 1:
        raise SystemExit(
            "ENUM_SIEVE(枚举+筛法)必须单独跑（GPU 串行化）:\n"
            "  CPU: --methods LLL,BKZ,ENUM --workers 8\n"
            "  GPU: --methods ENUM_SIEVE --workers 1"
        )

    if "ENUM_SIEVE" in methods:
        print(
            f"ENUM_SIEVE beta={args.beta_enum_sieve}  steps_mult={args.steps_mult}  "
            f"(persistent worker + prefetch pipeline)"
        )
        run_g6k_server(
            args.dims,
            args.seeds_per_dim,
            args.beta_enum_sieve,
            args.steps_mult,
            prefetch=args.g6k_prefetch,
        )
        print(f">> results in {RAW}")
        return

    # ----------------------------------------------------------------------
    # CPU path
    # ----------------------------------------------------------------------
    workers = args.workers or max(1, int(0.9 * (os.cpu_count() or 8)))

    betas = {"LLL": 0, "BKZ": args.beta_bkz, "ENUM": args.beta_enum}
    unknown = [m for m in methods if m not in betas]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}")

    tasks = build_tasks(dims, methods, args.seeds_per_dim, betas, args.steps_mult)

    print(f"methods={methods}  tasks={len(tasks)}  workers={workers}")

    run_cpu_pool(tasks, workers)

    print(f">> results in {RAW}")


if __name__ == "__main__":
    main()

