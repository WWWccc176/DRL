#!/usr/bin/env python3
"""
Persistent ENUM_SIEVE (GPU) worker.

ONE process, ONE CUDA context, processes EVERY (dim, file) back-to-back so the
GPU stays busy instead of paying python-import + CUDA-context startup per file.

Pipeline (async):
  prefetch thread : read + parse the NEXT file       (disk/CPU, releases GIL)
  main     thread : create_matrix + reduce_full(ENUM_SIEVE)  (GPU sieve, warm ctx)

Coverage (same rules as bench.py):
  * every INTEGER dim in the range, EVERY file counted;
  * first file (index 0) of each dim -> cosine CSV for the heatmap.
"""

import sys
import json
import time
import queue
import threading
import argparse
import faulthandler

faulthandler.enable()
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import g6k_env  # FIRST: chdir into g6k root (spherical_coding/)
import numpy as np

from dataset_io import files_for_dim, read_text
from lattice_metrics import parse_fplll, profile_metrics, cos_from_basis

ROOT = HERE.parents[1]
OUT = ROOT / "results" / "bench"
RAW = OUT / "raw"
COS = OUT / "cos"


def parse_dims(spec):
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))  # every integer dim
    return [int(x) for x in spec.split(",") if x]


def build_manifest(dims, seeds_per_dim):
    k = seeds_per_dim if (seeds_per_dim and seeds_per_dim > 0) else 10**9
    tasks = []
    for dim in dims:
        for i, (seed, path) in enumerate(files_for_dim(dim, k)):
            tasks.append(dict(dim=dim, seed=seed, path=str(path), first=(i == 0)))
    return tasks


def prefetch_worker(tasks, q, stop):
    """Read + parse each file ahead of the GPU; push (task, text, n)."""
    for t in tasks:
        if stop.is_set():
            break
        p = Path(t["path"])
        if not p.is_absolute():
            p = g6k_env.ORIG_CWD / p
        try:
            text = read_text(str(p.resolve()))
            n = len(parse_fplll(text))
            q.put((t, text, n))
        except Exception as e:  # bad file -> report, keep going
            q.put((t, None, str(e)))
    q.put(None)  # sentinel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="70-110")
    ap.add_argument("--seeds-per-dim", type=int, default=0)  # 0 -> ALL files
    ap.add_argument("--beta", type=int, default=60)
    ap.add_argument("--steps-mult", type=int, default=4)  # steps = mult*dim
    ap.add_argument(
        "--prefetch", type=int, default=2, help="files to prepare ahead of the GPU"
    )
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    dims = parse_dims(args.dims)
    tasks = build_manifest(dims, args.seeds_per_dim)
    total = len(tasks)
    print(
        f"[enum_sieve_server] dims {dims[0]}..{dims[-1]} ({len(dims)}), "
        f"files={total}, beta={args.beta}, steps_mult={args.steps_mult}, "
        f"prefetch={args.prefetch}",
        flush=True,
    )
    if total == 0:
        print("[enum_sieve_server] no files found, nothing to do", flush=True)
        return

    # warm the backend + CUDA context ONCE, then reuse for every file
    from backend_loader import import_backend
    from reducers import reduce_full

    be = import_backend()

    q = queue.Queue(maxsize=max(1, args.prefetch))
    stop = threading.Event()
    pf = threading.Thread(target=prefetch_worker, args=(tasks, q, stop), daemon=True)
    pf.start()

    done = 0
    t_start = time.time()
    while True:
        item = q.get()
        if item is None:
            break
        task, text, n_or_err = item
        tag = f"{task['dim']}_ENUM_SIEVE_{task['seed']}"

        if text is None:  # prefetch failure
            (RAW / f"ERR_{tag}.log").write_text(f"prefetch failed: {n_or_err}")
            print(f"[skip] dim{task['dim']} seed{task['seed']}: {n_or_err}", flush=True)
            continue
        n = n_or_err

        t0 = time.time()
        mid = be.create_matrix(text)
        total_steps = args.steps_mult * n
        try:
            rstats = reduce_full(be, mid, n, "ENUM_SIEVE", args.beta, total_steps)
        except Exception as e:
            be.free_matrix(mid)
            (RAW / f"ERR_{tag}.log").write_text(f"reduce_full failed: {e}")
            print(f"[FAIL] dim{task['dim']} seed{task['seed']}: {e}", flush=True)
            continue

        gs = np.asarray(be.evaluate_matrix(mid)["gs_log_norms"], float)
        pm = profile_metrics(gs, n)
        red = parse_fplll(be.dump_matrix(mid))
        cosL, maxc, meanc = cos_from_basis(red)
        be.free_matrix(mid)

        rec = dict(
            dim=task["dim"],
            seed=task["seed"],
            method="ENUM_SIEVE",
            beta=args.beta,
            steps_budget=total_steps,
            steps_done=rstats.get("steps", 0),
            time_s=round(time.time() - t0, 3),
            max_cos=maxc,
            mean_cos=meanc,
            accepted=rstats["accepted"],
            calls=rstats["calls"],
            **pm,
        )

        if task["first"]:
            COS.mkdir(parents=True, exist_ok=True)
            np.savetxt(
                COS / f"{tag}.csv", np.asarray(cosL, float), fmt="%.5f", delimiter=","
            )
        (RAW / f"{tag}.json").write_text(json.dumps(rec, indent=2))

        done += 1
        rate = done / max(1e-9, time.time() - t_start)
        print(
            f"[{done}/{total}] ENUM_SIEVE dim{task['dim']} seed{task['seed']} "
            f"b1/GH={rec.get('b1_gh', float('nan')):.4f} "
            f"t={rec['time_s']}s ({rate:.2f} files/s)",
            flush=True,
        )

    stop.set()
    print(
        f"[enum_sieve_server] done {done}/{total} in {time.time() - t_start:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
