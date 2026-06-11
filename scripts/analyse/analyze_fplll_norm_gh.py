#!/usr/bin/env python3
"""
Analyze SVP challenge dataset files with the existing Rust/C++ fplll backend.

Put this file at:
    DRL/scripts/analyse/analyse_fplll_norm_gh.py

Expected project layout:
    DRL/
    ├── dataset/      # svpchallengedim{dim}seed{seed}.txt
    ├── results/
    ├── rustcore/
    ├── scripts/
    │   └── analyse/
    └── src/

This script follows the same backend style as agent7:
    import my_project_backend
    create_matrix_lll_rust(...)
    reduce_rust(pool_id, "LOCAL_BKZ", beta, pos)
    evaluate_matrix_rust(pool_id)
    free_matrix_rust(pool_id)

It evaluates:
    lll
    bkz20
    bkz20_30
    bkz20_30_40

For BKZ baselines, because the current backend exposes LOCAL_BKZ, each beta is
implemented as one left-to-right LOCAL_BKZ sweep followed by LLL.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import os
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

METHOD_PIPELINES: Dict[str, Tuple[int, ...]] = {
    "lll": (),
    "bkz20": (20,),
    "bkz20_30": (20, 30),
    "bkz20_30_40": (20, 30, 40),
}

DATASET_PATTERNS = [
    re.compile(r"^svpchallengedim(?P<dim>\d+)seed(?P<seed>-?\d+)\.txt$"),
    re.compile(r"^svp_dim(?P<dim>\d+)_seed(?P<seed>-?\d+)\.txt$"),
    re.compile(r"^dim(?P<dim>\d+)_seed(?P<seed>-?\d+)\.txt$"),
]


@dataclass(frozen=True)
class MatrixFile:
    path: Path
    dim: int
    seed: int


def find_project_root(start: Optional[Path] = None) -> Path:
    if start is None:
        start = Path(__file__).resolve()
    start = start if start.is_dir() else start.parent
    for p in [start, *start.parents]:
        if (p / "dataset").is_dir() and (p / "rustcore").is_dir():
            (p / "results").mkdir(parents=True, exist_ok=True)
            return p
    raise RuntimeError(
        "Cannot find DRL project root. Run from inside DRL or pass --project-root /home/pyjast1123/DRL"
    )


def add_backend_search_paths(project_root: Path) -> None:
    candidates = [
        project_root,
        project_root / "rustcore",
        project_root / "rustcore" / "target" / "release",
        project_root / "rustcore" / "target" / "debug",
        project_root / "target" / "release",
        project_root / "target" / "debug",
    ]
    for base in [project_root / "rustcore" / "target", project_root / "target"]:
        if base.exists():
            for child in base.rglob("my_project_backend*.so"):
                candidates.append(child.parent)
    for p in candidates:
        if p.exists():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)


def import_backend(project_root: Path):
    add_backend_search_paths(project_root)
    try:
        return importlib.import_module("my_project_backend")
    except Exception as exc:
        head = "\n".join(f"  - {p}" for p in sys.path[:30])
        raise RuntimeError(
            "Cannot import my_project_backend.\n"
            "First build/install your Rust PyO3 backend, for example:\n"
            "  cd /home/pyjast1123/DRL/rustcore && maturin develop --release\n"
            "or:\n"
            "  cd /home/pyjast1123/DRL/rustcore && cargo build --release\n"
            "Current sys.path head:\n"
            f"{head}"
        ) from exc


def parse_matrix_filename(path: Path) -> Optional[Tuple[int, int]]:
    for pat in DATASET_PATTERNS:
        m = pat.match(path.name)
        if m:
            return int(m.group("dim")), int(m.group("seed"))
    return None


def scan_dataset(dataset_dir: Path, dim_min: int, dim_max: int) -> List[MatrixFile]:
    out: List[MatrixFile] = []
    for path in sorted(dataset_dir.glob("*.txt")):
        parsed = parse_matrix_filename(path)
        if parsed is None:
            continue
        dim, seed = parsed
        if dim_min <= dim <= dim_max:
            out.append(MatrixFile(path=path, dim=dim, seed=seed))
    out.sort(key=lambda x: (x.dim, x.seed, x.path.name))
    return out


def ratio_from_pool(backend: Any, pool_id: int, dim: int) -> Tuple[float, float, float]:
    info = backend.evaluate_matrix_rust(pool_id)
    gs = list(info["gs_log_norms"])
    if not gs:
        raise RuntimeError("evaluate_matrix_rust returned empty gs_log_norms")
    log_vol = float(sum(gs))
    log_gh = log_vol / float(dim) + 0.5 * math.log(
        float(dim) / (2.0 * math.pi * math.e)
    )
    log_b1 = float(gs[0])
    ratio = math.exp(log_b1 - log_gh)
    norm = math.exp(log_b1)
    gh = math.exp(log_gh)
    return ratio, norm, gh


def run_local_bkz_sweep(backend: Any, pool_id: int, dim: int, beta: int) -> None:
    for pos in range(0, max(1, dim - 1)):
        backend.reduce_rust(pool_id, "LOCAL_BKZ", int(beta), int(pos))


def run_one_method(
    backend: Any, matrix_text: str, dim: int, pipeline: Sequence[int]
) -> Tuple[float, float, float]:
    pool_id = backend.create_matrix_lll_rust(matrix_text)
    try:
        backend.reduce_rust(pool_id, "LLL", 2, 0)
        for beta in pipeline:
            run_local_bkz_sweep(backend, pool_id, dim, int(beta))
            backend.reduce_rust(pool_id, "LLL", 2, 0)
        return ratio_from_pool(backend, pool_id, dim)
    finally:
        try:
            backend.free_matrix_rust(pool_id)
        except Exception:
            pass


def worker(job: Tuple[str, int, int, str]) -> List[Dict[str, Any]]:
    file_path_s, dim, seed, project_root_s = job
    file_path = Path(file_path_s)
    project_root = Path(project_root_s)
    try:
        backend = import_backend(project_root)
        matrix_text = file_path.read_text(encoding="utf-8")
        rows: List[Dict[str, Any]] = []
        for method, pipeline in METHOD_PIPELINES.items():
            ratio, norm, gh = run_one_method(backend, matrix_text, dim, pipeline)
            rows.append(
                {
                    "dim": dim,
                    "seed": seed,
                    "method": method,
                    "pipeline": "+".join(map(str, pipeline)) if pipeline else "LLL",
                    "norm": norm,
                    "gh": gh,
                    "norm_over_gh": ratio,
                    "file": file_path.name,
                    "status": "ok",
                    "error": "",
                }
            )
        return rows
    except Exception as exc:
        return [
            {
                "dim": dim,
                "seed": seed,
                "method": "ERROR",
                "pipeline": "ERROR",
                "norm": "",
                "gh": "",
                "norm_over_gh": "",
                "file": file_path.name,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            }
        ]


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dim",
        "seed",
        "method",
        "pipeline",
        "norm",
        "gh",
        "norm_over_gh",
        "file",
        "status",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def ok_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        if r.get("status") != "ok":
            continue
        try:
            rr = dict(r)
            rr["dim"] = int(rr["dim"])
            rr["seed"] = int(rr["seed"])
            rr["norm"] = float(rr["norm"])
            rr["gh"] = float(rr["gh"])
            rr["norm_over_gh"] = float(rr["norm_over_gh"])
            out.append(rr)
        except Exception:
            pass
    return out


def plot_scatter(rows: List[Dict[str, Any]], results_dir: Path) -> List[Path]:
    import matplotlib.pyplot as plt

    clean = ok_rows(rows)
    by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in clean:
        by_method[str(r["method"])].append(r)
    figures: List[Path] = []
    for method in METHOD_PIPELINES:
        data = by_method.get(method, [])
        if not data:
            continue
        xs = [r["dim"] for r in data]
        ys = [r["norm_over_gh"] for r in data]
        plt.figure(figsize=(9.0, 5.5))
        plt.scatter(xs, ys, s=18, alpha=0.75)
        plt.xlabel("Dimension")
        plt.ylabel("norm / GH")
        plt.title(f"{method}: norm/GH by dimension and seed")
        plt.grid(True, alpha=0.30)
        plt.tight_layout()
        out = results_dir / f"svpchallenge_fplll_{method}_norm_over_gh.png"
        plt.savefig(out, dpi=180)
        plt.close()
        figures.append(out)
    return figures


def summarize_method(
    rows: List[Dict[str, Any]], method: str
) -> Optional[Tuple[int, float, float, float, float]]:
    vals = sorted(
        float(r["norm_over_gh"]) for r in ok_rows(rows) if r["method"] == method
    )
    if not vals:
        return None
    n = len(vals)
    mean = sum(vals) / n
    median = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    return n, mean, median, vals[0], vals[-1]


def write_report(
    rows: List[Dict[str, Any]],
    results_dir: Path,
    csv_path: Path,
    figures: List[Path],
    files_count: int,
) -> Path:
    clean = ok_rows(rows)
    errors = [r for r in rows if r.get("status") == "error"]
    dims = sorted({int(r["dim"]) for r in clean})
    pairs = sorted({(int(r["dim"]), int(r["seed"])) for r in clean})
    report = results_dir / "svpchallenge_fplll_baseline_report.md"
    lines: List[str] = []
    lines.append("# SVP Challenge FPLLL Baseline Report")
    lines.append("")
    lines.append(f"- Dataset files scanned: {files_count}")
    lines.append(f"- Successful result rows: {len(clean)}")
    lines.append(f"- Failed rows: {len(errors)}")
    if dims:
        lines.append(f"- Dimension range: {min(dims)}–{max(dims)}")
    lines.append(f"- Distinct dim+seed pairs: {len(pairs)}")
    lines.append(f"- CSV: `{csv_path.name}`")
    lines.append("")
    lines.append("## Method summary")
    lines.append("")
    lines.append("| method | count | mean norm/GH | median | min | max |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for method in METHOD_PIPELINES:
        s = summarize_method(rows, method)
        if s is None:
            continue
        n, mean, median, vmin, vmax = s
        lines.append(
            f"| {method} | {n} | {mean:.8g} | {median:.8g} | {vmin:.8g} | {vmax:.8g} |"
        )
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for fig in figures:
        lines.append(f"- `{fig.name}`")
    lines.append("")
    lines.append("## Backend note")
    lines.append("")
    lines.append(
        "This script uses the same Python-facing Rust/C++ backend style as agent7. "
        "Because the current backend exposes `LOCAL_BKZ` rather than a separate global BKZ call, "
        "`bkz20`, `bkz20_30`, and `bkz20_30_40` are implemented as sequential full left-to-right "
        "LOCAL_BKZ sweeps, with LLL after each beta sweep."
    )
    lines.append("")
    if errors:
        lines.append("## Errors")
        lines.append("")
        for r in errors[:30]:
            first_line = str(r.get("error", "")).splitlines()[0]
            lines.append(f"- `{r.get('file')}`: {first_line}")
        if len(errors) > 30:
            lines.append(f"- ... {len(errors) - 30} more errors omitted")
        lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def analyze(args: argparse.Namespace) -> None:
    project_root = (
        Path(args.project_root).expanduser().resolve()
        if args.project_root
        else find_project_root()
    )
    dataset_dir = (
        Path(args.dataset_dir).expanduser().resolve()
        if args.dataset_dir
        else project_root / "dataset"
    )
    results_dir = (
        Path(args.results_dir).expanduser().resolve()
        if args.results_dir
        else project_root / "results"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    files = scan_dataset(dataset_dir, args.dim_min, args.dim_max)
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise RuntimeError(
            f"No dataset files found in {dataset_dir} for dim range {args.dim_min}-{args.dim_max}. "
            "Expected names like svpchallengedim50seed0.txt"
        )
    csv_path = results_dir / "svpchallenge_fplll_baseline_norm_gh.csv"
    jobs = [(str(f.path), f.dim, f.seed, str(project_root)) for f in files]
    rows: List[Dict[str, Any]] = []
    workers = max(1, int(args.workers))
    if workers == 1:
        for i, job in enumerate(jobs, 1):
            print(f"[{i}/{len(jobs)}] {Path(job[0]).name}", flush=True)
            rows.extend(worker(job))
            write_csv(rows, csv_path)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(worker, job) for job in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                part = fut.result()
                rows.extend(part)
                name = part[0].get("file", "unknown") if part else "unknown"
                status = part[0].get("status", "empty") if part else "empty"
                print(f"[{i}/{len(jobs)}] {name} -> {status}", flush=True)
                write_csv(rows, csv_path)
    rows.sort(
        key=lambda r: (
            int(r.get("dim", 10**9)),
            int(r.get("seed", 10**9)),
            str(r.get("method", "")),
        )
    )
    write_csv(rows, csv_path)
    figures = plot_scatter(rows, results_dir)
    report = write_report(rows, results_dir, csv_path, figures, len(files))
    print(f"CSV:    {csv_path}")
    print(f"Report: {report}")
    for fig in figures:
        print(f"Figure: {fig}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate LLL/BKZ baselines and plot norm/GH scatter figures."
    )
    p.add_argument("--project-root", default=None, help="Default: auto-detect DRL root")
    p.add_argument(
        "--dataset-dir", default=None, help="Default: <project-root>/dataset"
    )
    p.add_argument(
        "--results-dir", default=None, help="Default: <project-root>/results"
    )
    p.add_argument("--dim-min", type=int, default=40)
    p.add_argument("--dim-max", type=int, default=122)
    p.add_argument(
        "--workers", type=int, default=max(1, min((os.cpu_count() or 2) // 2, 4))
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Debug: process only the first N files"
    )
    return p


def main() -> None:
    analyze(build_parser().parse_args())


if __name__ == "__main__":
    main()
