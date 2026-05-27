#!/usr/bin/env python3
"""
Analyze all DRL lattice-reduction result files for thesis Chapter 5.

Expected project structure:
DRL/
    results/
    a7/
        a7_dim50/
            dim50_seed0.txt
            dim50_seed1.txt
            ...
        a7_dim51/
            dim51_seed0.txt
            ...
        analysis_outputs/
            tables/
            figures/

This script:
1. Scans all results/a6up_dim*/dim*_seed*.txt files.
2. Parses Initial State and New Best Found blocks.
3. Finds per-seed final best results.
4. Finds per-dimension best/mean/worst/std/success count.
5. Generates thesis tables as .txt and .tex.
6. Generates figures for learning curves, LLL-vs-DRL comparison, final DRL results, success count, orthogonality metrics, and max-cosine metrics.
7. Reports which planned experiments cannot be produced from the existing result files alone.

Author: generated for DRL local-BKZ thesis analysis.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


TARGET_RATIO = 1.05
VERSION_NAME = "a7"

# ============================================================
# Path utilities
# ============================================================


def find_project_root() -> Path:
    """
    If this script is placed in scripts/, root is one level above.
    If this script is run from project root, use current working directory.
    """
    script_path = Path(__file__).resolve()
    if script_path.parent.name == "scripts":
        return script_path.parent.parent
    return Path.cwd().resolve()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ============================================================
# Parsing utilities
# ============================================================

FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?")
DIM_DIR_RE = re.compile(r"a6up_dim(\d+)$")
RESULT_FILE_RE = re.compile(r"dim(\d+)_seed(\d+)\.txt$")
EPISODE_RE = re.compile(r"Episode\s+(\d+)")


def extract_first_float(text: str) -> Optional[float]:
    match = FLOAT_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def extract_float_after_colon(text: str) -> Optional[float]:
    if ":" in text:
        tail = text.split(":")[-1].strip()
        return extract_first_float(tail)
    return extract_first_float(text)


def parse_b1_vector(line: str) -> Optional[List[int]]:
    """
    Parse line like:
    b₁ = [768, 389, ...]
    """
    if "[" not in line or "]" not in line:
        return None
    inside = line[line.find("[") + 1 : line.rfind("]")]
    vals = []
    for x in inside.split(","):
        x = x.strip()
        if x:
            try:
                vals.append(int(x))
            except ValueError:
                return None
    return vals if vals else None


def parse_metrics_block(lines: List[str], start_idx: int) -> Tuple[Dict[str, Any], int]:
    """
    Parse metrics after a header line until the next header or end of file.

    Returns:
        block: dict with ratio, defect, max_cos, min_cos, b1 if found
        next_idx: index of next header or end
    """
    block: Dict[str, Any] = {
        "ratio": None,
        "defect": None,
        "max_cos": None,
        "min_cos": None,
        "b1": None,
    }

    i = start_idx
    while i < len(lines):
        s = lines[i].strip()

        if s.startswith("--- ") and i != start_idx:
            break

        # Ratio line, including unicode form "Ratio (‖b₁‖/GH):"
        if "Ratio" in s and "GH" in s:
            val = extract_float_after_colon(s)
            if val is not None:
                block["ratio"] = val

        elif "Orthog" in s or "Defect" in s:
            val = extract_float_after_colon(s)
            if val is not None:
                block["defect"] = val

        elif "Max Cosine" in s:
            val = extract_float_after_colon(s)
            if val is not None:
                block["max_cos"] = val

        elif "Min Cosine" in s:
            val = extract_float_after_colon(s)
            if val is not None:
                block["min_cos"] = val

        elif "b₁" in s or "b1" in s:
            vec = parse_b1_vector(s)
            if vec is not None:
                block["b1"] = vec

        i += 1

    return block, i


def parse_result_file(path: Path) -> Dict[str, Any]:
    """
    Parse one dim{dim}_seed{seed}.txt file.

    Returns a dictionary containing:
        dim, seed, file, initial metrics, records, final metrics, first_success_ep
    """
    match = RESULT_FILE_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot infer dim/seed from filename: {path}")

    dim = int(match.group(1))
    seed = int(match.group(2))

    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    initial: Optional[Dict[str, Any]] = None
    records: List[Dict[str, Any]] = []

    i = 0
    while i < len(lines):
        s = lines[i].strip()

        if s.startswith("--- Initial State ---"):
            block, next_i = parse_metrics_block(lines, i + 1)
            block["episode"] = 0
            initial = block
            i = next_i
            continue

        if s.startswith("--- New Best Found"):
            ep_match = EPISODE_RE.search(s)
            episode = int(ep_match.group(1)) if ep_match else None
            block, next_i = parse_metrics_block(lines, i + 1)
            block["episode"] = episode
            if block.get("ratio") is not None:
                records.append(block)
            i = next_i
            continue

        i += 1

    if initial is None or initial.get("ratio") is None:
        raise ValueError(f"No valid initial state found in {path}")

    # Sort records by episode. If same episode appears multiple times,
    # keep all for accurate final-best detection, but curve builder will collapse.
    records.sort(
        key=lambda r: (r["episode"] if r["episode"] is not None else 10**18, r["ratio"])
    )

    if records:
        # The final best should be the minimum ratio ever recorded in the file.
        final = min(records, key=lambda r: r["ratio"])
    else:
        final = dict(initial)

    first_success_ep = None
    for r in records:
        if r.get("ratio") is not None and r["ratio"] <= TARGET_RATIO:
            first_success_ep = r.get("episode")
            break

    return {
        "dim": dim,
        "seed": seed,
        "file": str(path),
        "initial": initial,
        "records": records,
        "final": final,
        "first_success_ep": first_success_ep,
        "success": bool(
            final.get("ratio") is not None and final["ratio"] <= TARGET_RATIO
        ),
    }


def scan_all_results(
    version_results_dir: Path, version_name: str
) -> List[Dict[str, Any]]:
    """
    Scan result files under:
        results/{version_name}/{version_name}_dim*/dim*_seed*.txt
    """
    rows: List[Dict[str, Any]] = []

    dim_dir_re = re.compile(rf"{re.escape(version_name)}_dim(\d+)$")

    for dim_dir in sorted(version_results_dir.glob(f"{version_name}_dim*")):
        if not dim_dir.is_dir():
            continue
        if not dim_dir_re.search(dim_dir.name):
            continue

        for fpath in sorted(dim_dir.glob("dim*_seed*.txt")):
            try:
                parsed = parse_result_file(fpath)
                rows.append(parsed)
            except Exception as e:
                print(f"[Warning] Skipping {fpath}: {e}")

    return rows


# ============================================================
# Curve construction
# ============================================================


def collapse_records_by_episode(
    records: List[Dict[str, Any]],
) -> List[Tuple[int, float]]:
    """
    If multiple new-best records appear in the same episode, keep the lowest ratio.
    """
    ep_to_ratio: Dict[int, float] = {}
    for r in records:
        ep = r.get("episode")
        ratio = r.get("ratio")
        if ep is None or ratio is None:
            continue
        if ep not in ep_to_ratio:
            ep_to_ratio[ep] = ratio
        else:
            ep_to_ratio[ep] = min(ep_to_ratio[ep], ratio)

    return sorted(ep_to_ratio.items(), key=lambda x: x[0])


def build_step_curve(
    initial_ratio: float, records: List[Dict[str, Any]]
) -> Tuple[List[int], List[float]]:
    """
    Build post-step curve from initial ratio and New Best records.
    """
    collapsed = collapse_records_by_episode(records)

    if not collapsed:
        return [0, 1], [initial_ratio, initial_ratio]

    max_ep = max(ep for ep, _ in collapsed)
    episodes = list(range(0, max_ep + 2))

    ratios = []
    current = initial_ratio
    idx = 0
    for ep in episodes:
        while idx < len(collapsed) and collapsed[idx][0] <= ep:
            current = min(current, collapsed[idx][1])
            idx += 1
        ratios.append(current)

    return episodes, ratios


# ============================================================
# Table construction
# ============================================================


def mean_std(values: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None, None
    arr = np.asarray(vals, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=0))


def fmt(x: Any, digits: int = 6) -> str:
    if x is None:
        return "NA"
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if math.isnan(x):
            return "NA"
        return f"{x:.{digits}f}"
    return str(x)


def build_seed_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        initial = r["initial"]
        final = r["final"]
        out.append(
            {
                "dim": r["dim"],
                "seed": r["seed"],
                "initial_ratio": initial.get("ratio"),
                "final_ratio": final.get("ratio"),
                "ratio_improvement": initial.get("ratio") - final.get("ratio")
                if initial.get("ratio") is not None and final.get("ratio") is not None
                else None,
                "initial_defect": initial.get("defect"),
                "final_defect": final.get("defect"),
                "defect_improvement": initial.get("defect") - final.get("defect")
                if initial.get("defect") is not None and final.get("defect") is not None
                else None,
                "initial_max_cos": initial.get("max_cos"),
                "final_max_cos": final.get("max_cos"),
                "max_cos_improvement": initial.get("max_cos") - final.get("max_cos")
                if initial.get("max_cos") is not None
                and final.get("max_cos") is not None
                else None,
                "initial_min_cos": initial.get("min_cos"),
                "final_min_cos": final.get("min_cos"),
                "best_episode": final.get("episode"),
                "first_success_episode": r.get("first_success_ep"),
                "success": r["success"],
                "file": r["file"],
            }
        )

    out.sort(key=lambda x: (x["dim"], x["seed"]))
    return out


def build_dimension_summary(seed_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_dim: Dict[int, List[Dict[str, Any]]] = {}
    for row in seed_rows:
        by_dim.setdefault(row["dim"], []).append(row)

    out: List[Dict[str, Any]] = []
    for dim, items in sorted(by_dim.items()):
        init_ratios = [x["initial_ratio"] for x in items]
        final_ratios = [x["final_ratio"] for x in items]
        init_defects = [x["initial_defect"] for x in items]
        final_defects = [x["final_defect"] for x in items]
        init_maxcos = [x["initial_max_cos"] for x in items]
        final_maxcos = [x["final_max_cos"] for x in items]

        init_mean, init_std = mean_std(init_ratios)
        final_mean, final_std = mean_std(final_ratios)
        init_def_mean, _ = mean_std(init_defects)
        final_def_mean, _ = mean_std(final_defects)
        init_cos_mean, _ = mean_std(init_maxcos)
        final_cos_mean, _ = mean_std(final_maxcos)

        best_item = min(
            items,
            key=lambda x: (
                x["final_ratio"] if x["final_ratio"] is not None else float("inf")
            ),
        )
        worst_item = max(
            items,
            key=lambda x: (
                x["final_ratio"] if x["final_ratio"] is not None else -float("inf")
            ),
        )

        out.append(
            {
                "dim": dim,
                "num_seeds": len(items),
                "success_count": sum(1 for x in items if x["success"]),
                "initial_mean_ratio": init_mean,
                "initial_std_ratio": init_std,
                "final_mean_ratio": final_mean,
                "final_std_ratio": final_std,
                "best_ratio": best_item["final_ratio"],
                "best_seed": best_item["seed"],
                "worst_ratio": worst_item["final_ratio"],
                "worst_seed": worst_item["seed"],
                "mean_ratio_improvement": init_mean - final_mean
                if init_mean is not None and final_mean is not None
                else None,
                "initial_mean_defect": init_def_mean,
                "final_mean_defect": final_def_mean,
                "mean_defect_improvement": init_def_mean - final_def_mean
                if init_def_mean is not None and final_def_mean is not None
                else None,
                "initial_mean_max_cos": init_cos_mean,
                "final_mean_max_cos": final_cos_mean,
                "mean_max_cos_improvement": init_cos_mean - final_cos_mean
                if init_cos_mean is not None and final_cos_mean is not None
                else None,
            }
        )

    return out


def save_txt_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("No data.\n", encoding="utf-8")
        return

    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(fields)
        for row in rows:
            writer.writerow([fmt(row.get(k)) for k in fields])


def save_latex_table(
    path: Path,
    rows: List[Dict[str, Any]],
    caption: str,
    label: str,
    max_rows: Optional[int] = None,
) -> None:
    if not rows:
        path.write_text("% No data.\n", encoding="utf-8")
        return

    rows_to_write = rows[:max_rows] if max_rows is not None else rows
    fields = list(rows_to_write[0].keys())

    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{" + "l" * len(fields) + "}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(fields).replace("_", "\\_") + " \\\\\n")
        f.write("\\midrule\n")
        for row in rows_to_write:
            vals = [fmt(row.get(k), digits=5) for k in fields]
            vals = [v.replace("_", "\\_") for v in vals]
            f.write(" & ".join(vals) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}%\n")
        f.write("}\n")
        f.write("\\end{table}\n")


def save_best_per_dimension(path: Path, dim_summary: List[Dict[str, Any]]) -> None:
    lines = []
    lines.append("Best result per dimension")
    lines.append("=" * 80)
    for row in dim_summary:
        lines.append(
            f"Dim {row['dim']}: best seed={row['best_seed']}, "
            f"best ratio={fmt(row['best_ratio'])}, "
            f"mean ratio={fmt(row['final_mean_ratio'])}, "
            f"success={row['success_count']}/{row['num_seeds']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# Plotting
# ============================================================
EXCLUDE_PLOT_DIMS = {54, 59}


def filter_plot_dims_dim_summary(
    dim_summary: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Filter dimensions only for plotting. Tables still keep all parsed dimensions."""
    return [r for r in dim_summary if r["dim"] not in EXCLUDE_PLOT_DIMS]


def filter_plot_dims_seed_rows(seed_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter dimensions only for per-seed plots."""
    return [r for r in seed_rows if r["dim"] not in EXCLUDE_PLOT_DIMS]


def setup_episode_axis(ax) -> None:
    ax.set_xscale("symlog", base=2, linthresh=1.0)
    ax.xaxis.set_major_locator(ticker.SymmetricalLogLocator(base=2, linthresh=1.0))
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: "" if x < 0 else f"{int(x)}")
    )
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlim(left=0)


def save_figure(fig, path: Path) -> None:
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved figure] {path}")


def plot_best_learning_curves(rows: List[Dict[str, Any]], fig_dir: Path) -> None:
    by_dim: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        by_dim.setdefault(r["dim"], []).append(r)

    fig, ax = plt.subplots(figsize=(10, 6))

    plotted = False
    for dim, items in sorted(by_dim.items()):
        best = min(
            items,
            key=lambda r: (
                r["final"]["ratio"]
                if r["final"].get("ratio") is not None
                else float("inf")
            ),
        )
        eps, rats = build_step_curve(best["initial"]["ratio"], best["records"])
        ax.step(
            eps,
            rats,
            where="post",
            linewidth=1.2,
            label=f"dim={dim}, seed={best['seed']}",
        )
        plotted = True

    if not plotted:
        return

    ax.axhline(TARGET_RATIO, linestyle="--", linewidth=1.0, label=r"target $\rho=1.05$")
    setup_episode_axis(ax)
    ax.set_xlabel("Episode")
    ax.set_ylabel(r"$\|b_1\|/\mathrm{GH}$")
    ax.set_title("Best Learning Curve per Dimension")
    ax.grid(True, which="major", linestyle=":", linewidth=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, fig_dir / "fig_5_3_best_learning_curves.png")


def plot_success_count_curves(rows: List[Dict[str, Any]], fig_dir: Path) -> None:
    by_dim: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        by_dim.setdefault(r["dim"], []).append(r)

    fig, ax = plt.subplots(figsize=(10, 6))

    plotted = False
    for dim, items in sorted(by_dim.items()):
        success_eps = []
        max_ep = 1

        for r in items:
            eps, rats = build_step_curve(r["initial"]["ratio"], r["records"])
            max_ep = max(max_ep, max(eps))
            first = None
            for e, val in zip(eps, rats):
                if val <= TARGET_RATIO:
                    first = e
                    break
            success_eps.append(first)

        xs = list(range(0, max_ep + 1))
        ys = []
        for e in xs:
            count = sum(1 for s_ep in success_eps if s_ep is not None and s_ep <= e)
            ys.append(count)

        ax.step(xs, ys, where="post", linewidth=1.2, label=f"dim={dim}")
        plotted = True

    if not plotted:
        return

    setup_episode_axis(ax)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Number of successful seeds, target={TARGET_RATIO}")
    ax.set_title("Experimental Success Count During Training")
    ax.grid(True, which="major", linestyle=":", linewidth=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, fig_dir / "fig_5_3_experimental_success_count_curves.png")


def plot_initial_vs_final_ratio(
    dim_summary: List[Dict[str, Any]], fig_dir: Path
) -> None:
    dim_summary = filter_plot_dims_dim_summary(dim_summary)

    dims = [r["dim"] for r in dim_summary]
    init = [r["initial_mean_ratio"] for r in dim_summary]
    final = [r["final_mean_ratio"] for r in dim_summary]

    x = np.arange(len(dims))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, init, width, label="Initial LLL")
    ax.bar(x + width / 2, final, width, label="Final DRL-BKZ")
    ax.axhline(TARGET_RATIO, linestyle="--", linewidth=1.0, label=r"target $\rho=1.05$")

    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("Dimension")
    ax.set_ylabel(r"Mean $\|b_1\|/\mathrm{GH}$")
    ax.set_title("Initial LLL vs Final DRL-BKZ")
    ax.set_ylim(1.0, 1.75)
    ax.grid(axis="y", linestyle=":", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, fig_dir / "fig_5_4_initial_lll_vs_final_drl_bkz.png")


def plot_final_ratio_with_std(dim_summary: List[Dict[str, Any]], fig_dir: Path) -> None:
    dim_summary = filter_plot_dims_dim_summary(dim_summary)

    dims = [r["dim"] for r in dim_summary]
    means = [r["final_mean_ratio"] for r in dim_summary]
    stds = [
        r["final_std_ratio"] if r["final_std_ratio"] is not None else 0.0
        for r in dim_summary
    ]

    x = np.arange(len(dims))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, means, yerr=stds, capsize=4, label="Final DRL-BKZ mean ± std")
    ax.axhline(TARGET_RATIO, linestyle="--", linewidth=1.0, label=r"target $\rho=1.05$")

    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("Dimension")
    ax.set_ylabel(r"Final $\|b_1\|/\mathrm{GH}$")
    ax.set_title("End-to-End DRL-BKZ Results")
    ax.grid(axis="y", linestyle=":", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, fig_dir / "fig_5_5_final_drl_bkz_ratio_mean_std.png")


def plot_per_seed_final_scatter(seed_rows: List[Dict[str, Any]], fig_dir: Path) -> None:
    dims = sorted(set(r["dim"] for r in seed_rows))

    fig, ax = plt.subplots(figsize=(10, 6))

    for dim in dims:
        vals = [
            r["final_ratio"]
            for r in seed_rows
            if r["dim"] == dim and r["final_ratio"] is not None
        ]
        jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else [0.0]
        xs = [dim + j for j in jitter]
        ax.scatter(xs, vals, s=25, alpha=0.85)

    ax.axhline(TARGET_RATIO, linestyle="--", linewidth=1.0, label=r"target $\rho=1.05$")
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("Dimension")
    ax.set_ylabel(r"Final $\|b_1\|/\mathrm{GH}$")
    ax.set_title("Per-seed Final DRL Ratios")
    ax.set_ylim(bottom=0.5)
    ax.grid(axis="y", linestyle=":", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, fig_dir / "fig_5_5_per_seed_final_ratio_scatter.png")


def plot_orthogonality_metrics(
    dim_summary: List[Dict[str, Any]], fig_dir: Path
) -> None:
    dims = [r["dim"] for r in dim_summary]
    init_def = [r["initial_mean_defect"] for r in dim_summary]
    final_def = [r["final_mean_defect"] for r in dim_summary]

    x = np.arange(len(dims))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, init_def, width, label="Initial LLL")
    ax.bar(x + width / 2, final_def, width, label="Final DRL")

    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Mean logarithmic orthogonality defect")
    ax.set_title("Orthogonality Defect Before and After DRL")
    ax.grid(axis="y", linestyle=":", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, fig_dir / "fig_5_7_orthogonality_defect_initial_vs_final.png")


def plot_max_cos_metrics(dim_summary: List[Dict[str, Any]], fig_dir: Path) -> None:
    dims = [r["dim"] for r in dim_summary]
    init_cos = [r["initial_mean_max_cos"] for r in dim_summary]
    final_cos = [r["final_mean_max_cos"] for r in dim_summary]

    x = np.arange(len(dims))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, init_cos, width, label="Initial LLL")
    ax.bar(x + width / 2, final_cos, width, label="Final DRL")

    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Mean maximum cosine")
    ax.set_title("Maximum Cosine Before and After DRL")
    ax.grid(axis="y", linestyle=":", linewidth=0.5)
    ax.legend()
    ax.set_ylim(bottom=0.5)
    fig.tight_layout()
    save_figure(fig, fig_dir / "fig_5_7_max_cosine_initial_vs_final.png")


# ============================================================
# Optional diagnostics: action logs and heatmaps
# ============================================================

ACTION_RE_LIST = [
    re.compile(
        r"Action.*?beta\s*=?\s*(\d+).*?(?:pos|position|p)\s*=?\s*(\d+)", re.IGNORECASE
    ),
    re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)"),
]


def parse_actions_from_file(path: Path) -> List[Tuple[int, int]]:
    """
    Try to parse actions if action traces were saved in result files.
    Many current result files may not contain action logs, so this can return empty.
    """
    actions: List[Tuple[int, int]] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if (
            "action" not in line.lower()
            and "beta" not in line.lower()
            and "(" not in line
        ):
            continue
        for rgx in ACTION_RE_LIST:
            m = rgx.search(line)
            if m:
                try:
                    beta = int(m.group(1))
                    pos = int(m.group(2))
                    actions.append((beta, pos))
                    break
                except Exception:
                    pass
    return actions


def plot_action_distribution_if_available(
    rows: List[Dict[str, Any]], fig_dir: Path, report_lines: List[str]
) -> None:
    all_actions: List[Tuple[int, int]] = []
    for r in rows:
        actions = parse_actions_from_file(Path(r["file"]))
        all_actions.extend(actions)

    if not all_actions:
        report_lines.append(
            "Action distribution plot skipped: no action traces were found in the parsed result files."
        )
        return

    from collections import Counter

    cnt = Counter(all_actions)
    common = cnt.most_common(30)

    labels = [f"({b},{p})" for (b, p), _ in common]
    vals = [v for _, v in common]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(np.arange(len(labels)), vals)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Most Frequent Actions")
    ax.grid(axis="y", linestyle=":", linewidth=0.5)
    fig.tight_layout()
    save_figure(fig, fig_dir / "fig_5_2_action_distribution_if_available.png")
    report_lines.append("Action distribution plot generated from saved action traces.")


# ============================================================
# Missing experiment report
# ============================================================


def write_missing_experiment_report(path: Path) -> None:
    text = f"""
Experiments that can be generated directly from current result files
===================================================================
1. Training dynamics:
   - Best learning curve per dimension.
   - Success count curve per dimension.

2. Initial LLL vs final DRL:
   - The Initial State block gives the LLL-preprocessed basis quality.
   - The New Best Found blocks give the final/best DRL result.

3. End-to-end DRL result table:
   - mean ratio, std ratio, best ratio, worst ratio, success count.

4. Orthogonality metrics:
   - initial/final orthogonality defect.
   - initial/final maximum cosine.

5. Per-seed variability:
   - final ratio scatter plot over seeds and dimensions.

Experiments that require additional logs or additional runs
===========================================================
1. Fixed BKZ baseline:
   - Cannot be produced from DRL result files unless fixed-BKZ results are saved separately.
   - You should run a fixed BKZ baseline script and save one file per dimension/seed in the same format:
     Initial State + Final Fixed BKZ State, including ratio, defect, max cosine, min cosine.

2. Random local-BKZ baseline:
   - Cannot be inferred from DRL result files.
   - You need to run a random policy using the same action budget, periodic LLL rule, and terminal polish.

3. AxialCNN vs MLP ablation:
   - Requires training a separate MLP-DDQN agent under the same seed set and budget.
   - Existing DRL result files do not contain this comparison.

4. With/without cosine ablation:
   - Requires a separate training run with cosine-token features removed.
   - Existing result files do not contain this comparison.

5. Runtime and computational cost:
   - The current sample result format does not contain reliable runtime, BKZ call count, LLL call count, or neural-network overhead.
   - Add logging for wall-clock time, number of backend calls, selected beta values, number of LLL calls, and final polish time.

6. Cosine heatmaps:
   - A heatmap requires either the full basis matrix or the saved cosine matrix.
   - If result files only save b1 and scalar metrics, the script cannot reconstruct the cosine matrix.
   - To generate heatmaps, save final_basis or final_cosine_matrix for each best result.

7. Action-distribution analysis:
   - Requires action traces such as episode, step, beta, pos, reward.
   - If action traces are only printed to terminal and not saved, the script cannot reconstruct them.
"""
    path.write_text(text.strip() + "\n", encoding="utf-8")


# ============================================================
# Main
# ============================================================


def main() -> None:
    root = find_project_root()
    results_dir = root / "results"

    if not results_dir.is_dir():
        raise FileNotFoundError(f"Cannot find results directory: {results_dir}")

    version_name = VERSION_NAME
    version_results_dir = results_dir / version_name

    if not version_results_dir.is_dir():
        raise FileNotFoundError(
            f"Cannot find version results directory: {version_results_dir}"
        )

    # analysis_outputs 与 a7_dim50/a7_dim51/... 同级
    out_dir = version_results_dir / "analysis_outputs"
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"

    ensure_dir(out_dir)
    ensure_dir(table_dir)
    ensure_dir(fig_dir)

    rows = scan_all_results(results_dir)
    if not rows:
        print(f"No valid result files found under {version_results_dir}")
        return

    seed_summary = build_seed_summary(rows)
    dim_summary = build_dimension_summary(seed_summary)

    # Save tables
    save_txt_table(table_dir / "all_seed_summary.txt", seed_summary)
    save_txt_table(table_dir / "dimension_summary.txt", dim_summary)
    save_best_per_dimension(table_dir / "best_per_dimension.txt", dim_summary)

    save_latex_table(
        table_dir / "dimension_summary_latex.tex",
        dim_summary,
        caption="End-to-end DRL results by dimension.",
        label="tab:end_to_end_drl_results",
    )

    # A shorter thesis table with most important columns
    compact_dim_summary = []
    for r in dim_summary:
        compact_dim_summary.append(
            {
                "dim": r["dim"],
                "num_seeds": r["num_seeds"],
                "success_count": r["success_count"],
                "initial_mean_ratio": r["initial_mean_ratio"],
                "final_mean_ratio": r["final_mean_ratio"],
                "final_std_ratio": r["final_std_ratio"],
                "best_ratio": r["best_ratio"],
                "worst_ratio": r["worst_ratio"],
                "initial_mean_defect": r["initial_mean_defect"],
                "final_mean_defect": r["final_mean_defect"],
                "initial_mean_max_cos": r["initial_mean_max_cos"],
                "final_mean_max_cos": r["final_mean_max_cos"],
            }
        )

    save_txt_table(table_dir / "chapter5_main_table.txt", compact_dim_summary)
    save_latex_table(
        table_dir / "chapter5_main_table_latex.tex",
        compact_dim_summary,
        caption="Main experimental results of the DRL-controlled local BKZ framework.",
        label="tab:chapter5_main_results",
    )

    # Generate figures
    plot_best_learning_curves(rows, fig_dir)
    plot_success_count_curves(rows, fig_dir)
    plot_initial_vs_final_ratio(dim_summary, fig_dir)
    plot_final_ratio_with_std(dim_summary, fig_dir)
    plot_per_seed_final_scatter(seed_summary, fig_dir)
    plot_orthogonality_metrics(dim_summary, fig_dir)
    plot_max_cos_metrics(dim_summary, fig_dir)

    # Optional action distribution
    report_lines: List[str] = []
    plot_action_distribution_if_available(rows, fig_dir, report_lines)

    # Missing experiment notes
    write_missing_experiment_report(
        out_dir / "missing_experiments_and_required_logs.txt"
    )

    # Final analysis report
    report = []
    report.append("Analysis completed.")
    report.append("=" * 80)
    report.append(f"Project root: {root}")
    report.append(f"Results root: {results_dir}")
    report.append(f"Version name: {version_name}")
    report.append(f"Version results directory: {version_results_dir}")
    report.append(f"Analysis output directory: {out_dir}")
    report.append(f"Parsed result files: {len(rows)}")
    report.append(f"Parsed dimensions: {', '.join(str(r['dim']) for r in dim_summary)}")
    report.append("")
    report.append("Generated tables:")
    report.append(f"- {table_dir / 'all_seed_summary.txt'}")
    report.append(f"- {table_dir / 'dimension_summary.txt'}")
    report.append(f"- {table_dir / 'best_per_dimension.txt'}")
    report.append(f"- {table_dir / 'chapter5_main_table.txt'}")
    report.append(f"- {table_dir / 'chapter5_main_table_latex.tex'}")
    report.append("")
    report.append("Generated figures:")
    for p in sorted(fig_dir.glob("*.png")):
        report.append(f"- {p}")
    report.append("")
    report.extend(report_lines)
    report.append("")
    report.append(
        "See missing_experiments_and_required_logs.txt for experiments that cannot be generated from current result files alone."
    )

    (out_dir / "analysis_report.txt").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print("\n".join(report))


if __name__ == "__main__":
    main()
