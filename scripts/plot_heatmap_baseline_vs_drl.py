#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cosine heatmap comparison for Chapter 5.

Each selected row compares the same lattice instance under:
    Initial LLL | Fixed Local-BKZ | Final DRL-BKZ

This version uses the current matrix-pool backend API:
    create_matrix_rust
    create_matrix_lll_rust
    reduce_rust
    dump_matrix_rust
    clone_matrix_rust
    free_matrix_rust

It does NOT call the old run_reduction_rust API.
"""

import re
import sys
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "analysis_outputs" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(PROJECT_ROOT / "drl_app"))
sys.path.append(str(PROJECT_ROOT / "build"))

try:
    import my_project_backend

    print("C++ / Rust backend loaded successfully.")
except ImportError as e:
    print(f"Failed to import backend: {e}")
    sys.exit(1)


# ============================================================
# Config
# ============================================================

MAX_CASES = 2

PREFERRED_DIMS = [58, 57, 56, 55, 67, 54, 59]

# 这里必须和论文里 fixed BKZ baseline 的设置一致。
# 当前 C++ 后端没有 "BKZ" 方法名，只有 "LOCAL_BKZ"。
# 所以这里默认是从 LLL 后的 basis 出发，对 leading block 执行一次 LOCAL_BKZ-40。
FIXED_BKZ_STAGES = [
    ("LOCAL_BKZ", 40, 0),
]

FIXED_BKZ_LABEL = "Fixed Local-BKZ-40"

CMAP = "RdYlGn_r"
HEATMAP_VMIN = 0.0
HEATMAP_VMAX = 0.7


# ============================================================
# Regex
# ============================================================

SUMMARY_DIR_RE = re.compile(r"a6up_dim(\d+)$")
SEED_HEADER_RE = re.compile(r"^--- Seed\s+(\d+)\s+\(ratio=([0-9.]+)\)\s+---$")
ROW_RE = re.compile(r"\[([^\[\]]+)\]")


# ============================================================
# Matrix utilities
# ============================================================


def read_dataset_basis_string(dim: int, seed: int) -> str:
    fpath = DATASET_DIR / f"svpchallengedim{dim}seed{seed}.txt"

    if not fpath.exists():
        raise FileNotFoundError(f"Dataset file not found: {fpath}")

    return fpath.read_text(encoding="utf-8", errors="replace")


def parse_matrix_string_to_numpy(matrix_str: str) -> np.ndarray:
    """
    Parse fplll-style matrix string:
        [[1 2]
        [3 4]]
    or summary-style:
        [
          [1 2]
          [3 4]
        ]
    """
    rows = []

    row_blocks = ROW_RE.findall(matrix_str)

    for block in row_blocks:
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", block)
        if nums:
            rows.append([float(x) for x in nums])

    if not rows:
        raise ValueError("No matrix rows parsed.")

    arr = np.array(rows, dtype=float)

    if arr.ndim != 2:
        raise ValueError(f"Parsed matrix is not 2D: shape={arr.shape}")

    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Parsed matrix is not square: shape={arr.shape}")

    return arr


def compute_norms(B: np.ndarray) -> np.ndarray:
    return np.linalg.norm(B, axis=1)


def compute_log_det_abs(B: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(B)

    if sign == 0:
        raise ValueError("Singular basis matrix.")

    return float(logdet)


def compute_log_gh(B: np.ndarray) -> float:
    n = B.shape[0]
    logdet = compute_log_det_abs(B)
    return float(logdet / n + 0.5 * math.log(n / (2.0 * math.pi * math.e)))


def compute_ratio(B: np.ndarray) -> float:
    norms = compute_norms(B)
    return float(norms[0] / math.exp(compute_log_gh(B)))


def compute_log_defect(B: np.ndarray) -> float:
    norms = compute_norms(B)
    log_prod = float(np.sum(np.log(norms + 1e-300)))
    logdet = compute_log_det_abs(B)
    return float(log_prod - logdet)


# ============================================================
# Heatmap utilities
# ============================================================


def build_plot_matrix(full_cos_mat: np.ndarray):
    """
    Same visualization logic as your original plot_heatmap.py:
    take full_cos_mat[1:, :-1] and then its lower triangular part.
    """
    full_cos_mat = np.asarray(full_cos_mat, dtype=float)

    if full_cos_mat.ndim != 2:
        raise ValueError(f"cos_matrix is not 2D: shape={full_cos_mat.shape}")

    n = full_cos_mat.shape[0]

    if n <= 1:
        return full_cos_mat, 0.0, 0.0

    sub_mat = full_cos_mat[1:, :-1]
    plot_mat = np.tril(sub_mat)

    valid_mask = np.tril(np.ones_like(sub_mat), k=0).astype(bool)
    valid_vals = sub_mat[valid_mask]
    valid_vals = valid_vals[np.isfinite(valid_vals)]

    cos_max = float(np.max(valid_vals)) if len(valid_vals) > 0 else 0.0
    cos_avg = float(np.mean(valid_vals)) if len(valid_vals) > 0 else 0.0

    return plot_mat, cos_max, cos_avg


def panel_from_pool(matrix_id: int, ratio_override=None):
    """
    Extract cosine matrix from current backend matrix without modifying it.

    reduce_rust(matrix_id, "NONE", 0, 0) works because the C++ do_reduction()
    only handles "LLL" and "LOCAL_BKZ"; any other method leaves the matrix unchanged
    and still returns extracted float data.
    """
    info = my_project_backend.reduce_rust(matrix_id, "NONE", 0, 0)

    cos_mat = np.asarray(info["cos_matrix"], dtype=float)
    plot_mat, cos_max, cos_avg = build_plot_matrix(cos_mat)

    matrix_str = my_project_backend.dump_matrix_rust(matrix_id)
    B = parse_matrix_string_to_numpy(matrix_str)

    ratio = compute_ratio(B) if ratio_override is None else ratio_override
    log_defect = compute_log_defect(B)

    return {
        "plot_mat": plot_mat,
        "ratio": ratio,
        "log_defect": log_defect,
        "min_norm": float(info.get("min_norm", 0.0)),
        "cos_max": cos_max,
        "cos_avg": cos_avg,
    }


# ============================================================
# Summary parser
# ============================================================


def parse_summary_successful_matrices(summary_path: Path):
    """
    Parse successful final matrices from dimXX_summary.txt.

    Return:
        seed -> {
            "ratio": float,
            "matrix_str": str
        }
    """
    text = summary_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    successful = {}
    in_basis_section = False
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if "FULL BASIS MATRICES" in line:
            in_basis_section = True
            i += 1
            continue

        if not in_basis_section:
            i += 1
            continue

        m = SEED_HEADER_RE.match(line)

        if m:
            seed = int(m.group(1))
            ratio = float(m.group(2))

            matrix_lines = []
            i += 1

            while i < len(lines) and not lines[i].strip():
                i += 1

            bracket_balance = 0
            started = False

            while i < len(lines):
                s = lines[i].rstrip()

                if not started and s.strip().startswith("["):
                    started = True

                if started:
                    matrix_lines.append(s)
                    bracket_balance += s.count("[")
                    bracket_balance -= s.count("]")

                    if bracket_balance == 0:
                        break

                i += 1

            matrix_str = "\n".join(matrix_lines)

            # 验证能否解析成方阵
            try:
                _ = parse_matrix_string_to_numpy(matrix_str)
                successful[seed] = {
                    "ratio": ratio,
                    "matrix_str": matrix_str,
                }
            except Exception as e:
                print(
                    f"[Warning] Failed to parse final matrix in {summary_path}, seed={seed}: {e}"
                )

        i += 1

    return successful


# ============================================================
# Backend generation for one case
# ============================================================


def generate_case_panels(
    dim: int, seed: int, final_ratio: float, final_matrix_str: str
):
    """
    Generate three panels:
        Initial LLL
        Fixed Local-BKZ
        Final DRL-BKZ
    """
    ids_to_free = []

    try:
        raw_str = read_dataset_basis_string(dim, seed)

        # 1. Initial LLL
        lll_id = my_project_backend.create_matrix_lll_rust(raw_str)
        ids_to_free.append(lll_id)

        lll_panel = panel_from_pool(lll_id)

        # 2. Fixed Local-BKZ from the LLL state
        fixed_id = my_project_backend.clone_matrix_rust(lll_id)
        ids_to_free.append(fixed_id)

        for method, beta, pos in FIXED_BKZ_STAGES:
            print(f"  Fixed baseline call: method={method}, beta={beta}, pos={pos}")
            _ = my_project_backend.reduce_rust(fixed_id, method, int(beta), int(pos))

        fixed_panel = panel_from_pool(fixed_id)

        # 3. Final DRL-BKZ from summary
        drl_id = my_project_backend.create_matrix_rust(final_matrix_str)
        ids_to_free.append(drl_id)

        drl_panel = panel_from_pool(drl_id, ratio_override=final_ratio)

        return [lll_panel, fixed_panel, drl_panel]

    finally:
        for mid in ids_to_free:
            try:
                my_project_backend.free_matrix_rust(mid)
            except Exception:
                pass


# ============================================================
# Case collection and selection
# ============================================================


def collect_successful_cases():
    cases = []

    for dim_dir in sorted(RESULTS_DIR.glob("a6up_dim*")):
        if not dim_dir.is_dir():
            continue

        mdir = SUMMARY_DIR_RE.match(dim_dir.name)

        if not mdir:
            continue

        dim = int(mdir.group(1))
        summary_path = dim_dir / f"dim{dim}_summary.txt"

        if not summary_path.exists():
            print(f"[Skip] No summary file: {summary_path}")
            continue

        successful = parse_summary_successful_matrices(summary_path)

        if not successful:
            print(f"[Skip] No successful final basis in {summary_path.name}")
            continue

        best_seed = min(successful.keys(), key=lambda s: successful[s]["ratio"])
        best = successful[best_seed]

        dataset_path = DATASET_DIR / f"svpchallengedim{dim}seed{best_seed}.txt"

        if not dataset_path.exists():
            print(f"[Skip] Dataset file missing for dim={dim}, seed={best_seed}")
            continue

        cases.append(
            {
                "dim": dim,
                "seed": best_seed,
                "ratio": best["ratio"],
                "final_matrix_str": best["matrix_str"],
                "summary_path": summary_path,
            }
        )

    return cases


def select_cases(cases):
    case_by_dim = {c["dim"]: c for c in cases}

    selected = []

    for d in PREFERRED_DIMS:
        if d in case_by_dim:
            selected.append(case_by_dim[d])

        if len(selected) >= MAX_CASES:
            break

    if len(selected) < MAX_CASES:
        used_dims = {c["dim"] for c in selected}

        for c in sorted(cases, key=lambda x: (x["dim"], x["ratio"])):
            if c["dim"] not in used_dims:
                selected.append(c)
                used_dims.add(c["dim"])

            if len(selected) >= MAX_CASES:
                break

    return selected


# ============================================================
# Plotting
# ============================================================


def panel_xlabel(stats):
    return (
        f"$\\rho={stats['ratio']:.6f}$\n"
        f"$D={stats['log_defect']:.2f}$\n"
        f"Min={stats['min_norm']:.2e}\n"
        f"CosMax={stats['cos_max']:.4f}\n"
        f"CosAvg={stats['cos_avg']:.4f}"
    )


def plot_heatmap_comparison(selected_cases):
    panels = []

    for case in selected_cases:
        dim = case["dim"]
        seed = case["seed"]

        print(
            f"\nGenerating panels for dim={dim}, seed={seed}, final ratio={case['ratio']:.6f}"
        )

        try:
            stats_list = generate_case_panels(
                dim=dim,
                seed=seed,
                final_ratio=case["ratio"],
                final_matrix_str=case["final_matrix_str"],
            )

            for name, stats in zip(["LLL", "Fixed", "DRL"], stats_list):
                print(
                    f"  {name}: shape={stats['plot_mat'].shape}, "
                    f"rho={stats['ratio']:.6f}, "
                    f"D={stats['log_defect']:.2f}, "
                    f"CosMax={stats['cos_max']:.4f}"
                )

            panels.append(
                {
                    "dim": dim,
                    "seed": seed,
                    "stats": stats_list,
                }
            )

        except Exception as e:
            print(
                f"[Skip] Failed to generate panels for dim={dim}, seed={seed}: {repr(e)}"
            )
            continue

    if not panels:
        print("No valid panels were generated. Figure was not saved.")
        return

    nrows = len(panels)
    ncols = 3

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(15.5, 5.4 * nrows),
        constrained_layout=False,
    )

    if nrows == 1:
        axes = np.array([axes])

    plt.subplots_adjust(hspace=0.68, wspace=0.18, right=0.91)

    col_titles = ["Initial LLL", FIXED_BKZ_LABEL, "Final DRL-BKZ"]
    im = None

    for i, panel in enumerate(panels):
        dim = panel["dim"]
        seed = panel["seed"]

        for j in range(ncols):
            ax = axes[i, j]
            stats = panel["stats"][j]

            im = ax.imshow(
                stats["plot_mat"],
                cmap=CMAP,
                vmin=HEATMAP_VMIN,
                vmax=HEATMAP_VMAX,
                interpolation="nearest",
            )

            if i == 0:
                ax.set_title(col_titles[j], fontsize=15, fontweight="bold", pad=12)

            if j == 0:
                ax.set_ylabel(f"Dim {dim}, Seed {seed}", fontsize=14, fontweight="bold")

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(panel_xlabel(stats), fontsize=9.0, labelpad=7)

    if im is not None:
        cbar_ax = fig.add_axes([0.93, 0.16, 0.015, 0.68])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(r"$|\cos(\mathbf{b}_i,\mathbf{b}_j)|$", fontsize=13)

    output_path = OUTPUT_DIR / "fig_5_8_lll_fixed_bkz_drl_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved heatmap comparison figure to: {output_path}")


# ============================================================
# Main
# ============================================================


def main():
    cases = collect_successful_cases()

    if not cases:
        print("No successful cases found from summary files.")
        return

    selected = select_cases(cases)

    print("\nSelected heatmap cases:")
    for c in selected:
        print(f"  Dim={c['dim']}, Seed={c['seed']}, Final DRL ratio={c['ratio']:.6f}")

    plot_heatmap_comparison(selected)


if __name__ == "__main__":
    main()
