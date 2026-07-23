from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

from .io_utils import parse_fplll


# Color choices:
# - cividis: perceptually uniform and color-vision-deficiency friendly.
# - Okabe-Ito blue/vermillion: established colorblind-safe scientific palette.
# - action map: green -> blue -> purple, with larger beta mapped to deeper purple.
_HEATMAP_CMAP = "cividis"
_INIT_COLOR = "#0072B2"
_FINAL_COLOR = "#D55E00"
_ACTION_ANCHORS = ["#009E73", "#0072B2", "#6A3D9A"]


def _normalized_rows(basis: list[list[int]]) -> np.ndarray:
    """Convert arbitrary-size integer rows to stable unit float vectors."""
    if not basis:
        raise ValueError("empty basis")

    width = len(basis[0])
    if width == 0 or any(len(row) != width for row in basis):
        raise ValueError("basis rows have inconsistent lengths")

    normalized = np.empty((len(basis), width), dtype=np.float64)
    for index, row in enumerate(basis):
        maximum = max(abs(int(value)) for value in row)
        if maximum == 0:
            raise ValueError(f"basis row {index} is zero")
        scaled = np.fromiter(
            (float(int(value) / maximum) for value in row),
            dtype=np.float64,
            count=width,
        )
        norm = float(np.linalg.norm(scaled))
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError(f"basis row {index} has invalid norm")
        normalized[index] = scaled / norm
    return normalized


def cosine_matrix_from_basis(basis: list[list[int]]) -> np.ndarray:
    rows = _normalized_rows(basis)
    cosine = np.abs(rows @ rows.T)
    np.clip(cosine, 0.0, 1.0, out=cosine)
    return cosine


def _scaled_integer_matrix(basis: list[list[int]]) -> tuple[np.ndarray, int]:
    """Convert a large integer matrix to float64 after one global power-of-two scale.

    A single global scale preserves all relative row lengths and therefore the GSO
    profile. Mantissas are truncated to about 53 bits before ``ldexp`` so arbitrary
    Python integers never overflow during conversion.
    """
    if not basis:
        raise ValueError("empty basis")
    width = len(basis[0])
    if width == 0 or any(len(row) != width for row in basis):
        raise ValueError("basis rows have inconsistent lengths")

    scale_bits = max(abs(int(value)).bit_length() for row in basis for value in row)
    if scale_bits == 0:
        raise ValueError("zero basis")

    matrix = np.empty((len(basis), width), dtype=np.float64)
    for i, row in enumerate(basis):
        for j, raw in enumerate(row):
            value = int(raw)
            if value == 0:
                matrix[i, j] = 0.0
                continue
            sign = -1.0 if value < 0 else 1.0
            magnitude = abs(value)
            bits = magnitude.bit_length()
            shift = max(0, bits - 53)
            mantissa = float(magnitude >> shift)
            matrix[i, j] = sign * math.ldexp(mantissa, shift - scale_bits)
    return matrix, scale_bits


def gso_log_norms_from_basis(basis: list[list[int]]) -> np.ndarray:
    """Return natural-log Gram-Schmidt row norms in basis order."""
    matrix, scale_bits = _scaled_integer_matrix(basis)
    # QR on B^T performs Gram-Schmidt on the rows of B in their original order.
    _, r = np.linalg.qr(matrix.T, mode="reduced")
    diagonal = np.abs(np.diag(r))
    floor = np.finfo(np.float64).tiny
    return np.log(np.maximum(diagonal, floor)) + scale_bits * math.log(2.0)


def _cosine_stats(cosine: np.ndarray) -> tuple[float, float, float]:
    if cosine.shape[0] <= 1:
        return 0.0, 0.0, 0.0
    values = cosine[np.tril_indices(cosine.shape[0], -1)]
    if not values.size:
        return 0.0, 0.0, 0.0
    return float(np.max(values)), float(np.mean(values)), float(np.min(values))


def _scientific_rc() -> dict[str, Any]:
    return {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9.5,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }


def save_cosine_comparison(
    initial_cosine: np.ndarray,
    final_cosine: np.ndarray,
    output_path: Path,
    *,
    title: str,
    subtitle: str = "",
) -> None:
    """Save a 1x2 initial/final full cosine heatmap with one shared scale."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = final_cosine.shape[0]
    figure_width = max(10.5, min(16.0, 9.5 + n / 22.0))

    with plt.rc_context(_scientific_rc()):
        fig, axes = plt.subplots(1, 2, figsize=(figure_width, figure_width * 0.46))
        matrices = (initial_cosine, final_cosine)
        panel_titles = ("Initial: LLL + BKZ-20", "Final")
        image = None
        for ax, matrix, panel_title in zip(axes, matrices, panel_titles):
            image = ax.imshow(
                matrix,
                vmin=0.0,
                vmax=1.0,
                cmap=_HEATMAP_CMAP,
                interpolation="nearest",
                aspect="equal",
                origin="upper",
            )
            ax.set_title(panel_title)
            ax.set_xlabel("Basis vector index")
            ax.set_ylabel("Basis vector index")

        fig.suptitle(title + (f"\n{subtitle}" if subtitle else ""), y=1.02)
        colorbar = fig.colorbar(image, ax=axes, fraction=0.032, pad=0.035)
        colorbar.set_label(r"Absolute cosine $|\cos(b_i,b_j)|$")
        fig.savefig(output_path)
        plt.close(fig)


def save_gso_comparison(
    initial_gso: np.ndarray,
    final_gso: np.ndarray,
    output_path: Path,
    *,
    title: str,
    subtitle: str = "",
) -> None:
    """Plot initial and final log-GSO profiles on one scientific line chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_initial = np.arange(len(initial_gso))
    x_final = np.arange(len(final_gso))

    with plt.rc_context(_scientific_rc()):
        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        ax.plot(
            x_initial,
            initial_gso,
            color=_INIT_COLOR,
            linewidth=1.8,
            marker="o",
            markersize=2.8,
            markevery=max(1, len(initial_gso) // 18),
            label="Initial: LLL + BKZ-20",
        )
        ax.plot(
            x_final,
            final_gso,
            color=_FINAL_COLOR,
            linewidth=1.8,
            marker="s",
            markersize=2.6,
            markevery=max(1, len(final_gso) // 18),
            label="Final",
        )
        ax.set_title(title + (f"\n{subtitle}" if subtitle else ""))
        ax.set_xlabel("Basis vector index $i$")
        ax.set_ylabel(r"$\log \|b_i^*\|$")
        ax.grid(True, which="major", color="#D9D9D9", linewidth=0.65, alpha=0.75)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)


def save_action_path_plot(
    action_path: list[dict[str, Any]],
    dim: int,
    output_path: Path,
    *,
    title: str,
    subtitle: str = "",
) -> None:
    """Plot each action as a vertical interval [pos, pos+beta] at its path step."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle
    from matplotlib.cm import ScalarMappable

    output_path.parent.mkdir(parents=True, exist_ok=True)
    actions = [
        action
        for action in action_path
        if action.get("pos") is not None and action.get("beta") is not None
    ]
    if not actions:
        with plt.rc_context(_scientific_rc()):
            fig, ax = plt.subplots(figsize=(8.5, 5.4))
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0, dim)
            ax.set_xlabel("Action sequence step")
            ax.set_ylabel("Lattice dimension index")
            ax.set_title(title + (f"\n{subtitle}" if subtitle else ""))
            ax.text(
                0.5,
                0.5 * dim,
                "No actions in the saved best path",
                ha="center",
                va="center",
                color="#4D4D4D",
            )
            fig.tight_layout()
            fig.savefig(output_path)
            plt.close(fig)
        return

    betas = np.asarray([int(action["beta"]) for action in actions], dtype=float)
    beta_min = float(np.min(betas))
    beta_max = float(np.max(betas))
    norm = Normalize(vmin=beta_min, vmax=beta_max if beta_max > beta_min else beta_min + 1.0)
    cmap = LinearSegmentedColormap.from_list(
        "scientific_green_blue_purple",
        _ACTION_ANCHORS,
        N=256,
    )

    figure_width = max(8.5, min(18.0, 6.5 + 0.16 * len(actions)))
    with plt.rc_context(_scientific_rc()):
        fig, ax = plt.subplots(figsize=(figure_width, 5.4))
        for sequence_step, action in enumerate(actions, start=1):
            pos = int(action["pos"])
            beta = int(action["beta"])
            end = min(dim, pos + beta)
            rectangle = Rectangle(
                (sequence_step - 0.38, pos),
                0.76,
                max(0, end - pos),
                facecolor=cmap(norm(beta)),
                edgecolor="#2B2B2B",
                linewidth=0.35,
            )
            ax.add_patch(rectangle)

        ax.set_xlim(0.4, len(actions) + 0.6)
        ax.set_ylim(0, dim)
        ax.set_xlabel("Action sequence step")
        ax.set_ylabel("Lattice dimension index")
        ax.set_title(title + (f"\n{subtitle}" if subtitle else ""))
        ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)

        scalar = ScalarMappable(norm=norm, cmap=cmap)
        scalar.set_array([])
        colorbar = fig.colorbar(scalar, ax=ax, fraction=0.028, pad=0.025)
        colorbar.set_label(r"Block size $\beta$")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)


def _load_records(results_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for metadata_path in sorted(results_dir.glob("dim*/best_records/seed*.json")):
        try:
            record = json.loads(metadata_path.read_text(encoding="utf-8"))
            basis_path = results_dir / record["basis_file"]
            initial_basis_path = results_dir / record["initial_basis_file"]
            basis = parse_fplll(basis_path.read_text(encoding="utf-8"))
            initial_basis = parse_fplll(initial_basis_path.read_text(encoding="utf-8"))
            if not basis or not initial_basis:
                raise ValueError("initial or final basis is empty")
            record["basis_path"] = str(basis_path)
            record["initial_basis_path"] = str(initial_basis_path)
            record["basis"] = basis
            record["initial_basis"] = initial_basis
            records.append(record)
        except Exception as exc:
            print(f"[A11 analysis] skip {metadata_path}: {exc}", flush=True)
    return records


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_post_training_analysis(
    results_dir: str | os.PathLike[str], goal: float = 1.05
) -> dict[str, Any]:
    """Summarize best bases and generate cosine, GSO, and action-path figures."""
    results_path = Path(results_dir).resolve()
    output_dir = results_path / "analysis"
    heatmap_dir = output_dir / "cosine_heatmaps"
    gso_dir = output_dir / "gso_profiles"
    action_dir = output_dir / "action_paths"
    for directory in (output_dir, heatmap_dir, gso_dir, action_dir):
        directory.mkdir(parents=True, exist_ok=True)

    records = _load_records(results_path)
    seed_rows: list[dict[str, Any]] = []
    action_counter: Counter[tuple[Any, ...]] = Counter()

    for record in records:
        final_basis = record.pop("basis")
        initial_basis = record.pop("initial_basis")
        initial_cosine = cosine_matrix_from_basis(initial_basis)
        final_cosine = cosine_matrix_from_basis(final_basis)
        initial_gso = gso_log_norms_from_basis(initial_basis)
        final_gso = gso_log_norms_from_basis(final_basis)
        cos_max, cos_avg, cos_min = _cosine_stats(final_cosine)
        dim = int(record["dim"])
        seed_id = int(record["seed_id"])
        action_path = list(record.get("action_path") or [])
        common_subtitle = (
            f"ratio={float(record['ratio']):.8f}, episode={record.get('episode', 0)}, "
            f"actions={len(action_path)}"
        )

        heatmap_path = heatmap_dir / f"dim{dim}_seed{seed_id}.png"
        save_cosine_comparison(
            initial_cosine,
            final_cosine,
            heatmap_path,
            title=f"A11 cosine comparison: dim {dim}, seed {seed_id}",
            subtitle=common_subtitle,
        )

        gso_path = gso_dir / f"dim{dim}_seed{seed_id}.png"
        save_gso_comparison(
            initial_gso,
            final_gso,
            gso_path,
            title=f"A11 GSO profile: dim {dim}, seed {seed_id}",
            subtitle=common_subtitle,
        )

        action_plot_path = action_dir / f"dim{dim}_seed{seed_id}.png"
        save_action_path_plot(
            action_path,
            dim,
            action_plot_path,
            title=f"A11 action path: dim {dim}, seed {seed_id}",
            subtitle=common_subtitle,
        )

        accepted_actions = 0
        for action in action_path:
            key = (
                action.get("kind", "agent_action"),
                action.get("pos"),
                action.get("beta"),
                action.get("backend", "unknown"),
                bool(action.get("accepted", True)),
            )
            action_counter[key] += 1
            if bool(action.get("accepted", True)):
                accepted_actions += 1

        seed_rows.append(
            {
                "dim": dim,
                "seed": seed_id,
                "ratio": float(record["ratio"]),
                "success": int(float(record["ratio"]) < goal),
                "episode": int(record.get("episode", 0)),
                "defect": record.get("defect"),
                "saved_max_cos": record.get("max_cos"),
                "matrix_cos_max": cos_max,
                "matrix_cos_avg": cos_avg,
                "matrix_cos_min": cos_min,
                "action_path_length": len(action_path),
                "accepted_path_entries": accepted_actions,
                "basis_file": record["basis_file"],
                "initial_basis_file": record["initial_basis_file"],
                "heatmap_file": str(heatmap_path.relative_to(results_path)),
                "gso_file": str(gso_path.relative_to(results_path)),
                "action_plot_file": str(action_plot_path.relative_to(results_path)),
            }
        )

    seed_rows.sort(key=lambda row: (row["dim"], row["seed"]))
    _write_csv(
        output_dir / "seed_summary.csv",
        [
            "dim",
            "seed",
            "ratio",
            "success",
            "episode",
            "defect",
            "saved_max_cos",
            "matrix_cos_max",
            "matrix_cos_avg",
            "matrix_cos_min",
            "action_path_length",
            "accepted_path_entries",
            "basis_file",
            "initial_basis_file",
            "heatmap_file",
            "gso_file",
            "action_plot_file",
        ],
        seed_rows,
    )

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[int(row["dim"])].append(row)

    dimension_rows: list[dict[str, Any]] = []
    for dim in sorted(grouped):
        rows = grouped[dim]
        ratios = [float(row["ratio"]) for row in rows]
        path_lengths = [int(row["action_path_length"]) for row in rows]
        dimension_rows.append(
            {
                "dim": dim,
                "seeds": len(rows),
                "successes": sum(int(row["success"]) for row in rows),
                "best_ratio": min(ratios),
                "mean_ratio": mean(ratios),
                "std_ratio": pstdev(ratios) if len(ratios) > 1 else 0.0,
                "worst_ratio": max(ratios),
                "mean_action_path_length": mean(path_lengths),
                "mean_cos_max": mean(float(row["matrix_cos_max"]) for row in rows),
                "mean_cos_avg": mean(float(row["matrix_cos_avg"]) for row in rows),
            }
        )

    _write_csv(
        output_dir / "dimension_summary.csv",
        [
            "dim",
            "seeds",
            "successes",
            "best_ratio",
            "mean_ratio",
            "std_ratio",
            "worst_ratio",
            "mean_action_path_length",
            "mean_cos_max",
            "mean_cos_avg",
        ],
        dimension_rows,
    )

    action_rows = [
        {
            "kind": key[0],
            "pos": key[1],
            "beta": key[2],
            "backend": key[3],
            "accepted": int(key[4]),
            "count": count,
        }
        for key, count in sorted(
            action_counter.items(),
            key=lambda item: (-item[1], str(item[0])),
        )
    ]
    _write_csv(
        output_dir / "action_usage.csv",
        ["kind", "pos", "beta", "backend", "accepted", "count"],
        action_rows,
    )

    successes = sum(int(row["success"]) for row in seed_rows)
    report_lines = [
        "A11 automatic post-training analysis",
        "=" * 72,
        f"Results directory: {results_path}",
        f"Saved best bases analyzed: {len(seed_rows)}",
        f"Goal: ratio < {goal}",
        f"Goal reached: {successes}/{len(seed_rows)}",
        f"Dimensions: {sorted(grouped)}",
        "",
        "Generated:",
        f"- {output_dir / 'seed_summary.csv'}",
        f"- {output_dir / 'dimension_summary.csv'}",
        f"- {output_dir / 'action_usage.csv'}",
        f"- {heatmap_dir} ({len(seed_rows)} initial/final cosine figures)",
        f"- {gso_dir} ({len(seed_rows)} GSO figures)",
        f"- {action_dir} ({len(seed_rows)} action-path figures)",
    ]
    report_path = output_dir / "analysis_summary.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    summary = {
        "records": len(seed_rows),
        "successes": successes,
        "dimensions": sorted(grouped),
        "output_dir": str(output_dir),
        "report": str(report_path),
    }
    print("\n".join(report_lines), flush=True)
    return summary


def main() -> None:
    from .config import GOAL_THRESHOLD, RESULTS_DIR

    run_post_training_analysis(
        RESULTS_DIR,
        goal=GOAL_THRESHOLD,
    )


if __name__ == "__main__":
    main()
