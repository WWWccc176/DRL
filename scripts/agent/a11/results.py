from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


def append_training_log(results_dir, message: str):
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "training.log"), "a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def _basis_to_text(basis) -> str:
    if not basis:
        return "[]\n"
    rows = ["[" + " ".join(str(int(value)) for value in row) + "]" for row in basis]
    return "[\n" + "\n".join(rows) + "\n]\n"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return value.item()
    except Exception:
        return str(value)


def save_seed_result(results_dir, info, is_update):
    """Append a human-readable best record and refresh machine-readable artifacts.

    Every saved best record contains the full integer basis and the complete action
    path that produced it. The current best basis is also written as a standalone
    fplll-style ``.txt`` file so post-training analysis can generate cosine maps
    without importing the native backend.
    """
    dim = int(info["dim"])
    seed_id = int(info["seed_id"])
    episode = int(info.get("episode", 0))
    action_path = list(info.get("action_path") or [])
    basis = info.get("basis") or []
    initial_basis = info.get("initial_basis") or []

    dim_dir = Path(results_dir) / f"dim{dim}"
    dim_dir.mkdir(parents=True, exist_ok=True)
    filepath = dim_dir / f"seed{seed_id}.txt"

    with filepath.open("a" if is_update else "w", encoding="utf-8") as f:
        if not is_update:
            f.write(
                "=" * 60
                + f"\n Dim={dim} Seed={seed_id} File={info['seed_file']}\n"
                + "=" * 60
                + "\n--- Initial ---\n"
            )
        else:
            f.write(f"\n--- New Best (Episode {episode}) ---\n")

        f.write(f"  Ratio: {info['ratio']:.10f}\n")
        if info.get("defect") is not None:
            f.write(
                f"  Defect: {info['defect']:.10f}  MaxCos: {info['max_cos']:.8f}  "
                f"MinCos: {info['min_cos']:.8f}\n"
            )
        f.write(f"  b1 = {info.get('vector')}\n")
        f.write(f"  Action Path Length: {len(action_path)}\n")
        f.write(
            "  Action Path JSON: "
            + json.dumps(_json_safe(action_path), ensure_ascii=False, separators=(",", ":"))
            + "\n"
        )
        f.write("  Full Basis:\n")
        f.write(_basis_to_text(basis))

    basis_path = dim_dir / "best_basis" / f"seed{seed_id}.txt"
    initial_basis_path = dim_dir / "initial_basis" / f"seed{seed_id}.txt"
    path_json = dim_dir / "best_paths" / f"seed{seed_id}.json"
    metadata_path = dim_dir / "best_records" / f"seed{seed_id}.json"

    _atomic_write_text(basis_path, _basis_to_text(basis))
    _atomic_write_text(initial_basis_path, _basis_to_text(initial_basis))
    _atomic_write_text(
        path_json,
        json.dumps(_json_safe(action_path), ensure_ascii=False, indent=2) + "\n",
    )

    metadata = {
        "dim": dim,
        "seed_id": seed_id,
        "seed_file": info.get("seed_file"),
        "ratio": float(info["ratio"]),
        "defect": None if info.get("defect") is None else float(info["defect"]),
        "max_cos": None if info.get("max_cos") is None else float(info["max_cos"]),
        "min_cos": None if info.get("min_cos") is None else float(info["min_cos"]),
        "episode": episode,
        "vector": _json_safe(info.get("vector")),
        "action_path": _json_safe(action_path),
        "basis_file": str(basis_path.relative_to(Path(results_dir))),
        "initial_basis_file": str(initial_basis_path.relative_to(Path(results_dir))),
        "result_file": str(filepath.relative_to(Path(results_dir))),
    }
    _atomic_write_text(
        metadata_path,
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    )

    return {
        "result_file": str(filepath),
        "basis_file": str(basis_path),
        "initial_basis_file": str(initial_basis_path),
        "path_file": str(path_json),
        "metadata_file": str(metadata_path),
    }


def save_dimension_summary(results_dir, dim, all_infos, goal=1.05):
    dim_dir = os.path.join(results_dir, f"dim{dim}")
    os.makedirs(dim_dir, exist_ok=True)
    filepath = os.path.join(dim_dir, "summary.txt")
    infos = sorted(
        (info for info in all_infos if info["dim"] == dim),
        key=lambda x: x["seed_id"],
    )

    with open(filepath, "w", encoding="utf-8") as f:
        reached = [info for info in infos if info["ratio"] < goal]
        f.write("=" * 60 + f"\n DIM {dim} SUMMARY (goal<{goal})\n" + "=" * 60 + "\n")
        f.write(f"Reached: {len(reached)}/{len(infos)}\n\n")
        for info in infos:
            status = "✓" if info["ratio"] < goal else " "
            path_len = len(info.get("action_path") or [])
            f.write(
                f"  [{status}] seed{info['seed_id']:3d}: "
                f"{info['ratio']:.6f} (ep {info.get('episode', '?')}, actions {path_len})\n"
            )


def save_final_summary(results_dir, all_infos, goal=1.05):
    filepath = os.path.join(results_dir, "summary.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        reached = [info for info in all_infos if info["ratio"] < goal]
        f.write("=" * 60 + f"\n FINAL SUMMARY (goal<{goal})\n" + "=" * 60 + "\n")
        f.write(f"Reached: {len(reached)}/{len(all_infos)}\n\n")
        for info in sorted(all_infos, key=lambda x: (x["dim"], x["seed_id"])):
            status = "✓" if info["ratio"] < goal else " "
            path_len = len(info.get("action_path") or [])
            f.write(
                f"  [{status}] dim{info['dim']:3d} seed{info['seed_id']:3d}: "
                f"{info['ratio']:.6f} (ep {info.get('episode', '?')}, actions {path_len})\n"
            )


def plot_training_history(results_dir, history, goal_threshold):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history["loss"])
    axes[0].set_title("Loss")
    axes[0].grid(True)
    axes[1].plot(history["best_min"])
    axes[1].axhline(goal_threshold, ls="--")
    axes[1].set_title("Global best ratio")
    axes[1].grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "training.png"))
    plt.close(fig)
