from __future__ import annotations

import os


def append_training_log(results_dir, message: str):
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "training.log"), "a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def save_seed_result(results_dir, info, is_update):
    dim_dir = os.path.join(results_dir, f"dim{info['dim']}")
    os.makedirs(dim_dir, exist_ok=True)
    filepath = os.path.join(dim_dir, f"seed{info['seed_id']}.txt")

    with open(filepath, "a" if is_update else "w", encoding="utf-8") as f:
        if not is_update:
            f.write(
                "=" * 60
                + f"\n Dim={info['dim']} Seed={info['seed_id']} File={info['seed_file']}\n"
                + "=" * 60
                + "\n--- Initial ---\n"
            )
        else:
            f.write(f"\n--- New Best (Episode {info['episode']}) ---\n")
        f.write(f"  Ratio: {info['ratio']:.8f}\n")
        if info.get("defect") is not None:
            f.write(
                f"  Defect: {info['defect']:.8f}  MaxCos: {info['max_cos']:.6f}  "
                f"MinCos: {info['min_cos']:.6f}\n"
            )
        f.write(f"  b1 = {info.get('vector')}\n")


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
            f.write(
                f"  [{status}] seed{info['seed_id']:3d}: "
                f"{info['ratio']:.6f} (ep {info.get('episode', '?')})\n"
            )


def save_final_summary(results_dir, all_infos, goal=1.05):
    filepath = os.path.join(results_dir, "summary.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        reached = [info for info in all_infos if info["ratio"] < goal]
        f.write("=" * 60 + f"\n FINAL SUMMARY (goal<{goal})\n" + "=" * 60 + "\n")
        f.write(f"Reached: {len(reached)}/{len(all_infos)}\n\n")
        for info in sorted(all_infos, key=lambda x: (x["dim"], x["seed_id"])):
            status = "✓" if info["ratio"] < goal else " "
            f.write(
                f"  [{status}] dim{info['dim']:3d} seed{info['seed_id']:3d}: "
                f"{info['ratio']:.6f} (ep {info.get('episode', '?')})\n"
            )


def plot_training_history(results_dir, history, goal_threshold):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"])
    plt.title("loss")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history["best_min"])
    plt.axhline(goal_threshold, ls="--")
    plt.title("global best ratio")
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "training.png"))
    plt.close()
