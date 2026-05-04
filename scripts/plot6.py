#!/usr/bin/env python3
"""
Plot learning curves for Deep RL-based SVP from multiple dimensions.
Results are read from results/a6up_dim{dim}/dim{dim}_seed{seed}.txt,
and a step plot (horizontal lines between improvements) is produced.
"""

import os
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
# -------------------- Data parsing --------------------


def parse_file(filepath):
    """
    Parse a lattice reduction log file.
    Returns (initial_ratio, list_of_(episode, ratio)) sorted by episode.
    If the same episode appears multiple times, only the last (best) ratio is kept.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    initial_ratio = None
    records = []  # (ep, ratio)
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        # Initial state block
        if stripped.startswith("--- Initial State ---"):
            for j in range(i + 1, min(i + 10, len(lines))):
                if "Ratio" in lines[j] and "‖b₁‖/GH" in lines[j]:
                    val = float(lines[j].split(":")[-1].strip())
                    initial_ratio = val
                    break
        # New best block
        elif stripped.startswith("--- New Best Found (Episode"):
            match = re.search(r"Episode\s+(\d+)", stripped)
            if match:
                ep = int(match.group(1))
                for j in range(i + 1, min(i + 10, len(lines))):
                    if "Ratio" in lines[j] and "‖b₁‖/GH" in lines[j]:
                        val = float(lines[j].split(":")[-1].strip())
                        records.append((ep, val))
                        break
        i += 1

    if initial_ratio is None:
        raise ValueError(f"No initial ratio found in {filepath}")

    # Sort by episode and keep the last (lowest) ratio for each episode
    records.sort(key=lambda x: x[0])
    unique = []
    for ep, ratio in records:
        if unique and unique[-1][0] == ep:
            unique[-1] = (ep, ratio)  # replace with later (better) value
        else:
            unique.append((ep, ratio))

    return initial_ratio, unique


def build_step_curve(initial_ratio, best_records):
    """
    From initial ratio and a list of (ep, ratio) improvements,
    build episode and ratio arrays for a post-step plot.
    """
    if not best_records:
        max_ep = 1
    else:
        max_ep = best_records[-1][0]

    # 让横坐标从 0 开始
    episodes = list(range(0, max_ep + 2))
    ratios = [initial_ratio] * len(episodes)

    # 这样赋值后，0到ep之间保持initial_ratio，到了ep处发生下跌
    for ep, new_ratio in best_records:
        for idx in range(ep, len(episodes)):
            ratios[idx] = new_ratio

    return episodes, ratios


def get_best_curve_in_dim(dim, base_dir):
    """
    For dimension `dim` (except 67), find all txt files in the folder,
    parse each, and return the curve (episodes, ratios) that achieves the
    smallest final ratio.
    """
    folder = base_dir / f"a6up_dim{dim}"
    if not folder.is_dir():
        print(f"Warning: folder not found: {folder}")
        return None

    best_final = float("inf")
    best_curve = None

    for fpath in folder.glob("*.txt"):
        try:
            init, records = parse_file(fpath)
            eps, rats = build_step_curve(init, records)
            final_ratio = rats[-1]
            if final_ratio < best_final:
                best_final = final_ratio
                best_curve = (eps, rats)
        except Exception as e:
            print(f"Skipping {fpath}: {e}")

    return best_curve


# -------------------- Plotting --------------------


def exact_number_formatter(x, pos):
    """直接返回具体的数字而不是科学计数法"""
    if x < 0:
        return ""
    return f"{int(x)}"


def main():
    # Locate the project root (scripts/ folder is one level below root)
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    results_dir = root_dir / "results"

    # Dimensions to plot
    dims = [55, 56, 57, 58, 59, 67]
    curves = {}  # dim -> (episodes, ratios)

    for dim in dims:
        if dim == 67:
            fpath = results_dir / "a6up_dim67" / "dim67_seed13.txt"
            if fpath.is_file():
                try:
                    init, records = parse_file(fpath)
                    eps, rats = build_step_curve(init, records)
                    curves[dim] = (eps, rats)
                except Exception as e:
                    print(f"Failed to parse dim67: {e}")
            else:
                print(f"Warning: dim67 file not found: {fpath}")
        else:
            curve = get_best_curve_in_dim(dim, results_dir)
            if curve is not None:
                curves[dim] = curve
            else:
                print(f"No valid data for dim={dim}")

    if not curves:
        print("No data to plot!")
        return

    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color cycle
    sns.set_palette("muted")
    colors = sns.color_palette("muted", len(curves))
    for (dim, (eps, rats)), color in zip(sorted(curves.items()), colors):
        ax.step(eps, rats, where="post", linewidth=0.9, label=f"dim={dim}", color=color)

    # Red dashed line at y = 1.05
    ax.axhline(y=1.05, color="red", linestyle="--", linewidth=0.9, alpha=0.8, zorder=0)

    # ==========================
    # 核心修改：坐标轴处理
    # ==========================
    # 使用 symlog (对称对数)，允许坐标轴包含 0。
    # linthresh=1.0 意味着 0 到 1 之间是线性，1 之后是底数为 2 的对数缩放
    ax.set_xscale("symlog", base=2, linthresh=1.0)

    # 强制让 Locator 以底数为 2 的间隔进行标注 (0, 1, 2, 4, 8, 16...)
    ax.xaxis.set_major_locator(ticker.SymmetricalLogLocator(base=2, linthresh=1.0))

    # 使用自定义的格式化器显示具体的纯数字
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(exact_number_formatter))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    # 强制让 X 轴从 0 开始
    ax.set_xlim(left=0)

    # Grid
    ax.grid(True, which="major", linestyle=":", linewidth=0.4, alpha=0.7)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.2, alpha=0.4)

    # Axes labels & title
    ax.set_xlabel("Episode")
    ax.set_ylabel(r"$\|b_1\| / \mathrm{GH}$")
    ax.set_title("Deep Reinforcement Learning for SVP")

    # Legend
    ax.legend(title="Learning Curve", fontsize=9, title_fontsize=10)

    plt.tight_layout()
    # Save the plot in the results directory
    out_path = results_dir / "learning_curve.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Learning curve saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
