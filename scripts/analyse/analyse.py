#!/usr/bin/env python3
"""
analyse.py — aggregate results/bench/raw/*.json into stats + figures.

Outputs (results/bench/):
  per_dim_stats.csv / per_dim_stats.txt
  stats_by_method.csv
  figures/scatter_b1gh_vs_dim.png
  figures/scatter_maxcos_vs_dim.png
  figures/heatmap_cos_dim10.png

Robust to schema drift and to the old 'G6K' method name (now 'ENUM_SIEVE').
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results" / "bench"
RAW = OUT / "raw"
COS = OUT / "cos"
FIG = OUT / "figures"

METHOD_ORDER = ["LLL", "BKZ", "ENUM", "ENUM_SIEVE"]
METHOD_COLORS = {
    "LLL": "#1f77b4",
    "BKZ": "#2ca02c",
    "ENUM": "#ff7f0e",
    "ENUM_SIEVE": "#d62728",
}
METHOD_LABELS = {  # legend / title display names
    "LLL": "LLL",
    "BKZ": "BKZ",
    "ENUM": "ENUM",
    "ENUM_SIEVE": "ENUM+SIEVE",
}

CMAP = "RdYlGn_r"  # small = green, large = red
HM_VMIN, HM_VMAX = 0.0, 0.7


# --------------------------------------------------------------------------
# load
# --------------------------------------------------------------------------
def load_df():
    rows = []
    for p in sorted(RAW.glob("*.json")):
        if p.name.startswith("ERR_"):
            continue
        try:
            rows.append(json.loads(p.read_text()))
        except Exception as e:
            print(f"[warn] skip {p.name}: {e}")

    if not rows:
        raise SystemExit(f"No records in {RAW}. Run bench.py first.")

    df = pd.DataFrame(rows)

    # unify quality-metric name (old files used 'rhf')
    if "b1_gh" not in df.columns and "rhf" in df.columns:
        df["b1_gh"] = df["rhf"]

    # derive ||b1|| from log_b1 if present
    if "log_b1" in df.columns:
        df["b1"] = np.exp(pd.to_numeric(df["log_b1"], errors="coerce"))

    # unify method name (old files used 'G6K')
    if "method" in df.columns:
        df["method"] = df["method"].replace({"G6K": "ENUM_SIEVE"})

    df["dim"] = pd.to_numeric(df["dim"], errors="coerce").astype("Int64")
    if "method" not in df.columns:
        raise SystemExit("records have no 'method' column")
    return df


# --------------------------------------------------------------------------
# per-dimension statistics (norm & cosine: max / min / mean / var)
# --------------------------------------------------------------------------
NORM_COS_METRICS = [
    ("b1", "||b1||"),
    ("b1_gh", "||b1||/GH"),
    ("max_cos", "max|cos|"),
    ("mean_cos", "mean|cos|"),
]


def _stats(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    a = s.to_numpy(float)
    return float(a.max()), float(a.min()), float(a.mean()), float(a.var(ddof=0))


def per_dim_stats(df):
    metrics = [(k, lbl) for (k, lbl) in NORM_COS_METRICS if k in df.columns]
    recs = []
    for (method, dim), sub in df.groupby(["method", "dim"]):
        row = {"method": method, "dim": int(dim), "n_seeds": len(sub)}
        for key, _ in metrics:
            mx, mn, me, va = _stats(sub[key])
            row[f"{key}_max"] = mx
            row[f"{key}_min"] = mn
            row[f"{key}_mean"] = me
            row[f"{key}_var"] = va
        recs.append(row)

    out = pd.DataFrame(recs)
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    out["__mo"] = out["method"].map(order).fillna(99)
    out = out.sort_values(["__mo", "dim"]).drop(columns="__mo").reset_index(drop=True)
    return out


def save_stats(df):
    OUT.mkdir(parents=True, exist_ok=True)

    pds = per_dim_stats(df)
    pds.round(6).to_csv(OUT / "per_dim_stats.csv", index=False)
    (OUT / "per_dim_stats.txt").write_text(
        "Per-dimension statistics (norm & cosine: max / min / mean / var)\n"
        "  Methods: LLL / BKZ / ENUM / ENUM+SIEVE\n"
        "  ||b1||    = shortest vector length\n"
        "  ||b1||/GH = length over Gaussian-Heuristic (near 1 = good)\n"
        "  max|cos|, mean|cos| aggregated across seeds within each dimension\n"
        "  var = population variance (ddof=0)\n"
        + "=" * 110
        + "\n"
        + pds.round(6).to_string(index=False)
        + "\n",
        encoding="utf-8",
    )

    feats = [
        c
        for c in [
            "b1",
            "b1_gh",
            "log_b1",
            "slope",
            "logdet",
            "max_cos",
            "mean_cos",
            "time_s",
            "calls",
            "steps_done",
        ]
        if c in df.columns
    ]
    by_m = df.groupby("method")[feats].agg(["mean", "std", "min", "max"]).round(6)
    by_m.to_csv(OUT / "stats_by_method.csv")

    print(">> per_dim_stats.csv / per_dim_stats.txt / stats_by_method.csv")
    return pds


# --------------------------------------------------------------------------
# scatter plots
# --------------------------------------------------------------------------
def scatter(df, ycol, ylabel, fname, target=None):
    if ycol not in df.columns:
        print(f"[skip] scatter {fname}: column '{ycol}' missing")
        return

    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6.5))

    for m in [x for x in METHOD_ORDER if x in df.method.unique()]:
        sub = df[df.method == m]
        x = pd.to_numeric(sub["dim"], errors="coerce")
        y = pd.to_numeric(sub[ycol], errors="coerce")
        col = METHOD_COLORS.get(m)
        ax.scatter(x, y, s=22, alpha=0.5, label=METHOD_LABELS.get(m, m), color=col)
        g = sub.groupby("dim")[ycol].mean()  # mean trend line
        ax.plot(g.index.astype(int), g.values, lw=1.7, color=col)

    if target is not None:
        ax.axhline(target, ls="--", lw=1.0, color="gray", label=f"target={target}")

    ax.set_xlabel("Dimension")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs dimension")
    ax.grid(True, ls=":", lw=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f">> figures/{fname}")


# --------------------------------------------------------------------------
# cosine heatmaps (dims that are multiples of 10)
# --------------------------------------------------------------------------
def load_cos_display(dim, method):
    cand = sorted(COS.glob(f"{dim}_{method}_*.csv"))
    if not cand and method == "ENUM_SIEVE":  # old files used 'G6K'
        cand = sorted(COS.glob(f"{dim}_G6K_*.csv"))
    if not cand:
        return None
    try:
        C = np.loadtxt(cand[0], delimiter=",")
    except Exception:
        return None
    C = np.atleast_2d(np.asarray(C, float))
    if C.ndim != 2 or C.shape[0] != C.shape[1] or C.shape[0] < 2:
        return None
    C = np.abs(C)
    if np.allclose(np.triu(C, 1), 0.0):  # stored as lower-triangle only
        C = C + C.T
    np.fill_diagonal(C, 0.0)
    return C


def heatmap_panel(C):
    """Lower-tri of C[1:,:-1] plus (cos_max, cos_avg)."""
    sub = C[1:, :-1]
    plot_mat = np.tril(sub)
    mask = np.tril(np.ones_like(sub), 0).astype(bool)
    vals = sub[mask]
    vals = vals[np.isfinite(vals)]
    cmax = float(vals.max()) if len(vals) else 0.0
    cavg = float(vals.mean()) if len(vals) else 0.0
    return plot_mat, cmax, cavg


def plot_heatmaps(df):
    dims = sorted({int(d) for d in df.dim.dropna().unique() if int(d) % 10 == 0})
    methods = [m for m in METHOD_ORDER if m in df.method.unique()]

    grid, kept_dims = {}, []
    for d in dims:
        row = {}
        for m in methods:
            C = load_cos_display(d, m)
            if C is not None:
                row[m] = C
        if row:
            grid[d] = row
            kept_dims.append(d)

    kept_methods = [m for m in methods if any(m in grid[d] for d in kept_dims)]
    if not kept_dims or not kept_methods:
        print(
            "[skip] heatmap: no cosine matrices for multiples of 10 "
            "(only the FIRST file of each dim is saved to results/bench/cos)"
        )
        return

    FIG.mkdir(parents=True, exist_ok=True)
    nr, nc = len(kept_dims), len(kept_methods)
    fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 4.2 * nr), squeeze=False)

    im = None
    for i, d in enumerate(kept_dims):
        for j, m in enumerate(kept_methods):
            ax = axes[i][j]
            C = grid[d].get(m)
            if C is None:
                ax.axis("off")
                continue
            pm, cmax, cavg = heatmap_panel(C)
            im = ax.imshow(
                pm, cmap=CMAP, vmin=HM_VMIN, vmax=HM_VMAX, interpolation="nearest"
            )
            if i == 0:
                ax.set_title(
                    METHOD_LABELS.get(m, m), fontsize=14, fontweight="bold", pad=10
                )
            if j == 0:
                ax.set_ylabel(f"Dim {d}", fontsize=13, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"CosMax={cmax:.3f}\nCosAvg={cavg:.3f}", fontsize=9)

    if im is not None:
        fig.subplots_adjust(right=0.9)
        cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(r"$|\cos(\mathbf{b}_i,\mathbf{b}_j)|$", fontsize=12)

    fig.suptitle(
        "Cosine heatmaps (dims = multiples of 10) — "
        "LLL / BKZ / ENUM / ENUM+SIEVE  (green=small, red=large)",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(FIG / "heatmap_cos_dim10.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(">> figures/heatmap_cos_dim10.png")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    df = load_df()
    print(
        f"loaded {len(df)} records; "
        f"methods={sorted(df.method.unique())}  "
        f"dims={int(df.dim.min())}..{int(df.dim.max())}"
    )

    save_stats(df)
    scatter(df, "b1_gh", "||b1|| / GH", "scatter_b1gh_vs_dim.png", target=1.0)
    scatter(df, "max_cos", "max |cos(b_i,b_j)|", "scatter_maxcos_vs_dim.png")
    plot_heatmaps(df)

    print(f">> done. outputs in {OUT}")


if __name__ == "__main__":
    main()
