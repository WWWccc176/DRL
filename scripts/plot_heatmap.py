import matplotlib.pyplot as plt
import numpy as np
import os
import math
import sys
import concurrent.futures

# ================= 配置路径 =================
# 获取当前脚本所在目录 (scripts/)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # DRL/

# 这里的路径必须对应你的截图结构
sys.path.append(os.path.join(project_root, "drl_app"))
sys.path.append(os.path.join(project_root, "build"))  # 以防万一

try:
    import my_project_backend

    print("C++ 模块加载成功！")
except ImportError as e:
    print(f"无法加载 C++ 模块: {e}")
    sys.exit(1)

DATASET_DIR = os.path.join(project_root, "dataset")


# ================= 辅助函数 =================
def get_log_det(matrix_str):
    """计算原始行列式的对数，用于计算 Defect"""
    # 简单的字符串解析转numpy
    matrix_str = matrix_str.replace("[", "").replace("]", "").strip()
    rows = []
    for line in matrix_str.split("\n"):
        if line.strip():
            rows.append([float(x) for x in line.split()])
    arr = np.array(rows)
    # 使用 slogdet 计算 log(|det(L)|)
    sign, logdet = np.linalg.slogdet(arr)
    return logdet


# ================= 核心处理函数 (用于并发) =================
def process_single_dim(dim, stages):
    """处理单个维度的所有规约阶段"""
    filename = f"svpchallengedim{dim}seed0.txt"
    filepath = os.path.join(DATASET_DIR, filename)

    if not os.path.exists(filepath):
        print(f"⚠️ 跳过 Dim {dim}: 文件不存在")
        return dim, None, None

    print(f"🚀 开始处理 Dim {dim}...", flush=True)

    with open(filepath, "r") as f:
        raw_str = f.read()

    original_log_det = get_log_det(raw_str)
    current_str = raw_str

    local_plot_data = {}
    local_stats_data = {}

    for method, param in stages:
        label = "LLL" if method == "LLL" else f"BKZ-{param}"
        key = f"Dim{dim}_{label}"

        # 调用 Rust/C++ 后端
        res = my_project_backend.run_reduction_rust(current_str, method, param)  # type: ignore

        current_str = res["matrix_str"]
        local_plot_data[key] = res["cos_matrix"]

        log_prod = res["log_prod"]
        defect = math.exp((log_prod - original_log_det) / dim)
        geom_mean = math.exp(log_prod / dim)

        local_stats_data[key] = {
            "defect": defect,
            "min": res["min_norm"],
            "mean": geom_mean,
        }

    print(f"✅ Dim {dim} 规约完成.")
    return dim, local_plot_data, local_stats_data


# ================= 主程序 =================
dims = [43, 45, 47, 55, 57, 70]  # 对应你的 dataset
# 严格按照原代码的顺序定义阶段
stages = [
    ("LLL", 0),
    ("BKZ", 10),
    ("BKZ", 15),
    ("BKZ", 20),
    ("BKZ", 25),
    ("BKZ", 30),
    ("BKZ", 35),
]
stage_names = ["LLL"] + [f"BKZ-{p}" for m, p in stages if m == "BKZ"]

plot_data = {}  # 存放 Cosine Matrices
stats_data = {}

print(f"🚀 开始处理，数据源: {DATASET_DIR}")

for dim in dims:
    filename = f"svpchallengedim{dim}seed0.txt"
    filepath = os.path.join(DATASET_DIR, filename)

    if not os.path.exists(filepath):
        print(f"⚠️ 跳过 Dim {dim}: 文件不存在")
        continue

    print(f"Processing Dim {dim}...", end=" ", flush=True)

    # 1. 读取原始文件
    with open(filepath, "r") as f:
        raw_str = f.read()

    # 2. 计算基准 log_det (Defect 分母)
    original_log_det = get_log_det(raw_str)

    # 3. *** 关键修改：渐进式规约 (Cascading) ***
    # 我们维护一个 current_str，每一步的输出作为下一步的输入
    current_str = raw_str

    for method, param in stages:
        label = "LLL" if method == "LLL" else f"BKZ-{param}"
        key = f"Dim{dim}_{label}"

        # 调用 C++: 传入上一次的结果
        # C++ 会返回：{'matrix_str', 'log_prod', 'min_norm', 'cos_matrix'}
        res = my_project_backend.run_reduction_rust(current_str, method, param)  # type: ignore

        # 更新 current_str 供下一次循环使用
        current_str = res["matrix_str"]

        # 存储绘图数据 (C++ 直接算好了 Cosine Matrix，无需 Python 再算)
        plot_data[key] = res["cos_matrix"]

        # 计算统计量
        # Normalized Defect = exp( (log_prod - log_det) / n )
        log_prod = res["log_prod"]
        defect = math.exp((log_prod - original_log_det) / dim)

        # 几何平均长度 = exp( log_prod / n )
        geom_mean = math.exp(log_prod / dim)

        stats_data[key] = {"defect": defect, "min": res["min_norm"], "mean": geom_mean}

    print("Done.")

# ================= 绘图部分 (保持逻辑一致) =================
print("Generating Plot...")
valid_dims = [d for d in dims if f"Dim{d}_LLL" in plot_data]

if not valid_dims:
    print("❌ 没有有效数据。")
    sys.exit(0)

fig, axes = plt.subplots(
    nrows=len(valid_dims), ncols=len(stage_names), figsize=(28, 5 * len(valid_dims))
)
# 处理单行情况
if len(valid_dims) == 1:
    axes = np.expand_dims(axes, axis=0)

plt.subplots_adjust(hspace=0.65, wspace=0.15)
cmap = "RdYlGn_r"

im = None

for i, dim in enumerate(valid_dims):
    for j, stage_name in enumerate(stage_names):
        key = f"Dim{dim}_{stage_name}"
        ax = axes[i, j]

        if key in plot_data:
            # 取出 C++ 算好的 N x N Cosine 矩阵
            full_cos_mat = plot_data[key]

            # *** 关键可视化逻辑 (复刻原 notebook) ***
            # 构造 (n-1) x (n-1) 下三角矩阵
            # Row对应 v2...vn, Col对应 v1...vn-1
            # 即取 mat[1:, :-1] 并取下三角

            n = full_cos_mat.shape[0]
            if n > 1:
                sub_mat = full_cos_mat[1:, :-1]
                plot_mat = np.tril(sub_mat)

                # 统计非零部分的 Max/Avg
                valid_mask = np.tril(np.ones_like(sub_mat), k=0).astype(bool)
                valid_vals = sub_mat[valid_mask]

                cos_max = np.max(valid_vals) if len(valid_vals) > 0 else 0
                cos_mean = np.mean(valid_vals) if len(valid_vals) > 0 else 0
            else:
                plot_mat = full_cos_mat
                cos_max, cos_mean = 0, 0

            # 绘图
            im = ax.imshow(
                plot_mat, cmap=cmap, vmin=0, vmax=0.7, interpolation="nearest"
            )

            # 标签
            if i == 0:
                ax.set_title(stage_name, fontsize=18, fontweight="bold", pad=20)
            if j == 0:
                ax.set_ylabel(f"Dim {dim}", fontsize=18, fontweight="bold")

            ax.set_xticks([])
            ax.set_yticks([])

            # 底部文字
            s = stats_data.get(key, {})
            info_text = (
                f"$\\delta^*={s.get('defect', 0):.6f}$\n"
                f"Min: {s.get('min', 0):.2e}\n"
                f"CosMax: {cos_max:.4f}\n"
                f"CosAvg: {cos_mean:.4f}"
            )
            ax.set_xlabel(
                info_text, fontsize=11, fontweight="medium", labelpad=8, linespacing=1.4
            )
        else:
            ax.axis("off")

# Colorbar
if im is not None:
    cbar_ax = fig.add_axes((0.92, 0.15, 0.015, 0.7))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("| Cosine Similarity |", fontsize=16)
output_file = os.path.join(project_root, "lattice_evolution.png")
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"绘图完成，已保存至: {output_file}")
