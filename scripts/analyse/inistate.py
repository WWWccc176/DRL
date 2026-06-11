import os
import math
import numpy as np
import my_project_backend


# ------------------------------
# 内部工具函数 (直接提取自你的代码)
# ------------------------------
def matrix_to_string(basis):
    lines = [" ".join(str(x) for x in row) for row in basis]
    return "[" + "\n".join(f"[{l}]" for l in lines) + "]"


def string_to_matrix_fast(mat_str):
    content = mat_str.strip()[1:-1]
    if not content:
        return []
    rows = content.split("\n")
    return [
        [int(x) for x in r.replace("[", "").replace("]", "").split()]
        for r in rows
        if r.strip()
    ]


def parse_challenge_file(filepath):
    """读取格基数据集文件"""
    matrix = []
    with open(filepath, "r") as f:
        content = f.read().replace("[", "").replace("]", "")
        for line in content.strip().split("\n"):
            if line.strip():
                row = [int(x) for x in line.split()]
                matrix.append(row)
    return matrix


# ------------------------------
# 核心提取逻辑
# ------------------------------
def extract_and_save_initial_state(input_filepath, output_filepath):
    print(f"[*] 正在读取: {input_filepath}")

    # 1. 解析初始文件
    basis = parse_challenge_file(input_filepath)
    dim = len(basis)
    raw_matrix_str = matrix_to_string(basis)

    # 2. 传给 Rust Backend 创建矩阵池并进行初始 LLL
    print(f"[*] 正在调用 backend 进行 LLL 约化...")
    pool_id = my_project_backend.create_matrix_lll_rust(raw_matrix_str)

    # 3. 计算对数体积与高斯启发式界 (GH)
    init_eval = my_project_backend.evaluate_matrix_rust(pool_id)
    gs_logs = np.array(init_eval["gs_log_norms"], dtype=np.float64)
    log_vol = np.sum(gs_logs)
    log_GH = (log_vol / dim) + 0.5 * math.log(dim / (2 * math.pi * math.e))

    # 4. 获取余弦矩阵及 log_prod[cite: 1]
    init_info = my_project_backend.reduce_rust(pool_id, "LLL", 2, 0)

    # 5. 计算相关指标[cite: 1]
    log_defect = float(init_info["log_prod"] - log_vol)
    log_ratio = float(gs_logs[0] - log_GH)
    ratio = math.exp(log_ratio)

    C = np.array(init_info["cos_matrix"], dtype=np.float64)
    lower = C[np.tril_indices(dim, -1)]
    max_cos = float(np.clip(np.max(lower) if lower.size > 0 else 0.0, 0.0, 1.0))
    min_cos = float(np.clip(np.min(lower) if lower.size > 0 else 0.0, 0.0, 1.0))

    # 6. 获取第一行最短向量 b1[cite: 1]
    mat_str = my_project_backend.dump_matrix_rust(pool_id)
    mat_list = string_to_matrix_fast(mat_str)
    b1 = mat_list[0] if mat_list else []

    # 7. 写入指定格式
    with open(output_filepath, "w") as f:
        f.write("--- Initial State ---\n")
        f.write(f"  Ratio (‖b₁‖/GH):  {ratio:.8f}\n")
        f.write(f"  Orthog. Defect:    {log_defect:.8f}\n")
        f.write(f"  Max Cosine:        {max_cos:.8f}\n")
        f.write(f"  Min Cosine:        {min_cos:.8f}\n")
        f.write(f"  b₁ = {b1}\n")

    print(f"[+] 提取完成！结果已保存至: {output_filepath}")

    # 清理释放池中矩阵 (如果你的 backend 支持)
    if hasattr(my_project_backend, "free_matrix_rust"):
        my_project_backend.free_matrix_rust(pool_id)


if __name__ == "__main__":
    # 配置目录结构：获取当前脚本所在目录的上一级目录作为项目根目录
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 这样就能正确找到并列的 dataset 文件夹了
    DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

    # 结果输出到根目录的 results 文件夹（如果你想输出到 scripts/results，可以把 PROJECT_ROOT 改回单层 dirname）
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 文件路径
    input_file = os.path.join(DATASET_DIR, "svpchallengedim67seed13.txt")
    output_file = os.path.join(RESULTS_DIR, "initial_state_dim67seed13.txt")

    if not os.path.exists(input_file):
        print(f"[-] 错误: 找不到数据集文件 {input_file}。请检查路径。")
    else:
        extract_and_save_initial_state(input_file, output_file)
