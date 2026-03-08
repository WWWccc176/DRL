import my_project_backend  # 注意：这里 import 的名字是你 Cargo.toml 里 [lib] 下定义的 name
import numpy as np


def main():
    print("正在调用 Rust + C++ 后端...")

    # 1. 准备测试数据
    # 这是一个简单的 2x2 矩阵字符串格式，fplll 能识别
    matrix_str = "[ [10 2] [3 15] ]"
    method = "LLL"
    param = 0

    # 2. 调用函数
    # 函数名是你 lib.rs 里 m.add_function 注册的名字
    try:
        result = my_project_backend.run_reduction_rust(matrix_str, method, param)  # type:ignore

        print("\n=== 运行成功！结果如下 ===")
        print(f"Log Prod: {result['log_prod']}")
        print(f"Min Norm: {result['min_norm']}")
        print(f"Matrix String: {result['matrix_str']}")

        print("\nCosine Matrix (Numpy Array):")
        print(result["cos_matrix"])
        print(f"Shape: {result['cos_matrix'].shape}")

    except Exception as e:
        print(f"运行出错: {e}")


if __name__ == "__main__":
    main()
