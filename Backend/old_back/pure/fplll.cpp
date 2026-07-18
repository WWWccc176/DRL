//目前暂时用不到这个文件，仅备份用


#include "bridge.h" 
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h> // 用于直接返回 Numpy 数组
#include <fplll/fplll.h>
#include <sstream>
#include <string>
#include <cmath>
#include <vector>
#include <map>

namespace py = pybind11;
using namespace fplll;
using MyMatrix = ZZ_mat<mpz_t>;

// 辅助：字符串 -> 矩阵
MyMatrix parse_matrix(const std::string& input_str) {
    MyMatrix B;
    std::stringstream ss(input_str);
    ss >> B; 
    return B;
}

// 辅助：矩阵 -> 字符串
std::string dump_matrix(MyMatrix& B) {
    std::stringstream ss;
    ss << "[";
    for (int i = 0; i < B.get_rows(); ++i) {
        ss << "[";
        for (int j = 0; j < B.get_cols(); ++j) {
            ss << B[i][j];
            if (j + 1 < B.get_cols()) ss << " ";
        }
        ss << "]";
        if (i + 1 < B.get_rows()) ss << "\n";
    }
    ss << "]";
    return ss.str();
}

// 核心逻辑：计算统计量和余弦矩阵 (完全复刻 matrix_gen.cpp 的数学逻辑)
py::dict compute_results(MyMatrix& B) {
    int n = B.get_rows();
    int cols = B.get_cols();

    // 1. 计算范数和 LogProd
    std::vector<double> norms(n);
    mpz_t norm_sq_i;
    mpz_init(norm_sq_i);
    
    double sum_log_norm = 0.0;
    double min_norm = -1.0;

    for (int i = 0; i < n; ++i) {
        mpz_set_ui(norm_sq_i, 0);
        for (int k = 0; k < cols; ++k) {
            mpz_addmul(norm_sq_i, B[i][k].get_data(), B[i][k].get_data());
        }
        norms[i] = sqrt(mpz_get_d(norm_sq_i));
        sum_log_norm += log(norms[i]);
        
        if (min_norm < 0 || norms[i] < min_norm) min_norm = norms[i];
    }
    mpz_clear(norm_sq_i);

    // 2. 计算 Cosine Matrix (N x N)
    // 直接生成 numpy array 返回给 Python，避免 Python 再次计算
    py::array_t<double> cos_matrix({n, n});
    auto r = cos_matrix.mutable_unchecked<2>();
    
    mpz_t dot_prod;
    mpz_init(dot_prod);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                r(i, j) = 0.0; // 对角线设为0，与原代码一致
            } else {
                mpz_set_ui(dot_prod, 0);
                for (int k = 0; k < cols; ++k) {
                    mpz_addmul(dot_prod, B[i][k].get_data(), B[j][k].get_data());
                }
                double dot = mpz_get_d(dot_prod);
                // 加上 1e-20 防止除零，虽然 lattice basis 不应为0
                double val = dot / (norms[i] * norms[j] + 1e-20);
                r(i, j) = std::abs(val); // 原代码输出也是 abs
            }
        }
    }
    mpz_clear(dot_prod);

    // 3. 结果打包
    py::dict result;
    result["matrix_str"] = dump_matrix(B);
    result["log_prod"] = sum_log_norm;
    result["min_norm"] = min_norm;
    result["cos_matrix"] = cos_matrix;
    
    return result;
}

// 统一接口：执行规约并返回所有数据
py::dict run_reduction(std::string matrix_str, std::string method, int param) {
    MyMatrix B = parse_matrix(matrix_str);
    
    if (method == "LLL") {
        // LLL: param is ignored or treated as delta fixed at 0.99
        lll_reduction(B, 0.99);
    } else if (method == "BKZ") {
        // BKZ: param is block_size
        if (param < 2) lll_reduction(B, 0.99);
        else bkz_reduction(B, param, BKZ_DEFAULT);
    }
    
    return compute_results(B);
}

PYBIND11_MODULE(my_cpp_module, m) {
    m.doc() = "FPLLL wrapper with Stats and Cosine Matrix"; 
    m.def("run_reduction", &run_reduction, "Run reduction and return stats+matrix",
          py::arg("matrix_str"), py::arg("method"), py::arg("param")=0);
}

