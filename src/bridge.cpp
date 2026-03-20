////#include "bridge.h"
////#include <fplll/fplll.h>
////#include <sstream>
////#include <cmath>
////#include <algorithm>
////
////using namespace fplll;
////using MyMatrix = ZZ_mat<mpz_t>;
////
////// 辅助函数：复制过来的
////MyMatrix parse_matrix_core(const std::string& input_str) {
////    MyMatrix B;
////    std::stringstream ss(input_str);
////    ss >> B; 
////    return B;
////}
////
////std::string dump_matrix_core(MyMatrix& B) {
////    std::stringstream ss;
////    ss << "[";
////    for (int i = 0; i < B.get_rows(); ++i) {
////        ss << "[";
////        for (int j = 0; j < B.get_cols(); ++j) {
////            ss << B[i][j];
////            if (j + 1 < B.get_cols()) ss << " ";
////        }
////        ss << "]";
////        if (i + 1 < B.get_rows()) ss << "\n";
////    }
////    ss << "]";
////    return ss.str();
////}
////
////// 核心函数：不依赖 Python
////ReductionResult run_reduction_core(rust::String matrix_str, rust::String method, int param) {
////    std::string matrix_s(matrix_str);
////    std::string method_s(method);
////    
////    MyMatrix B = parse_matrix_core(matrix_s);
////    
////    // 1. 执行规约
////    if (method_s == "LLL") {
////        lll_reduction(B, 0.99);
////    } else if (method_s == "BKZ") {
////        if (param < 2) lll_reduction(B, 0.99);
////        else bkz_reduction(B, param, BKZ_DEFAULT);
////    }
////
////    // 2. 计算统计量
////    int n = B.get_rows();
////    int cols = B.get_cols();
////    std::vector<double> norms(n);
////    mpz_t norm_sq_i, dot_prod;
////    mpz_inits(norm_sq_i, dot_prod, nullptr);
////    
////    double sum_log_norm = 0.0;
////    double min_norm = -1.0;
////
////    for (int i = 0; i < n; ++i) {
////        mpz_set_ui(norm_sq_i, 0);
////        for (int k = 0; k < cols; ++k) {
////            mpz_addmul(norm_sq_i, B[i][k].get_data(), B[i][k].get_data());
////        }
////        norms[i] = sqrt(mpz_get_d(norm_sq_i));
////        sum_log_norm += log(norms[i]);
////        if (min_norm < 0 || norms[i] < min_norm) min_norm = norms[i];
////    }
////
////    // 3. 计算 Cosine Matrix (扁平化)
////    std::vector<double> cos_flat(n * n);
////    for (int i = 0; i < n; ++i) {
////        for (int j = 0; j < n; ++j) {
////            if (i == j) {
////                cos_flat[i * n + j] = 0.0;
////            } else {
////                mpz_set_ui(dot_prod, 0);
////                for (int k = 0; k < cols; ++k) {
////                    mpz_addmul(dot_prod, B[i][k].get_data(), B[j][k].get_data());
////                }
////                double dot = mpz_get_d(dot_prod);
////                double val = dot / (norms[i] * norms[j] + 1e-20);
////                cos_flat[i * n + j] = std::abs(val);
////            }
////        }
////    }
////    mpz_clears(norm_sq_i, dot_prod, nullptr);
////
////    // 4. 返回纯 C++ 结构体
////    rust::Vec<double> rust_cos_flat;
////    for (double val : cos_flat) {
////        rust_cos_flat.push_back(val);
////    }
////
////    return ReductionResult {
////        rust::String(dump_matrix_core(B)), 
////        sum_log_norm,
////        min_norm,
////        rust_cos_flat, // 传入 rust::Vec
////        n
////    };
////}
//#include "bridge.h"
//#include <fplll/fplll.h>
//#include <sstream>
//#include <cmath>
//#include <algorithm>
//#include <vector>
//
//using namespace fplll;
//using MyMatrix = ZZ_mat<mpz_t>;
//
//// 辅助函数
//MyMatrix parse_matrix_core(const std::string& input_str) {
//    MyMatrix B;
//    std::stringstream ss(input_str);
//    ss >> B; 
//    return B;
//}
//
//std::string dump_matrix_core(MyMatrix& B) {
//    std::stringstream ss;
//    ss << "[";
//    for (int i = 0; i < B.get_rows(); ++i) {
//        ss << "[";
//        for (int j = 0; j < B.get_cols(); ++j) {
//            ss << B[i][j];
//            if (j + 1 < B.get_cols()) ss << " ";
//        }
//        ss << "]";
//        if (i + 1 < B.get_rows()) ss << "\n";
//    }
//    ss << "]";
//    return ss.str();
//}
//
//// 核心函数
//ReductionResult run_reduction_core(rust::String matrix_str, rust::String method, int param) {
//    // 1. 类型转换：rust::String -> std::string
//    std::string matrix_s(matrix_str);
//    std::string method_s(method);
//    
//    // 2. 使用转换后的 std::string
//    MyMatrix B = parse_matrix_core(matrix_s);
//    
//    // 3. 逻辑处理
//    if (method_s == "LLL") {
//        lll_reduction(B, 0.99);
//    } else if (method_s == "BKZ") {
//        if (param < 2) lll_reduction(B, 0.99);
//        else bkz_reduction(B, param, BKZ_DEFAULT);
//    }
//
//    // 4. 计算统计量
//    int n = B.get_rows();
//    int cols = B.get_cols();
//    std::vector<double> norms(n);
//    mpz_t norm_sq_i, dot_prod;
//    mpz_inits(norm_sq_i, dot_prod, nullptr);
//    
//    double sum_log_norm = 0.0;
//    double min_norm = -1.0;
//
//    for (int i = 0; i < n; ++i) {
//        mpz_set_ui(norm_sq_i, 0);
//        for (int k = 0; k < cols; ++k) {
//            mpz_addmul(norm_sq_i, B[i][k].get_data(), B[i][k].get_data());
//        }
//        norms[i] = sqrt(mpz_get_d(norm_sq_i));
//        sum_log_norm += log(norms[i]);
//        if (min_norm < 0 || norms[i] < min_norm) min_norm = norms[i];
//    }
//
//    // 5. 计算 Cosine Matrix (扁平化)
//    //std::vector<double> cos_flat(n * n);
//    //for (int i = 0; i < n; ++i) {
//    //    for (int j = 0; j < n; ++j) {
//    //        if (i == j) {
//    //            cos_flat[i * n + j] = 0.0;
//    //        } else {
//    //            mpz_set_ui(dot_prod, 0);
//    //            for (int k = 0; k < cols; ++k) {
//    //                mpz_addmul(dot_prod, B[i][k].get_data(), B[j][k].get_data());
//    //            }
//    //            double dot = mpz_get_d(dot_prod);
//    //            double val = dot / (norms[i] * norms[j] + 1e-20);
//    //            cos_flat[i * n + j] = std::abs(val);
//    //        }
//    //    }
//    //}
//    //mpz_clears(norm_sq_i, dot_prod, nullptr);
//    std::vector<double> cos_flat(n * n);
//    for (int i = 0; i < n; ++i) {
//        for (int j = 0; j < n; ++j) {
//            // 只有当行号大于列号时 (i > j)，才是下三角区域
//            if (i > j) {
//                mpz_set_ui(dot_prod, 0);
//                // 计算点积
//                for (int k = 0; k < cols; ++k) {
//                    mpz_addmul(dot_prod, B[i][k].get_data(), B[j][k].get_data());
//                }
//                
//                double dot = mpz_get_d(dot_prod);
//                // 计算余弦值
//                double val = dot / (norms[i] * norms[j] + 1e-20);
//                
//                // 存入绝对值 (或者如果你想要原始余弦值，去掉 std::abs)
//                cos_flat[i * n + j] = std::abs(val); 
//            } else {
//                // 对角线 (i==j) 和 上三角 (i<j) 全部填 0
//                cos_flat[i * n + j] = 0.0;
//            }
//        }
//    }
//    mpz_clears(norm_sq_i, dot_prod, nullptr);
//
//    // 6. 转换 vector 并返回
//    rust::Vec<double> rust_cos_flat;
//    for (double val : cos_flat) {
//        rust_cos_flat.push_back(val);
//    }
//
//    return ReductionResult {
//        rust::String(dump_matrix_core(B)), 
//        sum_log_norm,
//        min_norm,
//        rust_cos_flat, 
//        n
//    };
//}
#include "bridge.h"
#include "rustcore/src/lib.rs.h"
#include <fplll/fplll.h>
#include <sstream>
#include <cmath>
#include <vector>

using namespace fplll;
using MyMatrix = ZZ_mat<mpz_t>;

// 补回丢失的解析函数
MyMatrix parse_matrix_core(const std::string& input_str) {
    MyMatrix B;
    std::stringstream ss(input_str);
    ss >> B; 
    return B;
}

// 补回丢失的序列化函数
std::string dump_matrix_core(MyMatrix& B) {
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

// 核心函数 (使用指针/内存数组传递，摆脱高频字符串解析)
ReductionResult run_reduction_core(rust::String matrix_str, rust::String method, int param) {
    std::string matrix_s(matrix_str);
    std::string method_s(method);
    
    MyMatrix B = parse_matrix_core(matrix_s);
    
    // 1. 执行约化算法
    if (method_s == "LLL") {
        lll_reduction(B, 0.99);
    } else if (method_s == "BKZ") {
        if (param < 2) lll_reduction(B, 0.99);
        else bkz_reduction(B, param, BKZ_DEFAULT);
    }

    int rows = B.get_rows();
    int cols = B.get_cols();
    
    // 2. 预分配内存，准备将计算结果直接以连续内存(指针)形式传给 Rust
    rust::Vec<double> flat_matrix;
    rust::Vec<double> row_log_scales;
    flat_matrix.reserve(rows * cols);
    row_log_scales.reserve(rows);

    const double LOG2 = 0.6931471805599453;

    // 3. 直接在 C++ 端提取大整数的高精度尾数和量级，避免字符串截断
    for (int i = 0; i < rows; ++i) {
        double max_log_val = -1e300;
        std::vector<double> temp_mantissas(cols, 0.0);
        std::vector<double> temp_logs(cols, -1e300);

        for (int j = 0; j < cols; ++j) {
            long exp = 0;
            // mpz_get_d_2exp 提取尾数(区间[0.5, 1))和指数，避免浮点溢出
            double mantissa = mpz_get_d_2exp(&exp, B[i][j].get_data());
            if (mantissa != 0.0) {
                // 计算该元素的真实自然对数: ln(|value|) = ln(|mantissa|) + exp * ln(2)
                double log_val = std::log(std::abs(mantissa)) + exp * LOG2;
                temp_logs[j] = log_val;
                temp_mantissas[j] = mantissa;
                if (log_val > max_log_val) {
                    max_log_val = log_val;
                }
            }
        }

        row_log_scales.push_back(max_log_val > -1e299 ? max_log_val : 0.0);

        // 生成缩放后的矩阵，映射到 [-1.0, 1.0] 范围内
        for (int j = 0; j < cols; ++j) {
            if (temp_logs[j] > -1e299) {
                double diff = temp_logs[j] - max_log_val;
                double sign = temp_mantissas[j] > 0 ? 1.0 : -1.0;
                flat_matrix.push_back(sign * std::exp(diff));
            } else {
                flat_matrix.push_back(0.0);
            }
        }
    }

    return ReductionResult {
        rust::String(dump_matrix_core(B)), // 仅保留给 Python 做外部存储或初始化用
        rows,
        cols,
        flat_matrix,
        row_log_scales
    };
}
