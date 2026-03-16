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

using namespace fplll;
using MyMatrix = ZZ_mat<mpz_t>;

// 辅助函数保持不变
MyMatrix parse_matrix_core(const std::string& input_str) {
    MyMatrix B;
    std::stringstream ss(input_str);
    ss >> B; 
    return B;
}

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

// 核心函数：大幅精简，剥离所有 f64 计算
ReductionResult run_reduction_core(rust::String matrix_str, rust::String method, int param) {
    std::string matrix_s(matrix_str);
    std::string method_s(method);
    
    MyMatrix B = parse_matrix_core(matrix_s);
    
    // 1. 仅保留核心的 LLL/BKZ 逻辑
    if (method_s == "LLL") {
        lll_reduction(B, 0.99);
    } else if (method_s == "BKZ") {
        if (param < 2) lll_reduction(B, 0.99);
        else bkz_reduction(B, param, BKZ_DEFAULT);
    }

    // 2. 只返回字符串和维度信息
    return ReductionResult {
        rust::String(dump_matrix_core(B)), 
        B.get_rows(),
        B.get_cols()
    };
}
