#pragma once
#include "rust/cxx.h"

// 精简后的结构体，只包含字符串和行列数
//struct ReductionResult {
//    rust::String matrix_str;
//    int rows;
//    int cols;
//};
struct ReductionResult;
// 函数声明保持不变
ReductionResult run_reduction_core(rust::String matrix_str, rust::String method, int param);
