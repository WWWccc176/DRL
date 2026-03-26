#pragma once
#include "rust/cxx.h"

// 精简后的结构体，只包含字符串和行列数
//struct ReductionResult {
//    rust::String matrix_str;
//    int rows;
//    int cols;
//};
struct ReductionResult;

// 【修复这里】：在最后加上 int pos 参数，与 bridge.cpp 和 lib.rs 保持一致！
ReductionResult run_reduction_core(rust::String matrix_str, rust::String method, int param, int pos);
