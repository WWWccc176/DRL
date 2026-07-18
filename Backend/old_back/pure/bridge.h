//#pragma once
//#include "rust/cxx.h"
//
//// 精简后的结构体，只包含字符串和行列数
////struct ReductionResult {
////    rust::String matrix_str;
////    int rows;
////    int cols;
////};
//struct ReductionResult;
//
//// 【修复这里】：在最后加上 int pos 参数，与 bridge.cpp 和 lib.rs 保持一致！
//ReductionResult run_reduction_core(rust::String matrix_str, rust::String method, int32_t param, int32_t pos);
#pragma once
#include "rust/cxx.h"

struct ReductionResult;

// ===== 新 API：矩阵池（零序列化） =====
int64_t pool_create(rust::String matrix_str);
int64_t pool_create_lll(rust::String matrix_str);   // 创建 + LLL
ReductionResult pool_reduce(int64_t id, rust::String method, int32_t param, int32_t pos);
rust::String pool_dump(int64_t id);                  // 仅保存时用
void pool_free(int64_t id);
int64_t pool_clone(int64_t id);                      // reset 时用
