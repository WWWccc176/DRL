#pragma once
#include <string>
#include "rust/cxx.h"
#include <vector>
#include <memory>

// 2. 引入 Rust 生成的头文件
// 路径格式通常是: "包名/src/文件名.rs.h"
#include "rustcore/src/lib.rs.h"

// 3. 修改函数签名：使用 rust::String 替代 std::string
ReductionResult run_reduction_core(rust::String matrix_str, rust::String method, int param);
