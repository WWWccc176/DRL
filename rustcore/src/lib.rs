//use pyo3::prelude::*;
//
//// 1. 定义 C++ 接口 (使用 cxx)
//#[cxx::bridge]
//mod ffi {
//    unsafe extern "C++" {
//        include!("src/your_header.h"); // C++ 头文件路径
//        fn cpp_function(x: i32) -> i32; // C++ 函数声明
//    }
//}
//
//// 2. 实现 Rust 逻辑
//#[pyfunction]
//// 3. 暴露给 Python 的模块
//#[pymodule]
//fn my_project_backend(_py: Python, m: &PyModule) -> PyResult<()> {
//    m.add_function(wrap_pyfunction!(rust_calculation, m)?)?;
//    Ok(())
//}
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict; // 引入 PyArrayMethods 以支持 reshape 等操作

// --- 1. C++ 桥接定义 ---
#[cxx::bridge]
mod ffi {
    // 对应 C++ 的结构体
    struct ReductionResult {
        matrix_str: String,
        log_prod: f64,
        min_norm: f64,
        cos_matrix_flat: Vec<f64>,
        rows: i32,
    }

    unsafe extern "C++" {
        include!("src/bridge.h");

        // C++ 函数声明
        fn run_reduction_core(matrix_str: String, method: String, param: i32) -> ReductionResult;
    }
}

// --- 2. Rust 包装逻辑 ---
#[pyfunction]
fn run_reduction_rust(
    py: Python,
    matrix_str: String,
    method: String,
    param: i32,
) -> PyResult<PyObject> {
    // A. 调用 C++
    let result = ffi::run_reduction_core(matrix_str, method, param);

    // B. 处理 Numpy 数组 (先生成 1D，再 Reshape 成 2D)
    let rows = result.rows as usize;
    let array_1d = PyArray1::from_vec(py, result.cos_matrix_flat);

    // reshape 返回的是 PyResult<Bound<PyArray...>>
    let cos_matrix_2d = array_1d.reshape((rows, rows))?;

    // C. 构建 Python 字典 (使用新的 Bound API)
    let dict = PyDict::new(py);
    dict.set_item("matrix_str", result.matrix_str)?;
    dict.set_item("log_prod", result.log_prod)?;
    dict.set_item("min_norm", result.min_norm)?;
    dict.set_item("cos_matrix", cos_matrix_2d)?;

    Ok(dict.into())
}

// --- 3. 模块注册 (适配 PyO3 0.23 Bound API) ---
#[pymodule]
fn my_project_backend(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_reduction_rust, m)?)?;
    Ok(())
}
