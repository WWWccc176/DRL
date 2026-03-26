//use numpy::{PyArray1, PyArrayMethods};
//use pyo3::prelude::*;
//use pyo3::types::PyDict;
//use rayon::prelude::*; // 引入 PyArrayMethods 以支持 reshape 等操作
//
//// --- 1. C++ 桥接定义 ---
//#[cxx::bridge]
//mod ffi {
//    struct ReductionResult {
//        matrix_str: String,
//        rows: i32,
//        cols: i32,
//    }
//
//    unsafe extern "C++" {
//        include!("src/bridge.h");
//        fn run_reduction_core(matrix_str: String, method: String, param: i32) -> ReductionResult;
//    }
//}
//
//// --- 2. Rust 包装逻辑 ---
////#[pyfunction]
////fn run_reduction_rust(
////    py: Python,
////    matrix_str: String,
////    method: String,
////    param: i32,
////) -> PyResult<PyObject> {
////    // A. 调用 C++
////    let result = ffi::run_reduction_core(matrix_str, method, param);
////
////    // B. 处理 Numpy 数组 (先生成 1D，再 Reshape 成 2D)
////    let rows = result.rows as usize;
////    let array_1d = PyArray1::from_vec(py, result.cos_matrix_flat);
////
////    // reshape 返回的是 PyResult<Bound<PyArray...>>
////    let cos_matrix_2d = array_1d.reshape((rows, rows))?;
////
////    // C. 构建 Python 字典 (使用新的 Bound API)
////    let dict = PyDict::new(py);
////    dict.set_item("matrix_str", result.matrix_str)?;
////    dict.set_item("log_prod", result.log_prod)?;
////    dict.set_item("min_norm", result.min_norm)?;
////    dict.set_item("cos_matrix", cos_matrix_2d)?;
////
////    Ok(dict.into())
////}
//
//#[pyfunction]
//fn run_reduction_rust(
//    py: Python,
//    matrix_str: String,
//    method: String,
//    param: i32,
//) -> PyResult<PyObject> {
//    // A. 调用 C++ 进行规约
//    let result = ffi::run_reduction_core(matrix_str, method, param);
//    let n = result.rows as usize;
//
//    // B. 并行解析字符串到 Rust 的 2D 向量中 (f64)
//    let lines: Vec<&str> = result
//        .matrix_str
//        .lines()
//        .filter(|l| !l.trim().is_empty())
//        .collect();
//    let matrix: Vec<Vec<f64>> = lines
//        .par_iter()
//        .map(|line| {
//            let clean_line: String = line.replace(['[', ']'], "");
//            clean_line
//                .split_whitespace()
//                .filter_map(|s| s.parse::<f64>().ok())
//                .collect()
//        })
//        .collect();
//
//    // C. 并行计算每个向量的模长 (Norms)
//    let norms: Vec<f64> = matrix
//        .par_iter()
//        .map(|row| {
//            let sq_norm: f64 = row.iter().map(|&x| x * x).sum();
//            sq_norm.sqrt()
//        })
//        .collect();
//
//    // 统计量：找出最小模长，并计算 log 模长之和
//    let min_norm = norms.iter().cloned().fold(f64::INFINITY, f64::min);
//    let log_prod: f64 = norms.iter().map(|&x| x.ln()).sum();
//
//    // D. 并行计算 Cosine Matrix 的下三角部分并扁平化
//    // flat_map 可以并行处理每一行，然后合并成一个 1D Vec
//    let cos_flat: Vec<f64> = (0..n)
//        .into_par_iter()
//        .flat_map(|i| {
//            let mut row_cos = vec![0.0; n];
//            for j in 0..i {
//                // 只算 i > j 的下三角部分
//                let dot: f64 = matrix[i]
//                    .iter()
//                    .zip(matrix[j].iter())
//                    .map(|(a, b)| a * b)
//                    .sum();
//
//                let val = dot / (norms[i] * norms[j] + 1e-20);
//                row_cos[j] = val.abs(); // 如果需要带符号，去掉 .abs()
//            }
//            row_cos
//        })
//        .collect();
//
//    // E. 重新构造给 Python 的 Numpy 2D 数组
//    let array_1d = PyArray1::from_vec(py, cos_flat);
//    let cos_matrix_2d = array_1d.reshape((n, n))?;
//
//    // F. 构建字典返回
//    let dict = PyDict::new(py);
//    dict.set_item("matrix_str", result.matrix_str)?;
//    dict.set_item("log_prod", log_prod)?;
//    dict.set_item("min_norm", min_norm)?;
//    dict.set_item("cos_matrix", cos_matrix_2d)?;
//
//    Ok(dict.into())
//}
////#[pyfunction]//rust函数就写在这里，前面加这个就好
//#[pyfunction]
//fn evaluate_state_rust(
//    py: Python,
//    matrix_str: String,
//    pos: usize,
//    beta: usize,
//) -> PyResult<PyObject> {
//    // 1. 并发解析字符串为 f64 矩阵
//    let lines: Vec<&str> = matrix_str
//        .lines()
//        .filter(|l| !l.trim().is_empty())
//        .collect();
//    let matrix: Vec<Vec<f64>> = lines
//        .par_iter()
//        .map(|line| {
//            let clean_line = line.replace(['[', ']'], "");
//            clean_line
//                .split_whitespace()
//                .filter_map(|s| s.parse::<f64>().ok())
//                .collect()
//        })
//        .collect();
//
//    let m = matrix.len();
//    if m == 0 {
//        return Ok(PyDict::new(py).into());
//    }
//    let n = matrix[0].len();
//
//    // 2. 快速计算全局 Gram-Schmidt Log Norms (串行极速计算)
//    let mut b_star = vec![vec![0.0; n]; m];
//    let mut gs_log_norms = vec![0.0; m];
//
//    for i in 0..m {
//        let mut v = matrix[i].clone();
//        for j in 0..i {
//            let denom: f64 = b_star[j].iter().map(|x| x * x).sum();
//            if denom > 1e-300 {
//                let dot: f64 = v.iter().zip(&b_star[j]).map(|(a, b)| a * b).sum();
//                let mu = dot / denom;
//                for k in 0..n {
//                    v[k] -= mu * b_star[j][k];
//                }
//            }
//        }
//        b_star[i] = v.clone();
//        let norm_sq: f64 = v.iter().map(|x| x * x).sum();
//        gs_log_norms[i] = if norm_sq > 1e-300 {
//            0.5 * norm_sq.ln()
//        } else {
//            -690.0
//        };
//    }
//
//    // 3. 计算局部 Block 的 Ortho Defect (处理 pos 到 pos+beta 的子块)
//    let end = std::cmp::min(pos + beta, m);
//    let mut local_log_defect = 0.0;
//
//    if pos < end {
//        // 原向量对数模长之和
//        let sum_log_orig: f64 = (pos..end)
//            .map(|i| {
//                let sq: f64 = matrix[i].iter().map(|x| x * x).sum();
//                if sq > 1e-300 { 0.5 * sq.ln() } else { -690.0 }
//            })
//            .sum();
//
//        // 局部子块的 GS 对数模长之和 (需要对这个子块重新独立做 GS)
//        let block_size = end - pos;
//        let mut local_b_star = vec![vec![0.0; n]; block_size];
//        let mut sum_log_gs = 0.0;
//
//        for i in 0..block_size {
//            let mut v = matrix[pos + i].clone();
//            for j in 0..i {
//                let denom: f64 = local_b_star[j].iter().map(|x| x * x).sum();
//                if denom > 1e-300 {
//                    let dot: f64 = v.iter().zip(&local_b_star[j]).map(|(a, b)| a * b).sum();
//                    let mu = dot / denom;
//                    for k in 0..n {
//                        v[k] -= mu * local_b_star[j][k];
//                    }
//                }
//            }
//            local_b_star[i] = v.clone();
//            let norm_sq: f64 = v.iter().map(|x| x * x).sum();
//            sum_log_gs += if norm_sq > 1e-300 {
//                0.5 * norm_sq.ln()
//            } else {
//                -690.0
//            };
//        }
//        local_log_defect = sum_log_orig - sum_log_gs;
//    }
//
//    // 4. 打包返回 Python
//    let gs_array = PyArray1::from_vec(py, gs_log_norms);
//    let dict = PyDict::new(py);
//    dict.set_item("gs_log_norms", gs_array)?;
//    dict.set_item("local_log_defect", local_log_defect)?;
//
//    Ok(dict.into())
//}
//
//// --- 3. 模块注册 (适配 PyO3 0.23 Bound API) ---
//#[pymodule]
//fn my_project_backend(m: &Bound<'_, PyModule>) -> PyResult<()> {
//    m.add_function(wrap_pyfunction!(run_reduction_rust, m)?)?;
//    m.add_function(wrap_pyfunction!(evaluate_state_rust, m)?)?;
//    Ok(())
//}
//use numpy::{PyArray1, PyArrayMethods};
//use pyo3::prelude::*;
//use pyo3::types::PyDict;
//use rayon::prelude::*;
//
//#[cxx::bridge]
//mod ffi {
//    struct ReductionResult {
//        matrix_str: String,
//        rows: i32,
//        cols: i32,
//    }
//    unsafe extern "C++" {
//        include!("src/bridge.h");
//        fn run_reduction_core(matrix_str: String, method: String, param: i32) -> ReductionResult;
//    }
//}
//
//// 核心优化：安全解析并自动缩放矩阵，防止 f64 平方时溢出 (超过 10^308)
//fn parse_and_scale_matrix(matrix_str: &str) -> (Vec<Vec<f64>>, Vec<f64>) {
//    let lines: Vec<&str> = matrix_str
//        .lines()
//        .filter(|l| !l.trim().is_empty())
//        .collect();
//    let mut matrix: Vec<Vec<f64>> = lines
//        .par_iter()
//        .map(|line| {
//            let clean_line = line.replace(['[', ']'], "");
//            clean_line
//                .split_whitespace()
//                .filter_map(|s| s.parse::<f64>().ok())
//                .collect()
//        })
//        .collect();
//
//    let m = matrix.len();
//    let mut row_log_scales = vec![0.0; m];
//
//    // 按行缩放：除以绝对值最大值，保证所有数在 [-1, 1] 之间，绝对不会溢出
//    for i in 0..m {
//        let max_val = matrix[i].iter().map(|x| x.abs()).fold(0.0, f64::max);
//        if max_val > 0.0 {
//            row_log_scales[i] = max_val.ln();
//            for x in &mut matrix[i] {
//                *x /= max_val;
//            }
//        }
//    }
//    (matrix, row_log_scales)
//}
//
//#[pyfunction]
//fn run_reduction_rust(
//    py: Python,
//    matrix_str: String,
//    method: String,
//    param: i32,
//) -> PyResult<PyObject> {
//    let result = ffi::run_reduction_core(matrix_str, method, param);
//    let n = result.rows as usize;
//
//    let (matrix, row_log_scales) = parse_and_scale_matrix(&result.matrix_str);
//
//    let mut log_prod = 0.0;
//    let mut min_log_norm = f64::INFINITY;
//    let mut scaled_norms = vec![0.0; n];
//
//    // 计算 Log Prod，利用对数定律完美还原缩放量
//    for i in 0..n {
//        let sq_norm: f64 = matrix[i].iter().map(|&x| x * x).sum();
//        scaled_norms[i] = sq_norm.sqrt();
//        let true_log_norm = if sq_norm > 1e-300 {
//            0.5 * sq_norm.ln() + row_log_scales[i]
//        } else {
//            -690.0
//        };
//
//        log_prod += true_log_norm;
//        if true_log_norm < min_log_norm {
//            min_log_norm = true_log_norm;
//        }
//    }
//
//    // Cosine 不受标量缩放影响，直接用缩放后的计算，极度安全
//    let cos_flat: Vec<f64> = (0..n)
//        .into_par_iter()
//        .flat_map(|i| {
//            let mut row_cos = vec![0.0; n];
//            for j in 0..i {
//                let dot: f64 = matrix[i].iter().zip(&matrix[j]).map(|(a, b)| a * b).sum();
//                let val = dot / (scaled_norms[i] * scaled_norms[j] + 1e-20);
//                row_cos[j] = val.abs();
//            }
//            row_cos
//        })
//        .collect();
//
//    let array_1d = PyArray1::from_vec(py, cos_flat);
//    let cos_matrix_2d = array_1d.reshape((n, n))?;
//
//    let dict = PyDict::new(py);
//    dict.set_item("matrix_str", result.matrix_str)?;
//    dict.set_item("log_prod", log_prod)?;
//    // 返回真实的 min_norm，如果特别大就是 inf，但不会影响 log_prod 的正常运转
//    dict.set_item("min_norm", min_log_norm.exp())?;
//    dict.set_item("cos_matrix", cos_matrix_2d)?;
//
//    Ok(dict.into())
//}
//
//#[pyfunction]
//fn evaluate_state_rust(
//    py: Python,
//    matrix_str: String,
//    pos: usize,
//    beta: usize,
//) -> PyResult<PyObject> {
//    let (matrix, row_log_scales) = parse_and_scale_matrix(&matrix_str);
//    let m = matrix.len();
//    let n = if m > 0 { matrix[0].len() } else { 0 };
//
//    if m == 0 {
//        return Ok(PyDict::new(py).into());
//    }
//
//    let mut b_star = vec![vec![0.0; n]; m];
//    let mut gs_log_norms = vec![0.0; m];
//
//    // 全局 Gram-Schmidt
//    for i in 0..m {
//        let mut v = matrix[i].clone();
//        for j in 0..i {
//            let denom: f64 = b_star[j].iter().map(|x| x * x).sum();
//            if denom > 1e-300 {
//                let dot: f64 = v.iter().zip(&b_star[j]).map(|(a, b)| a * b).sum();
//                let mu = dot / denom;
//                for k in 0..n {
//                    v[k] -= mu * b_star[j][k];
//                }
//            }
//        }
//        b_star[i] = v.clone();
//        let norm_sq: f64 = v.iter().map(|x| x * x).sum();
//        gs_log_norms[i] = if norm_sq > 1e-300 {
//            0.5 * norm_sq.ln() + row_log_scales[i]
//        } else {
//            -690.0
//        };
//    }
//
//    // 局部 Defect
//    let end = std::cmp::min(pos + beta, m);
//    let mut local_log_defect = 0.0;
//
//    if pos < end {
//        let sum_log_orig: f64 = (pos..end)
//            .map(|i| {
//                let sq: f64 = matrix[i].iter().map(|x| x * x).sum();
//                if sq > 1e-300 {
//                    0.5 * sq.ln() + row_log_scales[i]
//                } else {
//                    -690.0
//                }
//            })
//            .sum();
//
//        let block_size = end - pos;
//        let mut local_b_star = vec![vec![0.0; n]; block_size];
//        let mut sum_log_gs = 0.0;
//
//        for i in 0..block_size {
//            let mut v = matrix[pos + i].clone();
//            for j in 0..i {
//                let denom: f64 = local_b_star[j].iter().map(|x| x * x).sum();
//                if denom > 1e-300 {
//                    let dot: f64 = v.iter().zip(&local_b_star[j]).map(|(a, b)| a * b).sum();
//                    let mu = dot / denom;
//                    for k in 0..n {
//                        v[k] -= mu * local_b_star[j][k];
//                    }
//                }
//            }
//            local_b_star[i] = v.clone();
//            let norm_sq: f64 = v.iter().map(|x| x * x).sum();
//            sum_log_gs += if norm_sq > 1e-300 {
//                0.5 * norm_sq.ln() + row_log_scales[pos + i]
//            } else {
//                -690.0
//            };
//        }
//        local_log_defect = sum_log_orig - sum_log_gs;
//    }
//
//    let gs_array = PyArray1::from_vec(py, gs_log_norms);
//    let dict = PyDict::new(py);
//    dict.set_item("gs_log_norms", gs_array)?;
//    dict.set_item("local_log_defect", local_log_defect)?;
//
//    Ok(dict.into())
//}
//
//#[pymodule]
//fn my_project_backend(m: &Bound<'_, PyModule>) -> PyResult<()> {
//    m.add_function(wrap_pyfunction!(run_reduction_rust, m)?)?;
//    m.add_function(wrap_pyfunction!(evaluate_state_rust, m)?)?;
//    Ok(())
//}
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

#[cxx::bridge]
mod ffi {
    struct ReductionResult {
        matrix_str: String,
        rows: i32,
        cols: i32,
        flat_matrix: Vec<f64>,
        row_log_scales: Vec<f64>,
    }
    unsafe extern "C++" {
        include!("src/bridge.h");
        // 【修改】：增加 pos 参数
        fn run_reduction_core(
            matrix_str: String,
            method: String,
            param: i32,
            pos: i32,
        ) -> ReductionResult;
    }
}
/// 1. 解析并缩放矩阵 (防溢出核心)
pub fn parse_and_scale_matrix(matrix_str: &str) -> (Vec<Vec<f64>>, Vec<f64>) {
    let lines: Vec<&str> = matrix_str
        .lines()
        .filter(|l| !l.trim().is_empty())
        .collect();

    let mut matrix: Vec<Vec<f64>> = lines
        .par_iter()
        .map(|line| {
            let clean_line = line.replace(['[', ']'], "");
            clean_line
                .split_whitespace()
                .filter_map(|s| s.parse::<f64>().ok())
                .collect()
        })
        .collect();

    let m = matrix.len();
    let mut row_log_scales = vec![0.0; m];

    for i in 0..m {
        let max_val = matrix[i].iter().map(|x| x.abs()).fold(0.0, f64::max);
        if max_val > 0.0 {
            row_log_scales[i] = max_val.ln();
            for x in &mut matrix[i] {
                *x /= max_val;
            }
        }
    }
    (matrix, row_log_scales)
}

/// 2. 计算缩放后的范数、真实的 Log Prod 和 Min Log Norm
pub fn compute_norms_and_metrics(
    matrix: &[Vec<f64>],
    row_log_scales: &[f64],
) -> (Vec<f64>, f64, f64) {
    let n = matrix.len();
    let mut log_prod = 0.0;
    let mut min_log_norm = f64::INFINITY;
    let mut scaled_norms = vec![0.0; n];

    for i in 0..n {
        let sq_norm: f64 = matrix[i].iter().map(|&x| x * x).sum();
        scaled_norms[i] = sq_norm.sqrt();

        let true_log_norm = if sq_norm > 1e-300 {
            0.5 * sq_norm.ln() + row_log_scales[i]
        } else {
            -690.0
        };

        log_prod += true_log_norm;
        if true_log_norm < min_log_norm {
            min_log_norm = true_log_norm;
        }
    }
    (scaled_norms, log_prod, min_log_norm)
}

/// 3. 计算余弦相似度矩阵 (一维展平格式，利用多线程)
pub fn compute_cosine_matrix(matrix: &[Vec<f64>], scaled_norms: &[f64]) -> Vec<f64> {
    let n = matrix.len();
    (0..n)
        .into_par_iter()
        .flat_map(|i| {
            let mut row_cos = vec![0.0; n];
            for j in 0..i {
                let dot: f64 = matrix[i].iter().zip(&matrix[j]).map(|(a, b)| a * b).sum();
                let val = dot / (scaled_norms[i] * scaled_norms[j] + 1e-20);
                row_cos[j] = val.abs();
            }
            row_cos
        })
        .collect()
}

/// 4. 计算 Gram-Schmidt 正交化 (GSO) 及其真实的 Log Norms
pub fn compute_gram_schmidt(
    matrix: &[Vec<f64>],
    row_log_scales: &[f64],
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let m = matrix.len();
    let n = if m > 0 { matrix[0].len() } else { 0 };

    let mut b_star = vec![vec![0.0; n]; m];
    let mut gs_log_norms = vec![0.0; m];

    for i in 0..m {
        let mut v = matrix[i].clone();
        for j in 0..i {
            let denom: f64 = b_star[j].iter().map(|x| x * x).sum();
            if denom > 1e-300 {
                let dot: f64 = v.iter().zip(&b_star[j]).map(|(a, b)| a * b).sum();
                let mu = dot / denom;
                for k in 0..n {
                    v[k] -= mu * b_star[j][k];
                }
            }
        }
        b_star[i] = v.clone();
        let norm_sq: f64 = v.iter().map(|x| x * x).sum();
        gs_log_norms[i] = if norm_sq > 1e-300 {
            0.5 * norm_sq.ln() + row_log_scales[i]
        } else {
            -690.0
        };
    }
    (b_star, gs_log_norms)
}

/// 5. 计算局部正交缺陷 (Local Defect)
pub fn compute_local_defect(
    matrix: &[Vec<f64>],
    row_log_scales: &[f64],
    pos: usize,
    beta: usize,
) -> f64 {
    let m = matrix.len();
    let n = if m > 0 { matrix[0].len() } else { 0 };
    let end = std::cmp::min(pos + beta, m);

    if pos >= end {
        return 0.0;
    }

    let sum_log_orig: f64 = (pos..end)
        .map(|i| {
            let sq: f64 = matrix[i].iter().map(|x| x * x).sum();
            if sq > 1e-300 {
                0.5 * sq.ln() + row_log_scales[i]
            } else {
                -690.0
            }
        })
        .sum();

    let block_size = end - pos;
    let mut local_b_star = vec![vec![0.0; n]; block_size];
    let mut sum_log_gs = 0.0;

    for i in 0..block_size {
        let mut v = matrix[pos + i].clone();
        for j in 0..i {
            let denom: f64 = local_b_star[j].iter().map(|x| x * x).sum();
            if denom > 1e-300 {
                let dot: f64 = v.iter().zip(&local_b_star[j]).map(|(a, b)| a * b).sum();
                let mu = dot / denom;
                for k in 0..n {
                    v[k] -= mu * local_b_star[j][k];
                }
            }
        }
        local_b_star[i] = v.clone();
        let norm_sq: f64 = v.iter().map(|x| x * x).sum();
        sum_log_gs += if norm_sq > 1e-300 {
            0.5 * norm_sq.ln() + row_log_scales[pos + i]
        } else {
            -690.0
        };
    }

    sum_log_orig - sum_log_gs
}

// =====================================================================
// 第二层：PyO3 接口层 (负责调用底层函数并打包给 Python)
// =====================================================================
#[pyfunction]
fn run_reduction_rust(
    py: Python,
    matrix_str: String,
    method: String,
    param: i32,
    pos: i32, // 【修改】：接收 Python 传来的 pos
) -> PyResult<PyObject> {
    // 传入 pos 给 C++
    let result = ffi::run_reduction_core(matrix_str, method, param, pos);

    let n = result.rows as usize;
    let cols = result.cols as usize;

    let mut matrix = Vec::with_capacity(n);
    let mut idx = 0;
    for _ in 0..n {
        let mut row = Vec::with_capacity(cols);
        for _ in 0..cols {
            row.push(result.flat_matrix[idx]);
            idx += 1;
        }
        matrix.push(row);
    }
    let row_log_scales = result.row_log_scales;

    let (scaled_norms, log_prod, min_log_norm) =
        compute_norms_and_metrics(&matrix, &row_log_scales);
    let cos_flat = compute_cosine_matrix(&matrix, &scaled_norms);

    let array_1d = PyArray1::from_vec(py, cos_flat);
    let cos_matrix_2d = array_1d.reshape((n, n))?;

    let dict = PyDict::new(py);
    dict.set_item("matrix_str", result.matrix_str)?;
    dict.set_item("log_prod", log_prod)?;
    dict.set_item("min_norm", min_log_norm.exp())?;
    dict.set_item("cos_matrix", cos_matrix_2d)?;

    Ok(dict.into())
}

#[pyfunction]
fn evaluate_state_rust(
    py: Python,
    matrix_str: String,
    pos: usize,
    beta: usize,
) -> PyResult<PyObject> {
    // 1. 解析
    let (matrix, row_log_scales) = parse_and_scale_matrix(&matrix_str);

    if matrix.is_empty() {
        return Ok(PyDict::new(py).into());
    }

    // 2. 调用纯 Rust 函数计算 GSO 和 Defect
    let (_, gs_log_norms) = compute_gram_schmidt(&matrix, &row_log_scales);
    let local_log_defect = compute_local_defect(&matrix, &row_log_scales, pos, beta);

    // 3. 转换为 Python 对象
    let gs_array = PyArray1::from_vec(py, gs_log_norms);
    let dict = PyDict::new(py);
    dict.set_item("gs_log_norms", gs_array)?;
    dict.set_item("local_log_defect", local_log_defect)?;

    Ok(dict.into())
}

// =====================================================================
// 附加：如果你想在 Python 中单独调用这些拆分出来的纯数学功能，可以暴露它们
// =====================================================================

#[pyfunction]
fn compute_cosine_only_rust(py: Python, matrix_str: String) -> PyResult<PyObject> {
    let (matrix, row_log_scales) = parse_and_scale_matrix(&matrix_str);
    let (scaled_norms, _, _) = compute_norms_and_metrics(&matrix, &row_log_scales);
    let cos_flat = compute_cosine_matrix(&matrix, &scaled_norms);

    let n = matrix.len();
    let array_1d = PyArray1::from_vec(py, cos_flat);
    let cos_matrix_2d = array_1d.reshape((n, n))?;
    Ok(cos_matrix_2d.into_any().unbind())
}

#[pymodule]
fn my_project_backend(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // 原有的核心接口
    m.add_function(wrap_pyfunction!(run_reduction_rust, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_state_rust, m)?)?;

    // 暴露单独的工具函数给 Python (可选)
    m.add_function(wrap_pyfunction!(compute_cosine_only_rust, m)?)?;

    Ok(())
}
