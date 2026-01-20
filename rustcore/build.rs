// rust_core/build.rs
//fn main() {
//    cxx_build::bridge("src/lib.rs") // Rust 入口
//        .file("../src/your_cpp_file.cpp") // 指向你根目录 src 下的 C++ 文件
//        .flag_if_supported("-std=c++17")
//        .compile("my-cpp-lib"); // 编译成静态库
//
//    println!("cargo:rerun-if-changed=src/lib.rs");
//    println!("cargo:rerun-if-changed=../src/your_cpp_file.cpp");
//    println!("cargo:rerun-if-changed=../src/your_header.h");
//}
fn main() {
    // 1. 查找系统中的 fplll 库 (包括 gmp, mpfr 等依赖)
    let fplll = pkg_config::Config::new()
        .probe("fplll")
        .expect("Unable to find fplll library. Make sure it is installed.");

    // 2. 配置 C++ 构建
    let mut build = cxx_build::bridge("src/lib.rs");

    build
        .file("../src/bridge.cpp") // 编译我们刚写的纯 C++ 文件
        .include("..")
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-O3"); // 开启优化

    // 3. 将 fplll 的头文件路径加进去
    for path in &fplll.include_paths {
        build.include(path);
    }

    build.compile("my-cpp-lib");

    // 4. 重新运行触发器
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=../src/bridge.cpp");
    println!("cargo:rerun-if-changed=../src/bridge.h");
}
