cmake_minimum_required(VERSION 3.10)
project(drl_project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 开启最高优化，这对 fplll 速度至关重要
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# 1. 查找 Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 2. 查找 Pybind11
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED)

# 3. 查找 FPLLL
find_package(PkgConfig REQUIRED)
pkg_check_modules(FPLLL REQUIRED fplll)

# 4. 定义模块
# 注意：源文件路径根据你的截图应为 src/fplll.cpp
add_library(my_cpp_module MODULE src/fplll.cpp)

# 5. 链接
target_link_libraries(my_cpp_module PRIVATE pybind11::module ${FPLLL_LIBRARIES})
target_include_directories(my_cpp_module PRIVATE ${FPLLL_INCLUDE_DIRS})

# 6. 输出路径设置到 drl_app 文件夹
set_target_properties(my_cpp_module PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/drl_app"
    PREFIX "" 
)
