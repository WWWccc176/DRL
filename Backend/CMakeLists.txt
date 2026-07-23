cmake_minimum_required(VERSION 3.24)

project(my_project_backend LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CUDA 13.3 uses GCC 15 as nvcc host compiler on this machine, while the
# standalone C++ compiler is GCC 16. Disable IPO/LTO so GCC-version-specific
# LTO bytecode is never mixed during CUDA device-link / final shared-module link.
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)

set(LATTICE_GPU_NUM 1 CACHE STRING "Number of local CUDA GPUs used by BGJ engine")
set(LATTICE_CUDA_ARCH native CACHE STRING "CUDA architecture; use native on the build machine")
set(LATTICE_PWC_DRAM_GB 3 CACHE STRING "Pool cache DRAM budget in GiB")
set(LATTICE_BWC_DRAM_GB 6 CACHE STRING "Bucket cache DRAM budget in GiB")
set(LATTICE_SWC_DRAM_GB 2 CACHE STRING "Solution cache DRAM budget in GiB")
set(LATTICE_UT_TABLE_DRAM_GB 6 CACHE STRING "UID table DRAM budget in GiB")
set(LATTICE_UT_BUFFER_DRAM_GB 3 CACHE STRING "UID buffer DRAM budget in GiB")
option(LATTICE_ENABLE_NUMA "Enable NUMA interleaving when libnuma is available" ON)


# ============================================================
# Python / pybind11
# ============================================================

find_package(
    Python3
    COMPONENTS Interpreter Development
    REQUIRED
)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_RESULT
)

if(NOT PYBIND11_RESULT EQUAL 0)
    message(
        FATAL_ERROR
        "Cannot find pybind11 from Python: ${Python3_EXECUTABLE}"
    )
endif()

set(
    pybind11_DIR
    "${PYBIND11_CMAKE_DIR}"
    CACHE PATH
    "pybind11 CMake directory"
    FORCE
)

message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
message(STATUS "pybind11_DIR: ${pybind11_DIR}")

find_package(pybind11 CONFIG REQUIRED)


# ============================================================
# OpenMP / CUDA / Threads
# ============================================================

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)


# ============================================================
# Native libraries
# ============================================================

find_library(FPLLL_LIBRARY NAMES fplll)
if(NOT FPLLL_LIBRARY)
    message(FATAL_ERROR "Could not find fplll library")
endif()

find_library(NTL_LIBRARY NAMES ntl)
if(NOT NTL_LIBRARY)
    message(FATAL_ERROR "Could not find NTL library")
endif()

find_library(GMP_LIBRARY NAMES gmp)
if(NOT GMP_LIBRARY)
    message(FATAL_ERROR "Could not find GMP library")
endif()

find_library(GMPXX_LIBRARY NAMES gmpxx)
if(NOT GMPXX_LIBRARY)
    message(FATAL_ERROR "Could not find GMP C++ library (gmpxx)")
endif()

set(LATTICE_USE_NUMA 0)
if(LATTICE_ENABLE_NUMA)
    find_path(NUMA_INCLUDE_DIR NAMES numa.h)
    find_library(NUMA_LIBRARY NAMES numa)
    if(NUMA_INCLUDE_DIR AND NUMA_LIBRARY)
        set(LATTICE_USE_NUMA 1)
        message(STATUS "NUMA support: enabled")
    else()
        message(WARNING "libnuma was not found; building with NUMA interleaving disabled")
    endif()
else()
    message(STATUS "NUMA support: disabled by configuration")
endif()

message(STATUS "FPLLL_LIBRARY: ${FPLLL_LIBRARY}")
message(STATUS "NTL_LIBRARY: ${NTL_LIBRARY}")
message(STATUS "GMP_LIBRARY: ${GMP_LIBRARY}")
message(STATUS "GMPXX_LIBRARY: ${GMPXX_LIBRARY}")
if(LATTICE_USE_NUMA)
    message(STATUS "NUMA_INCLUDE_DIR: ${NUMA_INCLUDE_DIR}")
    message(STATUS "NUMA_LIBRARY: ${NUMA_LIBRARY}")
endif()


# ============================================================
# Host SIMD detection
#
# AVX-512 path requires all subsets used by this host code:
#   AVX512F + AVX512BW + AVX512DQ + AVX512VL
#
# Detection checks BOTH:
#   1. compiler flag support
#   2. the CPU visible to the build process
#
# If all are available, compile the dedicated AVX-512 sources.
# Otherwise compile the native fallback sources with -march=native.
# ============================================================

include(CheckCXXCompilerFlag)
include(CheckCXXSourceRuns)

check_cxx_compiler_flag("-mavx512f"  COMPILER_HAS_AVX512F)
check_cxx_compiler_flag("-mavx512bw" COMPILER_HAS_AVX512BW)
check_cxx_compiler_flag("-mavx512dq" COMPILER_HAS_AVX512DQ)
check_cxx_compiler_flag("-mavx512vl" COMPILER_HAS_AVX512VL)

set(CPU_HAS_REQUIRED_AVX512 FALSE)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT CMAKE_CROSSCOMPILING)
    check_cxx_source_runs(
        "
        int main() {
        #if defined(__GNUC__) || defined(__clang__)
            __builtin_cpu_init();
            const bool ok =
                __builtin_cpu_supports(\"avx512f\")  &&
                __builtin_cpu_supports(\"avx512bw\") &&
                __builtin_cpu_supports(\"avx512dq\") &&
                __builtin_cpu_supports(\"avx512vl\");
            return ok ? 0 : 1;
        #else
            return 1;
        #endif
        }
        "
        CPU_HAS_REQUIRED_AVX512
    )
endif()

if(
    CPU_HAS_REQUIRED_AVX512
    AND COMPILER_HAS_AVX512F
    AND COMPILER_HAS_AVX512BW
    AND COMPILER_HAS_AVX512DQ
    AND COMPILER_HAS_AVX512VL
)
    set(LATTICE_USE_AVX512 TRUE)
else()
    set(LATTICE_USE_AVX512 FALSE)
endif()

if(LATTICE_USE_AVX512)
    set(SIEVE_VEC_SOURCE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/sieve_engine/src/vec_avx512.cpp"
    )
    set(SIEVE_POOL_SOURCE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/sieve_engine/src/pool_hd_avx512.cpp"
    )

    message(STATUS "Host SIMD path: AVX-512 optimized")
    message(STATUS "  vec source:      vec_avx512.cpp")
    message(STATUS "  pool_hd source:  pool_hd_avx512.cpp")
else()
    set(SIEVE_VEC_SOURCE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/sieve_engine/src/vec_native.cpp"
    )
    set(SIEVE_POOL_SOURCE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/sieve_engine/src/pool_hd_native.cpp"
    )

    message(STATUS "Host SIMD path: native fallback (-march=native)")
    message(STATUS "  vec source:      vec_native.cpp")
    message(STATUS "  pool_hd source:  pool_hd_native.cpp")
endif()


# ============================================================
# Local BGJ / DH sieve engine
# ============================================================

file(
    GLOB
    SIEVE_CPP
    CONFIGURE_DEPENDS
    src/sieve_engine/src/*.cpp
)

file(
    GLOB
    SIEVE_CUDA
    CONFIGURE_DEPENDS
    src/sieve_engine/src/*.cu
)

# Never compile the original and both variants together: they export the same
# symbols. Remove all six candidates from the glob, then append exactly one
# vec implementation and one pool_hd implementation selected above.
list(
    FILTER SIEVE_CPP
    EXCLUDE REGEX
    ".*/(vec|vec_native|vec_avx512|pool_hd|pool_hd_native|pool_hd_avx512)\\.cpp$"
)

list(
    APPEND SIEVE_CPP
    "${SIEVE_VEC_SOURCE}"
    "${SIEVE_POOL_SOURCE}"
)

add_library(
    local_sieve_engine
    STATIC
    ${SIEVE_CPP}
    ${SIEVE_CUDA}
)

target_include_directories(
    local_sieve_engine
    PUBLIC
        src/sieve_engine/include
        src/sieve_engine/dep
        $<$<BOOL:${LATTICE_USE_NUMA}>:${NUMA_INCLUDE_DIR}>
)

target_compile_definitions(
    local_sieve_engine
    PUBLIC
        LATTICE_GPU_NUM=${LATTICE_GPU_NUM}
        LATTICE_PWC_DRAM_GB=${LATTICE_PWC_DRAM_GB}
        LATTICE_BWC_DRAM_GB=${LATTICE_BWC_DRAM_GB}
        LATTICE_SWC_DRAM_GB=${LATTICE_SWC_DRAM_GB}
        LATTICE_UT_TABLE_DRAM_GB=${LATTICE_UT_TABLE_DRAM_GB}
        LATTICE_UT_BUFFER_DRAM_GB=${LATTICE_UT_BUFFER_DRAM_GB}
        LATTICE_USE_NUMA=${LATTICE_USE_NUMA}
)

target_compile_options(
    local_sieve_engine
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-march=native>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
        $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)

if(LATTICE_USE_AVX512)
    target_compile_options(
        local_sieve_engine
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-mavx512f>
            $<$<COMPILE_LANGUAGE:CXX>:-mavx512bw>
            $<$<COMPILE_LANGUAGE:CXX>:-mavx512dq>
            $<$<COMPILE_LANGUAGE:CXX>:-mavx512vl>
    )
endif()

set_target_properties(
    local_sieve_engine
    PROPERTIES
        CUDA_ARCHITECTURES "${LATTICE_CUDA_ARCH}"
        CUDA_SEPARABLE_COMPILATION ON
        INTERPROCEDURAL_OPTIMIZATION OFF
)

target_link_libraries(
    local_sieve_engine
    PUBLIC
        ${NTL_LIBRARY}
        ${GMPXX_LIBRARY}
        ${GMP_LIBRARY}
        $<$<BOOL:${LATTICE_USE_NUMA}>:${NUMA_LIBRARY}>
        OpenMP::OpenMP_CXX
        Threads::Threads
        CUDA::cudart
)


# ============================================================
# Python backend module
# ============================================================

pybind11_add_module(
    my_project_backend
    NO_EXTRAS
    src/backend/lattice_backend.cpp
    src/backend/matrix_utils.cpp
    src/backend/bkz2_engine.cpp
    src/backend/enumeration_engine.cpp
    src/backend/extreme_reducer.cpp
    src/backend/sieve_bridge.cpp
    src/cuda/lattice_cuda.cu
)

target_include_directories(
    my_project_backend
    PRIVATE
        src/backend
        src/cuda
        src/sieve_engine/include
        src/sieve_engine/dep
        $<$<BOOL:${LATTICE_USE_NUMA}>:${NUMA_INCLUDE_DIR}>
)

target_compile_definitions(
    my_project_backend
    PRIVATE
        USE_CUDA=1
)

target_compile_options(
    my_project_backend
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-march=native>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

if(LATTICE_USE_AVX512)
    target_compile_options(
        my_project_backend
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-mavx512f>
            $<$<COMPILE_LANGUAGE:CXX>:-mavx512bw>
            $<$<COMPILE_LANGUAGE:CXX>:-mavx512dq>
            $<$<COMPILE_LANGUAGE:CXX>:-mavx512vl>
    )
endif()

set_target_properties(
    my_project_backend
    PROPERTIES
        CUDA_ARCHITECTURES "${LATTICE_CUDA_ARCH}"
        CUDA_SEPARABLE_COMPILATION ON
        INTERPROCEDURAL_OPTIMIZATION OFF
)

target_link_libraries(
    my_project_backend
    PRIVATE
        local_sieve_engine
        ${FPLLL_LIBRARY}
        ${NTL_LIBRARY}
        ${GMPXX_LIBRARY}
        ${GMP_LIBRARY}
        $<$<BOOL:${LATTICE_USE_NUMA}>:${NUMA_LIBRARY}>
        OpenMP::OpenMP_CXX
        Threads::Threads
        CUDA::cudart
)
