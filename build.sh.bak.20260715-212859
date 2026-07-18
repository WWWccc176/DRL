#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/amax/projects/DRL"
SRC_DIR="$PROJECT_ROOT/src"

PYTHON="${PYTHON:-/home/amax/.conda/envs/drl_env/bin/python3}"
CXX="${CXX:-g++}"

EXT_SUFFIX="$("$PYTHON" -c \
    'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"

MODULE_NAME="my_project_backend${EXT_SUFFIX}"
FINAL_MODULE="$PROJECT_ROOT/$MODULE_NAME"

BUILD_DIR="$PROJECT_ROOT/.backend_build"
TMP_MODULE="$BUILD_DIR/$MODULE_NAME"

COMMON_FLAGS=(
    -O3
    -std=c++17
    -fPIC
    -funroll-loops
    -fomit-frame-pointer
)

echo ">> Python: $PYTHON"
"$PYTHON" -V

if [[ ! -f "$SRC_DIR/lattice_backend.cpp" ]]; then
    echo "ERROR: missing $SRC_DIR/lattice_backend.cpp" >&2
    exit 1
fi

PYBIND_INCLUDES="$("$PYTHON" -m pybind11 --includes)"
FPLLL_CFLAGS="$(pkg-config --cflags fplll)"
FPLLL_LIBS="$(pkg-config --libs fplll)"

mkdir -p "$BUILD_DIR"

echo ">> 停止残留训练/测试进程"
pkill -f '/scripts/agent/a10.py' 2>/dev/null || true
pkill -f 'poison_test.py' 2>/dev/null || true

echo ">> 删除所有旧 my_project_backend 扩展"

SITE_PACKAGES="$("$PYTHON" - <<'PY'
import site
paths = site.getsitepackages()
print(paths[0] if paths else "")
PY
)"

SEARCH_ROOTS=("$PROJECT_ROOT")

if [[ -n "$SITE_PACKAGES" && -d "$SITE_PACKAGES" ]]; then
    SEARCH_ROOTS+=("$SITE_PACKAGES")
fi

for root in "${SEARCH_ROOTS[@]}"; do
    find "$root" -type f \
        \( -name 'my_project_backend*.so' \
        -o -name 'my_project_backend*.pyd' \) \
        -delete

    find "$root" -type d \
        -name 'my_project_backend' \
        -prune \
        -exec rm -rf {} +
done

rm -f \
    "$BUILD_DIR/lattice_backend.o" \
    "$BUILD_DIR/lattice_cuda.o" \
    "$TMP_MODULE"

if command -v nvcc >/dev/null 2>&1; then
    echo ">> 检测到 CUDA，编译 GPU 版本"

    CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"

    nvcc \
        -O3 \
        -std=c++17 \
        -Xcompiler=-fPIC \
        -c "$SRC_DIR/lattice_cuda.cu" \
        -o "$BUILD_DIR/lattice_cuda.o"

    # shellcheck disable=SC2086
    "$CXX" \
        "${COMMON_FLAGS[@]}" \
        -DUSE_CUDA \
        $PYBIND_INCLUDES \
        -I"$CUDA_HOME/include" \
        $FPLLL_CFLAGS \
        -c "$SRC_DIR/lattice_backend.cpp" \
        -o "$BUILD_DIR/lattice_backend.o"

    # shellcheck disable=SC2086
    "$CXX" \
        -shared \
        "$BUILD_DIR/lattice_backend.o" \
        "$BUILD_DIR/lattice_cuda.o" \
        -o "$TMP_MODULE" \
        $FPLLL_LIBS \
        -lgmp \
        -lmpfr \
        -L"$CUDA_HOME/lib64" \
        -Wl,-rpath,"$CUDA_HOME/lib64" \
        -lcudart \
        -lpthread
else
    echo ">> 未检测到 nvcc，编译 CPU-only 版本"

    # shellcheck disable=SC2086
    "$CXX" \
        "${COMMON_FLAGS[@]}" \
        $PYBIND_INCLUDES \
        $FPLLL_CFLAGS \
        -c "$SRC_DIR/lattice_backend.cpp" \
        -o "$BUILD_DIR/lattice_backend.o"

    # shellcheck disable=SC2086
    "$CXX" \
        -shared \
        "$BUILD_DIR/lattice_backend.o" \
        -o "$TMP_MODULE" \
        $FPLLL_LIBS \
        -lgmp \
        -lmpfr \
        -lpthread
fi

echo ">> 原子安装唯一后端"
mv "$TMP_MODULE" "$FINAL_MODULE"
chmod 755 "$FINAL_MODULE"

rm -f \
    "$BUILD_DIR/lattice_backend.o" \
    "$BUILD_DIR/lattice_cuda.o"

echo ">> 验证文件"
ls -lh "$FINAL_MODULE"

echo ">> 验证 Python 实际加载位置"
cd "$PROJECT_ROOT"

PYTHONPATH="$PROJECT_ROOT" "$PYTHON" - <<'PY'
import os
import my_project_backend as backend

expected_root = "/home/amax/projects/DRL"
loaded = os.path.realpath(backend.__file__)

print("loaded:", loaded)

if os.path.dirname(loaded) != expected_root:
    raise RuntimeError(
        "加载位置错误，期望项目根目录，实际为：" + loaded
    )

required = {
    "create_matrix",
    "create_matrix_lll",
    "clone_matrix",
    "free_matrix",
    "reduce",
    "evaluate_matrix",
    "dump_matrix",
    "dump_block",
    "insert_coeff_vector",
    "strategies_info",
}

missing = sorted(name for name in required if not hasattr(backend, name))

if missing:
    raise RuntimeError(
        "新后端缺少接口：" + ", ".join(missing)
    )

print("strategies:", backend.strategies_info())
print("backend verification: OK")
PY

echo ">> 检查是否还存在其他副本"

for root in "${SEARCH_ROOTS[@]}"; do
    find "$root" -type f \
        \( -name 'my_project_backend*.so' \
        -o -name 'my_project_backend*.pyd' \) \
        -print
done

echo ">> 构建完成：$FINAL_MODULE"  echo ">> 未检测到 nvcc, 编译 CPU-only 版本"
  $CXX $COMMON $PYINC $FPLLL_CFLAGS \
    -c "$SRC_DIR/lattice_backend.cpp" -o lattice_backend.o
  $CXX -shared lattice_backend.o -o "$OUT_MODULE" \
    $FPLLL_LIBS -lgmp -lmpfr -lpthread
fi

rm -f lattice_backend.o lattice_cuda.o
echo ">> 已生成 $OUT_MODULE"
#bash build.sh 直接敲这行
